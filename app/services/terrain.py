"""Terrain generation service from elevation & vector data.

Pipeline:
    elevation grid -> height scaling -> optional road/building/contour relief
    -> unified heightfield -> closed manifold solid -> binary STL

Coordinate system: WGS84 input -> local metric grid in metres -> scaled mm model.

The heightfield is extruded downwards to z=0 and capped, producing a closed,
watertight, 2-manifold solid that slices cleanly. Roads, buildings and contour
lines are folded into the same heightfield (see :mod:`app.services.features`)
so they can never introduce non-manifold self-intersections the way separate
unioned solids would.
"""

import logging
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TerrainSettings:
    """Configuration for terrain model generation."""

    # Geographic bounds (WGS84 decimal degrees) -- set when area selected on map
    west: float = -3.05  # min lon
    south: float = 12.69  # min lat
    east: float = -2.95  # max lon
    north: float = 12.79  # max lat

    # Physical model dimensions in millimeters
    model_width_mm: float = 300  # X dimension
    model_depth_mm: float = 300  # Y dimension

    # Model scaling
    resolution_m: float = 30.0  # elevation grid cell size in metres

    # Terrain height controls
    min_altitude_mm: float = 2.0  # thinnest terrain layer (always printable)
    max_altitude_mm: float = 50.0  # tallest terrain relief
    vertical_exaggeration: float = 1.0  # multiplier on true-scale relief
    elevation_offset_mm: float = 1.0  # lift of the terrain above the base plate

    # Base plate settings
    base_thickness_mm: float = 3.0  # thickness of bottom plate
    base_extension_mm: float = 0.0  # overhang around edges

    # Smoothing
    smoothing_passes: int = 2  # binomial blur iterations on elevation grid

    # Feature toggles
    include_roads: bool = False
    include_buildings: bool = False
    include_contours: bool = False
    road_height_mm: float = 1.2
    road_width_mm: float = 1.6
    road_min_width_mm: float = 0.8
    building_height_mm: float = 6.0
    building_min_height_mm: float = 3.0
    building_footprint_scale: float = 1.0
    contour_interval_m: float = 50.0
    contour_major_interval_m: float = 250.0
    contour_thickness_mm: float = 0.6
    contour_height_mm: float = 0.8
    contours_engraved: bool = False

    # Limits
    max_model_width_mm: float = 600
    max_model_depth_mm: float = 600
    min_resolution_m: float = 10
    max_resolution_m: float = 128

    # Grid caps (protect against pathological meshes)
    max_grid_nodes: int = 320
    min_grid_nodes: int = 16

    @property
    def width(self) -> float:
        """West-east geographic span in WGS84 degrees."""
        return self.east - self.west

    @property
    def height(self) -> float:
        """South-north geographic span in WGS84 degrees."""
        return self.north - self.south


class TerrainGenerationError(Exception):
    """Raised when terrain processing fails.

    Error types:
      elevation_fetch  -- elevation data unavailable for bounds
      mesh_generation  -- mesh geometry invalid (holes, non-manifold)
      output_validation -- output file check failed
    """


def validate_settings(settings: TerrainSettings) -> list[str]:
    """Return a list of human-readable problems with ``settings``.

    An empty list means the configuration is safe to generate.
    """
    problems: list[str] = []

    if settings.model_width_mm <= 0 or settings.model_depth_mm <= 0:
        problems.append("Model width and depth must be greater than zero.")

    if settings.max_altitude_mm <= settings.min_altitude_mm:
        problems.append(
            "Maximum terrain height must exceed the minimum terrain thickness."
        )

    if settings.base_thickness_mm < 0:
        problems.append("Base thickness cannot be negative.")

    if settings.vertical_exaggeration <= 0:
        problems.append("Vertical exaggeration must be greater than zero.")

    if settings.base_thickness_mm + settings.min_altitude_mm <= 0:
        problems.append(
            "Base thickness plus minimum terrain thickness leaves no printable material."
        )

    if settings.width <= 0 or settings.height <= 0:
        problems.append("Selected area has zero width or height.")

    if settings.include_contours and settings.contour_interval_m <= 0:
        problems.append("Contour interval must be greater than zero metres.")

    if settings.include_roads and settings.road_min_width_mm < 0.4:
        problems.append("Road width below 0.4 mm is unlikely to print.")

    return problems


def _grid_shape(settings: TerrainSettings, width_m: float, height_m: float) -> tuple[int, int]:
    """Rows × columns for the model grid at the requested ground resolution."""
    cols = int(round(width_m / max(settings.resolution_m, 1e-6)))
    rows = int(round(height_m / max(settings.resolution_m, 1e-6)))

    # Scale the whole grid down if it exceeds the node cap.
    limit = settings.max_grid_nodes
    if cols > limit or rows > limit:
        shrink = max(cols, rows) / limit
        cols = max(settings.min_grid_nodes, int(cols / shrink))
        rows = max(settings.min_grid_nodes, int(rows / shrink))

    return max(2, rows), max(2, cols)


def smooth_grid(grid, passes: int):
    """Apply ``passes`` of a 3×3 binomial filter, preserving NaN voids."""
    if passes <= 0:
        return grid

    kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
    out = grid.astype(np.float64, copy=True)
    valid = np.isfinite(out)

    for _ in range(passes):
        padded = np.pad(np.where(valid, out, 0.0), 1, mode="edge")
        weight = np.pad(valid.astype(np.float64), 1, mode="edge")

        num = sum(
            kernel[r, c] * padded[r : r + out.shape[0], c : c + out.shape[1]]
            for r in range(3)
            for c in range(3)
        )
        den = sum(
            kernel[r, c] * weight[r : r + out.shape[0], c : c + out.shape[1]]
            for r in range(3)
            for c in range(3)
        )
        out = np.divide(num, den, out=np.copy(out), where=den > 1e-9)

    return out


def scale_to_mm(
    elev_m,
    settings: TerrainSettings,
    width_m: float,
    height_m: float,
):
    """Convert metre elevations above the area minimum into model millimetres.

    Uses a single isotropic scale (mm per metre of ground) so the model keeps
    true proportions, then applies vertical exaggeration and clamps the result
    to the printable height window.
    """
    elev = np.nan_to_num(np.asarray(elev_m, dtype=np.float64), nan=0.0)
    floor_m = float(np.min(elev))
    relief_m = elev - floor_m

    # mm per metre of ground, isotropic so relief is not distorted.
    mm_per_m = settings.model_width_mm / max(width_m, 1e-6)
    mm_per_m_y = settings.model_depth_mm / max(height_m, 1e-6)
    mm_per_m = min(mm_per_m, mm_per_m_y)

    scaled = relief_m * mm_per_m * settings.vertical_exaggeration

    lo = settings.min_altitude_mm
    hi = settings.max_altitude_mm
    if hi > lo and float(scaled.max()) > lo:
        # Map the exaggerated relief onto [lo, hi] when it overflows the window,
        # preserving relative shape below the cap.
        peak = float(scaled.max())
        if peak > hi:
            scaled = scaled * (hi - lo) / peak + lo

    return np.clip(scaled, lo, hi)


def _grid_indices(rows: int, cols: int) -> List[Tuple[int, int]]:
    """Boundary vertex loop, wound clockwise as seen from +Z.

    Row index 0 is the southern edge (y grows northward). Clockwise-from-above
    is the reverse of the top surface's own boundary traversal, so skirt
    triangles can share those edges directly while keeping every face wound
    outward and the solid consistently oriented.
    """
    loop: List[Tuple[int, int]] = []
    loop += [(rows - 1, j) for j in range(cols)]                      # north, W→E
    loop += [(i, cols - 1) for i in range(rows - 2, -1, -1)]          # east,  N→S
    loop += [(0, j) for j in range(cols - 2, -1, -1)]                 # south, E→W
    loop += [(i, 0) for i in range(1, rows - 1)]                  # west,  S→N
    return loop


class MeshBuilder:
    """Build a closed triangle mesh from a terrain heightfield.

    Output: ``(vertices, triangles)`` where ``vertices`` is an ``(N, 3)`` array
    of millimetre coordinates and ``triangles`` an ``(M, 3)`` index array with
    outward-facing, counter-clockwise winding.
    """

    def __init__(self, settings: TerrainSettings):
        self.settings = settings

    # -- geometry helpers ---------------------------------------------------

    def surface_of(self, elev_grid) -> "np.ndarray":
        """Convert an elevation grid (metres) to the model's millimetre surface.

        Callers that need to rasterise features must build them against this
        array, because :meth:`build_with_features` folds feature heights into
        exactly this surface. It is deterministic, so recomputing it there
        yields the identical grid.
        """
        from app.utils.projection import bounds_to_meters
        from app.models import GeoBounds

        s = self.settings
        grid = np.asarray(elev_grid, dtype=np.float64)
        if grid.ndim != 2 or min(grid.shape) < 2:
            raise TerrainGenerationError(
                f"Elevation grid must be at least 2×2, received {grid.shape}."
            )

        bounds = GeoBounds(west=s.west, south=s.south, east=s.east, north=s.north)
        width_m, height_m = bounds_to_meters(bounds)

        # Surface height = base plate top + lift + relief.
        relief_mm = scale_to_mm(grid, s, width_m, height_m)
        return s.base_thickness_mm + s.elevation_offset_mm + relief_mm

    def build(self, elev_grid) -> Tuple["np.ndarray", "np.ndarray"]:
        """Build the watertight solid for ``elev_grid`` (metres, south→north)."""
        surface = self.surface_of(elev_grid)
        rows, cols = surface.shape
        return self._extrude(surface, cols, rows)

    def build_with_features(
        self,
        elev_grid,
        *,
        roads: Optional[dict] = None,
        buildings: Optional[dict] = None,
        contours: Optional[dict] = None,
    ):
        """Build the solid after folding optional features into the heightfield.

        ``roads``/``buildings`` carry already-rasterised ``(mask, z)`` pairs in
        millimetres, as produced by :mod:`app.services.features` when they are
        given ``self.surface_of(elev_grid)``; ``contours`` carries a signed
        offset mask in millimetres.
        """
        s = self.settings
        surface = self.surface_of(elev_grid)
        rows, cols = surface.shape

        stats = {"roads": 0, "buildings": 0, "contours": 0}
        floor = s.base_thickness_mm + s.min_altitude_mm

        if roads:
            surface, count = apply_raised(surface, roads, floor)
            stats["roads"] = count
        if buildings:
            surface, count = apply_raised(surface, buildings, floor)
            stats["buildings"] = count
        if contours:
            surface, count = apply_contours(surface, contours, floor)
            stats["contours"] = count

        vertices, triangles = self._extrude(surface, cols, rows)
        return vertices, triangles, surface, stats

    # -- extrusion ----------------------------------------------------------

    def _extrude(self, surface, cols: int, rows: int):
        """Turn a heightfield into a closed solid resting on z=0."""
        s = self.settings

        # Pad outward by the base extension so the skirt is vertical.
        pad_x = s.base_extension_mm
        pad_y = s.base_extension_mm
        x = np.linspace(-pad_x, s.model_width_mm + pad_x, cols)
        y = np.linspace(-pad_y, s.model_depth_mm + pad_y, rows)

        xx, yy = np.meshgrid(x, y)
        # The elevation grid's row 0 is the southern edge, matching y[0].
        zz = np.asarray(surface, dtype=np.float64)

        top_count = rows * cols

        top_verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        bottom_verts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(top_count)])
        vertices = np.vstack([top_verts, bottom_verts]).astype(np.float64)

        # Top surface: two triangles per quad, normals pointing +Z.
        quads = []
        for i in range(rows - 1):
            base = i * cols
            for j in range(cols - 1):
                a = base + j
                b = base + j + 1
                c = (i + 1) * cols + j + 1
                d = (i + 1) * cols + j
                quads.append((a, c, d))
                quads.append((a, b, c))
        top_tris = np.array(quads, dtype=np.int64)

        # Bottom cap: the same triangulation wound the other way, so its
        # boundary runs opposite to the skirt's bottom edges and seals them.
        cap = np.empty_like(top_tris)
        for k in range(0, len(top_tris), 2):
            t1, t2 = top_tris[k], top_tris[k + 1]
            cap[k] = (top_count + t1[0], top_count + t1[2], top_count + t1[1])
            cap[k + 1] = (top_count + t2[0], top_count + t2[2], top_count + t2[1])
        bottom_tris = cap

        # Skirt: one quad per boundary edge, extruded straight down to z=0.
        loop = _grid_indices(rows, cols)
        skirt = []
        n = len(loop)
        for k in range(n):
            a = loop[k][0] * cols + loop[k][1]
            b = loop[(k + 1) % n][0] * cols + loop[(k + 1) % n][1]
            skirt.append((a, b, top_count + b))
            skirt.append((a, top_count + b, top_count + a))
        skirt_tris = np.array(skirt, dtype=np.int64)

        triangles = np.vstack([top_tris, bottom_tris, skirt_tris])
        return vertices, triangles


def apply_raised(surface, feature: dict, floor_mm: float):
    """Blend a ``(mask, z)`` feature into the surface with ``max`` semantics."""
    mask = np.asarray(feature["mask"], dtype=bool)
    z = np.asarray(feature["z"], dtype=np.float64)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return surface, 0

    blended = np.maximum(surface, np.maximum(z, floor_mm))
    return np.where(mask, blended, surface), count


def apply_contours(surface, feature: dict, floor_mm: float):
    """Apply a signed contour offset mask (raised ridges or engraved grooves)."""
    mask = np.asarray(feature["mask"], dtype=bool)
    offset = np.asarray(feature["offset"], dtype=np.float64)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return surface, 0

    target = np.clip(surface + offset, floor_mm, None)
    return np.where(mask, target, surface), count


def export_to_stl(vertices, triangles, output_path) -> str:
    """Export mesh to a binary STL file.

    Parameters
    ----------
    vertices : array-like of (x, y, z) tuples in mm
    triangles : array-like of (i, j, k) triangle index triplets
    output_path : str or Path to write the binary STL

    Returns the output path as a string.
    """
    import pathlib

    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.int64)

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    # Face normals must be unit vectors per the binary STL spec. Winding gives
    # the outward direction, so normalising the cross product is enough.
    normals = np.cross(v0 - v1, v0 - v2)
    lengths = np.linalg.norm(normals, axis=1)
    safe = lengths > 1e-12
    normals[safe] /= lengths[safe][:, None]
    normals[~safe] = 0.0

    # Binary STL: 80-byte header, uint32 count, 50 bytes per facet.
    header = b"Terrain mesh generated by Map-Creator"
    header = header[:80] + b"\x00" * (80 - len(header[:80]))

    # 12 float32 per facet (normal + 3 vertices) then a uint16 attribute count.
    payload = np.zeros((len(tris), 12), dtype=np.float32)
    payload[:, 0:3] = normals
    payload[:, 3:6] = v0
    payload[:, 6:9] = v1
    payload[:, 9:12] = v2

    facets = np.zeros((len(tris), 50), dtype=np.uint8)
    facets[:, :48] = payload.view(np.uint8).reshape(len(tris), 48)

    with open(pathlib.Path(output_path), "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(tris)))
        f.write(facets.tobytes())

    return str(output_path)
