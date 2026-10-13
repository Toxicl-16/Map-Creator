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
    building_height_scale: float = 1.0
    building_min_height_mm: float = 3.0
    building_footprint_scale: float = 1.05
    contour_interval_m: float = 50.0
    contour_major_interval_m: float = 250.0
    contour_thickness_mm: float = 0.6
    contour_height_mm: float = 0.8
    contours_engraved: bool = False
    layer_terrain: bool = False
    layer_interval_m: float = 0.0  # 0 = match the contour interval

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

    if settings.include_buildings and settings.building_height_scale <= 0:
        problems.append("Building height scale must be greater than zero.")

    if settings.layer_terrain:
        interval = settings.layer_interval_m or settings.contour_interval_m
        if interval <= 0:
            problems.append("Layer interval must be greater than zero metres.")

    if settings.include_roads and settings.road_min_width_mm < 0.4:
        problems.append("Road width below 0.4 mm is unlikely to print.")

    return problems


    return max(2, rows), max(2, cols)


def upsample_bilinear(grid, factor: int):
    """Refine a grid by an integer ``factor`` using bilinear interpolation.

    The output spans exactly the same ground area as the input (its cell size
    is ``factor`` times smaller), so it can stand in for ``grid`` when building
    the mesh. Refinement adds no new terrain data; it gives the contour rings
    and contour geometry enough cells to read as smooth curves instead of
    grid-aligned staircases.
    """
    grid = np.asarray(grid, dtype=np.float64)
    if factor <= 1:
        return grid
    r, c = grid.shape
    if r < 2 or c < 2:
        return grid

    yy = np.arange((r - 1) * factor + 1, dtype=np.float64) / factor
    xx = np.arange((c - 1) * factor + 1, dtype=np.float64) / factor
    y0 = np.floor(yy).astype(np.intp)
    y1 = np.minimum(y0 + 1, r - 1)
    x0 = np.floor(xx).astype(np.intp)
    x1 = np.minimum(x0 + 1, c - 1)
    wy = (yy - y0)[:, None]
    wx = (xx - x0)[None, :]

    top = grid[y0][:, x0] * (1.0 - wx) + grid[y0][:, x1] * wx
    bot = grid[y1][:, x0] * (1.0 - wx) + grid[y1][:, x1] * wx
    return top * (1.0 - wy) + bot * wy


def detail_grid(
    grid,
    settings,
    *,
    target_cell_mm: float = 2.0,
    max_nodes: int = 55_000,
):
    """Bilinearly refine ``grid`` so model cells are about ``target_cell_mm``.

    Contour rings and rasterised feature ribbons are only as smooth as the grid
    they are drawn on, so a coarse elevation grid (say a 30 m sample projected
    onto a 250 mm model) makes contour rings and roads look blocky. This upsamples
    the grid by an integer factor to reach the target cell size, then backs the
    factor off until the refined node count fits ``max_nodes`` (keeping the
    exported mesh inside a printable triangle budget). Returns ``grid``
    unchanged when it is already fine enough or cannot be refined.
    """
    grid = np.asarray(grid, dtype=np.float64)
    rows, cols = grid.shape
    if rows < 2 or cols < 2:
        return grid

    cell_x = settings.model_width_mm / (cols - 1)
    cell_y = settings.model_depth_mm / (rows - 1)
    cell = min(cell_x, cell_y)
    if cell <= target_cell_mm:
        return grid

    factor = max(1, int(round(cell / target_cell_mm)))
    while factor > 1 and ((rows - 1) * factor + 1) * ((cols - 1) * factor + 1) > max_nodes:
        factor -= 1
    if factor <= 1:
        return grid
    return upsample_bilinear(grid, factor)


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

    The terrain surface is kept smooth; elevation is expressed through the
    contour rings folded in by :mod:`app.services.features`, not by quantising
    the surface into terraces (which produces sharp juts).
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


def _point_in_tri(q, a, b, c, eps=1e-9):
    """True when ``q`` lies in (or on) triangle a-b-c, given CCW winding."""
    def _cross(o, p, r):
        return (p[0] - o[0]) * (r[1] - o[1]) - (p[1] - o[1]) * (r[0] - o[0])
    if _cross(a, b, c) < -eps:
        a, b = b, a
    return (_cross(a, b, q) >= -eps) and (_cross(b, c, q) >= -eps) and (_cross(c, a, q) >= -eps)


def _signed_area(pts):
    total = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def _poly_triangulate(pts):
    """Ear-clip a simple (not necessarily convex) polygon.

    ``pts`` is a list of ``(u, v)`` coordinates in the polygon's plane. Returns
    triangle index triples oriented consistently with the input winding (CCW
    input yields CCW triangles, CW input yields CW triangles).

    The routine is written so it always returns ``n - 2`` triangles for a
    polygon of ``n`` (deduplicated) vertices: real feature walls routinely
    contain many collinear and near-duplicate levels, and a stalled ear clip
    would silently leave holes in the otherwise closed solid.
    """
    # Drop consecutive duplicate points (and a duplicate closing point) so
    # zero-length edges cannot confuse the ear tests.
    clean = []
    for p in pts:
        if not clean or abs(p[0] - clean[-1][0]) > 1e-9 or abs(p[1] - clean[-1][1]) > 1e-9:
            clean.append(p)
    if len(clean) > 1 and abs(clean[0][0] - clean[-1][0]) <= 1e-9 and abs(clean[0][1] - clean[-1][1]) <= 1e-9:
        clean.pop()
    pts = clean

    n = len(pts)
    if n < 3:
        return []
    ccw = _signed_area(pts) > 0
    order = list(range(n)) if ccw else list(reversed(range(n)))
    idx = order[:]
    tris = []

    while len(idx) > 3:
        ni = len(idx)
        chosen = None
        for k in range(ni):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % ni]
            a, b, c = pts[i0], pts[i1], pts[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-12:
                continue
            if any(_point_in_tri(pts[m], a, b, c) for m in idx if m not in (i0, i1, i2)):
                continue
            chosen = k
            break
        if chosen is None:
            # Degenerate / near-collinear region: fall back to the most convex
            # corner so we always make progress and finish with n - 2 triangles.
            best_cross = -1.0
            for k in range(ni):
                i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % ni]
                a, b, c = pts[i0], pts[i1], pts[i2]
                cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                if cross > best_cross:
                    best_cross = cross
                    chosen = k
            if chosen is None:
                chosen = 1
        i0, i1, i2 = idx[chosen - 1], idx[chosen], idx[(chosen + 1) % ni]
        tris.append((i0, i1, i2))
        idx.pop(chosen)

    if len(idx) == 3:
        tris.append(tuple(idx))

    return tris


def _cell_corner_max(grid: "np.ndarray") -> "np.ndarray":
    """Per-cell maximum of the four corner heights.

    Returns a ``(rows, cols)`` array whose entry ``[i, j]`` is the highest of
    the four ground corners of cell ``(i, j)``; the last row and column are
    padding and are never covered.
    """
    a = grid[:-1, :-1]
    b = grid[:-1, 1:]
    c = grid[1:, 1:]
    d = grid[1:, :-1]
    cells = np.maximum(np.maximum(a, b), np.maximum(c, d))
    out = np.full_like(grid, -np.inf)
    out[:-1, :-1] = cells
    return out


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

        ``roads`` carries a crisp ``raise`` offset (millimetres) that is added
        to the terrain, so a road reads as a raised rectangle with a flat top
        and vertical sides; legacy ``{mask, z}`` dicts are still accepted and
        folded with ``max`` semantics. ``contours`` carries a smooth signed
        ``offset`` (raised or engraved rings) that is *blended into the terrain
        surface*, so the terrain and its rings stay smooth.

        ``buildings`` are the other crisp feature: a covered cell becomes a
        flat-roofed block with vertical walls (see :meth:`_extrude_crisp`), so a
        house reads as a house rather than a lump. When no crisp feature is
        present the terrain is extruded as a plain smooth heightfield.

        The returned ``final`` grid is the fully folded per-vertex surface.
        """
        s = self.settings
        surface = self.surface_of(elev_grid)
        rows, cols = surface.shape

        stats = {"roads": 0, "buildings": 0, "contours": 0}
        floor = s.base_thickness_mm + s.min_altitude_mm

        # -- smooth features are blended straight into the terrain -----------
        terrain = np.array(surface, dtype=np.float64, copy=True)

        if roads:
            stats["roads"] = int(np.count_nonzero(roads["mask"]))
            raise_field = roads.get("raise")
            if raise_field is None:
                # Legacy {mask, z} input: raise to the block top where masked.
                mask = np.asarray(roads["mask"], dtype=bool)
                z = np.asarray(roads["z"], dtype=np.float64)
                raise_field = np.where(mask, np.maximum(z - surface, 0.0), 0.0)
            terrain = terrain + np.asarray(raise_field, dtype=np.float64)

        if contours:
            stats["contours"] = int(np.count_nonzero(contours["mask"]))
            terrain = np.clip(
                terrain + np.asarray(contours["offset"], dtype=np.float64), floor, None
            )

        final = terrain.copy()

        # -- buildings stay crisp -------------------------------------------
        covered = np.zeros((rows, cols), dtype=bool)
        roof = np.full((rows, cols), -np.inf, dtype=np.float64)
        if buildings:
            mask = np.asarray(buildings["mask"], dtype=bool)
            if mask.shape != covered.shape:
                raise TerrainGenerationError(
                    f"Feature mask {mask.shape} must match the {covered.shape} elevation grid."
                )
            covered |= mask
            roof = np.where(mask, np.asarray(buildings["z"], dtype=np.float64), -np.inf)
            stats["buildings"] = int(np.count_nonzero(mask))
            final = np.where(mask, np.maximum(final, np.maximum(roof, floor)), final)

        ground = terrain

        # A roof must clear its own terrain corners and the printable floor, or
        # a neighbouring slope could punch through a wall.
        if covered.any():
            # Snap the terrain onto the same fine grid as the roofs below so a
            # flat roof that sits exactly on a corner's terrain height compares
            # equal to that corner instead of differing by rounding noise.
            ground = np.round(ground, 5)
            corner_max = _cell_corner_max(ground)
            roof = np.where(
                covered,
                np.maximum(np.maximum(roof, floor), corner_max),
                -np.inf,
            )
            # Snap near-equal roofs onto a fine height grid so adjacent cells
            # that differ only by floating-point noise share a single corner
            # vertex instead of stacking two coincident roof points (which would
            # leave T-junction gaps in the mesh).
            roof = np.where(covered, np.round(roof, 5), -np.inf)

        # Drop a ring of covered cells so features never touch the model edge,
        # keeping the skirt on smooth terrain; stats keep the full counts.
        # Cells span rows-1 x cols-1, so the outermost live row is rows-1 and
        # the outermost live column is cols-2.
        covered_mesh = covered.copy()
        covered_mesh[0, :] = False
        covered_mesh[rows - 2, :] = False
        covered_mesh[:, 0] = False
        covered_mesh[:, cols - 2] = False

        if not covered_mesh.any():
            vertices, triangles = self._extrude(ground, cols, rows)
        else:
            vertices, triangles = self._extrude_crisp(ground, roof, covered_mesh, cols, rows)
        return vertices, triangles, final, stats

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

    def _extrude_crisp(self, ground, roof, covered, cols: int, rows: int):
        """Extrude a heightfield where covered cells become crisp flat-topped
        blocks with vertical walls instead of blended slope artefacts.

        ``ground`` is the per-vertex terrain surface (contours already folded
        in), ``roof`` the per-cell roof height for every covered cell and
        ``covered`` the (already eroded) per-cell boolean mask. Roof corners are
        shared by neighbouring cells of equal height; walls adopt the level set
        of each shared grid vertex so a wall is a single triangulated strip even
        when the two blocks beside it sit at different heights.
        """
        s = self.settings
        pad_x = s.base_extension_mm
        pad_y = s.base_extension_mm
        x = np.linspace(-pad_x, s.model_width_mm + pad_x, cols)
        y = np.linspace(-pad_y, s.model_depth_mm + pad_y, rows)
        xx, yy = np.meshgrid(x, y)

        zz = np.asarray(ground, dtype=np.float64)
        roof = np.asarray(roof, dtype=np.float64)
        covered = np.asarray(covered, dtype=bool)

        top_count = rows * cols
        ground_idx = np.arange(top_count).reshape(rows, cols)

        ground_verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)
        acc = ground_verts.tolist()
        roof_lookup: dict = {}

        def add_point(px, py, pz) -> int:
            acc.append([float(px), float(py), float(pz)])
            return len(acc) - 1

        def roof_vertex(i, j, h) -> int:
            key = (i * cols + j, h)
            idx = roof_lookup.get(key)
            if idx is None:
                idx = add_point(float(x[j]), float(y[i]), float(h))
                roof_lookup[key] = idx
            return idx

        def vertex_at(i, j, h) -> int:
            if abs(h - zz[i, j]) < 1e-9:
                return int(ground_idx[i, j])
            return roof_vertex(i, j, h)

        # Pinch corners: four covered cells around one vertex whose diagonal
        # pair (NE+SW or NW+SE) both sit above the other two would put the whole
        # vertical crease inside the solid (four faces on one edge). Crop the
        # shorter diagonal roof down to the top of the other diagonal so every
        # edge stays two-manifold; the roof takes a shallow folded crease.
        overrides: dict = {}

        def cell_top(cell, i, j) -> float:
            ci, cj = cell
            if not covered[ci, cj]:
                return float(zz[i, j])
            h = float(roof[ci, cj])
            ov = overrides.get((i * cols + j, cell))
            if ov is not None:
                h = min(h, ov)
            return h

        def incident(i, j):
            cells = []
            for ci in (i - 1, i):
                for cj in (j - 1, j):
                    if 0 <= ci < rows - 1 and 0 <= cj < cols - 1:
                        cells.append((ci, cj))
            return cells

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                diagonals = [
                    (((i - 1, j - 1), (i, j)), ((i - 1, j), (i, j - 1))),
                    (((i - 1, j), (i, j - 1)), ((i - 1, j - 1), (i, j))),
                ]
                for elevated, other in diagonals:
                    a_cell, b_cell = elevated
                    ta = cell_top(a_cell, i, j)
                    tb = cell_top(b_cell, i, j)
                    cap = max(cell_top(cell, i, j) for cell in other)
                    if min(ta, tb) > cap:
                        cell = b_cell if ta >= tb else a_cell
                        overrides[(i * cols + j, cell)] = cap

        def levels_at(i, j):
            vals = {float(zz[i, j])}
            for cell in incident(i, j):
                if covered[cell]:
                    vals.add(float(cell_top(cell, i, j)))
            return sorted(vals)

        top_tris = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                a = int(ground_idx[i, j])
                b = int(ground_idx[i, j + 1])
                c = int(ground_idx[i + 1, j + 1])
                d = int(ground_idx[i + 1, j])
                if covered[i, j]:
                    ra = vertex_at(i, j, cell_top((i, j), i, j))
                    rb = vertex_at(i, j + 1, cell_top((i, j), i, j + 1))
                    rc = vertex_at(i + 1, j + 1, cell_top((i, j), i + 1, j + 1))
                    rd = vertex_at(i + 1, j, cell_top((i, j), i + 1, j))
                    top_tris.append((ra, rc, rd))
                    top_tris.append((ra, rb, rc))
                else:
                    top_tris.append((a, c, d))
                    top_tris.append((a, b, c))

        def emit_wall(A, B, ia, ja, ib, jb, axis):
            ta_p = cell_top(A, ia, ja)
            ta_q = cell_top(A, ib, jb)
            tb_p = cell_top(B, ia, ja)
            tb_q = cell_top(B, ib, jb)
            if abs(ta_p - tb_p) < 1e-9 and abs(ta_q - tb_q) < 1e-9:
                return
            min_p, max_p = min(ta_p, tb_p), max(ta_p, tb_p)
            min_q, max_q = min(ta_q, tb_q), max(ta_q, tb_q)
            tall_a = (ta_p + ta_q) > (tb_p + tb_q)

            chain_p = [h for h in levels_at(ia, ja) if min_p - 1e-9 <= h <= max_p + 1e-9]
            chain_q = [h for h in levels_at(ib, jb) if min_q - 1e-9 <= h <= max_q + 1e-9]

            def u_of(i2, j2):
                return float(y[i2]) if axis == "col" else float(x[j2])

            vtx, uv = [], []
            vtx.append(vertex_at(ia, ja, min_p))
            uv.append((u_of(ia, ja), float(min_p)))
            for h in chain_q:
                vtx.append(vertex_at(ib, jb, h))
                uv.append((u_of(ib, jb), float(h)))
            for h in reversed(chain_p):
                if h == min_p:
                    continue
                vtx.append(vertex_at(ia, ja, h))
                uv.append((u_of(ia, ja), float(h)))
            if not tall_a:
                vtx.reverse()
                uv.reverse()
            for tri in _poly_triangulate(uv):
                top_tris.append((vtx[tri[0]], vtx[tri[1]], vtx[tri[2]]))

        # Column walls (constant x, cells west/east): shared vertex columns run
        # along the row axis, so the strip's u-coordinate is y. A wall is drawn
        # wherever exactly one of the neighbours is covered, and between two
        # covered cells of different heights.
        for i in range(rows - 1):
            for j in range(1, cols - 1):
                if covered[i, j] or covered[i, j - 1]:
                    emit_wall((i, j - 1), (i, j), i + 1, j, i, j, "col")
        # Row walls (constant y, cells south/north): the strip's u is x.
        for j in range(cols - 1):
            for i in range(1, rows - 1):
                if covered[i, j] or covered[i - 1, j]:
                    emit_wall((i - 1, j), (i, j), i, j + 1, i, j, "row")

        triangles = np.asarray(top_tris, dtype=np.int64)

        bottom_base = len(acc)
        bottom_verts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(top_count)])
        vertices = np.vstack([np.asarray(acc, dtype=np.float64), bottom_verts]).astype(np.float64)

        # Bottom cap mirrors the terrain quad triangulation, reversed so the
        # skirt's bottom ring seals against it with opposite orientation.
        cap = []
        for i in range(rows - 1):
            base = i * cols
            for j in range(cols - 1):
                a = base + j
                b = base + j + 1
                c = (i + 1) * cols + j + 1
                d = (i + 1) * cols + j
                cap.append((bottom_base + a, bottom_base + d, bottom_base + c))
                cap.append((bottom_base + a, bottom_base + c, bottom_base + b))
        cap_tris = np.asarray(cap, dtype=np.int64)

        loop = _grid_indices(rows, cols)
        skirt = []
        n = len(loop)
        for k in range(n):
            a = loop[k][0] * cols + loop[k][1]
            b = loop[(k + 1) % n][0] * cols + loop[(k + 1) % n][1]
            skirt.append((a, b, bottom_base + b))
            skirt.append((a, bottom_base + b, bottom_base + a))
        skirt_tris = np.asarray(skirt, dtype=np.int64)

        triangles = np.vstack([triangles, cap_tris, skirt_tris])
        return vertices, triangles


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
