"""Fold OpenStreetMap features and contour lines into the terrain heightfield.

Roads, buildings and contours are all expressed as modifications of the same
heightfield the terrain mesh is built from:

* **Roads** raise their ribbon to a flat-topped, vertical-sided strip, so a
  road reads as a clean raised rectangle rather than a rounded causeway.
* **Buildings** raise their footprint to a fixed height above the local
  ground. Untagged footprints use ``building_height_mm``; footprints that carry
  ``height``/``building:levels`` tags keep their relative stature, and
  ``height_scale`` scales the whole layer.
* **Contours** are marching-squares isolines of the elevation grid, blended into
  the surface as narrow smooth ridges (raised) or grooves (engraved), so the
  terrain and its rings stay smooth.

Merging into the heightfield — rather than unioning separate solids — is what
keeps the exported mesh watertight and 2-manifold.
"""

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ── model-space transform ─────────────────────────────────────────────────────


@dataclass
class ModelTransform:
    """Maps WGS84 lon/lat onto the model's millimetre XY plane.

    The model is centred on the selection, so model coordinates run from
    ``[-width/2, +width/2]`` × ``[-depth/2, +depth/2]``.
    """

    center_lon: float
    center_lat: float
    width_mm: float
    depth_mm: float
    span_east_m: float = 0.0
    span_north_m: float = 0.0
    to_meters: object = None

    @classmethod
    def create(cls, bounds, width_mm: float, depth_mm: float) -> "ModelTransform":
        from app.utils.projection import simple_equirect

        center_lon = (bounds.west + bounds.east) / 2.0
        center_lat = (bounds.south + bounds.north) / 2.0
        # simple_equirect returns (to_meters, from_meters).
        to_meters, _from_meters = simple_equirect(center_lon, center_lat)

        # Ground extent of the selection, so lon/lat offsets can be scaled onto
        # the model's millimetre plane instead of staying in metres.
        span_east_m = to_meters(bounds.east, center_lat)[0] - to_meters(bounds.west, center_lat)[0]
        span_north_m = to_meters(center_lon, bounds.north)[1] - to_meters(center_lon, bounds.south)[1]

        return cls(
            center_lon=center_lon,
            center_lat=center_lat,
            width_mm=width_mm,
            depth_mm=depth_mm,
            span_east_m=span_east_m,
            span_north_m=span_north_m,
            to_meters=to_meters,
        )

    @property
    def mm_per_east_m(self) -> float:
        if self.span_east_m == 0:
            return 1.0
        return self.width_mm / self.span_east_m

    @property
    def mm_per_north_m(self) -> float:
        if self.span_north_m == 0:
            return 1.0
        return self.depth_mm / self.span_north_m

    def to_model(self, lon: float, lat: float) -> tuple[float, float]:
        """Project lon/lat to model millimetres (origin at the model centre)."""
        east_m, north_m = self.to_meters(lon, lat)
        return east_m * self.mm_per_east_m, north_m * self.mm_per_north_m


def _grid_layout(rows: int, cols: int, width_mm: float, depth_mm: float):
    """Return per-axis millimetre coordinates and their spacing."""
    x = np.linspace(-width_mm / 2.0, width_mm / 2.0, cols)
    y = np.linspace(-depth_mm / 2.0, depth_mm / 2.0, rows)
    return x, y, (x[1] - x[0]) if cols > 1 else 1.0, (y[1] - y[0]) if rows > 1 else 1.0


def _cell_axes(rows: int, cols: int, width_mm: float, depth_mm: float):
    """Cell-centre axes: one sample per grid cell instead of per vertex.

    Feature masks are stamped per cell (a cell is covered when its centre is
    covered), which lets the mesh builder extrude flat roofs and vertical walls.
    The last row/column of a mask produced against these axes is never set.
    """
    x = np.linspace(-width_mm / 2.0, width_mm / 2.0, cols)
    y = np.linspace(-depth_mm / 2.0, depth_mm / 2.0, rows)
    return (x[:-1] + x[1:]) / 2.0, (y[:-1] + y[1:]) / 2.0


# ── OSM element helpers ───────────────────────────────────────────────────────


def _way_nodes(way: dict) -> list[tuple[float, float]]:
    """Extract ``(lat, lon)`` pairs from an Overpass way element."""
    geometry = way.get("geometry") or []
    nodes = [(g["lat"], g["lon"]) for g in geometry if "lat" in g and "lon" in g]
    if nodes:
        return nodes

    lat, lon = way.get("lat"), way.get("lon")
    if lat is None or lon is None:
        return []
    return [(lat, lon)]


def _building_height_m(tags: dict, default_m: float) -> float:
    """Resolve a building height in metres from OSM tags."""
    if not tags:
        return default_m

    for key in ("height", "building:height"):
        raw = tags.get(key)
        if raw:
            try:
                value = float(str(raw).replace("m", "").strip())
                if 1.0 <= value <= 400.0:
                    return value
            except ValueError:
                pass

    levels = tags.get("building:levels") or tags.get("levels")
    if levels:
        try:
            return float(str(levels).split(";")[0]) * 3.0
        except ValueError:
            pass

    kind = (tags.get("building") or "").lower()
    presets = {
        "garage": 2.5,
        "shed": 2.5,
        "hut": 2.0,
        "roof": 3.0,
        "carport": 2.4,
        "kiosk": 2.6,
        "chapel": 5.0,
        "church": 9.0,
        "temple": 8.0,
        "house": 5.5,
        "residential": 6.0,
        "apartments": 12.0,
        "commercial": 8.0,
        "office": 14.0,
        "retail": 6.0,
        "industrial": 8.0,
        "warehouse": 7.0,
    }
    return presets.get(kind, default_m)


def _outer_ring(way: dict) -> list[tuple[float, float]]:
    """Outer ring of a building way, closed and without interior nodes."""
    nodes = _way_nodes(way)
    if len(nodes) < 3:
        return []
    if nodes[0] == nodes[-1]:
        nodes = nodes[:-1]
    return nodes


# ── rasterisation primitives ──────────────────────────────────────────────────


def _segment_block(x_axis, y_axis, p0, p1, radius_mm):
    """Grid bounding box of the capsule around segment p0→p1, or None."""
    x0, y0 = p0
    x1, y1 = p1
    j0 = max(int(np.searchsorted(x_axis, min(x0, x1) - radius_mm)), 0)
    j1 = min(int(np.searchsorted(x_axis, max(x0, x1) + radius_mm)), len(x_axis))
    i0 = max(int(np.searchsorted(y_axis, min(y0, y1) - radius_mm)), 0)
    i1 = min(int(np.searchsorted(y_axis, max(y0, y1) + radius_mm)), len(y_axis))
    if j1 <= j0 or i1 <= i0:
        return None
    return i0, i1, j0, j1


def _smooth_falloff(dist, inner: float, outer: float):
    """C1 smoothstep weight: 1 inside ``inner``, 0 beyond ``outer``.

    Feathering feature ridges into the surrounding terrain with a C1 ramp is
    what stops roads and contour rings from reading as vertical-walled,
    staircased blocks: the ridge meets the ground with zero slope, so the
    surface stays smooth.
    """
    d = np.asarray(dist, dtype=np.float64)
    if outer <= inner:
        return (d <= inner).astype(np.float64)
    t = np.clip((d - inner) / (outer - inner), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def _segment_distance(gx, gy, x0: float, y0: float, x1: float, y1: float):
    """Distance from every point in the broadcast grid to segment p0→p1."""
    dx = x1 - x0
    dy = y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return np.hypot(gx - x0, gy - y0)
    t = np.clip(((gx - x0) * dx + (gy - y0) * dy) / length_sq, 0.0, 1.0)
    return np.hypot(gx - (x0 + t * dx), gy - (y0 + t * dy))


def _stamp_road(mask, x_axis, y_axis, p0, p1, half: float) -> None:
    """Rasterise a rectangular road strip around segment p0→p1.

    Every cell whose centre lies within ``half`` of the centreline is filled,
    giving a flat-topped ribbon with vertical sides: a raised rectangle rather
    than a rounded causeway.
    """
    block = _segment_block(x_axis, y_axis, p0, p1, half)
    if block is None:
        return
    i0, i1, j0, j1 = block
    gx = x_axis[j0:j1][None, :]
    gy = y_axis[i0:i1][:, None]
    dist = _segment_distance(gx, gy, p0[0], p0[1], p1[0], p1[1])
    mask[i0:i1, j0:j1] |= dist <= half


def _stamp_contour(
    mask, offset, x_axis, y_axis, p0, p1, inner, outer, height, engraved
) -> None:
    """Accumulate a smooth contour-ridge/groove around segment p0→p1."""
    block = _segment_block(x_axis, y_axis, p0, p1, outer)
    if block is None:
        return
    i0, i1, j0, j1 = block
    gx = x_axis[j0:j1][None, :]
    gy = y_axis[i0:i1][:, None]
    dist = _segment_distance(gx, gy, p0[0], p0[1], p1[0], p1[1])

    contribution = _smooth_falloff(dist, inner, outer) * abs(float(height))
    window = offset[i0:i1, j0:j1]
    if engraved:
        np.minimum(window, -contribution, out=window)
    else:
        np.maximum(window, contribution, out=window)
    mask[i0:i1, j0:j1] |= dist <= inner


def _fill_polygon(mask: np.ndarray, x_axis, y_axis, ring: Sequence[tuple[float, float]]) -> bool:
    """Even-odd scanline fill of a closed ring in model millimetres."""
    if len(ring) < 3:
        return False

    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    j0 = max(int(np.searchsorted(x_axis, min(xs))), 0)
    j1 = min(int(np.searchsorted(x_axis, max(xs))) + 1, len(x_axis))
    i0 = max(int(np.searchsorted(y_axis, min(ys))), 0)
    i1 = min(int(np.searchsorted(y_axis, max(ys))) + 1, len(y_axis))
    if j1 <= j0 or i1 <= i0:
        return False

    grid_x = x_axis[j0:j1][None, :]
    grid_y = y_axis[i0:i1][:, None]

    inside = np.zeros((i1 - i0, j1 - j0), dtype=bool)
    count = len(ring)
    for k in range(count):
        x0, y0 = ring[k]
        x1, y1 = ring[(k + 1) % count]
        if y0 == y1:
            continue
        straddles = (grid_y >= min(y0, y1)) & (grid_y < max(y0, y1))
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x0 + (grid_y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= straddles & (grid_x < x_cross)

    mask[i0:i1, j0:j1] |= inside
    return True


# ── public builders ───────────────────────────────────────────────────────────


def build_roads(
    ways: Iterable[dict],
    transform: ModelTransform,
    surface: np.ndarray,
    *,
    width_mm: float,
    height_mm: float,
    min_width_mm: float,
    smoothing_passes: int = 1,
) -> dict:
    """Rasterise OSM road centrelines into raised rectangular strips.

    A road is a flat-topped ribbon raised ``height_mm`` above the ground with
    vertical sides, so it reads as a clean raised rectangle rather than a
    rounded causeway. The returned ``raise`` field is a per-vertex offset in the
    same units as ``surface`` (millimetres in the pipeline), which the mesh
    builder folds into the terrain. ``mask`` records the ribbon footprint.
    """
    rows, cols = surface.shape
    x_axis, y_axis, _, _ = _grid_layout(rows, cols, transform.width_mm, transform.depth_mm)

    mask = np.zeros((rows, cols), dtype=bool)

    # Keep the strip at least a cell wide so a thin road never slips between
    # vertices on a coarse grid.
    cell_x = transform.width_mm / max(cols - 1, 1)
    cell_y = transform.depth_mm / max(rows - 1, 1)
    cell = float(min(cell_x, cell_y))
    half = max(width_mm, min_width_mm, cell) / 2.0

    for way in ways:
        nodes = _way_nodes(way)
        if len(nodes) < 2:
            continue

        model_pts = [transform.to_model(lon, lat) for lat, lon in nodes]
        clipped = [
            (x, y)
            for (x, y), (lat, lon) in zip(model_pts, nodes)
            if -transform.width_mm <= x <= transform.width_mm
            and -transform.depth_mm <= y <= transform.depth_mm
        ]
        if len(clipped) < 2:
            continue

        for k in range(len(clipped) - 1):
            _stamp_road(mask, x_axis, y_axis, clipped[k], clipped[k + 1], half)

    raise_field = np.where(mask, float(height_mm), 0.0)
    z = surface + raise_field
    return {"mask": mask, "z": np.where(mask, z, -np.inf), "raise": raise_field}


def build_buildings(
    ways: Iterable[dict],
    transform: ModelTransform,
    surface: np.ndarray,
    *,
    height_mm: float,
    min_height_mm: float,
    footprint_scale: float = 1.0,
    height_scale: float = 1.0,
    reference_height_m: float = 6.0,
) -> dict:
    """Rasterise building footprints into raised blocks.

    Untagged footprints are raised by ``height_mm``. Footprints carrying a real
    OSM height keep their *relative* stature — a church is taller than a shed —
    by scaling ``height_mm`` with the ratio of the tagged height to
    ``reference_height_m``, clamped to a sensible band so a single tower cannot
    swallow the model. ``height_scale`` then multiplies every building, so one
    control scales the whole layer up or down.

    ``footprint_scale`` grows or shrinks the outline around its centroid so
    footprints stay visible on coarse grids.
    """
    rows, cols = surface.shape
    x_axis, y_axis, _, _ = _grid_layout(rows, cols, transform.width_mm, transform.depth_mm)
    cx, cy = _cell_axes(rows, cols, transform.width_mm, transform.depth_mm)

    mask = np.zeros((rows, cols), dtype=bool)
    z = np.full((rows, cols), -np.inf, dtype=np.float64)

    scale = max(footprint_scale, 0.1)
    for way in ways:
        ring = _outer_ring(way)
        if not ring:
            continue

        model_ring = [transform.to_model(lon, lat) for lat, lon in ring]
        if footprint_scale != 1.0:
            cx_m = float(np.mean([p[0] for p in model_ring]))
            cy_m = float(np.mean([p[1] for p in model_ring]))
            model_ring = [
                (cx_m + (x - cx_m) * scale, cy_m + (y - cy_m) * scale) for x, y in model_ring
            ]

        before = mask.copy()
        if not _fill_polygon(mask, cx, cy, model_ring):
            continue
        cells = mask & ~before
        if not cells.any():
            continue

        # Take the highest ground the footprint actually covers, so a flat-topped
        # block never sinks into the uphill side of a slope.
        ci, cj = np.nonzero(cells)
        corners = np.stack([
            surface[ci, cj],
            surface[ci, cj + 1],
            surface[ci + 1, cj],
            surface[ci + 1, cj + 1],
        ])
        ground = float(corners.max())

        tags = way.get("tags") or {}
        real_m = _building_height_m(tags, default_m=0.0)
        ratio = 1.0
        if real_m > 0 and reference_height_m > 0:
            ratio = float(np.clip(real_m / reference_height_m, 0.25, 4.0))
        height = max(min_height_mm, height_mm * ratio) * max(height_scale, 0.0)

        block = _region_of(cx, cy, model_ring)
        if block is not None:
            i0, i1, j0, j1 = block
            local = z[i0:i1, j0:j1]
            local_cells = cells[i0:i1, j0:j1]
            np.maximum(
                local, ground + height, out=local, where=local_cells
            )

    return {"mask": mask, "z": np.where(mask, z, -np.inf)}


def _region_of(x_axis, y_axis, ring):
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    j0 = max(int(np.searchsorted(x_axis, min(xs))), 0)
    j1 = min(int(np.searchsorted(x_axis, max(xs))) + 1, len(x_axis))
    i0 = max(int(np.searchsorted(y_axis, min(ys))), 0)
    i1 = min(int(np.searchsorted(y_axis, max(ys))) + 1, len(y_axis))
    if j1 <= j0 or i1 <= i0:
        return None
    return i0, i1, j0, j1


def build_contours(
    elev_m: np.ndarray,
    *,
    width_mm: float,
    depth_mm: float,
    interval_m: float,
    major_every: int = 5,
    thickness_mm: float = 0.6,
    height_mm: float = 0.8,
    engraved: bool = False,
) -> dict:
    """Marching-squares isolines of ``elev_m`` as a signed offset mask.

    Isolines are computed on the elevation grid's index lattice, then mapped
    onto the model's millimetre plane so ``thickness_mm`` and ``height_mm``
    mean real millimetres rather than grid cells.

    Every ``major_every``-th contour is drawn thicker and taller so index and
    intermediate contours are distinguishable on the printed model.
    """
    rows, cols = elev_m.shape
    elev = np.nan_to_num(np.asarray(elev_m, dtype=np.float64), nan=0.0)

    empty = {"mask": np.zeros((rows, cols), dtype=bool), "offset": np.zeros((rows, cols))}
    if interval_m <= 0 or rows < 2 or cols < 2:
        return empty

    x_axis, y_axis, _, _ = _grid_layout(rows, cols, width_mm, depth_mm)

    lo = float(np.floor(elev.min() / interval_m))
    hi = float(np.ceil(elev.max() / interval_m))
    levels = np.arange(lo, hi + 1) * interval_m
    if levels.size == 0 or levels.size > 400:
        levels = np.linspace(elev.min(), elev.max(), 25)

    # Marching squares returns (col_index, row_index); row 0 is the south edge,
    # which is the same orientation y_axis uses.
    def to_model(col_idx: float, row_idx: float) -> tuple[float, float]:
        x = -width_mm / 2.0 + (col_idx / (cols - 1)) * width_mm
        y = -depth_mm / 2.0 + (row_idx / (rows - 1)) * depth_mm
        return x, y

    cell_x = width_mm / max(cols - 1, 1)
    cell_y = depth_mm / max(rows - 1, 1)
    cell = float(min(cell_x, cell_y))

    mask = np.zeros((rows, cols), dtype=bool)
    offset = np.zeros((rows, cols), dtype=np.float64)
    major_stride = max(major_every, 1)

    for k, level in enumerate(levels):
        is_major = int(round(level / interval_m)) % major_stride == 0
        thickness = max(0.0, thickness_mm * (1.8 if is_major else 1.0))
        height = height_mm * (1.9 if is_major else 1.0)
        if thickness <= 0:
            continue

        segments = _marching_squares(elev, level)
        if not segments:
            continue

        # Flat-top ridge (thickness) with a smooth shoulder, so the line reads
        # as a clean ring instead of a staircased block of cells.
        outer = thickness + max(cell, thickness)
        for (ax, ay), (bx, by) in segments:
            _stamp_contour(
                mask, offset, x_axis, y_axis,
                to_model(ax, ay), to_model(bx, by),
                thickness, outer, height, engraved,
            )

    if engraved:
        offset = -np.abs(offset)
    else:
        offset = np.abs(offset)

    return {"mask": mask, "offset": offset}


def _marching_squares(grid: np.ndarray, level: float) -> list[tuple[tuple, tuple]]:
    """Return line segments (in index coordinates) for one iso-level."""
    rows, cols = grid.shape
    segments: list[tuple[tuple, tuple]] = []

    a = grid[:-1, :-1]
    b = grid[:-1, 1:]
    c = grid[1:, 1:]
    d = grid[1:, :-1]

    def _cross(p, q, pv, qv):
        if (pv < level) == (qv < level):
            return None
        t = (level - pv) / (qv - pv)
        return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

    # Vectorised case selection keeps large grids fast.
    idx = np.argwhere(
        ((a < level).astype(np.uint8) << 3)
        | ((b < level).astype(np.uint8) << 2)
        | ((c < level).astype(np.uint8) << 1)
        | (d < level).astype(np.uint8)
    )
    for i, j in idx:
        p00 = (float(j), float(i))
        p10 = (float(j + 1), float(i))
        p11 = (float(j + 1), float(i + 1))
        p01 = (float(j), float(i + 1))
        v00, v10, v11, v01 = float(a[i, j]), float(b[i, j]), float(c[i, j]), float(d[i, j])

        top = _cross(p00, p10, v00, v10)
        right = _cross(p10, p11, v10, v11)
        bottom = _cross(p11, p01, v11, v01)
        left = _cross(p01, p00, v01, v00)

        code = (
            (1 if v00 < level else 0)
            | (2 if v10 < level else 0)
            | (4 if v11 < level else 0)
            | (8 if v01 < level else 0)
        )

        pairs = {
            1: [(left, top)],
            2: [(top, right)],
            3: [(left, right)],
            4: [(right, bottom)],
            5: [(left, top), (right, bottom)],
            6: [(top, bottom)],
            7: [(left, bottom)],
            8: [(left, bottom)],
            9: [(top, bottom)],
            10: [(left, top), (right, bottom)],
            11: [(top, right)],
            12: [(left, right)],
            13: [(top, right)],
            14: [(left, top)],
        }.get(code, [])

        for pair in pairs:
            if pair[0] is not None and pair[1] is not None:
                segments.append(pair)

    return segments
