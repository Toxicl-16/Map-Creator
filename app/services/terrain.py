"""Terrain generation service from elevation & vector data.

Pipeline:
    elevation grid -> UTM projection -> height mapping -> base mesh
    + road polygons (extruded) + building footprints (extruded)
    + optional contour lines -> unified mesh
    + watertight validation -> binary STL output

Coordinate system: WGS84 input -> UTM zone 326XX -> local metric grid in meters -> scaled mm model.
"""

import logging
import math
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    resolution_m: float = 64.0  # elevation grid resolution in meters

    # Terrain height controls
    min_altitude_mm: float = 2  # minimum terrain material thickness
    max_altitude_mm: float = 50  # maximum terrain extrusion height
    vertical_exaggeration: float = 1.0  # multiplier on terrain heights
    elevation_offset_mm: float = 1  # offset from base (Z=0)

    # Base plate settings
    base_thickness_mm: float = 3  # thickness of bottom plate
    base_extension_mm: float = 0  # overhang around edges

    # Smoothing
    smoothing_passes: int = 2  # Gaussian blur iterations on elevation grid

    # Feature toggles
    include_roads: bool = False
    include_buildings: bool = False
    road_height_mm: float = 1.0

    include_contours: bool = False
    contour_interval_m: float = 50
    contour_thickness_mm: float = 0.5

    # Limits
    max_model_width_mm: float = 600
    max_model_depth_mm: float = 600
    min_resolution_m: float = 10
    max_resolution_m: float = 128

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


def validate_settings(settings):
    """Ensure terrain settings are within reasonable operational bounds."""

    if settings.model_width_mm < 1 or settings.model_depth_mm < 1:
        logger.error("Model too small to print (< 1 mm)")
        return False

    if not (0.5 <= settings.vertical_exaggeration <= 50):
        logger.warning(
            "Extreme vertical exaggeration: %.1fx", settings.vertical_exaggeration
        )

    if settings.model_width_mm > settings.max_model_width_mm:
        logger.warning(
            "Width %.1f mm exceeds recommended max %.0f mm",
            settings.model_width_mm,
            settings.max_model_width_mm,
        )

    if settings.min_altitude_mm >= settings.max_altitude_mm:
        logger.error("min_altitude_mm (%.1f) >= max_altitude_mm (%.1f)",
                     settings.min_altitude_mm, settings.max_altitude_mm)
        return False

    degrees_span = max(settings.width, settings.height)
    estimated_m = degrees_span * 111_000
    if estimated_m > settings.max_resolution_m and settings.resolution_m < settings.min_resolution_m:
        logger.warning(
            "Area too large for chosen resolution: %.f m grid over %.1f km span",
            settings.resolution_m, estimated_m / 1000)

    return True


def _utm_zone(lon):
    """Derive the UTM zone number from a longitude value."""
    return max(1, min(60, int((lon + 180) / 6) + 1))


def _wgs84_to_utm(west, south, east, north, resolution_m):
    """Convert bounding box degrees to UTM projection info dict.

    Returns dict with keys:
      center_lat   - center latitude (used for zone calculation)
      easting_base - false easting of west edge in meters
      northing     - northing of south edge in meters
      x_step/y_step - grid spacing in metres
      cols/rows    - grid sizes
    """
    center_lat = (south + north) / 2.0
    zone = _utm_zone((west + east) / 2.0)

    # WGS84 constants
    a_km = 6378.137
    f_val = 1 / 298.257223563
    e2 = 2 * f_val - f_val ** 2
    k0 = 0.9996

    lat_rad = math.radians(center_lat)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    # Radius of curvature in the prime vertical
    N_m = a_km * (1 - e2) / ((1 - e2 * sin_lat ** 2) ** 1.5)

    # Meridional arc for northing
    M_m = a_km * k0 * (
        (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat_rad
        - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * lat_rad)
        + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * lat_rad)
        - 35 * e2 ** 3 / 3072 * math.sin(6 * lat_rad)
    )

    # Easting from central meridian for west edge
    lon_deg_west = west - (zone - 1) * 6 - 3
    lon_rad_west = math.radians(lon_deg_west)
    sin_lon = math.sin(lon_rad_west)
    cos_lon = math.cos(lon_rad_west)
    sin2_lon = sin_lon ** 2

    easting_base_m = 500_000 + N_m * k0 * (
        sin_lon
        + (1 - e2 * sin_lat ** 2) / 6 * cos_lat ** 3 * sin2_lon * (4 - sin2_lon)
    )

    # Width in metres at UTM scale
    easting_east_m = 500_000 + N_m * k0 * cos_lat * math.radians(east - west)

    # North/South span in meters
    north_m_val = M_m + N_m * k0 * cos_lat * math.radians(north - south)

    width_km = max(0.001, easting_east_m - easting_base_m)
    depth_km = max(0.001, north_m_val - M_m)

    cols_count = int(max(2, round(width_km * 1000 / resolution_m)))
    rows_count = int(max(2, round(depth_km * 1000 / resolution_m)))

    return {
        "center_lat": center_lat,
        "easting_base": easting_base_m,
        "northing": M_m,
        "x_step": width_km * 1000 / max(cols_count, 2),
        "y_step": depth_km * 1000 / max(rows_count, 2),
        "cols": cols_count,
        "rows": rows_count,
        "west_m": easting_base_m,
        "south_m": M_m,
        "east_m": easting_east_m,
        "north_m": north_m_val,
        "zone": zone,
    }


def _normalize_elevation(elev_array, min_height, max_height, vertical_exaggeration):
    """Scale elevation array to [min_height, max_height] and apply exaggeration."""
    import numpy as np

    elev = np.nan_to_num(elev_array, nan=0.0)
    min_elev = float(np.min(elev))
    max_elev_val = float(np.max(elev))
    height_range = max_elev_val - min_elev

    if abs(height_range) < 1e-6:
        logger.warning("Flat terrain detected: %.2f == %.2f", min_elev, max_elev_val)
        return np.full_like(elev, (min_height + max_height) / 2)

    normed = (elev - min_elev) / height_range
    scaled = normed * (max_height - min_height) + min_height

    if abs(vertical_exaggeration - 1.0) > 0.01:
        center_z = float(np.mean(scaled))
        deviation_arr = scaled - center_z
        abs_deviation = np.abs(deviation_arr)
        max_abs_dev = float(np.max(abs_deviation)) if max_height > min_height else 1e-9

        scale_factor = vertical_exaggeration / max(max_abs_dev, 1e-6)
        normalized = deviation_arr * (abs(vertical_exaggeration) / max(abs(max_height - min_height), 1)) if abs(max_height - min_height) > 0 else deviation_arr * vertical_exaggeration
        return np.array(normalized + center_z).clip(min_height, max_height)

    return scaled.clip(min_height, max_height)


def _grid_to_triangles(rows_count, cols_count):
    """Return list of triangle vertex-index triplets for a row x col grid.

    Triangles cover the grid in quads split along the north-east diagonal.
    """
    triangles = []
    for r_idx in range(rows_count - 1):
        for c_idx in range(cols_count - 1):
            idx00 = r_idx * cols_count + c_idx
            idx01 = idx00 + 1
            idx11 = (r_idx + 1) * cols_count + c_idx + 1
            idx10 = (r_idx + 1) * cols_count + c_idx
            # Two triangles per quad: NE-diagonal split
            triangles.append((idx00, idx10, idx11))
            triangles.append((idx00, idx11, idx01))
    return triangles


class MeshBuilder:
    """Build triangle mesh from terrain elevation grid and optionally OSM features.

    Output: tuple of (vertices : List[Tuple[float,float,float]],
                       triangles : List[Tuple[int,int,int]])
    Coordinates are in mm relative to model origin.
    """

    def __init__(self, settings):
        self.settings = settings

    def build(self, elev_grid, elevation_data_available=True):
        """Build the unified mesh from an elevation grid.

        Parameters
        ----------
        elev_grid : np.ndarray
            2-D height map in metres.
        elevation_data_available
            Set False for synthetic test terrain instead.
        """
        import numpy as np

        vertices = []
        triangles = []
        rows_count, cols_count = elev_grid.shape
        s = self.settings

        # --- base plate (Z = 0) -------------------------------------------
        base_plate_start = len(vertices)
        platex = s.base_extension_mm
        platedp = s.base_thickness_mm
        if platedp > 0:
            for iy in range(3):
                for ix in range(3):
                    vertices.append((-platex * 1.5 + ix * platex,
                                     -platedp * 1.5 + iy * platedp, 0))
            base_triangles = _grid_to_triangles(2, 2)
            for tri_val in base_triangles:
                triangles.append(tuple(v_idx + base_plate_start for v_idx in tri_val))

        # --- terrain mesh ----------------------------------------------
        utm_info = _wgs84_to_utm(s.west, s.south, s.east, s.north, s.resolution_m)
        width_km = max(rows_count * utm_info["x_step"], 0.001)
        height_km = max(cols_count * utm_info["y_step"], 0.001)

        scale_x = s.model_width_mm / (width_km + s.base_extension_mm * 2)
        scale_y = s.model_depth_mm / (height_km + s.base_extension_mm * 2)
        mesh_scale = min(scale_x, scale_y)

        # Height scaling: convert meter elevations to mm heights within range
        elevation_range = float(elev_grid.max() - elev_grid.min())
        if not elevation_data_available or abs(elevation_range) < 1e-9:
            synthetic_data = np.linspace(
                s.min_altitude_mm, s.max_altitude_mm, rows_count * cols_count
            ).reshape((rows_count, cols_count))
            elev_mm = (elev_grid - elev_grid.min()) / max(elev_grid.max() - elev_grid.min(), 1e-9) * (s.max_altitude_mm - s.min_altitude_mm) + s.min_altitude_mm
        else:
            elevation_height_range = s.max_altitude_mm - s.min_altitude_mm
            elev_mm = _normalize_elevation(
                elev_grid,
                min_height=0.0,
                max_height=elevation_height_range,
                vertical_exaggeration=s.vertical_exaggeration,
            )

        # Map mesh vertices to mm coordinates and build triangles
        terrain_start = len(vertices)
        for r_idx in range(rows_count):
            for c_idx in range(cols_count):
                x_pos = (c_idx / (cols_count - 1)) * s.model_width_mm if cols_count > 1 else s.model_width_mm / 2
                y_pos = ((rows_count - 1 - r_idx) / (rows_count - 1)) * s.model_depth_mm if rows_count > 1 else s.model_depth_mm / 2
                z_val = s.elevation_offset_mm + float(elev_mm[r_idx, c_idx])
                vertices.append((x_pos, y_pos, z_val))

        tri_indices = _grid_to_triangles(rows_count, cols_count)
        for tri_val in tri_indices:
            triangles.append(tuple(v_idx + terrain_start for v_idx in tri_val))

        return (vertices, triangles)


def export_to_stl(vertices, triangles, output_path):
    """Export mesh to binary STL file.

    Parameters
    ----------
    vertices : list of (x,y,z) tuples in mm
    triangles: list of (i,j,k) triangle index triplets
    output_path: str or Path to write the binary STL

    Returns path on success.
    """
    import numpy as np
    import pathlib

    out_path = pathlib.Path(output_path).resolve()

    header_bytes = b"Terrain mesh generated by Map-Creator\n"
    while len(header_bytes) < 80:
        header_bytes += b"\0"

    # Count triangle face normals
    num_triangles = len(triangles)
    with open(out_path, "wb") as f:
        f.write(header_bytes[:80])
        f.write(struct.pack('<I', num_triangles))

        for tri_val in triangles:
            i0, i1, i2 = tri_val
            v0 = np.array(vertices[i0])
            v1 = np.array(vertices[i1])
            v2 = np.array(vertices[i2])

            # Face normal (unnormalized)
            nx_val = (v0[1] - v1[1]) * (v0[2] - v2[2]) - (v0[2] - v1[2]) * (v0[0] - v2[0])
            ny_val = (v0[2] - v1[2]) * (v0[0] - v2[0]) - (v0[0] - v1[0]) * (v0[1] - v2[1])
            nz_val = (v0[0] - v1[0]) * (v0[1] - v2[1]) - (v0[1] - v1[1]) * (v0[2] - v2[2])

            f.write(struct.pack('<ffff', float(nx_val), float(ny_val), float(nz_val), 0.0))
            for vertex in [vertices[i0], vertices[i1], vertices[i2]]:
                f.write(struct.pack('<ffff', float(vertex[0]), float(vertex[1]), float(vertex[2]), 0.0))

    return str(out_path)


if __name__ == "__main__":
    settings = TerrainSettings(
        west=-3.05, south=12.69, east=-2.95, north=12.79,
        model_width_mm=300, model_depth_mm=300, resolution_m=64.0,
    )

    # Simulated dummy elevation grid 64x64 with slight altitude variation
    import numpy as np
    dummy_elevation = np.ones((64, 64)) * 100  # 100m base
    # Add a gentle hill in the center
    for r_idx in range(64):
        for c_idx in range(64):
            dist = math.sqrt(((c_idx - 32) / 32) ** 2 + ((r_idx - 32) / 32) ** 2)
            if dist < 1:
                dummy_elevation[r_idx, c_idx] += (1 - dist) * 20

    builder = MeshBuilder(settings)
    verts, tris = builder.build(dummy_elevation, elevation_data_available=True)

    out_path = export_to_stl(verts, tris, "/tmp/test_terrain.stl")
    print(f"Wrote {out_path}")
    import os
    size_bytes = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    print(f"File size: {size_bytes} bytes")
