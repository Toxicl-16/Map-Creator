"""Coordinate projection utilities.

CRITICAL: Never perform physical-distance calculations directly in lat/lon degrees.
All area/length computations use UTM (Universal Transverse Mercator) or a local
State Plane projection derived from the selected area's midpoint latitude.

Projection chain:
  WGS84 (lat/lon) → UTM zone for location → meters → physical model mm
"""
from __future__ import annotations

import math


# ─── Constants ──────────────────────────────────────────────────────────

_E = 6378137.0                    # WGS84 semi-major axis (m)
_F = 1 / 298.257223563            # WGS84 flattening
_E2 = 2 * _F - _F ** 2            # first eccentricity squared
_K0 = 0.9996                      # UTM scale factor on the central meridian


# ─── Math helpers ──────────────────────────────────────────────────────

def _deg2rad(d: float) -> float:
    return math.radians(d)


def _rad2deg(r: float) -> float:
    return math.degrees(r)


def utm_zone_for(lon: float, lat: float) -> int:
    """Return the UTM zone number (1-60) for a WGS84 point."""
    return max(1, min(60, math.floor((lon + 180) / 6) + 1))


def is_north_hemisphere(lat: float) -> bool:
    """Return True if latitude is in the northern hemisphere."""
    return lat >= 0


def lat_lon_to_utm(lon: float, lat: float) -> tuple[float, float, int]:
    """Convert WGS84 lat/lon (degrees) to UTM local coordinates (meters).

    Returns (easting, northing, utm_zone_number).
    Uses the full Transverse Mercator series expansion for accuracy at terrain-scale resolution.
    """
    zone = utm_zone_for(lon, lat)
    easting, northing, _ = _full_tm_forward(lon, lat, zone)
    return easting, northing, zone


def _meridian_arc(phi: float) -> float:
    """Meridian arc length from the equator to latitude ``phi`` (radians)."""
    e2, e4, e6 = _E2, _E2 ** 2, _E2 ** 3
    return _E * (
        (1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
        - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * phi)
        + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * phi)
        - (35 * e6 / 3072) * math.sin(6 * phi)
    )


def _full_tm_forward(lon: float, lat: float, zone: int) -> tuple[float, float, int]:
    """Full Transverse Mercator forward projection (WGS84 → UTM coords)."""
    return _tm_forward(lon, lat, (zone - 1) * 6 - 180 + 3)


def _tm_forward(lon: float, lat: float, lon0: float) -> tuple[float, float, int]:
    """Transverse Mercator forward projection about a central meridian.

    ``lon0`` is the longitude of the central meridian in degrees.
    """
    phi = _deg2rad(max(-89.9, min(89.9, lat)))
    lam = _deg2rad(lon - lon0)

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    tan_phi = math.tan(phi)

    nu = _E / math.sqrt(1.0 - _E2 * sin_phi ** 2)          # prime vertical radius
    eta2 = (_E2 / (1 - _E2)) * cos_phi ** 2                # second eccentricity²
    t2 = tan_phi ** 2

    e_easting = _K0 * nu * (
        lam * cos_phi
        + (1 - t2 + eta2) * lam ** 3 * cos_phi ** 3 / 6
        + (5 - 18 * t2 + t2 * t2 + 72 * eta2 - 58 * eta2 * t2)
        * lam ** 5 * cos_phi ** 5 / 120
    )

    n_northing = _K0 * (
        _meridian_arc(phi)
        + nu * tan_phi * (
            lam * lam * cos_phi ** 2 / 2
            + (5 - t2 + 9 * eta2 + 4 * eta2 ** 2)
            * lam ** 4 * cos_phi ** 4 / 24
            + (61 - 58 * t2 + t2 * t2 + 600 * eta2 - 330 * eta2 * t2)
            * lam ** 6 * cos_phi ** 6 / 720
        )
    )

    if lat < 0:
        n_northing += 10_000_000.0  # UTM false northing

    return e_easting + 500_000.0, n_northing, 0


def _inverse_meridian_arc(arc: float) -> float:
    """Solve M(phi) = ``arc`` for the footpoint latitude phi (radians)."""
    phi = arc / _E
    for _ in range(12):
        residual = _meridian_arc(phi) - arc
        sin_phi = math.sin(phi)
        rho = _E * (1 - _E2) / ((1 - _E2 * sin_phi ** 2) ** 1.5)
        step = residual / rho
        phi -= step
        if abs(step) < 1e-13:
            break
    return phi


def _utm_newton(
    east: float, north: float, lon0: float, *, southern: bool
) -> tuple[float, float]:
    """Solve the inverse Transverse Mercator against the forward projection."""
    x = east - 500_000.0

    # Seed from the footpoint latitude of the (unshifted) northing.
    phi = _inverse_meridian_arc((north - 10_000_000.0 if southern else north) / _K0)
    lam = lon0 + x / (_K0 * _E * math.cos(phi))

    h = 1e-8
    for _ in range(24):
        f_e, f_n, _ = _tm_forward(_rad2deg(lam), _rad2deg(phi), _rad2deg(lon0))
        r_e, r_n = f_e - east, f_n - north
        if abs(r_e) < 1e-7 and abs(r_n) < 1e-7:
            break

        # Numeric Jacobian columns: d/d(lam), d/d(phi).
        e_lam, n_lam, _ = _tm_forward(_rad2deg(lam + h), _rad2deg(phi), _rad2deg(lon0))
        e_phi, n_phi, _ = _tm_forward(_rad2deg(lam), _rad2deg(phi + h), _rad2deg(lon0))
        j11 = (e_lam - f_e) / h
        j21 = (n_lam - f_n) / h
        j12 = (e_phi - f_e) / h
        j22 = (n_phi - f_n) / h

        det = j11 * j22 - j12 * j21
        if abs(det) < 1e-12:
            break
        lam += (-r_e * j22 + r_n * j12) / det
        phi = max(_deg2rad(-89.9), min(_deg2rad(89.9), phi + (-r_n * j11 + r_e * j21) / det))

    return _rad2deg(lam), _rad2deg(phi)


def utm_to_lat_lon(
    east: float,
    north: float,
    zone: int,
    *,
    southern_hemisphere: bool,
) -> tuple[float, float]:
    """Convert UTM coordinates back to WGS84 lat/lon (degrees).

    ``southern_hemisphere`` is required. The 10,000,000 m false northing
    makes southern northings overlap the northern range almost entirely
    (33.87°S Sydney is north=6,250,948, which looks northern), and every
    southern northing is strictly below 10,000,000, so the hemisphere cannot
    be recovered from the value.
    """
    lon0 = _deg2rad((zone - 1) * 6 - 180 + 3)

    lon, lat = _utm_newton(east, north, lon0, southern=southern_hemisphere)

    if lon > 180:
        lon -= 360
    elif lon <= -180:
        lon += 360

    return lon, lat


# ── Simplified equirectangular projection (valid for small areas near center)

def simple_equirect(lon_center: float, lat_center: float):
    """Return a callable pair (to_meters, from_meters).

    For terrain generation where the study area is < 50 km, this approximation
    is perfectly adequate and avoids dependency on external projection libraries.
    The error compared to true UTM is < 0.15% for areas up to ~40 km across.
    """
    lat_rad = _deg2rad(lat_center)

    R_lat = _E * (1 - _E2) / ((1 - _E2 * math.sin(lat_rad) ** 2) ** 1.5)
    R_lon = _E * math.cos(lat_rad) / math.sqrt(1 - _E2 * math.sin(lat_rad) ** 2)

    def to_meters(lon: float, lat: float) -> tuple[float, float]:
        """(lon_deg, lat_deg) → (east_m, north_m) relative to center."""
        d_lon = math.radians(lon - lon_center)
        d_lat = _deg2rad(lat) - lat_rad
        east = d_lon * _K0 * R_lon
        north = d_lat * R_lat
        return east, north

    def from_meters(east_m: float, north_m: float) -> tuple[float, float]:
        """(east_m, north_m) relative to center → (lon_deg, lat_deg)."""
        lon = lon_center + _rad2deg(east_m / (_K0 * R_lon))
        lat = lat_center + _rad2deg(north_m / R_lat)
        return lon, lat

    return to_meters, from_meters


# ── Bounding box in meters

def bounds_to_meters(bounds):
    """Convert a GeoBounds to approximate width/height in meters.

    Uses the center latitude to compute metres-per-degree accurately.
    East–west distance is scaled by the radius of curvature in the prime
vertical; north–south distance by the meridional radius of curvature.
    """
    lat_rad = _deg2rad((bounds.south + bounds.north) / 2)

    # Meridional radius of curvature drives north-south distance; the radius of
    # curvature in the prime vertical scaled by cos(lat) drives east-west.
    R_lat = _E * (1 - _E2) / ((1 - _E2 * math.sin(lat_rad) ** 2) ** 1.5)
    R_lon = _E * math.cos(lat_rad) / math.sqrt(1 - _E2 * math.sin(lat_rad) ** 2)

    width_m = abs(_deg2rad(bounds.east - bounds.west)) * R_lon
    height_m = abs(_deg2rad(bounds.north - bounds.south)) * R_lat

    return width_m, height_m


def meters_to_degrees(width_m: float, height_m: float, center_lat: float) -> tuple[float, float]:
    """Convert meters at a given latitude back to approximate degrees."""
    lat_rad = _deg2rad(center_lat)

    R_lat = _E * (1 - _E2) / ((1 - _E2 * math.sin(lat_rad) ** 2) ** 1.5)
    R_lon = _E * math.cos(lat_rad) / math.sqrt(1 - _E2 * math.sin(lat_rad) ** 2)

    deg_w = _rad2deg(width_m / R_lon)
    deg_h = _rad2deg(height_m / R_lat)

    return deg_w, deg_h

