"""Projection math: WGS84 constants, UTM round trips, local metre frames."""

import math

import pytest

from app.utils.projection import (
    bounds_to_meters,
    is_north_hemisphere,
    lat_lon_to_utm,
    simple_equirect,
    utm_to_lat_lon,
)

PLACES = [
    ("Yosemite", 37.7327, -119.6057, True),
    ("San Francisco", 37.7749, -122.4194, True),
    ("London", 51.5074, -0.1278, True),
    ("Tokyo", 35.6762, 139.6503, True),
    ("Paris", 48.8566, 2.3522, True),
    ("Sydney", -33.8688, 151.2093, False),
    ("Cape Town", -33.9249, 18.4241, False),
    ("Rio", -22.9068, -43.1729, False),
]


def test_flattening_constants_are_physically_sane():
    """The flattening must be ~1/298.26, not its own inverse."""
    from app.utils import projection as p

    assert p._F == pytest.approx(1 / 298.257223563, rel=1e-9)
    assert p._E2 == pytest.approx(p._F * (2 - p._F), rel=1e-12)


@pytest.mark.parametrize("name,lat,lon,south", PLACES)
def test_utm_round_trip(name, lat, lon, south):
    east, north, zone = lat_lon_to_utm(lon, lat)
    back_lon, back_lat = utm_to_lat_lon(
        east, north, zone, southern_hemisphere=not is_north_hemisphere(lat)
    )

    assert back_lat == pytest.approx(lat, abs=1e-7), name
    assert back_lon == pytest.approx(lon, abs=1e-7), name


def test_utm_easting_is_near_zone_central_meridian():
    """Yosemite sits ~2.6° west of zone 11's central meridian, so <400k."""
    east, _north, zone = lat_lon_to_utm(-119.6057, 37.7327)
    assert zone == 11
    assert 200_000 < east < 400_000


def test_utm_easting_is_near_500k_on_the_central_meridian():
    east, _north, _zone = lat_lon_to_utm(-117.0, 37.7327)  # zone 11 CM is -117
    assert east == pytest.approx(500_000.0, abs=1_000)


def test_utm_southern_hemisphere_has_false_northing():
    """Sydney (33.87°S) is ~6.25M northings once the false northing is added."""
    _e, north, _z = lat_lon_to_utm(151.2093, -33.8688)
    assert is_north_hemisphere(-33.8688) is False
    assert 6_000_000 < north < 7_000_000


def test_utm_southern_flag_is_required_for_accuracy():
    """Sydney's northing is indistinguishable from a northern one."""
    east, north, zone = lat_lon_to_utm(151.2093, -33.8688)
    _lon, right_lat = utm_to_lat_lon(east, north, zone, southern_hemisphere=True)

    assert right_lat == pytest.approx(-33.8688, abs=1e-7)

    # Solving the same numbers as northern lands near the pole instead.
    _lon, wrong_lat = utm_to_lat_lon(east, north, zone, southern_hemisphere=False)
    assert abs(wrong_lat - right_lat) > 50


def test_utm_inverse_requires_the_hemisphere_flag():
    """The flag is mandatory: guessing would silently return garbage."""
    east, north, zone = lat_lon_to_utm(151.2093, -33.8688)
    with pytest.raises(TypeError):
        utm_to_lat_lon(east, north, zone)


def test_bounds_to_meters_matches_known_distance(yosemite_bounds):
    """Yosemite box is ~2.8 km east-west, ~3.3 km north-south."""
    width_m, height_m = bounds_to_meters(yosemite_bounds)

    assert 2_500 < width_m < 3_100
    assert 3_100 < height_m < 3_500


def test_bounds_to_meters_scales_with_latitude():
    """A degree of longitude is shorter at high latitude."""
    from app.models import GeoBounds

    equator = bounds_to_meters(GeoBounds(west=0, south=0, east=1, north=1))
    high = bounds_to_meters(GeoBounds(west=0, south=59, east=1, north=60))

    assert equator[0] == pytest.approx(111_195, rel=0.01)
    assert high[0] < equator[0] * 0.55
    # Meridional spacing is nearly constant, so height barely changes.
    assert high[1] == pytest.approx(equator[1], rel=0.02)


def test_simple_equirect_round_trip_and_center():
    to_meters, from_meters = simple_equirect(-119.6057, 37.7327)

    assert to_meters(-119.6057, 37.7327) == (0.0, 0.0)

    lon, lat = from_meters(1200.0, -800.0)
    back = to_meters(lon, lat)
    assert back[0] == pytest.approx(1200.0, abs=1e-6)
    assert back[1] == pytest.approx(-800.0, abs=1e-6)


def test_simple_equirect_is_north_positive():
    to_meters, _ = simple_equirect(0.0, 0.0)
    east, north = to_meters(0.01, 0.01)
    assert east > 0 and north > 0


def test_degree_inputs_are_not_treated_as_radians():
    """Regression: raw degrees were fed into metre conversions."""
    to_meters, _ = simple_equirect(-3.0, 12.7)
    east, north = to_meters(-2.95, 12.79)
    # ~5.2 km east, ~10 km north of the centre - not thousands of km.
    assert 4_000 < east < 6_000
    assert 9_000 < north < 11_000
    assert math.isfinite(east) and math.isfinite(north)