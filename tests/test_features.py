"""Feature rasterisation: model transform, roads, buildings, contours."""

import numpy as np
import pytest

from app.services.features import (
    ModelTransform,
    _building_height_m,
    _marching_squares,
    build_buildings,
    build_contours,
    build_roads,
)


@pytest.fixture
def transform(yosemite_bounds):
    return ModelTransform.create(yosemite_bounds, 300.0, 200.0)


def road_way():
    return {
        "type": "way",
        "tags": {"highway": "residential"},
        "geometry": [
            {"lat": 37.7215, "lon": -119.6165},
            {"lat": 37.7485, "lon": -119.5935},
        ],
    }


def building_way():
    return {
        "type": "way",
        "tags": {"building": "house"},
        "geometry": [
            {"lat": 37.7300, "lon": -119.6100},
            {"lat": 37.7300, "lon": -119.6080},
            {"lat": 37.7320, "lon": -119.6080},
            {"lat": 37.7320, "lon": -119.6100},
            {"lat": 37.7300, "lon": -119.6100},
        ],
    }


def test_transform_maps_corners_onto_the_model_plane(transform, yosemite_bounds):
    b = yosemite_bounds

    assert transform.to_model(b.west, b.south) == pytest.approx((-150.0, -100.0), abs=1e-6)
    assert transform.to_model(b.east, b.south) == pytest.approx((150.0, -100.0), abs=1e-6)
    assert transform.to_model(b.east, b.north) == pytest.approx((150.0, 100.0), abs=1e-6)
    assert transform.to_model(b.west, b.north) == pytest.approx((-150.0, 100.0), abs=1e-6)


def test_transform_origin_is_the_selection_centre(transform, yosemite_bounds):
    centre = (
        (yosemite_bounds.west + yosemite_bounds.east) / 2,
        (yosemite_bounds.south + yosemite_bounds.north) / 2,
    )
    assert transform.to_model(*centre) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_transform_output_is_millimetres_not_metres(transform, yosemite_bounds):
    """Regression: to_model used to return raw metre offsets."""
    east_m, north_m = transform.to_model(yosemite_bounds.east, yosemite_bounds.north)
    assert abs(east_m) <= transform.width_mm / 2 + 1e-6
    assert abs(north_m) <= transform.depth_mm / 2 + 1e-6


def test_transform_is_north_up(transform, yosemite_bounds):
    """Higher latitude must map to a larger model y."""
    low = transform.to_model(yosemite_bounds.west, yosemite_bounds.south)[1]
    high = transform.to_model(yosemite_bounds.west, yosemite_bounds.north)[1]
    assert high > low


def test_roads_cover_the_centreline(transform, synthetic_elevation):
    result = build_roads(
        [road_way()], transform, synthetic_elevation,
        width_mm=4.0, height_mm=0.6, min_width_mm=2.0,
    )
    mask, z = result["mask"], result["z"]

    assert mask.sum() > 0
    # Every masked cell has a finite height, so nothing collapses to the floor.
    assert np.isfinite(z[mask]).all()
    assert z[mask].min() >= synthetic_elevation.min()


def test_roads_follow_the_terrain_profile(transform, synthetic_elevation):
    result = build_roads(
        [road_way()], transform, synthetic_elevation,
        width_mm=4.0, height_mm=0.8, min_width_mm=2.0,
    )
    mask, z = result["mask"], result["z"]

    # The ribbon tracks the ground rather than sitting at a constant altitude.
    assert z[mask].std() > 1.0


def test_roads_width_scales_with_setting(transform, synthetic_elevation):
    narrow = build_roads(
        [road_way()], transform, synthetic_elevation,
        width_mm=2.0, height_mm=0.6, min_width_mm=1.0,
    )
    wide = build_roads(
        [road_way()], transform, synthetic_elevation,
        width_mm=8.0, height_mm=0.6, min_width_mm=8.0,
    )
    assert wide["mask"].sum() > narrow["mask"].sum()


def test_road_is_a_raised_rectangle(transform, synthetic_elevation):
    """A road is a flat-topped ribbon raised by the full height with no taper."""
    result = build_roads(
        [road_way()], transform, synthetic_elevation,
        width_mm=3.0, height_mm=2.0, min_width_mm=1.5,
    )
    raise_field = result["raise"]

    assert raise_field.max() == pytest.approx(2.0)
    # Crisp cross-section: every lifted vertex is at the full road height, so
    # the strip has vertical sides instead of a feathered shoulder.
    levels = np.unique(np.round(raise_field, 9))
    assert set(levels.tolist()) <= {0.0, 2.0}
    # The footprint and the raised cells agree exactly.
    assert np.array_equal(result["mask"], raise_field > 0)
    assert result["mask"].any()


def test_contour_rings_feather_smoothly(synthetic_elevation):
    result = build_contours(
        synthetic_elevation,
        width_mm=300.0, depth_mm=200.0,
        interval_m=50.0, thickness_mm=0.6, height_mm=0.8,
    )
    offset = result["offset"]
    peak = float(offset[result["mask"]].max())
    assert peak > 0
    # Shoulder cells carry intermediate heights between the line and the ground.
    assert np.any((offset > 1e-9) & (offset < peak - 1e-9))
    # Smooth ridges: nothing exceeds the tallest ring and nothing digs below 0.
    assert offset.min() >= 0
    assert offset.max() <= 1.9 * 0.8 + 1e-9


def test_roads_outside_the_model_are_ignored(transform, synthetic_elevation):
    far_away = {
        "type": "way",
        "tags": {"highway": "residential"},
        "geometry": [
            {"lat": 10.0, "lon": -119.6},
            {"lat": 10.1, "lon": -119.6},
        ],
    }
    result = build_roads(
        [far_away], transform, synthetic_elevation,
        width_mm=4.0, height_mm=0.6, min_width_mm=2.0,
    )
    assert result["mask"].sum() == 0


def test_buildings_raise_above_local_ground(transform, synthetic_elevation):
    result = build_buildings(
        [building_way()], transform, synthetic_elevation,
        height_mm=4.0, min_height_mm=1.5,
    )
    mask, z = result["mask"], result["z"]

    assert mask.sum() > 0
    assert np.isfinite(z[mask]).all()
    # Raised relative to the ground under the same footprint, not a global mean.
    ground = synthetic_elevation[mask]
    assert (z[mask] > ground).all()
    assert z[mask].std() == pytest.approx(0.0, abs=1e-9)


def test_footprint_spanning_the_model_edge_is_rasterised(yosemite_bounds, synthetic_elevation):
    """Regression: cell-centred axes are one shorter than the mask, and a
    footprint pinned to the far model corner used to write one extra row/column."""
    edge = ModelTransform.create(yosemite_bounds, 300.0, 200.0)
    way = {
        "type": "way",
        "tags": {"building": "yes"},
        "geometry": [
            {"lat": 37.7400, "lon": -119.6100},
            {"lat": 37.7400, "lon": -119.5900},
            {"lat": 37.7500, "lon": -119.5900},
            {"lat": 37.7500, "lon": -119.6100},
            {"lat": 37.7400, "lon": -119.6100},
        ],
    }
    result = build_buildings(
        [way], edge, synthetic_elevation, height_mm=4.0, min_height_mm=1.5
    )
    mask, z = result["mask"], result["z"]

    assert mask.sum() > 0
    assert np.isfinite(z[mask]).all()
    # The last row/column of the mask is never a cell centre, so it stays clear.
    assert not mask[-1, :].any()
    assert not mask[:, -1].any()


def test_building_height_prefers_explicit_tags():
    assert _building_height_m({"height": "12.5"}, 6.0) == pytest.approx(12.5)
    assert _building_height_m({"building:levels": "4"}, 6.0) == pytest.approx(12.0)
    assert _building_height_m({"building": "church"}, 6.0) == pytest.approx(9.0)


def test_building_height_ignores_nonsense_tags():
    assert _building_height_m({"height": "abc"}, 6.0) == pytest.approx(6.0)
    assert _building_height_m({"height": "900"}, 6.0) == pytest.approx(6.0)
    assert _building_height_m({}, 6.0) == pytest.approx(6.0)


def _building_z(bounds, height_scale=1.0, tags=None):
    """Peak height above ground for one footprint at a given scale."""
    way = building_way()
    if tags is not None:
        way["tags"] = tags
    transform = ModelTransform.create(bounds, 300.0, 200.0)
    surface = np.full((64, 64), 10.0)
    result = build_buildings(
        [way], transform, surface,
        height_mm=4.0, min_height_mm=1.0, height_scale=height_scale,
    )
    mask = result["mask"]
    assert mask.any()
    return float((result["z"][mask] - surface[mask]).max())


def test_building_height_scale_multiplies_every_block(yosemite_bounds):
    base = _building_z(yosemite_bounds, height_scale=1.0)
    doubled = _building_z(yosemite_bounds, height_scale=2.0)
    assert doubled == pytest.approx(2.0 * base)


def test_tagged_heights_keep_their_relative_stature(yosemite_bounds):
    plain = _building_z(yosemite_bounds, tags={"building": "yes"})
    church = _building_z(yosemite_bounds, tags={"building": "yes", "height": "24"})
    shed = _building_z(yosemite_bounds, tags={"building": "yes", "height": "3"})

    assert church > plain > shed


def test_extreme_tagged_heights_are_clamped(yosemite_bounds):
    plain = _building_z(yosemite_bounds, tags={"building": "yes"})
    tower = _building_z(yosemite_bounds, tags={"building": "yes", "height": "200"})
    assert tower <= 4.0 * plain + 1e-9


def test_contours_are_raised_by_default(synthetic_elevation):
    result = build_contours(
        synthetic_elevation, width_mm=300.0, depth_mm=200.0, interval_m=50.0
    )
    assert result["mask"].sum() > 0
    assert result["offset"].min() >= 0
    assert result["offset"].max() > 0


def test_contours_can_be_engraved_below_terrain(synthetic_elevation):
    result = build_contours(
        synthetic_elevation,
        width_mm=300.0,
        depth_mm=200.0,
        interval_m=50.0,
        engraved=True,
    )
    assert result["offset"].max() <= 0
    assert result["offset"].min() < 0


def test_contour_thickness_is_in_millimetres(synthetic_elevation):
    """Regression: thickness was applied in grid-index units."""
    thin = build_contours(
        synthetic_elevation,
        width_mm=300.0, depth_mm=200.0,
        interval_m=50.0, thickness_mm=0.5,
    )
    thick = build_contours(
        synthetic_elevation,
        width_mm=300.0, depth_mm=200.0,
        interval_m=50.0, thickness_mm=4.0,
    )
    assert thick["mask"].sum() > thin["mask"].sum()


def test_major_contours_are_taller(synthetic_elevation):
    result = build_contours(
        synthetic_elevation,
        width_mm=300.0, depth_mm=200.0,
        interval_m=50.0, height_mm=0.8, major_every=5,
    )
    heights = np.unique(result["offset"][result["mask"]])
    assert heights.max() > heights.min()


def test_zero_interval_produces_no_contours(synthetic_elevation):
    result = build_contours(
        synthetic_elevation, width_mm=300.0, depth_mm=200.0, interval_m=0
    )
    assert result["mask"].sum() == 0


def test_marching_squares_finds_the_50_percent_contour():
    grid = np.array([[0.0, 0.0], [0.0, 0.0]])
    grid[0, 0] = 0.0
    grid = np.array([[1.0, 0.0], [0.0, 0.0]])

    segments = _marching_squares(grid, 0.5)
    assert len(segments) == 1
    (p, q) = segments[0]
    assert all(0.0 <= v <= 1.0 for v in (*p, *q))


def test_marching_squares_returns_nothing_outside_the_range():
    grid = np.zeros((3, 3))
    assert _marching_squares(grid, 1.0) == []
    assert _marching_squares(grid, -1.0) == []


def test_road_ribbon_spans_a_coarse_grid_without_gaps(transform):
    """A road narrower than a cell must still form a connected ribbon."""
    rows = cols = 12
    surface = np.zeros((rows, cols))
    way = {
        "type": "way",
        "tags": {"highway": "residential"},
        "geometry": [
            {"lat": 37.735, "lon": -119.62 + 0.03 * k / 200.0} for k in range(201)
        ],
    }
    out = build_roads(
        [way], transform, surface,
        width_mm=3.0, height_mm=2.0, min_width_mm=1.5,
    )
    mask = out["mask"]
    assert mask.any()

    columns_hit = np.where(mask.any(axis=0))[0]
    assert columns_hit.size >= 3
    # Every column the road crosses is covered: no gaps along its length.
    assert np.all(np.diff(columns_hit) == 1)