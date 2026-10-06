"""Terrain solid: watertight topology, physical dimensions, feature folding."""

import numpy as np
import pytest

from app.services.terrain import (
    MeshBuilder,
    TerrainGenerationError,
    TerrainSettings,
    scale_to_mm,
)
from app.services.validation import validate_mesh


def test_mesh_is_watertight_and_manifold(builder, synthetic_elevation):
    vertices, triangles = builder.build(synthetic_elevation)
    result = validate_mesh(vertices, triangles)

    assert result.watertight, result.issues
    assert result.manifold, result.issues
    assert result.boundary_edges == 0
    assert result.non_manifold_edges == 0
    assert result.degenerate_triangles == 0
    assert result.issues == []


def test_mesh_has_positive_outward_volume(builder, synthetic_elevation):
    vertices, triangles = builder.build(synthetic_elevation)
    result = validate_mesh(vertices, triangles)

    # Positive volume means the winding is outward, not inverted.
    assert result.volume_mm3 > 0
    assert not any("winding is inverted" in issue for issue in result.issues)


def test_mesh_honours_requested_footprint(builder, synthetic_elevation):
    settings = builder.settings
    vertices, _triangles = builder.build(synthetic_elevation)
    result = validate_mesh(vertices, _triangles)

    assert result.bbox_mm[0] == pytest.approx(settings.model_width_mm, abs=1e-6)
    assert result.bbox_mm[1] == pytest.approx(settings.model_depth_mm, abs=1e-6)


def test_relief_respects_max_altitude(builder, synthetic_elevation):
    settings = builder.settings
    surface = builder.surface_of(synthetic_elevation)
    relief = surface.max() - settings.base_thickness_mm

    assert relief <= settings.max_altitude_mm + 1e-6


def test_relief_preserves_ordering(builder, synthetic_elevation):
    """Scaling must not invert or flatten the terrain's shape."""
    surface = builder.surface_of(synthetic_elevation)
    flat = surface.ravel()

    assert np.corrcoef(flat, synthetic_elevation.ravel())[0, 1] > 0.99


def test_flat_terrain_still_produces_a_printable_solid(builder):
    flat = np.full((12, 15), 100.0)
    vertices, triangles = builder.build(flat)
    result = validate_mesh(vertices, triangles)

    assert result.watertight
    assert result.manifold
    assert result.bbox_mm[2] > 0


def test_degenerate_grid_is_rejected(builder):
    with pytest.raises(TerrainGenerationError):
        builder.build(np.zeros((1, 10)))
    with pytest.raises(TerrainGenerationError):
        builder.build(np.zeros((4,)))


def test_winding_is_consistent_for_non_square_grids(builder, synthetic_elevation):
    """A tall, narrow grid must not flip any face."""
    settings = builder.settings
    settings.model_width_mm = 100
    settings.model_depth_mm = 400

    vertices, triangles = builder.build(synthetic_elevation)
    result = validate_mesh(vertices, triangles)

    assert result.watertight
    assert result.volume_mm3 > 0


def test_single_cell_grid_is_watertight(yosemite_bounds):
    """A 2x2 grid is the smallest legal input and must still seal."""
    settings = TerrainSettings(
        west=yosemite_bounds.west,
        south=yosemite_bounds.south,
        east=yosemite_bounds.east,
        north=yosemite_bounds.north,
    )
    vertices, triangles = MeshBuilder(settings).build(np.full((2, 2), 500.0))
    result = validate_mesh(vertices, triangles)

    assert result.watertight
    assert result.manifold


def test_features_are_folded_into_the_surface(builder, synthetic_elevation):
    from app.services.features import build_contours

    surface = builder.surface_of(synthetic_elevation)
    rows, cols = surface.shape

    # A full-grid road ribbon, as build_roads would produce.
    roads = {"mask": np.zeros((rows, cols), dtype=bool), "z": surface.copy()}
    roads["mask"][:5, :5] = True
    roads["z"][:5, :5] = surface[:5, :5] + 0.6

    contours = build_contours(
        synthetic_elevation,
        width_mm=300.0,
        depth_mm=200.0,
        interval_m=50.0,
    )

    _v, _t, final, stats = builder.build_with_features(
        synthetic_elevation, roads=roads, contours=contours
    )

    assert stats["roads"] == 25
    assert stats["contours"] == int(contours["mask"].sum())
    assert stats["contours"] > 0
    # Road cells rise to the ribbon height, or higher where a contour overlaps.
    road_delta = final[:5, :5] - surface[:5, :5]
    assert (road_delta >= 0.6 - 1e-6).all()
    assert np.allclose(road_delta[road_delta <= 0.6 + 1e-9], 0.6)
    # Contour cells away from the road ribbon stay within one contour height.
    away = contours["mask"] & ~roads["mask"]
    delta = np.abs(final[away] - surface[away])
    assert delta.max() <= contours["offset"].max() + 1e-6


def test_feature_folding_keeps_the_mesh_sealed(builder, synthetic_elevation, yosemite_bounds):
    from app.services.features import ModelTransform, build_roads

    surface = builder.surface_of(synthetic_elevation)
    transform = ModelTransform.create(yosemite_bounds, 300.0, 200.0)
    road = {
        "type": "way",
        "tags": {"highway": "residential"},
        "geometry": [
            {"lat": 37.7215, "lon": -119.6165},
            {"lat": 37.7485, "lon": -119.5935},
        ],
    }
    roads = build_roads(
        [road], transform, surface, width_mm=4.0, height_mm=0.6, min_width_mm=2.0
    )

    vertices, triangles, _final, _stats = builder.build_with_features(
        synthetic_elevation, roads=roads
    )
    result = validate_mesh(vertices, triangles)

    assert result.watertight
    assert result.manifold


def test_contours_can_be_engraved(builder, synthetic_elevation):
    from app.services.features import build_contours

    raised = build_contours(
        synthetic_elevation, width_mm=300.0, depth_mm=200.0, interval_m=50.0
    )
    engraved = build_contours(
        synthetic_elevation,
        width_mm=300.0,
        depth_mm=200.0,
        interval_m=50.0,
        engraved=True,
    )

    assert raised["offset"].min() >= 0
    assert engraved["offset"].max() <= 0
    # Engraving must not remove the lines themselves.
    assert np.array_equal(raised["mask"], engraved["mask"])


def test_scale_to_mm_is_monotonic(builder):
    """Relief large enough to clear the printability floor keeps its order."""
    settings = builder.settings
    grid = np.array([[0.0, 100.0], [200.0, 300.0]])
    scaled = scale_to_mm(grid, settings, 100.0, 100.0)

    assert scaled[0, 0] < scaled[0, 1] < scaled[1, 1]
    # Relief is fitted between the printability floor and the altitude cap.
    assert scaled.min() == pytest.approx(settings.min_altitude_mm, abs=1e-6)
    assert scaled.max() == pytest.approx(settings.max_altitude_mm, abs=1e-6)


def test_scale_to_mm_applies_printability_floor(builder):
    """Relief thinner than the floor is lifted to it rather than vanishing."""
    settings = builder.settings
    grid = np.array([[0.0, 10.0], [20.0, 30.0]])
    scaled = scale_to_mm(grid, settings, 1000.0, 1000.0)

    assert scaled.min() >= settings.min_altitude_mm - 1e-9
    assert scaled.max() >= settings.min_altitude_mm - 1e-9