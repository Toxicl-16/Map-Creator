"""Terrain solid: watertight topology, physical dimensions, feature folding."""

import numpy as np
import pytest

from app.services.terrain import (
    MeshBuilder,
    TerrainGenerationError,
    TerrainSettings,
    detail_grid,
    scale_to_mm,
    upsample_bilinear,
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


def test_upsample_bilinear_preserves_corners_and_size():
    grid = np.array([[0.0, 10.0], [20.0, 30.0]])
    up = upsample_bilinear(grid, 2)

    assert up.shape == (3, 3)
    assert up[0, 0] == pytest.approx(0.0)
    assert up[0, -1] == pytest.approx(10.0)
    assert up[-1, 0] == pytest.approx(20.0)
    assert up[-1, -1] == pytest.approx(30.0)
    # The added centre sample is the mean of the four corners.
    assert up[1, 1] == pytest.approx(15.0)


def test_detail_grid_refines_coarse_grids_toward_the_target():
    settings = TerrainSettings(model_width_mm=250, model_depth_mm=220)
    grid = np.linspace(100.0, 110.0, 20 * 15).reshape(15, 20)
    refined = detail_grid(grid, settings, target_cell_mm=2.0)

    assert refined.shape[0] > grid.shape[0]
    assert 250.0 / (refined.shape[1] - 1) <= 3.0  # nearer the 2 mm target


def test_detail_grid_leaves_an_already_fine_grid_alone():
    settings = TerrainSettings(model_width_mm=250, model_depth_mm=220)
    grid = np.zeros((200, 200))  # ~1.26 mm cells, finer than the target
    assert detail_grid(grid, settings).shape == grid.shape


def test_detail_grid_backs_off_to_respect_the_node_cap():
    settings = TerrainSettings(model_width_mm=500, model_depth_mm=500)
    grid = np.zeros((100, 100))
    refined = detail_grid(grid, settings, target_cell_mm=0.5, max_nodes=55_000)

    assert refined.size <= 55_000
    assert refined.shape[0] > grid.shape[0]  # refined as far as the cap allows


def test_detail_grid_will_not_refine_past_the_cap():
    settings = TerrainSettings(model_width_mm=500, model_depth_mm=500)
    grid = np.zeros((300, 300))  # 90k nodes; refining could only blow the cap
    assert detail_grid(grid, settings, target_cell_mm=0.5).shape == grid.shape


def test_layered_terrain_folds_smooth_contour_rings(builder, synthetic_elevation):
    """Elevation layers now draw smooth contour rings on smooth terrain."""
    from app.services.features import build_contours

    builder.settings.layer_terrain = True
    builder.settings.include_contours = True
    builder.settings.contour_interval_m = 10.0
    surface = builder.surface_of(synthetic_elevation)
    contours = build_contours(
        synthetic_elevation,
        width_mm=builder.settings.model_width_mm,
        depth_mm=builder.settings.model_depth_mm,
        interval_m=10.0,
    )
    vertices, triangles, final, stats = builder.build_with_features(
        synthetic_elevation, contours=contours
    )

    assert stats["contours"] > 0
    # The rings raise the surface but never cut terraces into it.
    assert (final >= surface - 1e-9).all()
    result = validate_mesh(vertices, triangles)
    assert result.watertight, result.issues


def test_scale_to_mm_stays_smooth_with_layers():
    settings = TerrainSettings(
        model_width_mm=300, model_depth_mm=300, resolution_m=30,
        min_altitude_mm=2, max_altitude_mm=42, base_thickness_mm=3,
        layer_terrain=True, layer_interval_m=20,
    )
    elev = np.linspace(2400, 2600, 400).reshape(20, 20)
    scaled = scale_to_mm(elev, settings, 1500.0, 1500.0)

    # No terracing: the surface keeps the terrain's own gradation rather than
    # collapsing onto a handful of discrete plateaus.
    assert len(np.unique(np.round(scaled, 6))) > 100
    assert np.corrcoef(scaled.ravel(), elev.ravel())[0, 1] > 0.99


def test_layered_mesh_is_still_watertight(builder, synthetic_elevation):
    builder.settings.layer_terrain = True
    builder.settings.layer_interval_m = 25.0

    vertices, triangles = builder.build(synthetic_elevation)
    result = validate_mesh(vertices, triangles)

    assert result.watertight, result.issues
    assert result.manifold, result.issues
    assert result.issues == []


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


def _interior_vertical_face_count(vertices, triangles, builder) -> int:
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.int64)
    w = builder.settings.model_width_mm
    d = builder.settings.model_depth_mm
    on_edge = (
        np.isclose(verts[:, 0], 0.0) | np.isclose(verts[:, 0], w)
        | np.isclose(verts[:, 1], 0.0) | np.isclose(verts[:, 1], d)
    )
    count = 0
    for tri in tris:
        pts = verts[tri]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-12 or abs(normal[2]) / norm >= 0.05:
            continue
        if not on_edge[tri].any():
            count += 1
    return count


def test_contours_alone_have_no_interior_vertical_walls(builder, synthetic_elevation):
    """The smoothed terrain and its contour rings must stay wall-free: only the
    outer skirt is vertical."""
    from app.services.features import build_contours

    contours = build_contours(
        synthetic_elevation, width_mm=300.0, depth_mm=200.0, interval_m=40.0
    )
    vertices, triangles, _final, _stats = builder.build_with_features(
        synthetic_elevation, contours=contours
    )

    assert _interior_vertical_face_count(vertices, triangles, builder) == 0


def test_road_raises_a_strip_above_the_smooth_terrain(builder, synthetic_elevation, yosemite_bounds):
    """Roads add a raised rectangular strip on top of the smooth surface."""
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
        [road], transform, surface, width_mm=4.0, height_mm=2.0, min_width_mm=2.0
    )
    vertices, triangles, final, _stats = builder.build_with_features(
        synthetic_elevation, roads=roads
    )

    assert roads["mask"].any()
    # The road footprint is lifted by the full road height above the terrain.
    lifted = final[roads["mask"]] - surface[roads["mask"]]
    assert np.allclose(lifted, 2.0)

    result = validate_mesh(vertices, triangles)
    assert result.watertight and result.manifold



def test_crisp_building_roof_is_flat(builder):
    """A covered cell renders as a flat roof with vertical walls, not a tent."""
    rows, cols = 18, 20
    flat = np.full((rows, cols), 100.0)
    surface = builder.surface_of(flat)
    h0 = float(surface.min())

    mask = np.zeros((rows, cols), dtype=bool)
    mask[6:8, 7:9] = True  # a 2x2 block of cells in the covered centre
    buildings = {"mask": mask, "z": np.where(mask, surface + 10.0, 0.0)}

    vertices, triangles, _final, _stats = builder.build_with_features(flat, buildings=buildings)
    result = validate_mesh(vertices, triangles)
    assert result.watertight, result.issues
    assert result.manifold, result.issues

    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.int64)
    horizontal = []
    for tri in tris:
        pts = verts[tri]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if np.linalg.norm(normal) > 1e-12 and abs(normal[2]) / np.linalg.norm(normal) > 0.999:
            horizontal.append(float(np.mean(pts[:, 2])))

    # Terrain is one flat level and the bottom cap is at z=0; the building adds
    # exactly one higher level, so there are three horizontal plateaus and the
    # roof is completely flat.
    heights = set(np.round(horizontal, 6))
    assert heights == {0.0, round(h0, 6), round(h0 + 10.0, 6)}
    roof_tris = [h for h in horizontal if h > h0 + 1.0]
    assert len(roof_tris) == 2 * int(mask.sum())  # two triangles per covered cell

    # The perimeter of the block is rimmed by vertical wall faces (a 2x2 block
    # has 8 outer cell edges, each a two-triangle quad dragging down from the
    # roof toward the terrain).
    walls = 0
    for tri in tris:
        pts = verts[tri]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if np.linalg.norm(normal) > 1e-12 and abs(normal[2]) / np.linalg.norm(normal) < 0.01:
            if (np.max(pts[:, 2]) > h0 + 5.0) and np.min(pts[:, 2]) < h0 + 5.0:
                walls += 1
    assert walls >= 16


def test_diagonal_corners_stay_manifold(builder, synthetic_elevation):
    """Two blocks touching at a single corner must not pinch the mesh."""
    surface = builder.surface_of(synthetic_elevation)
    rows, cols = surface.shape

    mask = np.zeros((rows, cols), dtype=bool)
    mask[25, 30] = True
    mask[26, 31] = True  # diagonally adjacent, sharing only a corner vertex
    buildings = {
        "mask": mask,
        "z": np.where(mask, surface + 5.0, 0.0),
    }

    vertices, triangles, _final, _stats = builder.build_with_features(
        synthetic_elevation, buildings=buildings
    )
    result = validate_mesh(vertices, triangles)

    assert result.watertight, result.issues
    assert result.manifold, result.issues
    assert result.non_manifold_edges == 0
    assert result.boundary_edges == 0


def test_adjacent_flat_roofs_level_with_terrain_stay_sealed(builder, synthetic_elevation):
    """Adjacent flat roofs that sit exactly on their terrain corner must not
    leave slivers: rounding the roofs but not the terrain used to spawn two
    near-coincident levels per shared vertex, which broke the wall strips."""
    surface = builder.surface_of(synthetic_elevation)
    rows, cols = surface.shape

    mask = np.zeros((rows, cols), dtype=bool)
    mask[15:45, 20:60] = True
    mask[20:25, 30:40] = False  # a courtyard so walls face one another
    # Roofs left below the floor: the builder lifts each to its own highest
    # terrain corner, so many shared vertices carry two near-equal levels.
    buildings = {"mask": mask, "z": np.where(mask, 0.0, 0.0)}

    vertices, triangles, _final, _stats = builder.build_with_features(
        synthetic_elevation, buildings=buildings
    )
    result = validate_mesh(vertices, triangles)

    assert result.watertight, result.issues
    assert result.manifold, result.issues
    assert result.non_manifold_edges == 0
    assert result.boundary_edges == 0


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