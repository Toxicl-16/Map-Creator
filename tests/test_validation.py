"""Validation and binary STL round-tripping."""

import struct

import numpy as np
import pytest

from app.services.terrain import export_to_stl
from app.services.validation import validate_mesh, validate_stl_file

STL_HEADER_BYTES = 80
STL_FACET_BYTES = 50


@pytest.fixture
def cube_mesh():
    """Axis-aligned unit cube: 8 vertices, 12 triangles, outward winding."""
    vertices = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 2, 1], [0, 3, 2],      # bottom, -Z
            [4, 5, 6], [4, 6, 7],      # top, +Z
            [0, 1, 5], [0, 5, 4],      # south, -Y
            [1, 2, 6], [1, 6, 5],      # east, +X
            [2, 3, 7], [2, 7, 6],      # north, +Y
            [3, 0, 4], [3, 4, 7],      # west, -X
        ],
        dtype=np.int64,
    )
    return vertices, triangles


def test_cube_is_watertight_with_volume_one(cube_mesh):
    vertices, triangles = cube_mesh
    result = validate_mesh(vertices, triangles)

    assert result.watertight
    assert result.manifold
    assert result.volume_mm3 == pytest.approx(1.0)
    assert result.bbox_mm == (1.0, 1.0, 1.0)
    assert result.issues == []


def test_inverted_winding_is_detected(cube_mesh):
    vertices, triangles = cube_mesh
    result = validate_mesh(vertices, triangles[:, ::-1])

    assert any("winding is inverted" in issue for issue in result.issues)
    assert not result.manifold


def test_open_surface_is_not_watertight():
    """A single quad has four boundary edges."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    result = validate_mesh(vertices, triangles)

    assert not result.watertight
    assert result.boundary_edges == 4


def test_degenerate_triangle_is_reported():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    triangles = np.array([[0, 1, 1], [0, 2, 3]], dtype=np.int64)
    result = validate_mesh(vertices, triangles)

    assert result.degenerate_triangles == 1
    assert not result.manifold


def test_empty_mesh_is_reported():
    result = validate_mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    assert result.issues == ["Mesh contains no triangles."]


def test_stl_round_trip_preserves_topology_and_volume(builder, synthetic_elevation, tmp_path):
    vertices, triangles = builder.build(synthetic_elevation)
    mesh = validate_mesh(vertices, triangles)

    path = tmp_path / "model.stl"
    export_to_stl(vertices, triangles, path)
    from_file = validate_stl_file(path, mesh.triangles)

    assert from_file.valid_stl
    assert from_file.watertight
    assert from_file.manifold
    assert from_file.triangles == mesh.triangles
    assert from_file.volume_mm3 == pytest.approx(mesh.volume_mm3, rel=1e-4)
    assert from_file.issues == []


def test_stl_file_size_is_exactly_header_plus_facets(cube_mesh, tmp_path):
    vertices, triangles = cube_mesh
    path = tmp_path / "cube.stl"
    export_to_stl(vertices, triangles, path)

    blob = path.read_bytes()
    (count,) = struct.unpack("<I", blob[STL_HEADER_BYTES : STL_HEADER_BYTES + 4])

    assert count == len(triangles)
    assert len(blob) == STL_HEADER_BYTES + 4 + count * STL_FACET_BYTES


def test_every_facet_is_50_bytes_and_finite(cube_mesh, tmp_path):
    """Regression: the attribute word must interleave, not trail the facets."""
    vertices, triangles = cube_mesh
    path = tmp_path / "cube.stl"
    export_to_stl(vertices, triangles, path)

    blob = path.read_bytes()
    (count,) = struct.unpack("<I", blob[STL_HEADER_BYTES : STL_HEADER_BYTES + 4])
    facets = np.frombuffer(
        blob, dtype=np.uint8, count=count * STL_FACET_BYTES, offset=STL_HEADER_BYTES + 4
    ).reshape(count, STL_FACET_BYTES)
    values = facets[:, :48].copy().view(np.float32).reshape(count, 12)

    assert np.isfinite(values).all()
    # Attributes are the trailing two bytes of each facet.
    assert not facets[:, 48:].any()


def test_stored_facet_normals_are_unit_length_and_point_outward(
    builder, synthetic_elevation, tmp_path
):
    """The binary STL spec requires unit normals agreeing with the winding."""
    vertices, triangles = builder.build(synthetic_elevation)
    path = tmp_path / "model.stl"
    export_to_stl(vertices, triangles, path)

    blob = path.read_bytes()
    (count,) = struct.unpack("<I", blob[STL_HEADER_BYTES : STL_HEADER_BYTES + 4])
    facets = np.frombuffer(
        blob, dtype=np.uint8, count=count * STL_FACET_BYTES, offset=STL_HEADER_BYTES + 4
    ).reshape(count, STL_FACET_BYTES)
    values = facets[:, :48].copy().view(np.float32).reshape(count, 12)

    stored = values[:, 0:3].astype(np.float64)
    corners = values[:, 3:12].astype(np.float64).reshape(count, 3, 3)

    magnitudes = np.linalg.norm(stored, axis=1)
    assert np.allclose(magnitudes, 1.0, atol=1e-3)

    # Recompute each normal from the stored winding; it must match.
    recomputed = np.cross(corners[:, 0] - corners[:, 1], corners[:, 0] - corners[:, 2])
    recomputed /= np.linalg.norm(recomputed, axis=1)[:, None]
    assert np.allclose(stored, recomputed, atol=1e-3)

    # Every normal must point away from the model interior, checked with the
    # divergence theorem: outward winding encloses a positive volume.
    volume = float(np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() / 6.0)
    assert volume > 0


def test_stl_vertex_table_is_recovered_by_welding(cube_mesh, tmp_path):
    """A facet soup has no index table; shared corners must be re-matched."""
    vertices, triangles = cube_mesh
    path = tmp_path / "cube.stl"
    export_to_stl(vertices, triangles, path)

    result = validate_stl_file(path, len(triangles))
    assert result.vertices == 8


def test_truncated_stl_is_rejected(cube_mesh, tmp_path):
    vertices, triangles = cube_mesh
    path = tmp_path / "cube.stl"
    export_to_stl(vertices, triangles, path)

    truncated = tmp_path / "truncated.stl"
    truncated.write_bytes(path.read_bytes()[:-10])

    result = validate_stl_file(truncated, len(triangles))
    assert not result.valid_stl
    assert any("size mismatch" in issue for issue in result.issues)


def test_triangle_count_mismatch_is_rejected(cube_mesh, tmp_path):
    vertices, triangles = cube_mesh
    path = tmp_path / "cube.stl"
    export_to_stl(vertices, triangles, path)

    result = validate_stl_file(path, len(triangles) + 1)
    assert not result.valid_stl
    assert any("facets" in issue for issue in result.issues)


def test_missing_file_is_reported(tmp_path):
    result = validate_stl_file(tmp_path / "nope.stl")
    assert not result.valid_stl
    assert any("Could not read" in issue for issue in result.issues)


def test_tiny_file_is_reported(tmp_path):
    path = tmp_path / "tiny.stl"
    path.write_bytes(b"\x00" * 10)

    result = validate_stl_file(path)
    assert any("too small" in issue for issue in result.issues)


def test_non_finite_coordinates_are_rejected(tmp_path):
    path = tmp_path / "bad.stl"
    count = 1
    payload = np.zeros((count, 12), dtype=np.float32)
    payload[0, 0] = np.nan
    facets = np.zeros((count, 50), dtype=np.uint8)
    facets[:, :48] = payload.view(np.uint8).reshape(count, 48)
    path.write_bytes(b"\x00" * 80 + struct.pack("<I", count) + facets.tobytes())

    result = validate_stl_file(path)
    assert any("non-finite" in issue for issue in result.issues)


def test_validation_result_serialises_for_the_api(builder, synthetic_elevation):
    vertices, triangles = builder.build(synthetic_elevation)
    payload = validate_mesh(vertices, triangles).as_dict()

    for key in ("watertight", "manifold", "valid_stl", "triangles", "volume_mm3", "issues"):
        assert key in payload
    assert isinstance(payload["issues"], list)
    assert payload["width_mm"] == pytest.approx(300.0, abs=1e-6)