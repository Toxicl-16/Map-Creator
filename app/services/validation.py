"""Mesh validation used to gate the STL result shown in the UI.

The checks are honest structural checks performed on the generated triangle
soup — nothing is assumed or hard-coded:

``watertight``   every undirected edge is used by exactly two triangles
``manifold``     no edge is used by three or more triangles, and no triangle
                 has a repeated vertex
``valid_stl``    the exported file parses, its header count matches the
                 geometry, and every coordinate is finite
"""

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

STL_HEADER_BYTES = 80
STL_FACET_BYTES = 50


@dataclass
class MeshValidation:
    """Outcome of validating a generated mesh."""

    watertight: bool = False
    manifold: bool = False
    valid_stl: bool = False
    triangles: int = 0
    vertices: int = 0
    boundary_edges: int = 0
    non_manifold_edges: int = 0
    degenerate_triangles: int = 0
    bbox_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    volume_mm3: float = 0.0
    issues: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "watertight": self.watertight,
            "manifold": self.manifold,
            "valid_stl": self.valid_stl,
            "triangles": self.triangles,
            "vertices": self.vertices,
            "boundary_edges": self.boundary_edges,
            "non_manifold_edges": self.non_manifold_edges,
            "degenerate_triangles": self.degenerate_triangles,
            "width_mm": round(self.bbox_mm[0], 2),
            "depth_mm": round(self.bbox_mm[1], 2),
            "height_mm": round(self.bbox_mm[2], 2),
            "volume_mm3": round(self.volume_mm3, 1),
            "issues": self.issues,
        }


def _edge_usage(triangles: np.ndarray) -> np.ndarray:
    """Count how many triangles reference each undirected edge."""
    tris = np.asarray(triangles, dtype=np.int64)
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return counts


def _signed_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Signed volume via the divergence theorem; positive for outward winding."""
    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _weld(coords: np.ndarray, decimals: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Recover shared vertices from a triangle soup by matching coordinates.

    STL files carry no vertex table, so the connectivity that the mesh builder
    knew about has to be rebuilt from repeated corner coordinates.
    """
    keys = np.round(np.asarray(coords, dtype=np.float64), decimals).reshape(-1, 3)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    return unique, np.asarray(inverse).reshape(-1, 3)


def _analyze_topology(result: MeshValidation, vertices: np.ndarray, triangles: np.ndarray) -> None:
    """Fill in every connectivity/geometry field on ``result``."""
    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)

    result.vertices = int(v.shape[0])
    result.triangles = int(t.shape[0])

    if result.triangles == 0 or result.vertices < 4:
        result.issues.append("Mesh contains no triangles.")
        return

    repeated = (
        (t[:, 0] == t[:, 1]) | (t[:, 1] == t[:, 2]) | (t[:, 2] == t[:, 0])
    )
    result.degenerate_triangles = int(np.count_nonzero(repeated))
    if result.degenerate_triangles:
        result.issues.append(
            f"{result.degenerate_triangles} triangle(s) reuse the same vertex."
        )

    non_finite = int(np.count_nonzero(~np.isfinite(v)))
    if non_finite:
        result.issues.append(f"{non_finite} vertex coordinate(s) are not finite.")
        return

    lo = v.min(axis=0)
    hi = v.max(axis=0)
    result.bbox_mm = (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2]))

    counts = _edge_usage(t)
    result.boundary_edges = int(np.count_nonzero(counts == 1))
    result.non_manifold_edges = int(np.count_nonzero(counts > 2))

    result.watertight = result.boundary_edges == 0
    result.manifold = result.non_manifold_edges == 0 and result.degenerate_triangles == 0

    volume = _signed_volume(v, t)
    result.volume_mm3 = abs(volume)
    if volume < 0:
        result.issues.append("Surface winding is inverted (negative volume).")
        result.manifold = False

    if not result.watertight:
        result.issues.append(
            f"{result.boundary_edges} open edge(s) — the surface is not sealed."
        )
    if result.non_manifold_edges:
        result.issues.append(
            f"{result.non_manifold_edges} edge(s) shared by more than two faces."
        )


def validate_mesh(vertices, triangles) -> MeshValidation:
    """Validate triangle connectivity and geometry."""
    result = MeshValidation()
    _analyze_topology(result, vertices, triangles)
    return result


def validate_stl_file(path: str | Path, expected_triangles: int | None = None) -> MeshValidation:
    """Parse a binary STL from disk and validate it.

    ``expected_triangles`` cross-checks the header count against the geometry
    the mesh builder reported, catching truncated writes.
    """
    path = Path(path)
    result = MeshValidation()

    try:
        blob = path.read_bytes()
    except OSError as exc:
        result.issues.append(f"Could not read STL file: {exc}")
        return result

    if len(blob) < STL_HEADER_BYTES + 4:
        result.issues.append("STL file is too small to be valid.")
        return result

    (count,) = struct.unpack("<I", blob[STL_HEADER_BYTES : STL_HEADER_BYTES + 4])
    expected_size = STL_HEADER_BYTES + 4 + count * STL_FACET_BYTES
    result.triangles = int(count)

    if len(blob) != expected_size:
        result.issues.append(
            f"STL size mismatch: header declares {count} facets "
            f"({expected_size} bytes) but file is {len(blob)} bytes."
        )
        return result

    if expected_triangles is not None and int(count) != int(expected_triangles):
        result.issues.append(
            f"STL holds {count} facets but {expected_triangles} were generated."
        )
        return result

    facets = np.frombuffer(
        blob, dtype=np.uint8, count=count * STL_FACET_BYTES, offset=STL_HEADER_BYTES + 4
    ).reshape(count, STL_FACET_BYTES)

    # Facet layout: 3×float32 normal, 9×float32 vertices, 1×uint16 attribute.
    values = facets[:, :48].copy().view(np.float32).reshape(count, 12)
    if not np.isfinite(values).all():
        result.issues.append("STL contains non-finite coordinates.")
        return result

    corners = values[:, 3:12].reshape(count, 3, 3).astype(np.float64)
    welded, indices = _weld(corners)

    result.valid_stl = True
    _analyze_topology(result, welded, indices)

    if result.issues:
        result.valid_stl = result.valid_stl and result.watertight and result.manifold

    return result
