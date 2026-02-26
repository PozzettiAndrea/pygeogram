"""Tests for pygeogram: CVT remeshing, mesh repair, mesh decimation."""

import numpy as np
import pytest


def make_icosphere():
    """Create a simple icosphere for testing."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio

    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)

    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)

    return vertices, faces


# ── CVT Remeshing ──────────────────────────────────────────────────


def test_remesh_smooth_basic():
    """Test basic CVT remeshing round-trip."""
    import pygeogram

    verts, faces = make_icosphere()
    v_out, f_out = pygeogram.remesh_smooth(verts, faces, nb_points=100)

    assert isinstance(v_out, np.ndarray)
    assert isinstance(f_out, np.ndarray)
    assert v_out.ndim == 2
    assert f_out.ndim == 2
    assert v_out.shape[1] == 3
    assert f_out.shape[1] == 3
    assert len(v_out) > 0
    assert len(f_out) > 0


def test_remesh_smooth_preserves_scale():
    """Test that remeshing roughly preserves the bounding box."""
    import pygeogram

    verts, faces = make_icosphere()
    v_out, f_out = pygeogram.remesh_smooth(verts, faces, nb_points=200)

    # Original is a unit sphere, output should be close
    original_extent = np.max(np.abs(verts))
    remeshed_extent = np.max(np.abs(v_out))
    assert abs(remeshed_extent - original_extent) < 0.2


def test_remesh_smooth_custom_params():
    """Test with custom Lloyd/Newton iteration counts."""
    import pygeogram

    verts, faces = make_icosphere()
    v_out, f_out = pygeogram.remesh_smooth(
        verts, faces, nb_points=50,
        nb_lloyd_iter=3, nb_newton_iter=10, newton_m=5,
        adjust=False,
    )

    assert v_out.shape[1] == 3
    assert f_out.shape[1] == 3
    assert len(v_out) > 0


def test_remesh_smooth_input_validation():
    """Test that invalid inputs raise errors."""
    import pygeogram

    # Wrong vertex shape
    bad_verts = np.zeros((10, 2), dtype=np.float64)
    good_faces = np.zeros((5, 3), dtype=np.int32)

    with pytest.raises((ValueError, RuntimeError)):
        pygeogram.remesh_smooth(bad_verts, good_faces, nb_points=10)


# ── Mesh Repair ────────────────────────────────────────────────────


def test_repair_colocate_vertices():
    """Duplicate vertices at same position should be merged."""
    import pygeogram

    verts, faces = make_icosphere()
    n_orig = len(verts)

    # Duplicate first 5 vertices (same position, new indices)
    dup_verts = np.vstack([verts, verts[:5]])
    # Rewire first face to use duplicated vertex
    dup_faces = faces.copy()
    dup_faces[0, 0] = n_orig  # face 0 vertex 0 → duplicate of vertex 0

    v_out, f_out = pygeogram.mesh_repair(dup_verts, dup_faces, colocate=True)

    # Should merge back to ~original vertex count
    assert len(v_out) <= n_orig
    assert f_out.shape[1] == 3


def test_repair_remove_duplicate_faces():
    """Duplicate faces should be removed."""
    import pygeogram

    verts, faces = make_icosphere()
    n_faces_orig = len(faces)

    # Append 5 duplicate faces
    dup_faces = np.vstack([faces, faces[:5]])

    v_out, f_out = pygeogram.mesh_repair(verts, dup_faces, remove_duplicates=True)

    assert len(f_out) <= n_faces_orig


def test_repair_degenerate_faces():
    """Degenerate faces (repeated vertex) should be removed."""
    import pygeogram

    verts, faces = make_icosphere()
    n_faces_orig = len(faces)

    # Add degenerate faces: two identical vertex indices
    degenerate = np.array([[0, 0, 1], [2, 3, 3]], dtype=np.int32)
    bad_faces = np.vstack([faces, degenerate])

    v_out, f_out = pygeogram.mesh_repair(verts, bad_faces)

    assert len(f_out) <= n_faces_orig


def test_repair_clean_mesh_passthrough():
    """A clean mesh should pass through repair mostly unchanged."""
    import pygeogram

    verts, faces = make_icosphere()
    v_out, f_out = pygeogram.mesh_repair(verts, faces)

    # Should have same or fewer (isolated vertex removal) vertices
    assert len(v_out) <= len(verts) + 1
    assert len(f_out) <= len(faces) + 1
    assert v_out.shape[1] == 3
    assert f_out.shape[1] == 3


def test_repair_with_epsilon():
    """Vertices within epsilon should be merged."""
    import pygeogram

    verts, faces = make_icosphere()
    n_orig = len(verts)

    # Add near-duplicate: vertex 0 shifted by tiny amount
    near_dup = verts[0:1] + 1e-10
    verts_noisy = np.vstack([verts, near_dup])
    # Rewire one face to use the near-duplicate
    faces_mod = faces.copy()
    faces_mod[0, 0] = n_orig

    v_out, f_out = pygeogram.mesh_repair(
        verts_noisy, faces_mod, colocate=True, colocate_epsilon=1e-6,
    )

    assert len(v_out) <= n_orig


# ── Mesh Decimation ────────────────────────────────────────────────


def test_decimate_reduces_count():
    """Decimation should reduce vertex and face count."""
    import pygeogram

    verts, faces = make_icosphere()
    v_out, f_out = pygeogram.mesh_decimate(verts, faces, nb_bins=3)

    assert len(v_out) < len(verts)
    assert len(f_out) < len(faces)
    assert v_out.shape[1] == 3
    assert f_out.shape[1] == 3


def test_decimate_more_bins_more_detail():
    """Higher nb_bins should preserve more vertices."""
    import pygeogram

    verts, faces = make_icosphere()
    v_lo, _ = pygeogram.mesh_decimate(verts, faces, nb_bins=3)
    v_hi, _ = pygeogram.mesh_decimate(verts, faces, nb_bins=10)

    assert len(v_hi) >= len(v_lo)


def test_decimate_preserves_bounds():
    """Decimation should roughly preserve the bounding box."""
    import pygeogram

    verts, faces = make_icosphere()
    v_out, f_out = pygeogram.mesh_decimate(verts, faces, nb_bins=5)

    orig_extent = np.max(np.abs(verts))
    dec_extent = np.max(np.abs(v_out))
    assert abs(dec_extent - orig_extent) < 0.5
