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


# ── Smoothing ─────────────────────────────────────────────────────


def test_smooth_basic():
    """Laplacian smoothing should return valid mesh of same topology."""
    import pygeogram

    verts, faces = make_icosphere()
    v_out, f_out = pygeogram.smooth(verts, faces, nb_iter=3)

    assert v_out.shape == verts.shape
    assert f_out.shape == faces.shape
    assert v_out.dtype == np.float64
    assert f_out.dtype == np.int32


def test_smooth_shrinks_mesh():
    """Laplacian smoothing should shrink a convex mesh slightly."""
    import pygeogram

    verts, faces = make_icosphere()
    v_out, _ = pygeogram.smooth(verts, faces, nb_iter=5)

    orig_extent = np.max(np.abs(verts))
    smoothed_extent = np.max(np.abs(v_out))
    # Smoothing a convex shape should shrink it
    assert smoothed_extent < orig_extent


def test_smooth_lsq_basic():
    """Least-squares smoothing should return valid mesh."""
    import pygeogram

    verts, faces = make_icosphere()
    v_out, f_out = pygeogram.smooth_lsq(verts, faces)

    assert v_out.shape[1] == 3
    assert f_out.shape[1] == 3
    assert len(v_out) == len(verts)
    assert len(f_out) == len(faces)


def test_compute_normals():
    """Normals should have correct shape and be roughly unit length."""
    import pygeogram

    verts, faces = make_icosphere()
    normals = pygeogram.compute_normals(verts, faces)

    assert normals.shape == verts.shape
    assert normals.dtype == np.float64

    # Normals on a unit sphere should be roughly unit length
    lengths = np.linalg.norm(normals, axis=1)
    assert np.all(lengths > 0.5)
    assert np.all(lengths < 2.0)


# ── Anisotropic Remeshing ─────────────────────────────────────────


def test_remesh_anisotropic_basic():
    """Anisotropic remeshing should produce valid output."""
    import pygeogram

    verts, faces = make_icosphere()
    v_out, f_out = pygeogram.remesh_anisotropic(
        verts, faces, nb_points=100, anisotropy=0.04,
    )

    assert isinstance(v_out, np.ndarray)
    assert isinstance(f_out, np.ndarray)
    assert v_out.ndim == 2 and v_out.shape[1] == 3
    assert f_out.ndim == 2 and f_out.shape[1] == 3
    assert len(v_out) > 0
    assert len(f_out) > 0


def test_remesh_anisotropic_preserves_scale():
    """Anisotropic remeshing should roughly preserve bounding box."""
    import pygeogram

    verts, faces = make_icosphere()
    v_out, _ = pygeogram.remesh_anisotropic(
        verts, faces, nb_points=200, anisotropy=0.04,
    )

    original_extent = np.max(np.abs(verts))
    remeshed_extent = np.max(np.abs(v_out))
    assert abs(remeshed_extent - original_extent) < 0.2


# ── Surface Reconstruction ────────────────────────────────────────


def make_point_cloud(n=500):
    """Create a noisy sphere point cloud for reconstruction tests."""
    # Fibonacci sphere for even distribution
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    points = np.column_stack([x, y, z]).astype(np.float64)
    return points


def test_co3ne_reconstruct_basic():
    """Co3Ne should reconstruct faces from a point cloud."""
    import pygeogram

    points = make_point_cloud(500)
    v_out, f_out = pygeogram.co3ne_reconstruct(
        points, nb_neighbors=20, nb_iterations=0, radius=5.0,
    )

    assert v_out.ndim == 2 and v_out.shape[1] == 3
    assert f_out.ndim == 2 and f_out.shape[1] == 3
    assert len(f_out) > 0  # should have produced triangles


def test_co3ne_compute_normals():
    """Co3Ne should estimate normals for a point cloud."""
    import pygeogram

    points = make_point_cloud(200)
    normals = pygeogram.co3ne_compute_normals(points, nb_neighbors=20)

    assert normals.shape == points.shape
    assert normals.dtype == np.float64

    # Normals should be non-zero
    lengths = np.linalg.norm(normals, axis=1)
    assert np.all(lengths > 0.01)


def test_poisson_reconstruct_basic():
    """Poisson reconstruction should produce a mesh from points + normals."""
    import pygeogram

    points = make_point_cloud(500)
    # Normals pointing outward (same as positions for a unit sphere)
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)

    v_out, f_out = pygeogram.poisson_reconstruct(points, normals, depth=5)

    assert v_out.ndim == 2 and v_out.shape[1] == 3
    assert f_out.ndim == 2 and f_out.shape[1] == 3
    assert len(v_out) > 0
    assert len(f_out) > 0


def test_reconstruction_input_validation():
    """Invalid inputs should raise errors."""
    import pygeogram

    bad_points = np.zeros((10, 2), dtype=np.float64)

    with pytest.raises((ValueError, RuntimeError)):
        pygeogram.co3ne_reconstruct(bad_points)

    with pytest.raises((ValueError, RuntimeError)):
        pygeogram.co3ne_compute_normals(bad_points)

    good_points = np.zeros((10, 3), dtype=np.float64)
    bad_normals = np.zeros((10, 2), dtype=np.float64)

    with pytest.raises((ValueError, RuntimeError)):
        pygeogram.poisson_reconstruct(good_points, bad_normals)


# ── Boolean Operations ───────────────────────────────────────────


def make_box(center=(0, 0, 0), size=1.0):
    """Create a simple axis-aligned box mesh for boolean tests."""
    cx, cy, cz = center
    h = size / 2.0
    vertices = np.array([
        [cx - h, cy - h, cz - h],
        [cx + h, cy - h, cz - h],
        [cx + h, cy + h, cz - h],
        [cx - h, cy + h, cz - h],
        [cx - h, cy - h, cz + h],
        [cx + h, cy - h, cz + h],
        [cx + h, cy + h, cz + h],
        [cx - h, cy + h, cz + h],
    ], dtype=np.float64)
    faces = np.array([
        # bottom
        [0, 2, 1], [0, 3, 2],
        # top
        [4, 5, 6], [4, 6, 7],
        # front
        [0, 1, 5], [0, 5, 4],
        # back
        [2, 3, 7], [2, 7, 6],
        # left
        [0, 4, 7], [0, 7, 3],
        # right
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)
    return vertices, faces


def test_boolean_union_basic():
    """Union of two overlapping boxes should produce valid mesh."""
    import pygeogram

    va, fa = make_box(center=(0, 0, 0), size=1.0)
    vb, fb = make_box(center=(0.5, 0, 0), size=1.0)

    v_out, f_out = pygeogram.mesh_union(va, fa, vb, fb)

    assert v_out.ndim == 2 and v_out.shape[1] == 3
    assert f_out.ndim == 2 and f_out.shape[1] == 3
    assert len(v_out) > 0
    assert len(f_out) > 0


def test_boolean_intersection_basic():
    """Intersection of two overlapping boxes should produce valid mesh."""
    import pygeogram

    va, fa = make_box(center=(0, 0, 0), size=1.0)
    vb, fb = make_box(center=(0.5, 0, 0), size=1.0)

    v_out, f_out = pygeogram.mesh_intersection(va, fa, vb, fb)

    assert v_out.ndim == 2 and v_out.shape[1] == 3
    assert f_out.ndim == 2 and f_out.shape[1] == 3
    assert len(v_out) > 0
    assert len(f_out) > 0


def test_boolean_difference_basic():
    """Difference of two overlapping boxes should produce valid mesh."""
    import pygeogram

    va, fa = make_box(center=(0, 0, 0), size=1.0)
    vb, fb = make_box(center=(0.5, 0, 0), size=1.0)

    v_out, f_out = pygeogram.mesh_difference(va, fa, vb, fb)

    assert v_out.ndim == 2 and v_out.shape[1] == 3
    assert f_out.ndim == 2 and f_out.shape[1] == 3
    assert len(v_out) > 0
    assert len(f_out) > 0


def test_boolean_union_larger_than_parts():
    """Union bounding box should encompass both inputs."""
    import pygeogram

    va, fa = make_box(center=(0, 0, 0), size=1.0)
    vb, fb = make_box(center=(0.5, 0, 0), size=1.0)

    v_out, _ = pygeogram.mesh_union(va, fa, vb, fb)

    # Union should span from -0.5 to 1.0 along X
    assert np.min(v_out[:, 0]) < -0.4
    assert np.max(v_out[:, 0]) > 0.9


def test_boolean_intersection_smaller_than_parts():
    """Intersection should be smaller than either input."""
    import pygeogram

    va, fa = make_box(center=(0, 0, 0), size=1.0)
    vb, fb = make_box(center=(0.5, 0, 0), size=1.0)

    v_out, _ = pygeogram.mesh_intersection(va, fa, vb, fb)

    # Intersection should be in the overlap region [0, 0.5] along X
    assert np.min(v_out[:, 0]) > -0.1
    assert np.max(v_out[:, 0]) < 0.6


def test_boolean_input_validation():
    """Invalid mesh shapes should raise errors."""
    import pygeogram

    bad_verts = np.zeros((10, 2), dtype=np.float64)
    good_verts = np.zeros((8, 3), dtype=np.float64)
    good_faces = np.zeros((12, 3), dtype=np.int32)

    with pytest.raises(ValueError):
        pygeogram.mesh_union(bad_verts, good_faces, good_verts, good_faces)

    with pytest.raises(ValueError):
        pygeogram.mesh_union(good_verts, good_faces, bad_verts, good_faces)


# ── Mesh I/O ─────────────────────────────────────────────────────


def test_mesh_save_and_load_obj(tmp_path):
    """Save and load OBJ should round-trip vertex/face data."""
    import pygeogram

    verts, faces = make_icosphere()
    filepath = str(tmp_path / "test.obj")

    pygeogram.mesh_save(verts, faces, filepath)
    v_loaded, f_loaded = pygeogram.mesh_load(filepath)

    assert v_loaded.shape == verts.shape
    assert f_loaded.shape == faces.shape
    assert v_loaded.dtype == np.float64
    assert f_loaded.dtype == np.int32
    np.testing.assert_allclose(v_loaded, verts, atol=1e-6)


def test_mesh_save_and_load_ply(tmp_path):
    """Save and load PLY should round-trip."""
    import pygeogram

    verts, faces = make_icosphere()
    filepath = str(tmp_path / "test.ply")

    pygeogram.mesh_save(verts, faces, filepath)
    v_loaded, f_loaded = pygeogram.mesh_load(filepath)

    assert v_loaded.shape == verts.shape
    assert f_loaded.shape == faces.shape
    np.testing.assert_allclose(v_loaded, verts, atol=1e-6)


def test_mesh_save_and_load_off(tmp_path):
    """Save and load OFF should round-trip."""
    import pygeogram

    verts, faces = make_icosphere()
    filepath = str(tmp_path / "test.off")

    pygeogram.mesh_save(verts, faces, filepath)
    v_loaded, f_loaded = pygeogram.mesh_load(filepath)

    assert v_loaded.shape == verts.shape
    assert f_loaded.shape == faces.shape
    np.testing.assert_allclose(v_loaded, verts, atol=1e-6)


def test_mesh_save_and_load_stl(tmp_path):
    """Save and load STL should preserve faces (vertices may be duplicated)."""
    import pygeogram

    verts, faces = make_icosphere()
    filepath = str(tmp_path / "test.stl")

    pygeogram.mesh_save(verts, faces, filepath)
    v_loaded, f_loaded = pygeogram.mesh_load(filepath)

    # STL duplicates vertices per face, so vertex count may differ
    assert v_loaded.ndim == 2 and v_loaded.shape[1] == 3
    assert f_loaded.ndim == 2 and f_loaded.shape[1] == 3
    assert len(f_loaded) == len(faces)


def test_mesh_io_input_validation():
    """Invalid mesh shapes should raise errors on save."""
    import pygeogram

    bad_verts = np.zeros((10, 2), dtype=np.float64)
    good_faces = np.zeros((5, 3), dtype=np.int32)

    with pytest.raises(ValueError):
        pygeogram.mesh_save(bad_verts, good_faces, "/tmp/bad.obj")


def test_mesh_load_nonexistent():
    """Loading a nonexistent file should raise an error."""
    import pygeogram

    with pytest.raises(RuntimeError):
        pygeogram.mesh_load("/tmp/nonexistent_mesh_file_xyz.obj")


# ── UV Parameterization ──────────────────────────────────────────


def test_make_atlas_basic():
    """UV atlas should return per-face-corner UV coordinates."""
    import pygeogram

    verts, faces = make_icosphere()
    uvs = pygeogram.make_atlas(verts, faces)

    assert uvs.ndim == 3
    assert uvs.shape == (len(faces), 3, 2)
    assert uvs.dtype == np.float64


def test_make_atlas_uv_range():
    """UV coordinates should be in [0, 1] after packing."""
    import pygeogram

    verts, faces = make_icosphere()
    uvs = pygeogram.make_atlas(verts, faces)

    assert np.all(uvs >= -0.01)  # small tolerance
    assert np.all(uvs <= 1.01)


def test_make_atlas_parameterizers():
    """Different parameterizer methods should all produce valid output."""
    import pygeogram

    verts, faces = make_icosphere()

    for param in [pygeogram.PARAM_PROJECTION, pygeogram.PARAM_LSCM, pygeogram.PARAM_ABF]:
        uvs = pygeogram.make_atlas(verts, faces, parameterizer=param)
        assert uvs.shape == (len(faces), 3, 2)
        assert np.all(np.isfinite(uvs))


def test_make_atlas_no_packing():
    """Atlas with PACK_NONE should still produce UVs (just not packed)."""
    import pygeogram

    verts, faces = make_icosphere()
    uvs = pygeogram.make_atlas(verts, faces, packer=pygeogram.PACK_NONE)

    assert uvs.shape == (len(faces), 3, 2)
    assert np.all(np.isfinite(uvs))


def test_make_atlas_input_validation():
    """Invalid mesh shapes should raise errors."""
    import pygeogram

    bad_verts = np.zeros((10, 2), dtype=np.float64)
    good_faces = np.zeros((5, 3), dtype=np.int32)

    with pytest.raises(ValueError):
        pygeogram.make_atlas(bad_verts, good_faces)
