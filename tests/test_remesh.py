"""Tests for pygeogram CVT remeshing."""

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
