"""
High-level Python wrapper for geogram mesh smoothing and normal computation.
"""

import numpy as np
from numpy.typing import NDArray

from pygeogram._pygeogram import (
    simple_laplacian_smooth as _simple_laplacian_smooth,
    mesh_smooth as _mesh_smooth,
    compute_normals as _compute_normals,
)


def smooth(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    nb_iter: int = 1,
    normals_only: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Smooth a mesh using iterative Laplacian relaxation.

    Moves each vertex toward the barycenter of its neighbors. Fast but can
    shrink the mesh with many iterations.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Input vertex positions.
    faces : array-like, shape (M, 3)
        Input triangle face indices.
    nb_iter : int, default 1
        Number of smoothing iterations.
    normals_only : bool, default False
        If True, only smooth stored normals, not vertex positions.

    Returns
    -------
    vertices_out : ndarray, shape (N, 3)
        Smoothed vertex positions (float64).
    faces_out : ndarray, shape (M, 3)
        Triangle face indices (int32), unchanged.
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {f.shape}")
    if nb_iter < 1:
        raise ValueError(f"nb_iter must be >= 1, got {nb_iter}")

    return _simple_laplacian_smooth(v, f, nb_iter, normals_only)


def smooth_lsq(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Smooth a mesh using least-squares Laplacian optimization.

    Higher quality than simple Laplacian smoothing. Uses OpenNL solver
    to minimize the Laplacian energy.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Input vertex positions.
    faces : array-like, shape (M, 3)
        Input triangle face indices.

    Returns
    -------
    vertices_out : ndarray, shape (N, 3)
        Smoothed vertex positions (float64).
    faces_out : ndarray, shape (M, 3)
        Triangle face indices (int32), unchanged.
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {f.shape}")

    return _mesh_smooth(v, f)


def compute_normals(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
) -> NDArray[np.float64]:
    """
    Compute vertex normals for a surface mesh.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Input vertex positions.
    faces : array-like, shape (M, 3)
        Input triangle face indices.

    Returns
    -------
    normals : ndarray, shape (N, 3)
        Vertex normals (float64).
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {f.shape}")

    return _compute_normals(v, f)
