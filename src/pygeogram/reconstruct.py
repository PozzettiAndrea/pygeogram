"""
High-level Python wrapper for geogram surface reconstruction algorithms.
"""

import numpy as np
from numpy.typing import NDArray

from pygeogram._pygeogram import (
    co3ne_reconstruct as _co3ne_reconstruct,
    co3ne_compute_normals as _co3ne_compute_normals,
    poisson_reconstruct as _poisson_reconstruct,
)


def co3ne_reconstruct(
    vertices: NDArray[np.float64],
    nb_neighbors: int = 30,
    nb_iterations: int = 3,
    radius: float = 5.0,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Reconstruct a triangle mesh from a point cloud using Co3Ne.

    Smoothes the point cloud and reconstructs triangles in one pass using
    the Concurrent Co-Cones algorithm.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Input point cloud positions.
    nb_neighbors : int, default 30
        Number of neighbors for tangent plane estimation.
    nb_iterations : int, default 3
        Number of point cloud smoothing iterations before reconstruction.
    radius : float, default 5.0
        Maximum distance for connecting neighbors with triangles.

    Returns
    -------
    vertices_out : ndarray, shape (P, 3)
        Reconstructed vertex positions (float64).
    faces_out : ndarray, shape (Q, 3)
        Reconstructed triangle face indices (int32).
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if nb_neighbors < 1:
        raise ValueError(f"nb_neighbors must be >= 1, got {nb_neighbors}")

    return _co3ne_reconstruct(v, nb_neighbors, nb_iterations, radius)


def co3ne_compute_normals(
    vertices: NDArray[np.float64],
    nb_neighbors: int = 30,
    reorient: bool = False,
) -> NDArray[np.float64]:
    """
    Compute normals for a point cloud using Co3Ne.

    Estimates normals from the nearest neighbors best-approximating plane.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Input point cloud positions.
    nb_neighbors : int, default 30
        Number of neighbors for tangent plane estimation.
    reorient : bool, default False
        If True, propagate orientation over the KNN graph for consistent normals.

    Returns
    -------
    normals : ndarray, shape (N, 3)
        Estimated normals (float64).
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if nb_neighbors < 1:
        raise ValueError(f"nb_neighbors must be >= 1, got {nb_neighbors}")

    return _co3ne_compute_normals(v, nb_neighbors, reorient)


def poisson_reconstruct(
    vertices: NDArray[np.float64],
    normals: NDArray[np.float64],
    depth: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Reconstruct a surface from an oriented point cloud using Poisson reconstruction.

    Solves the Poisson equation on an adaptive octree to extract an isosurface.
    Requires pre-computed oriented normals.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Input point cloud positions.
    normals : array-like, shape (N, 3)
        Oriented normals, one per point.
    depth : int, default 8
        Octree depth. Higher = more detail. Use 10-11 for highly detailed models.

    Returns
    -------
    vertices_out : ndarray, shape (P, 3)
        Reconstructed vertex positions (float64).
    faces_out : ndarray, shape (Q, 3)
        Reconstructed triangle face indices (int32).
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    n = np.ascontiguousarray(normals, dtype=np.float64)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if n.ndim != 2 or n.shape[1] != 3:
        raise ValueError(f"normals must have shape (N, 3), got {n.shape}")
    if v.shape[0] != n.shape[0]:
        raise ValueError(
            f"vertices and normals must have same number of rows, "
            f"got {v.shape[0]} and {n.shape[0]}"
        )
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    return _poisson_reconstruct(v, n, depth)
