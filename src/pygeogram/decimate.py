"""
High-level Python wrapper for geogram vertex-clustering mesh decimation.
"""

import numpy as np
from numpy.typing import NDArray

from pygeogram._pygeogram import (
    mesh_decimate as _mesh_decimate,
    MESH_DECIMATE_DUP_F,
    MESH_DECIMATE_DEG_3,
    MESH_DECIMATE_KEEP_B,
)


def mesh_decimate(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    nb_bins: int,
    *,
    remove_duplicates: bool = True,
    remove_degree3: bool = True,
    keep_borders: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Simplify a mesh using vertex clustering (spatial binning).

    Groups vertices into a 3D grid of nb_bins^3 cells, merges vertices
    within each cell, and rebuilds face connectivity.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Input vertex positions.
    faces : array-like, shape (M, 3)
        Input triangle face indices.
    nb_bins : int
        Grid resolution per axis. Higher = more detail preserved.
        Typical range: 50-200.
    remove_duplicates : bool, default True
        Remove duplicate facets after clustering.
    remove_degree3 : bool, default True
        Remove degree-3 vertices (simplifies topology).
    keep_borders : bool, default True
        Preserve boundary vertices.

    Returns
    -------
    vertices_out : ndarray, shape (P, 3)
        Simplified vertex positions (float64).
    faces_out : ndarray, shape (Q, 3)
        Simplified triangle face indices (int32).
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {f.shape}")
    if nb_bins < 1:
        raise ValueError(f"nb_bins must be >= 1, got {nb_bins}")

    mode = 0
    if remove_duplicates:
        mode |= MESH_DECIMATE_DUP_F
    if remove_degree3:
        mode |= MESH_DECIMATE_DEG_3
    if keep_borders:
        mode |= MESH_DECIMATE_KEEP_B

    return _mesh_decimate(v, f, nb_bins, mode)
