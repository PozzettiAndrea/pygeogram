"""
High-level Python wrapper for geogram mesh repair.
"""

import numpy as np
from numpy.typing import NDArray

from pygeogram._pygeogram import (
    mesh_repair as _mesh_repair,
    MESH_REPAIR_COLOCATE,
    MESH_REPAIR_DUP_F,
    MESH_REPAIR_TRIANGULATE,
    MESH_REPAIR_RECONSTRUCT,
)


def mesh_repair(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    *,
    colocate: bool = True,
    remove_duplicates: bool = True,
    triangulate: bool = True,
    reconstruct: bool = False,
    colocate_epsilon: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Repair a triangle mesh by fixing common defects.

    Merges colocated (duplicate) vertices, removes duplicate and degenerate
    facets, triangulates non-triangular faces, and fixes non-manifold topology.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Input vertex positions.
    faces : array-like, shape (M, 3)
        Input triangle face indices.
    colocate : bool, default True
        Merge vertices at the same position (within colocate_epsilon).
    remove_duplicates : bool, default True
        Remove duplicate and degenerate facets.
    triangulate : bool, default True
        Triangulate non-triangular facets.
    reconstruct : bool, default False
        Post-process result of Co3Ne reconstruction algorithm.
    colocate_epsilon : float, default 0.0
        Tolerance for merging colocated vertices. 0.0 = exact match only.

    Returns
    -------
    vertices_out : ndarray, shape (P, 3)
        Cleaned vertex positions (float64).
    faces_out : ndarray, shape (Q, 3)
        Cleaned triangle face indices (int32).
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {f.shape}")

    mode = 0
    if colocate:
        mode |= MESH_REPAIR_COLOCATE
    if remove_duplicates:
        mode |= MESH_REPAIR_DUP_F
    if triangulate:
        mode |= MESH_REPAIR_TRIANGULATE
    if reconstruct:
        mode |= MESH_REPAIR_RECONSTRUCT

    return _mesh_repair(v, f, mode, colocate_epsilon)
