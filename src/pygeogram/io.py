"""
High-level Python wrapper for geogram mesh I/O.
"""

import numpy as np
from numpy.typing import NDArray

from pygeogram._pygeogram import (
    mesh_load as _mesh_load,
    mesh_save as _mesh_save,
)


def mesh_load(
    filename: str,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Load a mesh from a file.

    Supports OBJ, PLY, OFF, STL, mesh/meshb formats. Format is detected
    from the file extension. Non-triangular faces are automatically triangulated.

    Parameters
    ----------
    filename : str
        Path to the mesh file.

    Returns
    -------
    vertices : ndarray, shape (N, 3)
        Vertex positions (float64).
    faces : ndarray, shape (M, 3)
        Triangle face indices (int32).
    """
    return _mesh_load(filename)


def mesh_save(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    filename: str,
) -> None:
    """
    Save a mesh to a file.

    Supports OBJ, PLY, OFF, STL, mesh/meshb formats. Format is detected
    from the file extension.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Vertex positions.
    faces : array-like, shape (M, 3)
        Triangle face indices.
    filename : str
        Path to the output file.
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {f.shape}")

    _mesh_save(v, f, filename)
