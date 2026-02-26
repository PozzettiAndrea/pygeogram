"""
High-level Python wrapper for geogram UV parameterization / texture atlas.
"""

import numpy as np
from numpy.typing import NDArray

from pygeogram._pygeogram import (
    mesh_make_atlas as _mesh_make_atlas,
    PARAM_PROJECTION,
    PARAM_LSCM,
    PARAM_SPECTRAL_LSCM,
    PARAM_ABF,
    PACK_NONE,
    PACK_TETRIS,
    PACK_XATLAS,
)

# Re-export constants for user convenience
__all__ = [
    "make_atlas",
    "PARAM_PROJECTION",
    "PARAM_LSCM",
    "PARAM_SPECTRAL_LSCM",
    "PARAM_ABF",
    "PACK_NONE",
    "PACK_TETRIS",
    "PACK_XATLAS",
]


def make_atlas(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    hard_angles_threshold: float = 45.0,
    parameterizer: int = PARAM_ABF,
    packer: int = PACK_XATLAS,
) -> NDArray[np.float64]:
    """
    Generate UV texture coordinates for a mesh.

    Decomposes the mesh into charts, parameterizes each chart to minimize
    distortion, and packs them into [0,1] texture space.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Input vertex positions.
    faces : array-like, shape (M, 3)
        Input triangle face indices.
    hard_angles_threshold : float, default 45.0
        Dihedral angle threshold (degrees) for chart boundaries.
    parameterizer : int, default PARAM_ABF
        Chart flattening method:
        - PARAM_PROJECTION (0): projection onto fitted plane (fast, low quality)
        - PARAM_LSCM (1): Least Squares Conformal Maps
        - PARAM_SPECTRAL_LSCM (2): Spectral LSCM (less distortion)
        - PARAM_ABF (3): Angle-Based Flattening++ (best quality)
    packer : int, default PACK_XATLAS
        Chart packing method:
        - PACK_NONE (0): no packing
        - PACK_TETRIS (1): built-in Tetris packing
        - PACK_XATLAS (2): XAtlas library packing

    Returns
    -------
    uvs : ndarray, shape (M, 3, 2)
        UV coordinates per face corner. uvs[i, j] is the (u, v) for
        face i, corner j.
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {f.shape}")

    return _mesh_make_atlas(v, f, hard_angles_threshold, parameterizer, packer)
