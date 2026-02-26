"""
High-level Python wrapper for geogram CVT remeshing.
"""

import numpy as np
from numpy.typing import NDArray

from pygeogram._pygeogram import remesh_smooth as _remesh_smooth


def remesh_smooth(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    nb_points: int,
    nb_lloyd_iter: int = 5,
    nb_newton_iter: int = 30,
    newton_m: int = 7,
    adjust: bool = True,
    adjust_max_edge_distance: float = 0.5,
    adjust_border_importance: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Remesh a surface using Centroidal Voronoi Tessellation (CVT).

    Produces a high-quality isotropic remeshing by computing a CVT on the input
    surface. The algorithm distributes points evenly using Lloyd relaxation and
    Newton optimization, then extracts the dual surface.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Input vertex positions.
    faces : array-like, shape (M, 3)
        Input triangle face indices.
    nb_points : int
        Target number of output vertices. Use 0 to keep same count as input.
    nb_lloyd_iter : int, default 5
        Number of Lloyd relaxation iterations (initial uniform distribution).
    nb_newton_iter : int, default 30
        Number of Newton optimization iterations (refines point placement).
    newton_m : int, default 7
        Number of evaluations for Hessian approximation in L-BFGS.
    adjust : bool, default True
        Adjust output vertices to better approximate the input surface.
    adjust_max_edge_distance : float, default 0.5
        Max search distance for surface adjustment, relative to average edge length.
    adjust_border_importance : float, default 2.0
        Importance of boundary fitting during adjustment.

    Returns
    -------
    vertices_out : ndarray, shape (P, 3)
        Output vertex positions (float64).
    faces_out : ndarray, shape (Q, 3)
        Output triangle face indices (int32).
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {f.shape}")

    return _remesh_smooth(
        v, f, nb_points,
        nb_lloyd_iter, nb_newton_iter, newton_m,
        adjust, adjust_max_edge_distance, adjust_border_importance,
    )
