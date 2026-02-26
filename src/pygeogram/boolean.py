"""
High-level Python wrapper for geogram boolean mesh operations.
"""

import numpy as np
from numpy.typing import NDArray

from pygeogram._pygeogram import mesh_boolean as _mesh_boolean


def mesh_union(
    vertices_a: NDArray[np.float64],
    faces_a: NDArray[np.int32],
    vertices_b: NDArray[np.float64],
    faces_b: NDArray[np.int32],
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Compute the union of two closed surface meshes.

    Parameters
    ----------
    vertices_a, faces_a : array-like
        First mesh (vertices shape (N,3), faces shape (M,3)).
    vertices_b, faces_b : array-like
        Second mesh (vertices shape (P,3), faces shape (Q,3)).

    Returns
    -------
    vertices_out : ndarray, shape (R, 3)
        Result vertex positions (float64).
    faces_out : ndarray, shape (S, 3)
        Result triangle face indices (int32).
    """
    return _bool_op(vertices_a, faces_a, vertices_b, faces_b, "A+B")


def mesh_intersection(
    vertices_a: NDArray[np.float64],
    faces_a: NDArray[np.int32],
    vertices_b: NDArray[np.float64],
    faces_b: NDArray[np.int32],
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Compute the intersection of two closed surface meshes.

    Parameters
    ----------
    vertices_a, faces_a : array-like
        First mesh (vertices shape (N,3), faces shape (M,3)).
    vertices_b, faces_b : array-like
        Second mesh (vertices shape (P,3), faces shape (Q,3)).

    Returns
    -------
    vertices_out : ndarray, shape (R, 3)
        Result vertex positions (float64).
    faces_out : ndarray, shape (S, 3)
        Result triangle face indices (int32).
    """
    return _bool_op(vertices_a, faces_a, vertices_b, faces_b, "A*B")


def mesh_difference(
    vertices_a: NDArray[np.float64],
    faces_a: NDArray[np.int32],
    vertices_b: NDArray[np.float64],
    faces_b: NDArray[np.int32],
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Subtract mesh B from mesh A.

    Parameters
    ----------
    vertices_a, faces_a : array-like
        First mesh (vertices shape (N,3), faces shape (M,3)).
    vertices_b, faces_b : array-like
        Second mesh (vertices shape (P,3), faces shape (Q,3)).

    Returns
    -------
    vertices_out : ndarray, shape (R, 3)
        Result vertex positions (float64).
    faces_out : ndarray, shape (S, 3)
        Result triangle face indices (int32).
    """
    return _bool_op(vertices_a, faces_a, vertices_b, faces_b, "A-B")


def _bool_op(vertices_a, faces_a, vertices_b, faces_b, operation):
    va = np.ascontiguousarray(vertices_a, dtype=np.float64)
    fa = np.ascontiguousarray(faces_a, dtype=np.int32)
    vb = np.ascontiguousarray(vertices_b, dtype=np.float64)
    fb = np.ascontiguousarray(faces_b, dtype=np.int32)

    for name, v, f in [("A", va, fa), ("B", vb, fb)]:
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(
                f"vertices_{name.lower()} must have shape (N, 3), got {v.shape}"
            )
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(
                f"faces_{name.lower()} must have shape (M, 3), got {f.shape}"
            )

    return _mesh_boolean(va, fa, vb, fb, operation)
