"""
pygeogram — Python bindings for geogram geometry processing.

Exposes:
- remesh_smooth: CVT (Centroidal Voronoi Tessellation) surface remeshing
- remesh_anisotropic: Curvature-adapted anisotropic CVT remeshing
- mesh_repair: Fix topology, merge duplicates, remove degenerates
- mesh_decimate: Vertex-clustering mesh simplification
- smooth: Iterative Laplacian mesh smoothing
- smooth_lsq: Least-squares Laplacian mesh smoothing
- compute_normals: Vertex normal computation for surface meshes
- co3ne_reconstruct: Point cloud to mesh via Co3Ne
- co3ne_compute_normals: Point cloud normal estimation via Co3Ne
- poisson_reconstruct: Poisson surface reconstruction from oriented points
"""

from pygeogram.remesh import remesh_smooth, remesh_anisotropic
from pygeogram.repair import mesh_repair
from pygeogram.decimate import mesh_decimate
from pygeogram.smooth import smooth, smooth_lsq, compute_normals
from pygeogram.reconstruct import (
    co3ne_reconstruct,
    co3ne_compute_normals,
    poisson_reconstruct,
)

__version__ = "0.1.0"
__all__ = [
    "remesh_smooth",
    "remesh_anisotropic",
    "mesh_repair",
    "mesh_decimate",
    "smooth",
    "smooth_lsq",
    "compute_normals",
    "co3ne_reconstruct",
    "co3ne_compute_normals",
    "poisson_reconstruct",
    "__version__",
]
