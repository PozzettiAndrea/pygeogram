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
- mesh_union: Boolean union of two meshes
- mesh_intersection: Boolean intersection of two meshes
- mesh_difference: Boolean difference (A - B) of two meshes
- mesh_load: Load mesh from file (OBJ, PLY, OFF, STL, mesh/meshb)
- mesh_save: Save mesh to file
- make_atlas: Generate UV texture coordinates (LSCM/ABF parameterization)
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
from pygeogram.boolean import mesh_union, mesh_intersection, mesh_difference
from pygeogram.io import mesh_load, mesh_save
from pygeogram.parameterize import (
    make_atlas,
    PARAM_PROJECTION,
    PARAM_LSCM,
    PARAM_SPECTRAL_LSCM,
    PARAM_ABF,
    PACK_NONE,
    PACK_TETRIS,
    PACK_XATLAS,
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
    "mesh_union",
    "mesh_intersection",
    "mesh_difference",
    "mesh_load",
    "mesh_save",
    "make_atlas",
    "PARAM_PROJECTION",
    "PARAM_LSCM",
    "PARAM_SPECTRAL_LSCM",
    "PARAM_ABF",
    "PACK_NONE",
    "PACK_TETRIS",
    "PACK_XATLAS",
    "__version__",
]
