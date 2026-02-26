"""
pygeogram — Python bindings for geogram geometry processing.

Currently exposes:
- remesh_smooth: CVT (Centroidal Voronoi Tessellation) surface remeshing
- mesh_repair: Fix topology, merge duplicates, remove degenerates
- mesh_decimate: Vertex-clustering mesh simplification
"""

from pygeogram.remesh import remesh_smooth
from pygeogram.repair import mesh_repair
from pygeogram.decimate import mesh_decimate

__version__ = "0.1.0"
__all__ = ["remesh_smooth", "mesh_repair", "mesh_decimate", "__version__"]
