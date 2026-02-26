"""
pygeogram — Python bindings for geogram geometry processing.

Currently exposes:
- remesh_smooth: CVT (Centroidal Voronoi Tessellation) surface remeshing
"""

from pygeogram.remesh import remesh_smooth

__version__ = "0.1.0"
__all__ = ["remesh_smooth", "__version__"]
