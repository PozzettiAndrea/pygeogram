# pygeogram

Python bindings for [geogram](https://github.com/BrunoLevy/geogram) geometry processing.

Currently exposes **CVT (Centroidal Voronoi Tessellation) remeshing** — geogram's
high-quality isotropic surface remesher by Bruno Levy (INRIA).

**[Live demo with visual results](https://pozzettiandrea.github.io/pygeogram/)**

## Installation

```bash
pip install pygeogram
```

## Exposed API

| Python function | C++ source | Description |
|---|---|---|
| `pygeogram.remesh_smooth()` | [`GEO::remesh_smooth()`](https://github.com/BrunoLevy/geogram/blob/main/src/lib/geogram/mesh/mesh_remesh.h) | CVT isotropic surface remeshing |

Input/output is raw numpy arrays — works directly with trimesh, open3d, pyvista, or any mesh library.

### `pygeogram.remesh_smooth(vertices, faces, nb_points, ...)`

Remesh a triangle surface using Centroidal Voronoi Tessellation. Distributes points
evenly via Lloyd relaxation + Newton/L-BFGS optimization, then extracts the dual surface.

```python
import numpy as np
import pygeogram

# From numpy arrays
v_out, f_out = pygeogram.remesh_smooth(
    vertices,       # (N, 3) float64
    faces,          # (M, 3) int32
    nb_points=5000, # target vertex count
)

# From trimesh
import trimesh
mesh = trimesh.load("model.stl")
v_out, f_out = pygeogram.remesh_smooth(mesh.vertices, mesh.faces, nb_points=5000)
result = trimesh.Trimesh(vertices=v_out, faces=f_out)
```

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `vertices` | (required) | Input vertex positions, shape `(N, 3)`, float64 |
| `faces` | (required) | Input triangle indices, shape `(M, 3)`, int32 |
| `nb_points` | (required) | Target number of output vertices |
| `nb_lloyd_iter` | 5 | Lloyd relaxation iterations (initial uniform distribution) |
| `nb_newton_iter` | 30 | Newton optimization iterations (refines placement) |
| `newton_m` | 7 | Hessian evaluations for L-BFGS |
| `adjust` | True | Adjust vertices to better approximate input surface |
| `adjust_max_edge_distance` | 0.5 | Max search distance for adjustment |
| `adjust_border_importance` | 2.0 | Importance of boundary fitting |

#### Returns

- `vertices_out` — `(P, 3)` float64 array
- `faces_out` — `(Q, 3)` int32 array

## Roadmap

Future bindings to expose (mirroring [GraphiteThree](https://github.com/BrunoLevy/GraphiteThree)):

| Priority | C++ function | Purpose |
|---|---|---|
| Next | `GEO::mesh_decimate()` | Mesh decimation/simplification |
| Next | `GEO::mesh_repair()` | Mesh repair (remove duplicates, fix topology) |
| Planned | `GEO::compute_normals()` | Per-vertex/face normal computation |
| Planned | `GEO::mesh_boolean()` | Boolean operations (union, intersection, difference) |
| Planned | `GEO::mesh_parameterize()` | UV parameterization (LSCM, ABF) |
| Planned | `GEO::Delaunay` | Delaunay triangulation (2D/3D) |
| Planned | `GEO::CentroidalVoronoiTesselation` | Direct CVT access |

## Building from source

Requires CMake, a C++17 compiler, and Python 3.10+.

```bash
git clone --recursive https://github.com/PozzettiAndrea/pygeogram.git
cd pygeogram
cd geogram && git submodule update --init --recursive && cd ..
pip install .
```

## License

BSD-3-Clause — same as geogram.

## Credits

- [geogram](https://github.com/BrunoLevy/geogram) by Bruno Levy (INRIA)
- CVT remeshing: Yan et al. "Isotropic Remeshing with Fast and Exact Computation
  of Restricted Voronoi Diagram" (CGF 2009)
