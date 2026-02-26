# pygeogram

<div align="center">
<a href="https://pozzettiandrea.github.io/pygeogram/">
<img src="https://pozzettiandrea.github.io/pygeogram/preview.png" alt="CVT Remeshing Demo" width="800">
</a>
<br>
<b><a href="https://pozzettiandrea.github.io/pygeogram/">View Live Demo →</a></b>
</div>

Python bindings for [geogram](https://github.com/BrunoLevy/geogram) geometry processing.

Currently exposes **CVT (Centroidal Voronoi Tessellation) remeshing** — geogram's
high-quality isotropic surface remesher by Bruno Levy (INRIA).

## Installation

Pre-built wheels for Linux, macOS (Apple Silicon), and Windows:

```bash
pip install pygeogram --extra-index-url https://pozzettiandrea.github.io/pygeogram/wheels/
```

## Currently Exposed API

| Python function | C++ source | Description |
|---|---|---|
| `pygeogram.remesh_smooth()` | [`GEO::remesh_smooth()`](https://github.com/BrunoLevy/geogram/blob/main/src/lib/geogram/mesh/mesh_remesh.h) | CVT isotropic surface remeshing |

Input/output is raw numpy arrays. Works directly with trimesh, open3d, pyvista, or any mesh library.

### `pygeogram.remesh_smooth(vertices, faces, nb_points, ...)`

Remesh a triangle surface using Centroidal Voronoi Tessellation.

```python
import pygeogram
import trimesh

mesh = trimesh.load("model.stl")
v_out, f_out = pygeogram.remesh_smooth(
    mesh.vertices, mesh.faces,
    nb_points=5000,
)
result = trimesh.Trimesh(vertices=v_out, faces=f_out)
```

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
