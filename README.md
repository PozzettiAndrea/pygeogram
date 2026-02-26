# pygeogram

<div align="center">
<a href="https://pozzettiandrea.github.io/pygeogram/">
<img src="https://pozzettiandrea.github.io/pygeogram/preview.png" alt="CVT Remeshing Demo" width="800">
</a>
<br>
<b><a href="https://pozzettiandrea.github.io/pygeogram/">View Live Demo →</a></b>
</div>

Python bindings for [geogram](https://github.com/BrunoLevy/geogram) geometry processing.

Currently exposes **CVT (Centroidal Voronoi Tessellation) remeshing**.

## Installation

```bash
pip install pygeogram --find-links https://github.com/PozzettiAndrea/pygeogram/releases/latest/download/
```

## Currently Exposed API

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

## License

BSD-3-Clause — same as geogram.

## Credits

- [geogram](https://github.com/BrunoLevy/geogram) by Bruno Levy (INRIA)
- CVT remeshing: Yan et al. "Isotropic Remeshing with Fast and Exact Computation
  of Restricted Voronoi Diagram" (CGF 2009)