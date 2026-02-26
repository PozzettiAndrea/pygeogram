# pygeogram

<div align="center">
<a href="https://pozzettiandrea.github.io/pygeogram/">
<img src="https://pozzettiandrea.github.io/pygeogram/preview.png" alt="CVT Remeshing Demo" width="800">
</a>
<br>
<b><a href="https://pozzettiandrea.github.io/pygeogram/">View Live Demo →</a></b>
</div>

Python bindings for [geogram](https://github.com/BrunoLevy/geogram) geometry processing.

## Installation

```bash
pip install pygeogram --find-links https://github.com/PozzettiAndrea/pygeogram/releases/latest/download/
```

## API

### CVT Remeshing

```python
import pygeogram

v_out, f_out = pygeogram.remesh_smooth(
    vertices, faces,
    nb_points=5000,
    nb_lloyd_iter=5,      # Lloyd relaxation iterations
    nb_newton_iter=30,    # Newton optimization iterations
)
```

### Mesh Repair

```python
v_clean, f_clean = pygeogram.mesh_repair(
    vertices, faces,
    colocate=True,          # merge duplicate vertices
    remove_duplicates=True, # remove duplicate/degenerate faces
    triangulate=True,       # triangulate non-triangle faces
    colocate_epsilon=0.0,   # vertex merge tolerance
)
```

### Mesh Decimation

```python
v_simple, f_simple = pygeogram.mesh_decimate(
    vertices, faces,
    nb_bins=100,          # grid resolution (higher = more detail)
    keep_borders=True,    # preserve boundary vertices
)
```

## License

BSD-3-Clause — same as geogram.

## Credits

- [geogram](https://github.com/BrunoLevy/geogram) by Bruno Levy (INRIA)
- CVT remeshing: Yan et al. "Isotropic Remeshing with Fast and Exact Computation
  of Restricted Voronoi Diagram" (CGF 2009)