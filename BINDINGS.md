# Binding Coverage

## Mapped

| Function | Description |
|----------|-------------|
| `remesh_smooth` | CVT isotropic remeshing |
| `remesh_anisotropic` | Curvature-adapted CVT remeshing |
| `mesh_repair` | Fix topology, merge colocated vertices, remove degenerate faces |
| `mesh_decimate` | Vertex clustering mesh simplification |
| `smooth` | Iterative Laplacian smoothing |
| `smooth_lsq` | Least-squares Laplacian smoothing |
| `compute_normals` | Vertex normal computation |
| `co3ne_reconstruct` | Point cloud to mesh via Co3Ne |
| `co3ne_compute_normals` | Point cloud normal estimation |
| `poisson_reconstruct` | Poisson surface reconstruction |
| `mesh_union` | Boolean union of two meshes |
| `mesh_intersection` | Boolean intersection of two meshes |
| `mesh_difference` | Boolean difference of two meshes |
| `mesh_load` | Load mesh from file (OBJ, PLY, OFF, STL, mesh/meshb) |
| `mesh_save` | Save mesh to file |
| `make_atlas` | UV parameterization (LSCM, ABF, XAtlas packing) |

## Not Mapped

| Capability | Notes |
|------------|-------|
| Tetrahedralization / volume meshing | 3D mesh generation |
| Delaunay / Voronoi diagrams | Used internally by CVT but not directly exposed |
| Geodesic distance computation | Shortest paths on surface |
| Mesh connectivity / adjacency queries | Neighbor queries, manifold checking |
| Curvature-flow / bilateral smoothing | Advanced smoothing variants |
| Vertex / face attribute management | Custom per-element data |
| Surface-surface intersection | Imported in C++ but no Python wrapper |
