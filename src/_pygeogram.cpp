// Python bindings for geogram geometry processing via nanobind.
//
// Exposes: remesh_smooth (CVT), remesh_anisotropic, mesh_repair,
//          mesh_decimate, smoothing, normals, Co3Ne + Poisson reconstruction.
// Follows the same pattern as pymeshfix's _meshfix.cpp.

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <atomic>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "array_support.h"

#include <geogram/basic/common.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_remesh.h>
#include <geogram/mesh/mesh_repair.h>
#include <geogram/mesh/mesh_decimate.h>
#include <geogram/mesh/mesh_geometry.h>
#include <geogram/mesh/mesh_smoothing.h>
#include <geogram/points/co3ne.h>
#include <geogram/third_party/PoissonRecon/poisson_geogram.h>

namespace nb = nanobind;

// Ensure GEO::initialize() is called exactly once.
static std::atomic<bool> geo_initialized{false};

static void ensure_geogram_initialized() {
    bool expected = false;
    if (geo_initialized.compare_exchange_strong(expected, true)) {
        GEO::initialize();
        // Import all standard arg groups that geogram algorithms may need
        GEO::CmdLine::import_arg_group("standard");
        GEO::CmdLine::import_arg_group("algo");
        GEO::CmdLine::import_arg_group("remesh");
        GEO::CmdLine::import_arg_group("opt");
        GEO::CmdLine::import_arg_group("pre");
        GEO::CmdLine::import_arg_group("post");
        GEO::CmdLine::import_arg_group("co3ne");
        // Suppress geogram's own logging to stderr
        GEO::CmdLine::set_arg("log:quiet", "true");
    }
}

// Convert numpy arrays (N,3) vertices + (M,3) faces into a GEO::Mesh.
static void numpy_to_geomesh(
    const NDArray<const double, 2> verts,
    const NDArray<const int, 2> faces,
    GEO::Mesh &M
) {
    const size_t nv = verts.shape(0);
    const size_t nf = faces.shape(0);

    if (verts.shape(1) != 3) {
        throw std::runtime_error("Vertex array must have shape (N, 3)");
    }
    if (faces.shape(1) != 3) {
        throw std::runtime_error("Face array must have shape (M, 3)");
    }

    M.clear();
    M.vertices.create_vertices(static_cast<GEO::index_t>(nv));
    for (size_t i = 0; i < nv; ++i) {
        GEO::vec3 &p = M.vertices.point(static_cast<GEO::index_t>(i));
        p[0] = verts(i, 0);
        p[1] = verts(i, 1);
        p[2] = verts(i, 2);
    }

    M.facets.create_triangles(static_cast<GEO::index_t>(nf));
    for (size_t i = 0; i < nf; ++i) {
        GEO::index_t fi = static_cast<GEO::index_t>(i);
        M.facets.set_vertex(fi, 0, static_cast<GEO::index_t>(faces(i, 0)));
        M.facets.set_vertex(fi, 1, static_cast<GEO::index_t>(faces(i, 1)));
        M.facets.set_vertex(fi, 2, static_cast<GEO::index_t>(faces(i, 2)));
    }

    M.facets.connect();
}

// Convert a GEO::Mesh back to numpy arrays: (vertices, faces).
static nb::tuple geomesh_to_numpy(const GEO::Mesh &M) {
    const GEO::index_t nv = M.vertices.nb();
    const GEO::index_t nf = M.facets.nb();

    NDArray<double, 2> verts_arr = MakeNDArray<double, 2>({static_cast<int>(nv), 3});
    double *verts = verts_arr.data();

    for (GEO::index_t i = 0; i < nv; ++i) {
        const GEO::vec3 &p = M.vertices.point(i);
        verts[i * 3 + 0] = p[0];
        verts[i * 3 + 1] = p[1];
        verts[i * 3 + 2] = p[2];
    }

    NDArray<int, 2> faces_arr = MakeNDArray<int, 2>({static_cast<int>(nf), 3});
    int *faces = faces_arr.data();

    for (GEO::index_t f = 0; f < nf; ++f) {
        faces[f * 3 + 0] = static_cast<int>(M.facets.vertex(f, 0));
        faces[f * 3 + 1] = static_cast<int>(M.facets.vertex(f, 1));
        faces[f * 3 + 2] = static_cast<int>(M.facets.vertex(f, 2));
    }

    return nb::make_tuple(verts_arr, faces_arr);
}

// Wrapper for GEO::remesh_smooth (CVT remeshing).
static nb::tuple py_remesh_smooth(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int nb_points,
    int nb_lloyd_iter,
    int nb_newton_iter,
    int newton_m,
    bool adjust,
    double adjust_max_edge_distance,
    double adjust_border_importance
) {
    ensure_geogram_initialized();

    if (nb_points < 1) {
        throw std::runtime_error("nb_points must be >= 1");
    }

    GEO::Mesh M_in;
    numpy_to_geomesh(vertices, faces, M_in);

    // Repair input mesh for robustness
    GEO::mesh_repair(M_in, GEO::MESH_REPAIR_DEFAULT);

    GEO::Mesh M_out;
    GEO::remesh_smooth(
        M_in, M_out,
        static_cast<GEO::index_t>(nb_points),
        0, // dim=0 means use M_in.vertices.dimension()
        static_cast<GEO::index_t>(nb_lloyd_iter),
        static_cast<GEO::index_t>(nb_newton_iter),
        static_cast<GEO::index_t>(newton_m),
        adjust,
        adjust_max_edge_distance,
        adjust_border_importance
    );

    return geomesh_to_numpy(M_out);
}

// Wrapper for GEO::mesh_repair (in-place repair, returns cleaned mesh).
static nb::tuple py_mesh_repair(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int mode,
    double colocate_epsilon
) {
    ensure_geogram_initialized();

    GEO::Mesh M;
    numpy_to_geomesh(vertices, faces, M);

    GEO::mesh_repair(M, static_cast<GEO::MeshRepairMode>(mode), colocate_epsilon);

    return geomesh_to_numpy(M);
}

// Wrapper for GEO::mesh_decimate_vertex_clustering.
static nb::tuple py_mesh_decimate(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int nb_bins,
    int mode
) {
    ensure_geogram_initialized();

    if (nb_bins < 1) {
        throw std::runtime_error("nb_bins must be >= 1");
    }

    GEO::Mesh M;
    numpy_to_geomesh(vertices, faces, M);

    GEO::mesh_decimate_vertex_clustering(
        M,
        static_cast<GEO::index_t>(nb_bins),
        static_cast<GEO::MeshDecimateMode>(mode)
    );

    return geomesh_to_numpy(M);
}


// Convert numpy (N,3) vertices into a GEO::Mesh with only points (no facets).
static void numpy_points_to_geomesh(
    const NDArray<const double, 2> verts,
    GEO::Mesh &M
) {
    const size_t nv = verts.shape(0);
    if (verts.shape(1) != 3) {
        throw std::runtime_error("Vertex array must have shape (N, 3)");
    }

    M.clear();
    M.vertices.create_vertices(static_cast<GEO::index_t>(nv));
    for (size_t i = 0; i < nv; ++i) {
        GEO::vec3 &p = M.vertices.point(static_cast<GEO::index_t>(i));
        p[0] = verts(i, 0);
        p[1] = verts(i, 1);
        p[2] = verts(i, 2);
    }
}

// Convert numpy (N,3) vertices + (N,3) normals into a GEO::Mesh with
// a "normal" vertex attribute (required by Poisson reconstruction).
static void numpy_points_with_normals_to_geomesh(
    const NDArray<const double, 2> verts,
    const NDArray<const double, 2> normals,
    GEO::Mesh &M
) {
    const size_t nv = verts.shape(0);
    if (verts.shape(1) != 3) {
        throw std::runtime_error("Vertex array must have shape (N, 3)");
    }
    if (normals.shape(0) != nv || normals.shape(1) != 3) {
        throw std::runtime_error("Normals array must have shape (N, 3) matching vertices");
    }

    M.clear();
    M.vertices.create_vertices(static_cast<GEO::index_t>(nv));
    for (size_t i = 0; i < nv; ++i) {
        GEO::vec3 &p = M.vertices.point(static_cast<GEO::index_t>(i));
        p[0] = verts(i, 0);
        p[1] = verts(i, 1);
        p[2] = verts(i, 2);
    }

    // Create "normal" vector attribute with dimension 3
    GEO::Attribute<double> normal_attr;
    normal_attr.create_vector_attribute(M.vertices.attributes(), "normal", 3);
    for (size_t i = 0; i < nv; ++i) {
        normal_attr[i * 3 + 0] = normals(i, 0);
        normal_attr[i * 3 + 1] = normals(i, 1);
        normal_attr[i * 3 + 2] = normals(i, 2);
    }
}

// ── Smoothing wrappers ───────────────────────────────────────────

// Wrapper for GEO::simple_Laplacian_smooth.
// simple_Laplacian_smooth requires 6D vertices (position + normals),
// so we call compute_normals first to augment the mesh.
static nb::tuple py_simple_laplacian_smooth(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int nb_iter,
    bool normals_only
) {
    ensure_geogram_initialized();

    if (nb_iter < 1) {
        throw std::runtime_error("nb_iter must be >= 1");
    }

    GEO::Mesh M;
    numpy_to_geomesh(vertices, faces, M);

    // Augment to 6D (compute_normals sets dimension to 6)
    GEO::compute_normals(M);

    GEO::simple_Laplacian_smooth(
        M,
        static_cast<GEO::index_t>(nb_iter),
        normals_only
    );

    // Project back to 3D for output
    M.vertices.set_dimension(3);

    return geomesh_to_numpy(M);
}

// Wrapper for GEO::mesh_smooth (least-squares Laplacian via OpenNL).
static nb::tuple py_mesh_smooth(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces
) {
    ensure_geogram_initialized();

    GEO::Mesh M;
    numpy_to_geomesh(vertices, faces, M);

    GEO::mesh_smooth(M);

    return geomesh_to_numpy(M);
}

// Wrapper for GEO::compute_normals on a surface mesh.
// Returns normals (N,3) as a numpy array.
static NDArray<double, 2> py_compute_normals(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces
) {
    ensure_geogram_initialized();

    GEO::Mesh M;
    numpy_to_geomesh(vertices, faces, M);

    // compute_normals augments mesh to 6D (xyz + normal xyz)
    GEO::compute_normals(M);

    const GEO::index_t nv = M.vertices.nb();
    NDArray<double, 2> normals_arr = MakeNDArray<double, 2>({static_cast<int>(nv), 3});
    double *out = normals_arr.data();

    // After compute_normals, dimension is 6: coords 3,4,5 are the normal
    for (GEO::index_t i = 0; i < nv; ++i) {
        const double *p = M.vertices.point_ptr(i);
        out[i * 3 + 0] = p[3];
        out[i * 3 + 1] = p[4];
        out[i * 3 + 2] = p[5];
    }

    return normals_arr;
}

// ── Anisotropic remeshing wrapper ────────────────────────────────

static nb::tuple py_remesh_anisotropic(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int nb_points,
    double anisotropy,
    int nb_lloyd_iter,
    int nb_newton_iter,
    int newton_m,
    bool adjust,
    double adjust_max_edge_distance,
    double adjust_border_importance
) {
    ensure_geogram_initialized();

    if (nb_points < 1) {
        throw std::runtime_error("nb_points must be >= 1");
    }
    if (anisotropy <= 0.0) {
        throw std::runtime_error("anisotropy must be > 0");
    }

    GEO::Mesh M_in;
    numpy_to_geomesh(vertices, faces, M_in);

    GEO::mesh_repair(M_in, GEO::MESH_REPAIR_DEFAULT);

    // Augment to 6D with scaled normals for anisotropic CVT
    GEO::set_anisotropy(M_in, anisotropy);

    GEO::Mesh M_out;
    GEO::remesh_smooth(
        M_in, M_out,
        static_cast<GEO::index_t>(nb_points),
        6, // dim=6 for anisotropic
        static_cast<GEO::index_t>(nb_lloyd_iter),
        static_cast<GEO::index_t>(nb_newton_iter),
        static_cast<GEO::index_t>(newton_m),
        adjust,
        adjust_max_edge_distance,
        adjust_border_importance
    );

    // Project output back to 3D
    GEO::unset_anisotropy(M_out);

    return geomesh_to_numpy(M_out);
}

// ── Surface reconstruction wrappers ──────────────────────────────

// Wrapper for Co3Ne_smooth_and_reconstruct (all-in-one).
static nb::tuple py_co3ne_reconstruct(
    const NDArray<const double, 2> vertices,
    int nb_neighbors,
    int nb_iterations,
    double radius
) {
    ensure_geogram_initialized();

    if (nb_neighbors < 1) {
        throw std::runtime_error("nb_neighbors must be >= 1");
    }

    GEO::Mesh M;
    numpy_points_to_geomesh(vertices, M);

    GEO::Co3Ne_smooth_and_reconstruct(
        M,
        static_cast<GEO::index_t>(nb_neighbors),
        static_cast<GEO::index_t>(nb_iterations),
        radius
    );

    return geomesh_to_numpy(M);
}

// Wrapper for Co3Ne_compute_normals (point cloud normals).
static NDArray<double, 2> py_co3ne_compute_normals(
    const NDArray<const double, 2> vertices,
    int nb_neighbors,
    bool reorient
) {
    ensure_geogram_initialized();

    if (nb_neighbors < 1) {
        throw std::runtime_error("nb_neighbors must be >= 1");
    }

    GEO::Mesh M;
    numpy_points_to_geomesh(vertices, M);

    bool ok = GEO::Co3Ne_compute_normals(
        M,
        static_cast<GEO::index_t>(nb_neighbors),
        reorient
    );

    if (!ok) {
        throw std::runtime_error("Co3Ne normal computation failed");
    }

    const GEO::index_t nv = M.vertices.nb();
    GEO::Attribute<double> normal_attr(M.vertices.attributes(), "normal");

    NDArray<double, 2> normals_arr = MakeNDArray<double, 2>({static_cast<int>(nv), 3});
    double *out = normals_arr.data();

    for (GEO::index_t i = 0; i < nv; ++i) {
        out[i * 3 + 0] = normal_attr[i * 3 + 0];
        out[i * 3 + 1] = normal_attr[i * 3 + 1];
        out[i * 3 + 2] = normal_attr[i * 3 + 2];
    }

    return normals_arr;
}

// Wrapper for Poisson surface reconstruction.
static nb::tuple py_poisson_reconstruct(
    const NDArray<const double, 2> vertices,
    const NDArray<const double, 2> normals,
    int depth
) {
    ensure_geogram_initialized();

    if (depth < 1) {
        throw std::runtime_error("depth must be >= 1");
    }

    GEO::Mesh M_points;
    numpy_points_with_normals_to_geomesh(vertices, normals, M_points);

    GEO::Mesh M_surface;
    GEO::PoissonReconstruction poisson;
    poisson.set_depth(static_cast<GEO::index_t>(depth));
    poisson.reconstruct(&M_points, &M_surface);

    return geomesh_to_numpy(M_surface);
}

NB_MODULE(_pygeogram, m) {
    m.doc() = "Python bindings for geogram geometry processing library";

    m.def(
        "remesh_smooth",
        &py_remesh_smooth,
        R"doc(
Remesh a surface using Centroidal Voronoi Tessellation (CVT).

Produces a high-quality isotropic remeshing by computing a CVT on the input
surface. The algorithm uses Lloyd relaxation followed by Newton optimization
to distribute points evenly, then extracts the dual surface.

Parameters
----------
vertices : numpy.ndarray[np.float64]
    Vertex array of shape (N, 3).
faces : numpy.ndarray[np.int32]
    Face array of shape (M, 3), triangles only.
nb_points : int
    Target number of output vertices. Use 0 to keep the same count as input.
nb_lloyd_iter : int, default: 5
    Number of Lloyd relaxation iterations (initial uniform distribution).
nb_newton_iter : int, default: 30
    Number of Newton optimization iterations (refines point placement).
newton_m : int, default: 7
    Number of evaluations for Hessian approximation in L-BFGS.
adjust : bool, default: True
    If True, adjusts output vertices to better approximate the input surface.
adjust_max_edge_distance : float, default: 0.5
    Max search distance for surface adjustment, relative to average edge length.
adjust_border_importance : float, default: 2.0
    Importance of boundary fitting during adjustment.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    (vertices, faces) — output vertex array (P, 3) and face array (Q, 3).

Examples
--------
>>> import pygeogram
>>> v_out, f_out = pygeogram.remesh_smooth(vertices, faces, nb_points=5000)
)doc",
        nb::arg("vertices"),
        nb::arg("faces"),
        nb::arg("nb_points"),
        nb::arg("nb_lloyd_iter") = 5,
        nb::arg("nb_newton_iter") = 30,
        nb::arg("newton_m") = 7,
        nb::arg("adjust") = true,
        nb::arg("adjust_max_edge_distance") = 0.5,
        nb::arg("adjust_border_importance") = 2.0
    );

    // --- MeshRepairMode constants ---
    m.attr("MESH_REPAIR_COLOCATE") = static_cast<int>(GEO::MESH_REPAIR_COLOCATE);
    m.attr("MESH_REPAIR_DUP_F") = static_cast<int>(GEO::MESH_REPAIR_DUP_F);
    m.attr("MESH_REPAIR_TRIANGULATE") = static_cast<int>(GEO::MESH_REPAIR_TRIANGULATE);
    m.attr("MESH_REPAIR_RECONSTRUCT") = static_cast<int>(GEO::MESH_REPAIR_RECONSTRUCT);
    m.attr("MESH_REPAIR_QUIET") = static_cast<int>(GEO::MESH_REPAIR_QUIET);
    m.attr("MESH_REPAIR_DEFAULT") = static_cast<int>(GEO::MESH_REPAIR_DEFAULT);

    m.def(
        "mesh_repair",
        &py_mesh_repair,
        R"doc(
Repair a triangle mesh: merge colocated vertices, remove duplicate/degenerate
facets, triangulate, and fix non-manifold topology.

Parameters
----------
vertices : numpy.ndarray[np.float64]
    Vertex array of shape (N, 3).
faces : numpy.ndarray[np.int32]
    Face array of shape (M, 3).
mode : int
    Bitwise OR of MESH_REPAIR_* flags. Default: MESH_REPAIR_DEFAULT (7).
colocate_epsilon : float, default: 0.0
    Tolerance for merging colocated vertices. 0.0 = exact match only.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    (vertices, faces) — cleaned vertex array (P, 3) and face array (Q, 3).
)doc",
        nb::arg("vertices"),
        nb::arg("faces"),
        nb::arg("mode") = static_cast<int>(GEO::MESH_REPAIR_DEFAULT),
        nb::arg("colocate_epsilon") = 0.0
    );

    // --- MeshDecimateMode constants ---
    m.attr("MESH_DECIMATE_FAST") = static_cast<int>(GEO::MESH_DECIMATE_FAST);
    m.attr("MESH_DECIMATE_DUP_F") = static_cast<int>(GEO::MESH_DECIMATE_DUP_F);
    m.attr("MESH_DECIMATE_DEG_3") = static_cast<int>(GEO::MESH_DECIMATE_DEG_3);
    m.attr("MESH_DECIMATE_KEEP_B") = static_cast<int>(GEO::MESH_DECIMATE_KEEP_B);
    m.attr("MESH_DECIMATE_DEFAULT") = static_cast<int>(GEO::MESH_DECIMATE_DEFAULT);

    m.def(
        "mesh_decimate",
        &py_mesh_decimate,
        R"doc(
Simplify a mesh using vertex clustering (spatial binning).

Groups vertices into a 3D grid of nb_bins^3 cells, merges vertices within
each cell, and rebuilds the face connectivity.

Parameters
----------
vertices : numpy.ndarray[np.float64]
    Vertex array of shape (N, 3).
faces : numpy.ndarray[np.int32]
    Face array of shape (M, 3).
nb_bins : int
    Grid resolution. Higher = more detail preserved. Typical: 50-200.
mode : int
    Bitwise OR of MESH_DECIMATE_* flags. Default: MESH_DECIMATE_DEFAULT (7).

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    (vertices, faces) — simplified vertex array (P, 3) and face array (Q, 3).
)doc",
        nb::arg("vertices"),
        nb::arg("faces"),
        nb::arg("nb_bins"),
        nb::arg("mode") = static_cast<int>(GEO::MESH_DECIMATE_DEFAULT)
    );

    // --- Smoothing ---
    m.def(
        "simple_laplacian_smooth",
        &py_simple_laplacian_smooth,
        R"doc(
Smooth a mesh using iterative Laplacian relaxation.

Moves each vertex toward the barycenter of its neighbors. Fast but can
shrink the mesh with many iterations.

Parameters
----------
vertices : numpy.ndarray[np.float64]
    Vertex array of shape (N, 3).
faces : numpy.ndarray[np.int32]
    Face array of shape (M, 3), triangles only.
nb_iter : int
    Number of smoothing iterations.
normals_only : bool
    If True, only smooth stored normals, not vertex positions.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    (vertices, faces) — smoothed vertex array (N, 3) and face array (M, 3).
)doc",
        nb::arg("vertices"),
        nb::arg("faces"),
        nb::arg("nb_iter"),
        nb::arg("normals_only")
    );

    m.def(
        "mesh_smooth",
        &py_mesh_smooth,
        R"doc(
Smooth a mesh using least-squares Laplacian optimization.

Higher quality than simple Laplacian smoothing. Uses OpenNL linear solver.
Vertices with the "selection" attribute set to true are locked in place.

Parameters
----------
vertices : numpy.ndarray[np.float64]
    Vertex array of shape (N, 3).
faces : numpy.ndarray[np.int32]
    Face array of shape (M, 3), triangles only.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    (vertices, faces) — smoothed vertex array (N, 3) and face array (M, 3).
)doc",
        nb::arg("vertices"),
        nb::arg("faces")
    );

    m.def(
        "compute_normals",
        &py_compute_normals,
        R"doc(
Compute vertex normals for a surface mesh.

Parameters
----------
vertices : numpy.ndarray[np.float64]
    Vertex array of shape (N, 3).
faces : numpy.ndarray[np.int32]
    Face array of shape (M, 3), triangles only.

Returns
-------
numpy.ndarray[np.float64]
    Normal array of shape (N, 3).
)doc",
        nb::arg("vertices"),
        nb::arg("faces")
    );

    // --- Anisotropic Remeshing ---
    m.def(
        "remesh_anisotropic",
        &py_remesh_anisotropic,
        R"doc(
Remesh a surface using anisotropic Centroidal Voronoi Tessellation (CVT).

Similar to remesh_smooth but adapts triangle shape to surface curvature.
Augments the mesh to 6D (position + scaled normals) before CVT optimization,
producing elongated triangles in flat regions and smaller triangles in
high-curvature areas.

Parameters
----------
vertices : numpy.ndarray[np.float64]
    Vertex array of shape (N, 3).
faces : numpy.ndarray[np.int32]
    Face array of shape (M, 3), triangles only.
nb_points : int
    Target number of output vertices.
anisotropy : float, default: 0.04
    Anisotropy factor. Controls how much triangle shape adapts to curvature.
    Typical range: 0.02–0.1. Lower = more anisotropic.
nb_lloyd_iter : int, default: 5
    Number of Lloyd relaxation iterations.
nb_newton_iter : int, default: 30
    Number of Newton optimization iterations.
newton_m : int, default: 7
    Number of evaluations for Hessian approximation in L-BFGS.
adjust : bool, default: True
    Adjust output vertices to better approximate the input surface.
adjust_max_edge_distance : float, default: 0.5
    Max search distance for surface adjustment.
adjust_border_importance : float, default: 2.0
    Importance of boundary fitting during adjustment.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    (vertices, faces) — output vertex array (P, 3) and face array (Q, 3).
)doc",
        nb::arg("vertices"),
        nb::arg("faces"),
        nb::arg("nb_points"),
        nb::arg("anisotropy") = 0.04,
        nb::arg("nb_lloyd_iter") = 5,
        nb::arg("nb_newton_iter") = 30,
        nb::arg("newton_m") = 7,
        nb::arg("adjust") = true,
        nb::arg("adjust_max_edge_distance") = 0.5,
        nb::arg("adjust_border_importance") = 2.0
    );

    // --- Surface Reconstruction ---
    m.def(
        "co3ne_reconstruct",
        &py_co3ne_reconstruct,
        R"doc(
Reconstruct a triangle mesh from a point cloud using Co3Ne.

Smoothes the point cloud and reconstructs triangles in one pass using
the Concurrent Co-Cones algorithm.

Parameters
----------
vertices : numpy.ndarray[np.float64]
    Point cloud of shape (N, 3).
nb_neighbors : int, default: 30
    Number of neighbors for tangent plane estimation.
nb_iterations : int, default: 3
    Number of point cloud smoothing iterations.
radius : float, default: 5.0
    Maximum distance for connecting neighbors with triangles.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    (vertices, faces) — reconstructed vertex array and face array.
)doc",
        nb::arg("vertices"),
        nb::arg("nb_neighbors") = 30,
        nb::arg("nb_iterations") = 3,
        nb::arg("radius") = 5.0
    );

    m.def(
        "co3ne_compute_normals",
        &py_co3ne_compute_normals,
        R"doc(
Compute normals for a point cloud using Co3Ne.

Estimates normals from the nearest neighbors best-approximating plane.

Parameters
----------
vertices : numpy.ndarray[np.float64]
    Point cloud of shape (N, 3).
nb_neighbors : int, default: 30
    Number of neighbors for tangent plane estimation.
reorient : bool, default: False
    If True, propagate orientation over the KNN graph for consistent normals.

Returns
-------
numpy.ndarray[np.float64]
    Normal array of shape (N, 3).
)doc",
        nb::arg("vertices"),
        nb::arg("nb_neighbors") = 30,
        nb::arg("reorient") = false
    );

    m.def(
        "poisson_reconstruct",
        &py_poisson_reconstruct,
        R"doc(
Reconstruct a surface from oriented point cloud using Poisson reconstruction.

Solves the Poisson equation on an adaptive octree to extract an isosurface.
Requires pre-computed normals.

Parameters
----------
vertices : numpy.ndarray[np.float64]
    Point cloud of shape (N, 3).
normals : numpy.ndarray[np.float64]
    Normal array of shape (N, 3), one normal per point.
depth : int, default: 8
    Octree depth. Higher = more detail. Use 10–11 for highly detailed models.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    (vertices, faces) — reconstructed vertex array and face array.
)doc",
        nb::arg("vertices"),
        nb::arg("normals"),
        nb::arg("depth") = 8
    );
}
