// Python bindings for geogram CVT remeshing via nanobind.
//
// Exposes geogram's remesh_smooth() (Centroidal Voronoi Tessellation) to Python.
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
}
