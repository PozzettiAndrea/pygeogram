"""
Generate visual demo of pygeogram for GitHub Pages.

Shows CVT remeshing, mesh repair, and mesh decimation — each with
the actual Python code used alongside before/after renders.
"""

import os
import shutil
import time
import textwrap
import numpy as np

import pyvista as pv

pv.OFF_SCREEN = True

import pygeogram

OUT_DIR = os.path.join(os.path.dirname(__file__), "_site")

# Dark theme
BG_COLOR = "#1a1a2e"
MESH_COLOR_IN = "#4fc3f7"
MESH_COLOR_OUT = "#81c784"
MESH_COLOR_WARN = "#ffb74d"
EDGE_COLOR = "#222244"
TEXT_COLOR = "#e0e0e0"


def pv_mesh_from_numpy(verts, faces):
    n = len(faces)
    pv_faces = np.column_stack([np.full(n, 3, dtype=np.int32), faces]).ravel()
    return pv.PolyData(verts, pv_faces)


def render_mesh(mesh, filename, title, color=MESH_COLOR_IN,
                window_size=(800, 600)):
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.add_mesh(mesh, color=color, show_edges=True, edge_color=EDGE_COLOR,
                line_width=0.5, lighting=True, smooth_shading=True)
    pl.add_text(title, position="upper_left", font_size=12, color=TEXT_COLOR)
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()


def get_mesh():
    """Get Stanford bunny, fallback to icosphere."""
    try:
        bunny = pv.examples.download_bunny()
        verts = np.array(bunny.points, dtype=np.float64)
        faces = np.array(bunny.faces.reshape(-1, 4)[:, 1:], dtype=np.int32)
        return verts, faces, "bunny.stl"
    except Exception:
        sphere = pv.Icosphere(nsub=4, radius=1.0)
        verts = np.array(sphere.points, dtype=np.float64)
        faces = np.array(sphere.faces.reshape(-1, 4)[:, 1:], dtype=np.int32)
        return verts, faces, "sphere.stl"


def run_demo(name, func, verts_in, faces_in, code, after_label="Output"):
    """Run a pygeogram function, render before/after, return demo dict."""
    t0 = time.perf_counter()
    verts_out, faces_out = func(verts_in, faces_in)
    elapsed = time.perf_counter() - t0

    mesh_in = pv_mesh_from_numpy(verts_in, faces_in)
    mesh_out = pv_mesh_from_numpy(verts_out, faces_out)

    prefix = os.path.join(OUT_DIR, name)
    render_mesh(mesh_in, f"{prefix}_before.png",
                f"Input: {len(verts_in):,} verts, {len(faces_in):,} faces")
    render_mesh(mesh_out, f"{prefix}_after.png",
                f"{after_label}: {len(verts_out):,} verts, {len(faces_out):,} faces  ({elapsed:.2f}s)",
                color=MESH_COLOR_OUT)

    return {
        "name": name,
        "verts_in": len(verts_in),
        "faces_in": len(faces_in),
        "verts_out": len(verts_out),
        "faces_out": len(faces_out),
        "elapsed": elapsed,
        "code": code,
        "after_label": after_label,
    }


def html_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


TEMPLATE_DIR = os.path.dirname(__file__)


def _render_demo(d):
    code_html = html_escape(d["code"])
    label = d.get("after_label", "Output")
    return f"""
    <section class="demo">
      <div class="demo-grid">
        <div class="demo-code">
          <pre><code>{code_html}</code></pre>
          <p class="timing">{d['elapsed']:.2f}s &mdash; {d['verts_in']:,} &rarr; {d['verts_out']:,} verts, {d['faces_in']:,} &rarr; {d['faces_out']:,} faces</p>
        </div>
        <div class="demo-images">
          <div class="comparison">
            <div class="panel">
              <img src="{d['name']}_before.png" alt="Before">
              <span class="label">Input</span>
            </div>
            <div class="panel">
              <img src="{d['name']}_after.png" alt="After">
              <span class="label">{label}</span>
            </div>
          </div>
        </div>
      </div>
    </section>"""


def generate_html(sections):
    sections_html = ""
    for section in sections:
        sections_html += f"""
    <h2 class="section-title">{section['title']}</h2>
    <p class="section-sub">{section['subtitle']}</p>"""
        for d in section["demos"]:
            sections_html += _render_demo(d)

    with open(os.path.join(TEMPLATE_DIR, "template.html")) as f:
        template = f.read()

    html = template.replace("{{sections}}", sections_html)

    with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
        f.write(html)


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    verts, faces, mesh_name = get_mesh()
    sections = []

    # ── CVT Remeshing ──────────────────────────────────────────────
    remesh_demos = []

    remesh_demos.append(run_demo("default",
        lambda v, f: pygeogram.remesh_smooth(v, f, nb_points=2000),
        verts, faces,
        textwrap.dedent(f"""\
            import pygeogram
            import trimesh

            mesh = trimesh.load("{mesh_name}")
            v, f = pygeogram.remesh_smooth(
                mesh.vertices, mesh.faces,
                nb_points=2000,
            )"""),
        after_label="CVT Remeshed"))

    remesh_demos.append(run_demo("high_quality",
        lambda v, f: pygeogram.remesh_smooth(v, f, nb_points=3000,
            nb_lloyd_iter=10, nb_newton_iter=50),
        verts, faces,
        textwrap.dedent(f"""\
            v, f = pygeogram.remesh_smooth(
                mesh.vertices, mesh.faces,
                nb_points=3000,
                nb_lloyd_iter=10,
                nb_newton_iter=50,
            )"""),
        after_label="CVT Remeshed"))

    remesh_demos.append(run_demo("low_poly",
        lambda v, f: pygeogram.remesh_smooth(v, f, nb_points=500,
            nb_lloyd_iter=3, nb_newton_iter=10),
        verts, faces,
        textwrap.dedent(f"""\
            v, f = pygeogram.remesh_smooth(
                mesh.vertices, mesh.faces,
                nb_points=500,
                nb_lloyd_iter=3,
                nb_newton_iter=10,
            )"""),
        after_label="CVT Remeshed"))

    sections.append({
        "title": "CVT Remeshing",
        "subtitle": "Isotropic remeshing via Centroidal Voronoi Tessellation",
        "demos": remesh_demos,
    })

    # ── Mesh Repair ────────────────────────────────────────────────
    repair_demos = []

    # Manufacture defects: duplicate vertices + duplicate faces
    n_orig_v = len(verts)
    n_dup_v = min(500, n_orig_v)
    n_dup_f = min(200, len(faces))
    broken_verts = np.vstack([verts, verts[:n_dup_v]])
    # Rewire duplicated faces to use the new (duplicate) vertex indices
    dup_faces = faces[:n_dup_f].copy()
    for i in range(len(dup_faces)):
        for j in range(3):
            if dup_faces[i, j] < n_dup_v:
                dup_faces[i, j] += n_orig_v
    broken_faces = np.vstack([faces, dup_faces])

    repair_demos.append(run_demo("repair",
        lambda v, f: pygeogram.mesh_repair(v, f),
        broken_verts, broken_faces,
        textwrap.dedent(f"""\
            import pygeogram
            import numpy as np

            # Simulate corrupt mesh: {n_dup_v} duplicate
            # vertices + {n_dup_f} duplicate faces
            v = np.vstack([v_in, v_in[:{n_dup_v}]])
            f = np.vstack([f_in, f_in[:{n_dup_f}]])

            v_clean, f_clean = pygeogram.mesh_repair(
                v, f,
                colocate=True,
                remove_duplicates=True,
            )"""),
        after_label="Repaired"))

    sections.append({
        "title": "Mesh Repair",
        "subtitle": "Fix topology, merge duplicate vertices, remove degenerate faces",
        "demos": repair_demos,
    })

    # ── Mesh Decimation ────────────────────────────────────────────
    decimate_demos = []

    decimate_demos.append(run_demo("decimate_med",
        lambda v, f: pygeogram.mesh_decimate(v, f, nb_bins=100),
        verts, faces,
        textwrap.dedent(f"""\
            import pygeogram

            v, f = pygeogram.mesh_decimate(
                mesh.vertices, mesh.faces,
                nb_bins=100,
            )"""),
        after_label="Decimated"))

    decimate_demos.append(run_demo("decimate_low",
        lambda v, f: pygeogram.mesh_decimate(v, f, nb_bins=30),
        verts, faces,
        textwrap.dedent(f"""\
            # Aggressive decimation
            v, f = pygeogram.mesh_decimate(
                mesh.vertices, mesh.faces,
                nb_bins=30,
            )"""),
        after_label="Decimated"))

    sections.append({
        "title": "Vertex Clustering Decimation",
        "subtitle": "Fast mesh simplification via spatial binning",
        "demos": decimate_demos,
    })

    # ── Smoothing ─────────────────────────────────────────────────
    smooth_demos = []

    smooth_demos.append(run_demo("smooth_laplacian",
        lambda v, f: pygeogram.smooth(v, f, nb_iter=5),
        verts, faces,
        textwrap.dedent(f"""\
            import pygeogram

            v, f = pygeogram.smooth(
                mesh.vertices, mesh.faces,
                nb_iter=5,
            )"""),
        after_label="Smoothed"))

    smooth_demos.append(run_demo("smooth_lsq",
        lambda v, f: pygeogram.smooth_lsq(v, f),
        verts, faces,
        textwrap.dedent(f"""\
            # Least-squares Laplacian (higher quality)
            v, f = pygeogram.smooth_lsq(
                mesh.vertices, mesh.faces,
            )"""),
        after_label="Smoothed (LSQ)"))

    sections.append({
        "title": "Mesh Smoothing",
        "subtitle": "Laplacian and least-squares smoothing",
        "demos": smooth_demos,
    })

    # ── Anisotropic Remeshing ─────────────────────────────────────
    aniso_demos = []

    aniso_demos.append(run_demo("aniso_default",
        lambda v, f: pygeogram.remesh_anisotropic(v, f, nb_points=2000),
        verts, faces,
        textwrap.dedent(f"""\
            import pygeogram

            v, f = pygeogram.remesh_anisotropic(
                mesh.vertices, mesh.faces,
                nb_points=2000,
                anisotropy=0.04,
            )"""),
        after_label="Anisotropic Remeshed"))

    aniso_demos.append(run_demo("aniso_strong",
        lambda v, f: pygeogram.remesh_anisotropic(v, f, nb_points=2000,
            anisotropy=0.02),
        verts, faces,
        textwrap.dedent(f"""\
            # Stronger anisotropy (more curvature-adapted)
            v, f = pygeogram.remesh_anisotropic(
                mesh.vertices, mesh.faces,
                nb_points=2000,
                anisotropy=0.02,
            )"""),
        after_label="Anisotropic Remeshed"))

    sections.append({
        "title": "Anisotropic Remeshing",
        "subtitle": "Curvature-adapted CVT remeshing \u2014 elongated triangles in flat regions, finer in curved areas",
        "demos": aniso_demos,
    })

    # ── Surface Reconstruction ────────────────────────────────────
    recon_demos = []

    # Create a point cloud from the mesh vertices (subsample for visual clarity)
    rng = np.random.default_rng(42)
    n_pts = min(5000, len(verts))
    idx = rng.choice(len(verts), size=n_pts, replace=False)
    points = verts[idx]

    # We need a special render for point clouds (no faces)
    prefix_pc = os.path.join(OUT_DIR, "pointcloud")
    pc_pv = pv.PolyData(points)
    pl = pv.Plotter(off_screen=True, window_size=(800, 600))
    pl.add_mesh(pc_pv, color=MESH_COLOR_IN, point_size=3, render_points_as_spheres=True)
    pl.add_text(f"Input: {n_pts:,} points", position="upper_left", font_size=12, color=TEXT_COLOR)
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(f"{prefix_pc}_before.png", transparent_background=False)
    pl.close()

    # Co3Ne reconstruction
    t0 = time.perf_counter()
    v_recon, f_recon = pygeogram.co3ne_reconstruct(points, nb_neighbors=20, nb_iterations=0, radius=5.0)
    elapsed_co3ne = time.perf_counter() - t0

    mesh_recon = pv_mesh_from_numpy(v_recon, f_recon)
    render_mesh(mesh_recon, f"{prefix_pc}_after.png",
                f"Co3Ne: {len(v_recon):,} verts, {len(f_recon):,} faces  ({elapsed_co3ne:.2f}s)",
                color=MESH_COLOR_OUT)

    recon_demos.append({
        "name": "pointcloud",
        "verts_in": n_pts,
        "faces_in": 0,
        "verts_out": len(v_recon),
        "faces_out": len(f_recon),
        "elapsed": elapsed_co3ne,
        "code": textwrap.dedent(f"""\
            import pygeogram
            import numpy as np

            # {n_pts:,} points sampled from surface
            v, f = pygeogram.co3ne_reconstruct(
                points,
                nb_neighbors=20,
                radius=5.0,
            )"""),
        "after_label": "Co3Ne Reconstructed",
    })

    # Poisson reconstruction
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)  # outward normals for roughly centered mesh
    # Use Co3Ne to get better normals
    try:
        normals = pygeogram.co3ne_compute_normals(points, nb_neighbors=20, reorient=True)
    except Exception:
        pass

    t0 = time.perf_counter()
    v_poisson, f_poisson = pygeogram.poisson_reconstruct(points, normals, depth=6)
    elapsed_poisson = time.perf_counter() - t0

    mesh_poisson = pv_mesh_from_numpy(v_poisson, f_poisson)
    prefix_poisson = os.path.join(OUT_DIR, "poisson")
    # Reuse the point cloud input image
    shutil.copy(f"{prefix_pc}_before.png", f"{prefix_poisson}_before.png")
    render_mesh(mesh_poisson, f"{prefix_poisson}_after.png",
                f"Poisson: {len(v_poisson):,} verts, {len(f_poisson):,} faces  ({elapsed_poisson:.2f}s)",
                color=MESH_COLOR_OUT)

    recon_demos.append({
        "name": "poisson",
        "verts_in": n_pts,
        "faces_in": 0,
        "verts_out": len(v_poisson),
        "faces_out": len(f_poisson),
        "elapsed": elapsed_poisson,
        "code": textwrap.dedent(f"""\
            # Poisson reconstruction (needs normals)
            normals = pygeogram.co3ne_compute_normals(
                points, nb_neighbors=20, reorient=True,
            )
            v, f = pygeogram.poisson_reconstruct(
                points, normals, depth=6,
            )"""),
        "after_label": "Poisson Reconstructed",
    })

    sections.append({
        "title": "Surface Reconstruction",
        "subtitle": "Reconstruct triangle meshes from point clouds \u2014 Co3Ne and Poisson methods",
        "demos": recon_demos,
    })

    # ── Boolean Operations ────────────────────────────────────────
    bool_demos = []

    # Create two overlapping meshes: original + shifted copy
    shift = np.zeros_like(verts)
    bbox = verts.max(axis=0) - verts.min(axis=0)
    shift[:, 0] = bbox[0] * 0.4  # shift 40% of bounding box along X
    verts_b = verts + shift

    # Custom render for boolean: show both inputs
    prefix_bool = os.path.join(OUT_DIR, "bool_union")
    pl = pv.Plotter(off_screen=True, window_size=(800, 600))
    mesh_a_pv = pv_mesh_from_numpy(verts, faces)
    mesh_b_pv = pv_mesh_from_numpy(verts_b, faces)
    pl.add_mesh(mesh_a_pv, color=MESH_COLOR_IN, show_edges=True,
                edge_color=EDGE_COLOR, line_width=0.5, opacity=0.7,
                lighting=True, smooth_shading=True)
    pl.add_mesh(mesh_b_pv, color=MESH_COLOR_WARN, show_edges=True,
                edge_color=EDGE_COLOR, line_width=0.5, opacity=0.7,
                lighting=True, smooth_shading=True)
    pl.add_text(f"Input: 2 overlapping meshes", position="upper_left",
                font_size=12, color=TEXT_COLOR)
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(f"{prefix_bool}_before.png", transparent_background=False)
    pl.close()

    # Union
    t0 = time.perf_counter()
    v_union, f_union = pygeogram.mesh_union(verts, faces, verts_b, faces)
    elapsed_union = time.perf_counter() - t0

    mesh_union_pv = pv_mesh_from_numpy(v_union, f_union)
    render_mesh(mesh_union_pv, f"{prefix_bool}_after.png",
                f"Union: {len(v_union):,} verts, {len(f_union):,} faces  ({elapsed_union:.2f}s)",
                color=MESH_COLOR_OUT)

    bool_demos.append({
        "name": "bool_union",
        "verts_in": len(verts) + len(verts_b),
        "faces_in": len(faces) * 2,
        "verts_out": len(v_union),
        "faces_out": len(f_union),
        "elapsed": elapsed_union,
        "code": textwrap.dedent(f"""\
            import pygeogram

            # Combine two overlapping meshes
            v, f = pygeogram.mesh_union(
                verts_a, faces_a,
                verts_b, faces_b,
            )"""),
        "after_label": "Union (A + B)",
    })

    # Intersection — reuse same input image
    prefix_isect = os.path.join(OUT_DIR, "bool_isect")
    shutil.copy(f"{prefix_bool}_before.png", f"{prefix_isect}_before.png")

    t0 = time.perf_counter()
    v_isect, f_isect = pygeogram.mesh_intersection(verts, faces, verts_b, faces)
    elapsed_isect = time.perf_counter() - t0

    mesh_isect_pv = pv_mesh_from_numpy(v_isect, f_isect)
    render_mesh(mesh_isect_pv, f"{prefix_isect}_after.png",
                f"Intersection: {len(v_isect):,} verts, {len(f_isect):,} faces  ({elapsed_isect:.2f}s)",
                color=MESH_COLOR_OUT)

    bool_demos.append({
        "name": "bool_isect",
        "verts_in": len(verts) + len(verts_b),
        "faces_in": len(faces) * 2,
        "verts_out": len(v_isect),
        "faces_out": len(f_isect),
        "elapsed": elapsed_isect,
        "code": textwrap.dedent(f"""\
            # Keep only the overlapping region
            v, f = pygeogram.mesh_intersection(
                verts_a, faces_a,
                verts_b, faces_b,
            )"""),
        "after_label": "Intersection (A * B)",
    })

    # Difference — reuse same input image
    prefix_diff = os.path.join(OUT_DIR, "bool_diff")
    shutil.copy(f"{prefix_bool}_before.png", f"{prefix_diff}_before.png")

    t0 = time.perf_counter()
    v_diff, f_diff = pygeogram.mesh_difference(verts, faces, verts_b, faces)
    elapsed_diff = time.perf_counter() - t0

    mesh_diff_pv = pv_mesh_from_numpy(v_diff, f_diff)
    render_mesh(mesh_diff_pv, f"{prefix_diff}_after.png",
                f"Difference: {len(v_diff):,} verts, {len(f_diff):,} faces  ({elapsed_diff:.2f}s)",
                color=MESH_COLOR_OUT)

    bool_demos.append({
        "name": "bool_diff",
        "verts_in": len(verts) + len(verts_b),
        "faces_in": len(faces) * 2,
        "verts_out": len(v_diff),
        "faces_out": len(f_diff),
        "elapsed": elapsed_diff,
        "code": textwrap.dedent(f"""\
            # Subtract B from A
            v, f = pygeogram.mesh_difference(
                verts_a, faces_a,
                verts_b, faces_b,
            )"""),
        "after_label": "Difference (A - B)",
    })

    sections.append({
        "title": "Boolean Operations",
        "subtitle": "CSG union, intersection, and difference on closed surface meshes",
        "demos": bool_demos,
    })

    # ── Mesh I/O ──────────────────────────────────────────────────
    io_demos = []

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save and reload through OBJ
        obj_path = os.path.join(tmpdir, "mesh.obj")
        t0 = time.perf_counter()
        pygeogram.mesh_save(verts, faces, obj_path)
        v_loaded, f_loaded = pygeogram.mesh_load(obj_path)
        elapsed_io = time.perf_counter() - t0
        obj_size = os.path.getsize(obj_path)

    # Render original vs loaded (should be identical)
    mesh_loaded_pv = pv_mesh_from_numpy(v_loaded, f_loaded)
    prefix_io = os.path.join(OUT_DIR, "io_roundtrip")
    render_mesh(pv_mesh_from_numpy(verts, faces), f"{prefix_io}_before.png",
                f"Original: {len(verts):,} verts, {len(faces):,} faces")
    render_mesh(mesh_loaded_pv, f"{prefix_io}_after.png",
                f"Loaded: {len(v_loaded):,} verts, {len(f_loaded):,} faces  ({elapsed_io:.2f}s, {obj_size // 1024}KB)",
                color=MESH_COLOR_OUT)

    io_demos.append({
        "name": "io_roundtrip",
        "verts_in": len(verts),
        "faces_in": len(faces),
        "verts_out": len(v_loaded),
        "faces_out": len(f_loaded),
        "elapsed": elapsed_io,
        "code": textwrap.dedent(f"""\
            import pygeogram

            # Save to OBJ (also supports PLY, OFF, STL)
            pygeogram.mesh_save(verts, faces, "mesh.obj")

            # Load back
            v, f = pygeogram.mesh_load("mesh.obj")"""),
        "after_label": "Loaded from OBJ",
    })

    sections.append({
        "title": "Mesh I/O",
        "subtitle": "Load and save meshes in OBJ, PLY, OFF, STL, and mesh/meshb formats",
        "demos": io_demos,
    })

    # ── UV Parameterization ───────────────────────────────────────
    uv_demos = []

    t0 = time.perf_counter()
    uvs = pygeogram.make_atlas(verts, faces)
    elapsed_uv = time.perf_counter() - t0

    # Render original mesh colored by UV coordinates
    prefix_uv = os.path.join(OUT_DIR, "uv_atlas")
    render_mesh(pv_mesh_from_numpy(verts, faces), f"{prefix_uv}_before.png",
                f"Input: {len(verts):,} verts, {len(faces):,} faces")

    # Create UV visualization: build a flat mesh from UVs
    n_faces_uv = len(faces)
    uv_verts_flat = uvs.reshape(-1, 2)  # (M*3, 2)
    uv_verts_3d = np.column_stack([uv_verts_flat, np.zeros(len(uv_verts_flat))]).astype(np.float64)
    uv_faces_flat = np.arange(n_faces_uv * 3, dtype=np.int32).reshape(-1, 3)
    uv_mesh_pv = pv_mesh_from_numpy(uv_verts_3d, uv_faces_flat)

    pl = pv.Plotter(off_screen=True, window_size=(800, 600))
    pl.add_mesh(uv_mesh_pv, color=MESH_COLOR_OUT, show_edges=True,
                edge_color=EDGE_COLOR, line_width=0.5, lighting=True)
    pl.add_text(f"UV Atlas: {n_faces_uv:,} faces, {elapsed_uv:.2f}s",
                position="upper_left", font_size=12, color=TEXT_COLOR)
    pl.set_background(BG_COLOR)
    pl.camera_position = "xy"
    pl.screenshot(f"{prefix_uv}_after.png", transparent_background=False)
    pl.close()

    uv_demos.append({
        "name": "uv_atlas",
        "verts_in": len(verts),
        "faces_in": len(faces),
        "verts_out": len(verts),
        "faces_out": len(faces),
        "elapsed": elapsed_uv,
        "code": textwrap.dedent(f"""\
            import pygeogram

            # Generate UV atlas (ABF + XAtlas packing)
            uvs = pygeogram.make_atlas(
                verts, faces,
                parameterizer=pygeogram.PARAM_ABF,
                packer=pygeogram.PACK_XATLAS,
            )
            # uvs.shape == (n_faces, 3, 2)"""),
        "after_label": "UV Layout",
    })

    sections.append({
        "title": "UV Parameterization",
        "subtitle": "Generate texture atlas with LSCM/ABF parameterization and XAtlas packing",
        "demos": uv_demos,
    })

    generate_html(sections)

    # Preview image for README (first remesh before/after)
    try:
        from PIL import Image
        d = remesh_demos[0]
        before = Image.open(os.path.join(OUT_DIR, f"{d['name']}_before.png"))
        after = Image.open(os.path.join(OUT_DIR, f"{d['name']}_after.png"))
        w, h = before.size
        grid = Image.new("RGB", (w * 2, h), "#0d1117")
        grid.paste(before, (0, 0))
        grid.paste(after, (w, 0))
        grid.save(os.path.join(OUT_DIR, "preview.png"))
    except Exception as e:
        print(f"Skipping preview: {e}")

    print(f"Demo: {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f} ({sz // 1024}KB)")


if __name__ == "__main__":
    main()
