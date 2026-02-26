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


def generate_html(sections):
    all_sections_html = ""
    for section in sections:
        all_sections_html += f"""
    <h2 class="section-title">{section['title']}</h2>
    <p class="section-sub">{section['subtitle']}</p>
"""
        for d in section["demos"]:
            code_html = html_escape(d["code"])
            label = d.get("after_label", "Output")
            all_sections_html += f"""
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
    </section>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>pygeogram — Geometry Processing</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0d1117;
      color: #c9d1d9;
      padding: 2rem;
      max-width: 1400px;
      margin: 0 auto;
    }}
    a {{ color: #58a6ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    header {{
      text-align: center;
      margin-bottom: 2.5rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid #21262d;
    }}
    header h1 {{ font-size: 2.2rem; margin-bottom: 0.3rem; color: #f0f6fc; }}
    header .sub {{ color: #8b949e; font-size: 1.1rem; margin-bottom: 1rem; }}
    .install {{
      background: #161b22;
      color: #aed581;
      padding: 0.8rem 1.5rem;
      border-radius: 6px;
      border: 1px solid #21262d;
      font-family: "SF Mono", "Fira Code", monospace;
      font-size: 0.95rem;
      display: inline-block;
      margin: 0.8rem 0;
    }}
    .links {{ margin-top: 0.8rem; color: #8b949e; }}

    .section-title {{
      font-size: 1.5rem;
      color: #f0f6fc;
      margin: 2.5rem 0 0.3rem 0;
      padding-top: 1.5rem;
      border-top: 1px solid #21262d;
    }}
    .section-sub {{
      color: #8b949e;
      font-size: 0.95rem;
      margin-bottom: 1rem;
    }}

    .demo {{
      background: #161b22;
      border: 1px solid #21262d;
      border-radius: 8px;
      margin-bottom: 1.5rem;
      overflow: hidden;
    }}
    .demo-grid {{
      display: flex;
      flex-wrap: wrap;
    }}
    .demo-code {{
      flex: 0 0 420px;
      padding: 1.5rem;
      border-right: 1px solid #21262d;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }}
    .demo-code pre {{
      background: #0d1117;
      border: 1px solid #21262d;
      border-radius: 6px;
      padding: 1rem;
      overflow-x: auto;
      font-size: 0.85rem;
      line-height: 1.5;
    }}
    .demo-code code {{
      font-family: "SF Mono", "Fira Code", monospace;
      color: #c9d1d9;
    }}
    .timing {{
      color: #81c784;
      font-family: "SF Mono", "Fira Code", monospace;
      font-size: 0.85rem;
      margin-top: 0.8rem;
    }}
    .demo-images {{
      flex: 1;
      min-width: 500px;
      padding: 1rem;
    }}
    .comparison {{
      display: flex;
      gap: 0.5rem;
    }}
    .panel {{
      position: relative;
      flex: 1;
    }}
    .panel img {{
      width: 100%;
      border-radius: 4px;
      border: 1px solid #21262d;
    }}
    .label {{
      position: absolute;
      bottom: 6px;
      right: 6px;
      background: rgba(0,0,0,0.7);
      color: #c9d1d9;
      padding: 0.15rem 0.4rem;
      border-radius: 3px;
      font-size: 0.75rem;
    }}
    footer {{
      text-align: center;
      color: #484f58;
      margin-top: 2rem;
      font-size: 0.85rem;
    }}
    footer a {{ color: #484f58; }}
    @media (max-width: 900px) {{
      .demo-code {{ flex: 1 1 100%; border-right: none; border-bottom: 1px solid #21262d; }}
      .demo-images {{ min-width: unset; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>pygeogram</h1>
    <p class="sub">Python bindings for <a href="https://github.com/BrunoLevy/geogram">geogram</a> &mdash; geometry processing</p>
    <div class="install">pip install pygeogram --find-links https://github.com/PozzettiAndrea/pygeogram/releases/latest/download/</div>
    <p class="links">
      <a href="https://github.com/PozzettiAndrea/pygeogram">GitHub</a> &middot;
      <a href="https://pypi.org/project/pygeogram/">PyPI</a>
    </p>
  </header>

  {all_sections_html}

  <footer>
    Generated automatically by CI &middot;
    <a href="https://github.com/BrunoLevy/geogram">geogram</a> by Bruno Levy (INRIA)
  </footer>
</body>
</html>
"""
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
