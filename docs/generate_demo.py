"""
Generate visual demo of pygeogram CVT remeshing for GitHub Pages.

Shows 3 examples of the same mesh with different parameters,
alongside the actual code used.
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


def run_example(name, verts_in, faces_in, kwargs, code):
    t0 = time.perf_counter()
    verts_out, faces_out = pygeogram.remesh_smooth(verts_in, faces_in, **kwargs)
    elapsed = time.perf_counter() - t0

    mesh_in = pv_mesh_from_numpy(verts_in, faces_in)
    mesh_out = pv_mesh_from_numpy(verts_out, faces_out)

    prefix = os.path.join(OUT_DIR, name)
    render_mesh(mesh_in, f"{prefix}_before.png",
                f"Input: {len(verts_in):,} verts, {len(faces_in):,} faces")
    render_mesh(mesh_out, f"{prefix}_after.png",
                f"Output: {len(verts_out):,} verts, {len(faces_out):,} faces  ({elapsed:.2f}s)",
                color=MESH_COLOR_OUT)

    return {
        "name": name,
        "verts_in": len(verts_in),
        "faces_in": len(faces_in),
        "verts_out": len(verts_out),
        "faces_out": len(faces_out),
        "elapsed": elapsed,
        "code": code,
    }


def html_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_html(demos):
    demo_sections = ""
    for d in demos:
        code_html = html_escape(d["code"])
        demo_sections += f"""
    <section class="demo">
      <div class="demo-grid">
        <div class="demo-code">
          <pre><code>{code_html}</code></pre>
          <p class="timing">{d['elapsed']:.2f}s &mdash; {d['verts_in']:,} &rarr; {d['verts_out']:,} verts</p>
        </div>
        <div class="demo-images">
          <div class="comparison">
            <div class="panel">
              <img src="{d['name']}_before.png" alt="Before">
              <span class="label">Input</span>
            </div>
            <div class="panel">
              <img src="{d['name']}_after.png" alt="After">
              <span class="label">CVT Remeshed</span>
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
  <title>pygeogram — Voronoi Remeshing</title>
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
      flex: 0 0 400px;
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
    <p class="sub">Python bindings for <a href="https://github.com/BrunoLevy/geogram">geogram</a> &mdash; Voronoi remeshing</p>
    <div class="install">pip install pygeogram --find-links https://github.com/PozzettiAndrea/pygeogram/releases/latest/download/</div>
    <p class="links">
      <a href="https://github.com/PozzettiAndrea/pygeogram">GitHub</a> &middot;
      <a href="https://pypi.org/project/pygeogram/">PyPI</a>
    </p>
  </header>

  {demo_sections}

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
    demos = []

    # Example 1: Default parameters
    demos.append(run_example("default", verts, faces,
        {"nb_points": 2000},
        textwrap.dedent(f"""\
            import pygeogram
            import trimesh

            mesh = trimesh.load("{mesh_name}")
            v, f = pygeogram.remesh_smooth(
                mesh.vertices, mesh.faces,
                nb_points=2000,
            )""")))

    # Example 2: High quality — more Newton iterations
    demos.append(run_example("high_quality", verts, faces,
        {"nb_points": 3000, "nb_lloyd_iter": 10, "nb_newton_iter": 50},
        textwrap.dedent(f"""\
            v, f = pygeogram.remesh_smooth(
                mesh.vertices, mesh.faces,
                nb_points=3000,
                nb_lloyd_iter=10,
                nb_newton_iter=50,
            )""")))

    # Example 3: Fast low-poly — fewer iterations
    demos.append(run_example("low_poly", verts, faces,
        {"nb_points": 500, "nb_lloyd_iter": 3, "nb_newton_iter": 10},
        textwrap.dedent(f"""\
            v, f = pygeogram.remesh_smooth(
                mesh.vertices, mesh.faces,
                nb_points=500,
                nb_lloyd_iter=3,
                nb_newton_iter=10,
            )""")))

    generate_html(demos)

    # Preview image for README
    try:
        from PIL import Image
        imgs = []
        for d in demos:
            imgs.append(Image.open(os.path.join(OUT_DIR, f"{d['name']}_before.png")))
            imgs.append(Image.open(os.path.join(OUT_DIR, f"{d['name']}_after.png")))
        w, h = imgs[0].size
        grid = Image.new("RGB", (w * 2, h), "#0d1117")
        grid.paste(imgs[0], (0, 0))
        grid.paste(imgs[1], (w, 0))
        grid.save(os.path.join(OUT_DIR, "preview.png"))
    except Exception as e:
        print(f"Skipping preview: {e}")

    print(f"Demo: {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f} ({sz // 1024}KB)")


if __name__ == "__main__":
    main()
