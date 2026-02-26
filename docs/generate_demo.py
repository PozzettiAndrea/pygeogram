"""
Generate visual demo of pygeogram CVT remeshing for GitHub Pages.

Renders before/after comparison images using pyvista offscreen rendering.
Outputs: docs/_site/index.html + images
"""

import os
import shutil
import time
import numpy as np

import pyvista as pv

pv.OFF_SCREEN = True

import pygeogram


OUT_DIR = os.path.join(os.path.dirname(__file__), "_site")

# Dark theme colors
BG_COLOR = "#1a1a2e"
MESH_COLOR_IN = "#4fc3f7"
MESH_COLOR_OUT = "#81c784"
EDGE_COLOR = "#222244"
TEXT_COLOR = "#e0e0e0"


def make_icosphere(subdivisions=3):
    """Create an icosphere using pyvista."""
    sphere = pv.Icosphere(nsub=subdivisions, radius=1.0)
    verts = np.array(sphere.points, dtype=np.float64)
    faces_pv = sphere.faces.reshape(-1, 4)[:, 1:]
    faces = np.array(faces_pv, dtype=np.int32)
    return verts, faces, sphere


def pv_mesh_from_numpy(verts, faces):
    """Convert numpy arrays to pyvista PolyData."""
    n = len(faces)
    pv_faces = np.column_stack([np.full(n, 3, dtype=np.int32), faces]).ravel()
    return pv.PolyData(verts, pv_faces)


def render_mesh(mesh, filename, title, color=MESH_COLOR_IN,
                window_size=(800, 600)):
    """Render a mesh with edges to a PNG file."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.add_mesh(
        mesh,
        color=color,
        show_edges=True,
        edge_color=EDGE_COLOR,
        line_width=0.5,
        lighting=True,
        smooth_shading=True,
    )
    pl.add_text(title, position="upper_left", font_size=12, color=TEXT_COLOR)
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()


def run_demo(name, verts_in, faces_in, nb_points, nb_lloyd_iter=5,
             nb_newton_iter=30):
    """Run remeshing and generate comparison images."""
    # Time the remeshing
    t0 = time.perf_counter()
    verts_out, faces_out = pygeogram.remesh_smooth(
        verts_in, faces_in,
        nb_points=nb_points,
        nb_lloyd_iter=nb_lloyd_iter,
        nb_newton_iter=nb_newton_iter,
    )
    elapsed = time.perf_counter() - t0

    mesh_in = pv_mesh_from_numpy(verts_in, faces_in)
    mesh_out = pv_mesh_from_numpy(verts_out, faces_out)

    prefix = os.path.join(OUT_DIR, name)

    render_mesh(
        mesh_in,
        f"{prefix}_before.png",
        f"Input: {len(verts_in)} verts, {len(faces_in)} faces",
        color=MESH_COLOR_IN,
    )
    render_mesh(
        mesh_out,
        f"{prefix}_after.png",
        f"CVT: {len(verts_out)} verts, {len(faces_out)} faces  ({elapsed:.2f}s)",
        color=MESH_COLOR_OUT,
    )

    return {
        "name": name,
        "verts_in": len(verts_in),
        "faces_in": len(faces_in),
        "verts_out": len(verts_out),
        "faces_out": len(faces_out),
        "nb_points": nb_points,
        "elapsed": elapsed,
    }


def render_preview(demos):
    """Render a combined preview image for the README."""
    # Use the first demo's before/after side by side
    from PIL import Image
    images = []
    for d in demos[:2]:
        before = Image.open(os.path.join(OUT_DIR, f"{d['name']}_before.png"))
        after = Image.open(os.path.join(OUT_DIR, f"{d['name']}_after.png"))
        images.extend([before, after])

    # 2x2 grid
    w, h = images[0].size
    grid = Image.new("RGB", (w * 2, h * 2))
    for i, img in enumerate(images[:4]):
        grid.paste(img, ((i % 2) * w, (i // 2) * h))
    grid.save(os.path.join(OUT_DIR, "preview.png"))


def generate_html(demos):
    """Generate the index.html page."""
    demo_sections = ""
    for d in demos:
        demo_sections += f"""
    <section class="demo">
      <h2>{d['name'].replace('_', ' ').title()}</h2>
      <p>
        Input: <strong>{d['verts_in']}</strong> verts, <strong>{d['faces_in']}</strong> faces
        &rarr; target <strong>{d['nb_points']}</strong> points
        &rarr; Output: <strong>{d['verts_out']}</strong> verts, <strong>{d['faces_out']}</strong> faces
        &mdash; <span class="timing">{d['elapsed']:.2f}s</span>
      </p>
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
    </section>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>pygeogram — CVT Remeshing Demo</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0d1117;
      color: #c9d1d9;
      padding: 2rem;
      max-width: 1200px;
      margin: 0 auto;
    }}
    a {{ color: #58a6ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    header {{
      text-align: center;
      margin-bottom: 2rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid #21262d;
    }}
    header h1 {{ font-size: 2rem; margin-bottom: 0.5rem; color: #f0f6fc; }}
    header p {{ color: #8b949e; font-size: 1.1rem; }}
    .badge {{
      display: inline-block;
      background: #1b4332;
      color: #81c784;
      padding: 0.2rem 0.6rem;
      border-radius: 4px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-top: 0.5rem;
    }}
    .install-block {{
      background: #161b22;
      color: #aed581;
      padding: 1rem 1.5rem;
      border-radius: 6px;
      border: 1px solid #21262d;
      font-family: monospace;
      font-size: 0.95rem;
      margin: 1rem auto;
      max-width: 600px;
      text-align: left;
      white-space: pre;
    }}
    .demo {{
      background: #161b22;
      border: 1px solid #21262d;
      border-radius: 8px;
      padding: 1.5rem;
      margin-bottom: 2rem;
    }}
    .demo h2 {{ margin-bottom: 0.5rem; color: #f0f6fc; }}
    .demo p {{ color: #8b949e; margin-bottom: 1rem; }}
    .timing {{
      color: #81c784;
      font-weight: 600;
      font-family: monospace;
    }}
    .comparison {{
      display: flex;
      gap: 1rem;
      justify-content: center;
      flex-wrap: wrap;
    }}
    .panel {{
      position: relative;
      flex: 1;
      min-width: 300px;
      max-width: 560px;
    }}
    .panel img {{
      width: 100%;
      border-radius: 4px;
      border: 1px solid #21262d;
    }}
    .label {{
      position: absolute;
      bottom: 8px;
      right: 8px;
      background: rgba(0,0,0,0.7);
      color: #c9d1d9;
      padding: 0.2rem 0.5rem;
      border-radius: 3px;
      font-size: 0.8rem;
    }}
    pre.code {{
      background: #161b22;
      color: #c9d1d9;
      padding: 1rem;
      border-radius: 6px;
      border: 1px solid #21262d;
      overflow-x: auto;
      font-size: 0.9rem;
    }}
    pre.code .kw {{ color: #ff7b72; }}
    pre.code .fn {{ color: #d2a8ff; }}
    pre.code .str {{ color: #a5d6ff; }}
    pre.code .num {{ color: #79c0ff; }}
    pre.code .comment {{ color: #8b949e; }}
    footer {{
      text-align: center;
      color: #484f58;
      margin-top: 2rem;
      font-size: 0.85rem;
    }}
    footer a {{ color: #484f58; }}
  </style>
</head>
<body>
  <header>
    <h1>pygeogram</h1>
    <p>Python bindings for <a href="https://github.com/BrunoLevy/geogram">geogram</a> geometry processing</p>
    <span class="badge">CVT Remeshing</span>
    <div class="install-block">pip install pygeogram</div>
    <p style="margin-top:1rem;">
      <a href="https://github.com/PozzettiAndrea/pygeogram">GitHub</a> &middot;
      <a href="https://pypi.org/project/pygeogram/">PyPI</a>
    </p>
  </header>

  {demo_sections}

  <section class="demo">
    <h2>Quick Start</h2>
    <pre class="code"><code><span class="kw">import</span> pygeogram
<span class="kw">import</span> trimesh

mesh = trimesh.<span class="fn">load</span>(<span class="str">"model.stl"</span>)
v_out, f_out = pygeogram.<span class="fn">remesh_smooth</span>(
    mesh.vertices, mesh.faces,
    nb_points=<span class="num">5000</span>,
)
result = trimesh.<span class="fn">Trimesh</span>(vertices=v_out, faces=f_out)</code></pre>
  </section>

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

    demos = []

    # Demo 1: Icosphere (low poly → uniform remesh)
    verts, faces, _ = make_icosphere(subdivisions=2)
    demos.append(run_demo("icosphere", verts, faces, nb_points=200))

    # Demo 2: Icosphere (higher res → simplification)
    verts, faces, _ = make_icosphere(subdivisions=4)
    demos.append(run_demo("simplification", verts, faces, nb_points=300))

    # Demo 3: Noisy sphere (shows CVT cleanup)
    verts, faces, _ = make_icosphere(subdivisions=3)
    noise = np.random.RandomState(42).randn(*verts.shape) * 0.05
    verts_noisy = verts + noise
    demos.append(run_demo("noisy_sphere", verts_noisy, faces, nb_points=400))

    # Demo 4: Stanford bunny
    try:
        bunny = pv.examples.download_bunny()
        bunny_verts = np.array(bunny.points, dtype=np.float64)
        bunny_faces_pv = bunny.faces.reshape(-1, 4)[:, 1:]
        bunny_faces = np.array(bunny_faces_pv, dtype=np.int32)
        demos.append(run_demo("stanford_bunny", bunny_verts, bunny_faces,
                              nb_points=2000))
    except Exception as e:
        print(f"Skipping bunny demo: {e}")

    generate_html(demos)

    # Generate preview image for README
    try:
        render_preview(demos)
        print("  Generated preview.png")
    except Exception as e:
        print(f"  Skipping preview: {e}")

    print(f"Demo generated in {OUT_DIR}/")
    print(f"  {len(demos)} demos, files:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"    {f}")


if __name__ == "__main__":
    main()
