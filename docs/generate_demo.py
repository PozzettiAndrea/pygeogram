"""
Generate visual demo of pygeogram CVT remeshing for GitHub Pages.

Renders before/after comparison images using pyvista offscreen rendering.
Outputs: docs/_site/index.html + images
"""

import os
import shutil
import numpy as np

import pyvista as pv

pv.OFF_SCREEN = True

import pygeogram


OUT_DIR = os.path.join(os.path.dirname(__file__), "_site")


def make_icosphere(subdivisions=3):
    """Create an icosphere using pyvista."""
    sphere = pv.Icosphere(nsub=subdivisions, radius=1.0)
    verts = np.array(sphere.points, dtype=np.float64)
    faces_pv = sphere.faces.reshape(-1, 4)[:, 1:]  # strip leading 3
    faces = np.array(faces_pv, dtype=np.int32)
    return verts, faces, sphere


def pv_mesh_from_numpy(verts, faces):
    """Convert numpy arrays to pyvista PolyData."""
    n = len(faces)
    pv_faces = np.column_stack([np.full(n, 3, dtype=np.int32), faces]).ravel()
    return pv.PolyData(verts, pv_faces)


def render_mesh(mesh, filename, title, color="#4fc3f7", edge_color="black",
                show_edges=True, window_size=(800, 600)):
    """Render a mesh to a PNG file."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.add_mesh(
        mesh,
        color=color,
        show_edges=show_edges,
        edge_color=edge_color,
        line_width=0.5,
        lighting=True,
        smooth_shading=True,
    )
    pl.add_text(title, position="upper_left", font_size=14, color="black")
    pl.set_background("white")
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()


def render_wireframe(mesh, filename, title, color="#4fc3f7",
                     window_size=(800, 600)):
    """Render a wireframe view to a PNG file."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.add_mesh(
        mesh,
        style="wireframe",
        color=color,
        line_width=1.0,
    )
    pl.add_text(title, position="upper_left", font_size=14, color="black")
    pl.set_background("white")
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()


def run_demo(name, verts_in, faces_in, nb_points, nb_lloyd_iter=5,
             nb_newton_iter=30):
    """Run remeshing and generate comparison images."""
    verts_out, faces_out = pygeogram.remesh_smooth(
        verts_in, faces_in,
        nb_points=nb_points,
        nb_lloyd_iter=nb_lloyd_iter,
        nb_newton_iter=nb_newton_iter,
    )

    mesh_in = pv_mesh_from_numpy(verts_in, faces_in)
    mesh_out = pv_mesh_from_numpy(verts_out, faces_out)

    prefix = os.path.join(OUT_DIR, name)

    render_mesh(
        mesh_in,
        f"{prefix}_before.png",
        f"Input: {len(verts_in)} verts, {len(faces_in)} faces",
    )
    render_mesh(
        mesh_out,
        f"{prefix}_after.png",
        f"CVT remesh: {len(verts_out)} verts, {len(faces_out)} faces",
        color="#81c784",
    )
    render_wireframe(
        mesh_in,
        f"{prefix}_wire_before.png",
        f"Input wireframe",
        color="#4fc3f7",
    )
    render_wireframe(
        mesh_out,
        f"{prefix}_wire_after.png",
        f"CVT wireframe",
        color="#81c784",
    )

    return {
        "name": name,
        "verts_in": len(verts_in),
        "faces_in": len(faces_in),
        "verts_out": len(verts_out),
        "faces_out": len(faces_out),
        "nb_points": nb_points,
    }


def generate_html(demos):
    """Generate the index.html page."""
    demo_sections = ""
    for d in demos:
        demo_sections += f"""
    <section class="demo">
      <h2>{d['name'].replace('_', ' ').title()}</h2>
      <p>
        Input: <strong>{d['verts_in']}</strong> vertices, <strong>{d['faces_in']}</strong> faces
        &rarr; CVT remesh to <strong>{d['nb_points']}</strong> target points
        &rarr; Output: <strong>{d['verts_out']}</strong> vertices, <strong>{d['faces_out']}</strong> faces
      </p>

      <h3>Shaded</h3>
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

      <h3>Wireframe</h3>
      <div class="comparison">
        <div class="panel">
          <img src="{d['name']}_wire_before.png" alt="Before wireframe">
          <span class="label">Input</span>
        </div>
        <div class="panel">
          <img src="{d['name']}_wire_after.png" alt="After wireframe">
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
      background: #f5f5f5;
      color: #333;
      padding: 2rem;
      max-width: 1200px;
      margin: 0 auto;
    }}
    header {{
      text-align: center;
      margin-bottom: 2rem;
      padding-bottom: 1.5rem;
      border-bottom: 2px solid #e0e0e0;
    }}
    header h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
    header p {{ color: #666; font-size: 1.1rem; }}
    header a {{ color: #1976d2; text-decoration: none; }}
    header a:hover {{ text-decoration: underline; }}
    .badge {{
      display: inline-block;
      background: #e8f5e9;
      color: #2e7d32;
      padding: 0.2rem 0.6rem;
      border-radius: 4px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-top: 0.5rem;
    }}

    /* Platform tabs */
    .platform-tabs {{
      display: flex;
      justify-content: center;
      gap: 0;
      margin: 1.5rem 0;
    }}
    .platform-tabs button {{
      padding: 0.6rem 1.5rem;
      border: 2px solid #1976d2;
      background: white;
      color: #1976d2;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .platform-tabs button:first-child {{ border-radius: 6px 0 0 6px; }}
    .platform-tabs button:last-child {{ border-radius: 0 6px 6px 0; }}
    .platform-tabs button:not(:last-child) {{ border-right: none; }}
    .platform-tabs button.active {{
      background: #1976d2;
      color: white;
    }}
    .platform-tabs button:hover:not(.active) {{
      background: #e3f2fd;
    }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}
    .install-block {{
      background: #263238;
      color: #aed581;
      padding: 1rem 1.5rem;
      border-radius: 6px;
      font-family: monospace;
      font-size: 0.95rem;
      margin: 0.5rem auto;
      max-width: 600px;
      text-align: left;
      white-space: pre;
    }}
    .install-block .comment {{ color: #78909c; }}

    .demo {{
      background: white;
      border-radius: 8px;
      padding: 1.5rem;
      margin-bottom: 2rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .demo h2 {{ margin-bottom: 0.5rem; }}
    .demo h3 {{ margin: 1rem 0 0.5rem; color: #555; font-size: 1rem; }}
    .demo p {{ color: #666; margin-bottom: 1rem; }}
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
      border: 1px solid #e0e0e0;
    }}
    .label {{
      position: absolute;
      bottom: 8px;
      right: 8px;
      background: rgba(0,0,0,0.6);
      color: white;
      padding: 0.2rem 0.5rem;
      border-radius: 3px;
      font-size: 0.8rem;
    }}
    pre.code {{
      background: #263238;
      color: #eee;
      padding: 1rem;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 0.9rem;
    }}
    footer {{
      text-align: center;
      color: #999;
      margin-top: 2rem;
      font-size: 0.85rem;
    }}
    footer a {{ color: #999; }}
  </style>
</head>
<body>
  <header>
    <h1>pygeogram</h1>
    <p>Python bindings for <a href="https://github.com/BrunoLevy/geogram">geogram</a> geometry processing</p>
    <span class="badge">CVT Remeshing</span>

    <div class="platform-tabs">
      <button class="active" onclick="showTab('linux')">Linux</button>
      <button onclick="showTab('macos')">macOS</button>
      <button onclick="showTab('windows')">Windows</button>
    </div>

    <div id="tab-linux" class="tab-content active">
      <div class="install-block">pip install pygeogram</div>
      <p style="color:#888; font-size:0.85rem; margin-top:0.4rem;">
        manylinux x86_64 &amp; aarch64 &middot; Python 3.10&ndash;3.13
      </p>
    </div>
    <div id="tab-macos" class="tab-content">
      <div class="install-block">pip install pygeogram</div>
      <p style="color:#888; font-size:0.85rem; margin-top:0.4rem;">
        macOS 11+ &middot; x86_64 &amp; Apple Silicon (arm64) &middot; Python 3.10&ndash;3.13
      </p>
    </div>
    <div id="tab-windows" class="tab-content">
      <div class="install-block">pip install pygeogram</div>
      <p style="color:#888; font-size:0.85rem; margin-top:0.4rem;">
        Windows AMD64 &middot; Python 3.10&ndash;3.13
      </p>
    </div>

    <p style="margin-top:1rem;">
      <a href="https://github.com/PozzettiAndrea/pygeogram">GitHub</a> &middot;
      <a href="https://pypi.org/project/pygeogram/">PyPI</a>
    </p>
  </header>

  {demo_sections}

  <section class="demo">
    <h2>Quick Start</h2>
    <pre class="code"><code>import pygeogram
import trimesh

mesh = trimesh.load("model.stl")
v_out, f_out = pygeogram.remesh_smooth(
    mesh.vertices, mesh.faces,
    nb_points=5000,
)
result = trimesh.Trimesh(vertices=v_out, faces=f_out)</code></pre>
  </section>

  <footer>
    Generated automatically by CI &middot;
    <a href="https://github.com/BrunoLevy/geogram">geogram</a> by Bruno Levy (INRIA)
  </footer>

  <script>
    function showTab(platform) {{
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.platform-tabs button').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + platform).classList.add('active');
      document.querySelector('.platform-tabs button[onclick*="' + platform + '"]').classList.add('active');
    }}
  </script>
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

    # Demo 2: Icosphere (higher res → fewer points = simplification)
    verts, faces, _ = make_icosphere(subdivisions=4)
    demos.append(run_demo("simplification", verts, faces, nb_points=300))

    # Demo 3: Noisy sphere (shows CVT's smoothing quality)
    verts, faces, _ = make_icosphere(subdivisions=3)
    noise = np.random.RandomState(42).randn(*verts.shape) * 0.05
    verts_noisy = verts + noise
    demos.append(run_demo("noisy_sphere", verts_noisy, faces, nb_points=400))

    # Demo 4: Stanford bunny if available
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
    print(f"Demo generated in {OUT_DIR}/")
    print(f"  {len(demos)} demos, files:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"    {f}")


if __name__ == "__main__":
    main()
