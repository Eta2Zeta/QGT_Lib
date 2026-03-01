#!/usr/bin/env python3
"""
Load multiple PLY isosurface meshes and render them with a discrete RdBu_r color scheme
where colors are assigned by ORDER (index), not by the actual level values.

Usage examples:
  python view_isosurfaces_ply.py ./meshes/*.ply
  python view_isosurfaces_ply.py --alpha 0.35 --scale 1.0 ./meshes/*.ply
  python view_isosurfaces_ply.py --sort natural ./meshes/*.ply
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from pathlib import Path
import re

# ✅ EDIT THIS
MESH_DIR = Path("/Users/home/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/3D_QGT_results/RuO2Hamiltonian/data_set_6")   # or Path("relative/path")
GLOB_PAT = "*.ply"                                     # or "band*_level*.ply"

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]




# Fast 3D rendering
try:
    import pyvista as pv
except ImportError as e:
    raise SystemExit(
        "This script requires pyvista. Install with:\n"
        "  pip install pyvista\n"
        "If you also want better windowing on some platforms:\n"
        "  pip install pyvistaqt\n"
    ) from e

# For RdBu_r colors
try:
    import matplotlib.cm as cm
except ImportError as e:
    raise SystemExit(
        "This script requires matplotlib (only for colormap). Install with:\n"
        "  pip install matplotlib\n"
    ) from e


def natural_key(s: str):
    """Sort strings with embedded numbers naturally."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_mesh(path: Path) -> pv.PolyData:
    """Load a PLY mesh as PyVista PolyData."""
    mesh = pv.read(str(path))
    # Ensure PolyData (triangles); pyvista usually gives PolyData already.
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface().triangulate()
    else:
        mesh = mesh.triangulate()
    return mesh


def rd_bu_colors(n: int):
    """
    Return n RGBA colors evenly spaced from RdBu_r.
    IMPORTANT: colors are by index order, not by any value.
    """
    base = cm.get_cmap("RdBu_r")
    cols = base(np.linspace(0.0, 1.0, n))
    # pyvista likes RGB in [0,1]; alpha handled separately
    return cols[:, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.35, help="Mesh opacity (0..1).")
    ap.add_argument("--scale", type=float, default=1.0, help="Uniform scale applied to all meshes.")
    ap.add_argument(
        "--sort",
        choices=["none", "name", "natural"],
        default="natural",
        help="How to order meshes before assigning colors (color follows this order).",
    )
    ap.add_argument("--show_axes", action="store_true", help="Show axes widget.")
    ap.add_argument("--wireframe", action="store_true", help="Overlay wireframe for debugging.")
    ap.add_argument("--background", default="white", help="Background color (e.g. 'white' or '#111111').")
    ap.add_argument("--specular", type=float, default=0.2, help="Specular strength (0..1).")
    ap.add_argument("--smooth_shading", action="store_true", help="Enable smooth shading.")
    args = ap.parse_args()

    # Build list of files
    paths = sorted(MESH_DIR.glob(GLOB_PAT), key=lambda p: natural_key(p.name))

    if not paths:
        raise FileNotFoundError(f"No PLY files found in {MESH_DIR} matching {GLOB_PAT}")

    print("Found meshes:")
    for p in paths:
        print(" ", p)


    # order matters because colors are assigned by index
    if args.sort == "name":
        paths.sort(key=lambda x: x.name.lower())
    elif args.sort == "natural":
        paths.sort(key=lambda x: natural_key(x.name))
    # else: keep input order

    n = len(paths)
    colors = rd_bu_colors(n)

    plotter = pv.Plotter()
    plotter.set_background(args.background)

    print(f"Loading {n} meshes...")
    for i, p in enumerate(paths):
        mesh = load_mesh(p)

        if args.scale != 1.0:
            mesh = mesh.copy(deep=True)
            mesh.points *= float(args.scale)

        color = tuple(colors[i].tolist())

        # Add mesh
        plotter.add_mesh(
            mesh,
            color=color,
            opacity=float(args.alpha),
            specular=float(args.specular),
            smooth_shading=bool(args.smooth_shading),
            name=p.name,
        )

        # Optional wireframe overlay
        if args.wireframe:
            plotter.add_mesh(
                mesh,
                color="black",
                style="wireframe",
                opacity=min(1.0, float(args.alpha) + 0.25),
                line_width=1.0,
                name=p.name + "_wire",
            )

        print(f"  [{i+1}/{n}] {p.name}  -> color_index={i}")

    if args.show_axes:
        plotter.show_axes()

    # A small legend mapping "index -> filename"
    legend_entries = [(f"{i}: {paths[i].name}", tuple(colors[i].tolist())) for i in range(n)]
    plotter.add_legend(legend_entries, bcolor=None)

    plotter.show(title="PLY Isosurfaces (colors by order index)")


if __name__ == "__main__":
    main()