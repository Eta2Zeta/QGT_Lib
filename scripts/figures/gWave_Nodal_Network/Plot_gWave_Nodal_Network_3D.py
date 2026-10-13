"""Trace the exact SG194 middle-band nodal network and export offline Plotly HTML.

Run from anywhere after placing this directory in results/Plot_Specific/:
    python Plot_gWave_Nodal_Network_3D.py
    python Plot_gWave_Nodal_Network_3D.py --t4 0.5 --grid 601

Uses the repository Hamiltonian, not stored eigenvalue grids. Assumes static
J=(0,0,Jz), Jz != 0 and lamb_z != 0. The lamb_z=0 limit has nodal SURFACES
and is deliberately rejected by this curve tracer. Nonzero t3 gives no exact
middle-band crossings, and produces an empty, explicitly labelled plot.

Horizontal contours and each lattice vertical surface are traced separately.
Both roots of the height equation are retained; all sampled vertices are
checked by direct diagonalization. A finite contour grid cannot guarantee
discovery of arbitrarily small components: increase --grid to check geometry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import contourpy
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import root


REPO_ROOT = Path(os.environ.get("QGT_LIB_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO_ROOT))
from Library.Hamiltonian.gWaveAltermagnetHamiltonian import gWaveAltermagnetHamiltonian
from Library.data_paths import legacy_dir


# Edit these defaults or override them on the command line.
PARAMETERS = dict(t1=0.3, t2=0.3, t3=0.0, t4=0.3, mu=0.0,
                  Jx=0.0, Jy=0.0, Jz=0.2, lamb=0.1, lamb_z=0.5)
SQRT3 = np.sqrt(3.0)
XMAX, YMAX = 4 * np.pi / 3, 2 * np.pi / SQRT3
VERTICAL_LABELS = ("fₓ = 0", "fₓ − √3 fᵧ = 0", "fₓ + √3 fᵧ = 0")
COLORS = ("#d76823", "#8350ae", "#008977")


@dataclass
class Curve:
    family: str
    xyz: np.ndarray


def form_factors(xy):
    """Exact lattice f_x, f_y and their Cartesian gradients."""
    xy = np.asarray(xy, dtype=float)
    x, y = xy[..., 0], xy[..., 1]
    sx, cx = np.sin(x / 2), np.cos(x / 2)
    sy, cy = np.sin(SQRT3 * y / 2), np.cos(SQRT3 * y / 2)
    fx = np.sin(x) + sx * cy
    fy = SQRT3 * cx * sy
    mixed = -SQRT3 / 2 * sx * sy
    gx = np.stack((np.cos(x) + cx * cy / 2, mixed), axis=-1)
    gy = np.stack((mixed, 1.5 * cx * cy), axis=-1)
    return fx, fy, gx, gy


def level_and_gradient(xy, family, target):
    fx, fy, gx, gy = form_factors(xy)
    if family == "horizontal":
        return fx * fx + fy * fy - target, 2 * (fx[..., None] * gx + fy[..., None] * gy)
    factor = (0.0, -SQRT3, SQRT3)[int(family)]
    return fx + factor * fy, gx + factor * gy


def project_level(xy, family, target):
    """Newton correction normal to a 2D contour, using exact derivatives."""
    xy = np.array(xy, dtype=float, copy=True)
    for _ in range(15):
        value, grad = level_and_gradient(xy, family, target)
        denom = np.sum(grad * grad, axis=-1)
        step = np.divide(value, denom, out=np.zeros_like(value), where=denom > 1e-24)
        xy -= step[..., None] * grad
        if np.max(np.abs(value), initial=0) < 2e-14:
            break
    return xy


def in_bz(xy):
    x, y = np.asarray(xy)[..., 0], np.asarray(xy)[..., 1]
    return (np.abs(y) <= YMAX + 1e-11) & (np.abs(x) + np.abs(y) / SQRT3 <= XMAX + 1e-11)


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return zip(edges[::2], edges[1::2])


def height_roots(xy, params):
    """Both q=cos²(kz/2) roots; NaN marks absent/out-of-range roots."""
    fx, fy, _, _ = form_factors(xy)
    G = fy * (fy * fy - 3 * fx * fx)
    A = params["t4"] ** 2 * G ** 2
    B = params["lamb"] ** 2 * (fx * fx + fy * fy) ** 2
    C = 4 * A + B
    j2 = params["Jz"] ** 2
    discriminant = C * C - 16 * A * j2
    valid_discriminant = discriminant >= 0
    sd = np.sqrt(np.maximum(discriminant, 0))
    low = np.divide(2 * j2, C + sd, out=np.full_like(C, np.nan), where=C + sd > 0)
    high = np.divide(C + sd, 8 * A, out=np.full_like(C, np.nan), where=A > 0)
    values = np.stack((low, high), axis=-1)
    valid = valid_discriminant[..., None] & (values >= 0) & (values <= 1)
    return np.where(valid, values, np.nan)


def trace_network(params=None, grid=501):
    """Return (curves, horizontal/vertical junctions) in one first-BZ prism."""
    p = dict(PARAMETERS if params is None else params)
    if p["Jz"] == 0 or p.get("Jx", 0) != 0 or p.get("Jy", 0) != 0:
        raise ValueError("This tracer requires J=(0,0,Jz) with nonzero Jz.")
    if grid < 51:
        raise ValueError("Use at least 51 contour-grid points per direction.")
    if p["t3"] != 0:
        return [], np.empty((0, 3))
    if p["lamb_z"] == 0:
        raise ValueError("lamb_z=0 gives a nodal surface; this script traces curves for nonzero lamb_z.")

    target = abs(p["Jz"] / p["lamb"]) if p["lamb"] else np.inf
    xs = np.linspace(-XMAX, XMAX, grid)
    ys = np.linspace(-YMAX, YMAX, grid)
    xx, yy = np.meshgrid(xs, ys)
    mesh = np.stack((xx, yy), axis=-1)
    curves, planar_paths = [], []
    spacing = 2 * XMAX / (grid - 1) / 2

    for family in ("horizontal", "0", "1", "2"):
        if family == "horizontal" and not np.isfinite(target):
            continue
        field, _ = level_and_gradient(mesh, family, target)
        paths = contourpy.contour_generator(x=xs, y=ys, z=field).lines(0)
        for path in paths:
            length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            if length[-1] < spacing:
                continue
            s = np.linspace(0, length[-1], max(3, int(length[-1] / spacing) + 1))
            xy = np.column_stack([np.interp(s, length, path[:, dim]) for dim in (0, 1)])
            xy = project_level(xy, family, target)
            if family == "horizontal":
                planar_paths.append(xy)
                for start, stop in runs(in_bz(xy)):
                    if stop - start > 1:
                        curves.append(Curve(family, np.column_stack((xy[start:stop], np.zeros(stop-start)))))
                continue

            roots = height_roots(xy, p)
            for branch in (0, 1):
                valid = np.isfinite(roots[:, branch]) & in_bz(xy)
                for start, stop in runs(valid):
                    if stop - start < 2:
                        continue
                    points = [*xy[start:stop]]
                    # Locate ends where a height root enters [0,1] or where
                    # the curve reaches the BZ face. Avoid false connecting lines.
                    for at_start, good, bad in ((True, start, start-1), (False, stop-1, stop)):
                        if bad < 0 or bad >= len(xy):
                            continue
                        a, b = xy[good].copy(), xy[bad].copy()
                        for _ in range(40):
                            mid = project_level((a+b)/2, family, target)
                            acceptable = in_bz(mid) and np.isfinite(height_roots(mid, p)[branch])
                            if acceptable:
                                a = mid
                            else:
                                b = mid
                        if at_start:
                            points.insert(0, a)
                        else:
                            points.append(a)
                    points = np.asarray(points)
                    q = height_roots(points, p)[:, branch]
                    z = 2 * np.arccos(np.sqrt(q))
                    for sign in (-1, 1):
                        curves.append(Curve(family, np.column_stack((points, sign*z))))

    junctions = []
    for path in planar_paths:
        for family in ("0", "1", "2"):
            values, _ = level_and_gradient(path, family, target)
            candidates = np.flatnonzero(values[:-1] * values[1:] <= 0)
            for i in candidates:
                def equations(xy):
                    return [level_and_gradient(xy, "horizontal", target)[0],
                            level_and_gradient(xy, family, target)[0]]
                solution = root(equations, (path[i] + path[i+1])/2, tol=1e-11)
                point = solution.x
                if np.linalg.norm(equations(point)) < 1e-10 and in_bz(point):
                    if not any(np.linalg.norm(point-other) < 1e-6 for other in junctions):
                        junctions.append(point)
    junctions = np.column_stack((junctions, np.zeros(len(junctions)))) if junctions else np.empty((0,3))
    return curves, junctions


def validate_points(xyz, params):
    """Use the actual four-band model; avoid subtractive analytic radicands."""
    H = gWaveAltermagnetHamiltonian(**params)
    x, y, z = np.asarray(xyz).T
    matrices = H.compute_static_vectorized(x, y, z)
    energies = np.linalg.eigvalsh(matrices)
    eps, tx, tz, lx, ly, lz = H._model_terms(x, y, z)
    gap = energies[:, 2] - energies[:, 1]
    stable_gap = 2*np.sqrt(tx*tx + lz*lz + (np.sqrt(tz*tz+lx*lx+ly*ly)-abs(params["Jz"]))**2)
    return np.column_stack((eps, gap, stable_gap))


def bz_wireframe():
    angles = np.arange(7)*np.pi/3
    vertices = np.column_stack((XMAX*np.cos(angles), XMAX*np.sin(angles)))
    pieces = [np.column_stack((vertices, np.full(7,z))) for z in (-np.pi, np.pi)]
    pieces += [np.array([[x,y,-np.pi],[x,y,np.pi]]) for x,y in vertices[:-1]]
    return np.concatenate([np.vstack((piece, [np.nan]*3)) for piece in pieces])


def make_figure(curves, junctions, params, validation):
    fig = go.Figure()
    for family, name, color in [("horizontal", "kz = 0 crossings", "#2463bb")] + [
            (str(i), "Vertical: " + VERTICAL_LABELS[i], COLORS[i]) for i in range(3)]:
        xyz_parts, data_parts = [], []
        for curve in curves:
            if curve.family == family:
                xyz_parts.append(np.vstack((curve.xyz, [np.nan]*3)))
                data_parts.append(np.vstack((validation[id(curve)], [np.nan]*3)))
        if not xyz_parts:
            continue
        xyz, data = np.concatenate(xyz_parts), np.concatenate(data_parts)
        fig.add_trace(go.Scatter3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], customdata=data,
            mode="lines", name=name, line=dict(color=color, width=5), connectgaps=False,
            hovertemplate="kx=%{x:.6f}<br>ky=%{y:.6f}<br>kz=%{z:.6f}"
                          "<br>Crossing energy=%{customdata[0]:.6f}"
                          "<br>4-band gap=%{customdata[1]:.2e}<extra>%{fullData.name}</extra>"))
    if len(junctions):
        data = validate_points(junctions, params)
        fig.add_trace(go.Scatter3d(x=junctions[:,0],y=junctions[:,1],z=junctions[:,2],
            mode="markers", name="Planar / vertical junctions", customdata=data,
            marker=dict(size=5,color="#192d43",symbol="diamond"),
            hovertemplate="Junction<br>kx=%{x:.7f}<br>ky=%{y:.7f}<br>kz=%{z:.7f}"
                          "<br>Energy=%{customdata[0]:.6f}<br>Gap=%{customdata[1]:.2e}<extra></extra>"))
    edges = bz_wireframe()
    fig.add_trace(go.Scatter3d(x=edges[:,0],y=edges[:,1],z=edges[:,2],mode="lines",
        name="First Brillouin zone",line=dict(color="#9aa8b7",width=2),hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode="markers+text",text=["Γ"],
        textposition="top center", marker=dict(size=3,color="#657487"), showlegend=False, hoverinfo="skip"))
    axis = dict(showbackground=True,backgroundcolor="#f6f8fb",gridcolor="#dfe5ed",zerolinecolor="#b8c3cf")
    full = {"scene.xaxis.range":[-XMAX,XMAX],"scene.yaxis.range":[-YMAX,YMAX],"scene.zaxis.range":[-np.pi,np.pi]}
    close = {"scene.xaxis.range":[-1.55,1.55],"scene.yaxis.range":[-1.55,1.55],"scene.zaxis.range":[-np.pi,np.pi]}
    fig.update_layout(template="plotly_white", height=800, margin=dict(l=0,r=0,b=0,t=70),
        title=dict(text="SG194 · exact middle-band crossings",x=0.02),
        legend=dict(x=0.01,y=0.98,bgcolor="rgba(255,255,255,0.85)",groupclick="togglegroup"),
        scene=dict(xaxis=dict(title="kx",range=[-XMAX,XMAX],**axis),
                   yaxis=dict(title="ky",range=[-YMAX,YMAX],**axis),
                   zaxis=dict(title="kz",range=[-np.pi,np.pi],**axis),aspectmode="data",
                   camera=dict(eye=dict(x=1.6,y=1.7,z=1.15))),
        updatemenus=[dict(type="buttons",direction="right",x=.48,y=1.08,showactive=False,
                         buttons=[dict(label="Full BZ",method="relayout",args=[full]),
                                  dict(label="Γ region",method="relayout",args=[close])])])
    if not curves:
        fig.add_annotation(text="No exact middle-band crossings for nonzero t₃ and Jz",showarrow=False,
                           x=.5,y=.1,xref="paper",yref="paper",font=dict(size=17,color="#9a3412"))
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ("t1","t2","t3","t4","mu","Jz","lamb","lamb_z"):
        parser.add_argument("--"+key.replace("_","-"),type=float,default=PARAMETERS[key],dest=key)
    parser.add_argument("--grid",type=int,default=501)
    parser.add_argument("--output",type=Path,default=Path(legacy_dir("Plot_Specific","gWave_Nodal_Network","gWave_nodal_network_3d.html")))
    args = parser.parse_args(argv)
    params = dict(PARAMETERS, **{k:getattr(args,k) for k in PARAMETERS if hasattr(args,k)})
    if not all(np.isfinite(v) for v in params.values()):
        parser.error("Model parameters must be finite.")
    curves, junctions = trace_network(params, args.grid)
    validation = {id(c):validate_points(c.xyz,params) for c in curves}
    all_gaps = np.concatenate([v[:,1:] for v in validation.values()]) if validation else np.zeros((0,2))
    max_gap = float(np.max(all_gaps,initial=0))
    if max_gap > 1e-8:
        raise RuntimeError(f"Nodal validation failed: largest gap {max_gap:.3g}. Increase --grid.")
    fig = make_figure(curves,junctions,params,validation)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    parameter_text = ", ".join(f"{k}={params[k]:g}" for k in ("Jz","t3","t4","lamb","lamb_z"))
    note = (f"<p><b>{parameter_text}</b> · {sum(len(c.xyz) for c in curves):,} checked curve vertices · "
            f"maximum gap {max_gap:.2e}</p><p>Drag to rotate; scroll to zoom; click legend entries to hide branches. "
            "Hover for momentum, crossing energy and direct four-band gap. Coordinates use the model's lattice convention.</p>"
            "<p>These are crossings of the middle two bands E₂=E₃, not the lower/upper doublet degeneracies. "
            "The straight segments interpolate numerically sampled curves. Boundary pieces belong to the same periodic BZ. "
            "Crossing energy varies along the network; it is not fixed to the Fermi energy.</p>")
    html = fig.to_html(full_html=True, include_plotlyjs=True,config={"responsive":True,"displaylogo":False},
                       div_id="nodal-network")
    html = html.replace("<head>", "<head><title>SG194 nodal network</title>"
                        "<meta name='viewport' content='width=device-width, initial-scale=1'>")
    html = html.replace("<body>","<body style='margin:0;font-family:Arial,sans-serif;color:#25354a'>"
                        "<div style='padding:12px 24px;max-width:1200px;font-size:14px'>"+note+"</div>")
    args.output.write_text(html)
    report = dict(parameters=params,grid=args.grid,curve_pieces=len(curves),
                  vertices=sum(len(c.xyz) for c in curves),max_middle_gap=max_gap,
                  junctions=junctions.tolist(),html=args.output.name,
                  assumptions="Static J along z; nonzero Jz and lamb_z; first hexagonal BZ; E2=E3 only")
    args.output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    arrays = {f"curve_{i:03d}":c.xyz for i,c in enumerate(curves)}
    arrays["families"] = np.array([c.family for c in curves])
    arrays["junctions"] = junctions
    np.savez_compressed(args.output.with_suffix(".npz"),**arrays)
    print(json.dumps({k:v for k,v in report.items() if k not in ("junctions","parameters")},indent=2))
    print(f"HTML: {args.output}")


if __name__ == "__main__":
    main()
