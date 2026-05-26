import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle
from Library.plotting_utils import load_qgt, filter_entries_by_omega
import sys
import Library.Hamiltonian.Hamiltonian
from Library.Hamiltonian.ChiralHamiltonian import ChiralHamiltonian
from Library.Hamiltonian.SquareLatticeHamiltonian import SquareLatticeHamiltonian
import Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected
from Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected import ChiralHamiltonianChiralBasisProjected
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json


sys.modules["Library.Hamiltonian_v2"] = Library.Hamiltonian.Hamiltonian
sys.modules["Library.Hamiltonian.Chiral_Hamiltonian_Projected"] = Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected

Library.Hamiltonian.Hamiltonian.ChiralHamiltonian = ChiralHamiltonian
Library.Hamiltonian.Hamiltonian.SquareLatticeHamiltonian = SquareLatticeHamiltonian
Library.Hamiltonian.Hamiltonian.RhombohedralGrapheneHamiltonian = ChiralHamiltonianChiralBasisProjected
Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected.ChiralHamiltonianProjected = ChiralHamiltonianChiralBasisProjected
Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected.Chiral_Hamiltonian_Projected = ChiralHamiltonianChiralBasisProjected
Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected.RhombohedralGrapheneHamiltonian = ChiralHamiltonianChiralBasisProjected

mpl.rcParams.update({
    "font.size": 8,        # base font size
    "axes.titlesize": 8,   # ax.set_title
    "axes.labelsize": 8,   # ax.set_xlabel/set_ylabel
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 8, # fig.suptitle
})


def dynamic_2d_qgt_1d_sweep(
        dataset_path,
        sweep_param,
        param_min=None, param_max=None,
        quantity="trace",
        cmap="inferno",
        output_html=None):
    """
    Interactive HTML plot (Plotly slider) of a 2-D QGT quantity as one
    parameter is swept, loading data from the N-D bundle produced by
    ``Calc_QGT_2D_nd_parameter_sweep.py``.

    The swept axis is identified by name from the bundle's ``names`` array
    (saved as ``axis_{i}_{name}``).  All other parameter axes (if any) are
    fixed at their middle index.

    Parameters
    ----------
    dataset_path : str
        Absolute path **or** relative to the project root
        (e.g. ``"results/2D_QGT_ND/SquareLatticeHamiltonian/dataset2"``).
    sweep_param : str
        Name of the parameter to sweep (must match a name in the bundle's
        ``names`` array, e.g. ``"omega"``, ``"V"``, ``"A0"``).
    param_min / param_max : float | None
        Optional range filter on the swept parameter (inclusive).
    quantity : str
        ``"trace"`` (default), ``"berry"``, or ``"imqxy"``.
    cmap : str
        Plotly colorscale name (e.g. ``"inferno"``, ``"RdBu_r"``).
    output_html : str | None
        Destination .html file.  Defaults to
        ``<dataset_path>/dynamic_2d_{quantity}_vs_{sweep_param}.html``.
    """


    # ---------- resolve path ----------
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.getcwd(), dataset_path)

    bundle_path = os.path.join(dataset_path, "qgt_nd_bundle.npz")
    json_path = os.path.join(dataset_path, "parameters.json")

    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"ND bundle not found: {bundle_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"parameters.json not found: {json_path}")

    # ---------- load bundle ----------
    bundle = np.load(bundle_path, allow_pickle=True)
    with open(json_path, "r", encoding="utf-8") as f:
        pjson = json.load(f)

    names = list(bundle["names"])           # list of swept parameter names
    shape = list(bundle["shape"])           # lengths along each param axis
    kx    = bundle["kx"]                   # (Ny, Nx)
    ky    = bundle["ky"]                   # (Ny, Nx)

    # ---------- find sweep axis ----------
    if sweep_param not in names:
        raise ValueError(
            f"Parameter '{sweep_param}' not found in bundle.\n"
            f"Available parameters: {names}"
        )
    sweep_axis = names.index(sweep_param)
    param_values = bundle[f"axis_{sweep_axis}_{sweep_param}"]  # 1-D array

    # ---------- pick QGT field grid ----------
    # trace_grid shape: (*shape, Ny, Nx)
    q = quantity.lower()
    if q == "trace":
        field_grid = bundle["trace_grid"]
        label_q    = "QGT Trace"
    elif q == "berry":
        field_grid = -2.0 * bundle["g_xy_imag_grid"]
        label_q    = "Berry Curvature \u03a9"
    elif q == "imqxy":
        field_grid = bundle["g_xy_imag_grid"]
        label_q    = "Im(Q_xy)"
    else:
        raise ValueError(f"Unknown quantity '{quantity}'. Use 'trace', 'berry', or 'imqxy'.")

    # ---------- fix all non-sweep axes at their midpoint ----------
    # Build a tuple of slices/indices: sweep axis → slice(None), others → midpoint
    n_params = len(names)
    mid_idx = tuple(
        slice(None) if ax == sweep_axis else shape[ax] // 2
        for ax in range(n_params)
    )
    # Result: field_grid[mid_idx] has shape (n_sweep, Ny, Nx)
    field_2d_series = field_grid[mid_idx]

    # ---------- range filter on sweep parameter ----------
    mask = np.ones(len(param_values), dtype=bool)
    if param_min is not None:
        mask &= param_values >= param_min
    if param_max is not None:
        mask &= param_values <= param_max

    param_values    = param_values[mask]
    field_2d_series = field_2d_series[mask]    # (n_filtered, Ny, Nx)

    if len(param_values) == 0:
        raise ValueError(
            f"No slices of '{sweep_param}' fall within "
            f"[{param_min}, {param_max}]."
        )

    # ---------- colour limits ----------
    finite_vals = field_2d_series[np.isfinite(field_2d_series)]
    vmin, vmax = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
    if q != "trace":                       # symmetric colour bar for signed quantities
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    # ---------- hamiltonian name ----------
    ham_name = pjson["hamiltonian_name"]

    # Fixed-param annotation for the plot title (all axes except sweep)
    fixed_labels = [
        f"{names[ax]}={bundle[f'axis_{ax}_{names[ax]}'][shape[ax] // 2]:.4g}"
        for ax in range(n_params)
        if ax != sweep_axis
    ]
    fixed_str = ", ".join(fixed_labels) if fixed_labels else ""

    # ---------- build Plotly figure ----------
    fig = go.Figure()

    for i, pval in enumerate(param_values):
        fig.add_trace(go.Heatmap(
            z=field_2d_series[i].tolist(),
            x=kx[0, :].tolist(),
            y=ky[:, 0].tolist(),
            colorscale=cmap,
            zmin=vmin, zmax=vmax,
            visible=(i == 0),
            name=f"{sweep_param}={pval:.4g}",
            colorbar=dict(title=label_q, thickness=18),
            hovertemplate="kx: %{x:.3f}<br>ky: %{y:.3f}<br>Value: %{z:.4g}<extra></extra>",
        ))

    steps = []
    for i, pval in enumerate(param_values):
        visible = [False] * len(param_values)
        visible[i] = True
        steps.append(dict(
            method="update",
            args=[{"visible": visible},
                  {"title": f"{ham_name} \u2014 {label_q}<br>"
                            f"{sweep_param} = {pval:.6g}"
                            + (f"  ({fixed_str})" if fixed_str else "")}],
            label=f"{pval:.3g}",
        ))

    fig.update_layout(
        title=(f"{ham_name} \u2014 {label_q}<br>"
               f"{sweep_param} = {param_values[0]:.6g}"
               + (f"  ({fixed_str})" if fixed_str else "")),
        xaxis_title="kx",
        yaxis_title="ky",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        sliders=[dict(
            active=0,
            currentvalue={"prefix": f"{sweep_param}: ", "font": {"size": 14}},
            pad={"t": 50},
            steps=steps,
        )],
        width=700, height=680,
        margin=dict(l=60, r=80, t=100, b=80),
    )

    # ---------- build parameters sidebar ----------
    def _fmt_val(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def _section(title, rows_html):
        return (
            f'<div class="section">'
            f'<div class="section-title">{title}</div>'
            f'{rows_html}'
            f'</div>'
        )

    def _row(key, val):
        return f'<div class="row"><span class="key">{key}</span><span class="val">{val}</span></div>'

    ham_rows = "".join(
        _row(k, _fmt_val(v))
        for k, v in sorted(pjson["parameters"].items())
    )

    scan_ranges = pjson["scan_ranges"]
    scan_spacing = pjson["scan_spacing"]
    scan_rows = "".join(
        _row(
            name,
            f'{_fmt_val(info["min"])} → {_fmt_val(info["max"])}'
            + f'  ({scan_spacing[name]["count"]} pts, {scan_spacing[name]["scale"]})'
        )
        for name, info in scan_ranges.items()
    )

    kg = pjson["k_grid"]
    kgrid_rows = (
        _row("kx", f'{_fmt_val(kg["kx_min"])} → {_fmt_val(kg["kx_max"])}') +
        _row("ky", f'{_fmt_val(kg["ky_min"])} → {_fmt_val(kg["ky_max"])}') +
        _row("mesh", str(kg["mesh"]))
    )

    band_row = _row("band index", str(pjson["band_index"]))

    sidebar_html = f"""
<div class="sidebar">
  <div class="sidebar-header">{pjson["hamiltonian_name"]}</div>
  {_section("Hamiltonian Parameters", ham_rows)}
  {_section("Sweep Range", scan_rows)}
  {_section("k-Grid", kgrid_rows + band_row)}
</div>"""

    # ---------- write HTML ----------
    if output_html is None:
        output_html = os.path.join(
            dataset_path, f"dynamic_2d_{q}_vs_{sweep_param}.html"
        )

    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    full_html = f"""
    <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8"/>
        <title>{ham_name} \u2014 {label_q} vs {sweep_param}</title>
        <style>
            *, *::before, *::after {{ box-sizing: border-box; }}
            body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #f0f2f7;
            margin: 0; padding: 24px;
            display: flex; justify-content: center; align-items: flex-start; gap: 24px;
            }}
            .plot-wrapper {{ flex: 0 0 auto; }}
            .sidebar {{
            flex: 0 0 260px;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            padding: 0 0 16px 0;
            overflow: hidden;
            align-self: flex-start;
            margin-top: 8px;
            }}
            .sidebar-header {{
            background: linear-gradient(135deg, #3b4fa8 0%, #5b6fd6 100%);
            color: #fff;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.03em;
            padding: 14px 16px;
            word-break: break-word;
            }}
            .section {{ padding: 10px 16px 4px 16px; }}
            .section-title {{
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #8892b0;
            margin-bottom: 6px;
            border-bottom: 1px solid #e8ecf4;
            padding-bottom: 4px;
            }}
            .row {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 3px 0;
            font-size: 12.5px;
            border-bottom: 1px solid #f3f5fb;
            }}
            .row:last-child {{ border-bottom: none; }}
            .key {{
            color: #4a5568;
            font-weight: 600;
            padding-right: 8px;
            white-space: nowrap;
            }}
            .val {{
            color: #2d3748;
            text-align: right;
            font-variant-numeric: tabular-nums;
            word-break: break-all;
            }}
        </style>
        </head>
        <body>
        <div class="plot-wrapper">{plot_html}</div>
        {sidebar_html}
        </body>
    </html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Saved interactive HTML to: {output_html}")
    return output_html
