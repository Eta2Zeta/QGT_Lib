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


def dynamic_2d_qgt_1para_sweep_joined(
    left_dataset_path,
    right_dataset_path,
    sweep_param,
    *,
    quantity="trace",
    param_min_left=None,
    param_max_left=None,
    param_min_right=None,
    param_max_right=None,
    symmetric_cbar=None,   # None → True for berry/imqxy, False for trace
    drop_overlap=True,
    tol=1e-9,
    cmap="inferno",
    label_left="left",
    label_right="right",
    output_html=None,
):
    """
    HTML version of a joined 1-parameter N-D QGT sweep, generalised to any
    sweep parameter and reading from two N-D bundles.

    The left dataset is sorted ascending in ``sweep_param``; the right dataset
    is sorted *descending* (reversed) and appended — producing a continuous
    slider that sweeps up from the left range then back down through the right
    range.  This is useful for comparing e.g. left vs right circular
    polarisation where each covers a different frequency window.

    Parameters
    ----------
    left_dataset_path / right_dataset_path : str
        Paths to the two ND bundle folders (absolute or relative to CWD).
    sweep_param : str
        Parameter name swept in both bundles (e.g. ``"omega"``).
    quantity : str
        ``"trace"``, ``"berry"``, or ``"imqxy"``.
    param_min_left / param_max_left : float | None
        Range filter applied independently to the left dataset.
    param_min_right / param_max_right : float | None
        Range filter applied independently to the right dataset.
    symmetric_cbar : bool | None
        Force a symmetric colour bar.  ``None`` → True for berry/imqxy.
    drop_overlap : bool
        If the highest ``sweep_param`` value is shared between datasets,
        drop the duplicate from the right side (default True).
    label_left / label_right : str
        Short labels shown in slider step annotations and the sidebar.
    output_html : str | None
        Output .html path.  Defaults to
        ``<left_dataset_path>/dynamic_2d_{quantity}_{sweep_param}_joined.html``.
    """
    import json as _json
    import plotly.graph_objects as go

    # ---- helpers -------------------------------------------------------
    def _resolve(p):
        return p if os.path.isabs(p) else os.path.join(os.getcwd(), p)

    def _load(dataset_path):
        dataset_path = _resolve(dataset_path)
        bundle_path  = os.path.join(dataset_path, "qgt_nd_bundle.npz")
        json_path    = os.path.join(dataset_path, "parameters.json")
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"ND bundle not found: {bundle_path}")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"parameters.json not found: {json_path}")
        bundle   = np.load(bundle_path, allow_pickle=True)
        with open(json_path, "r", encoding="utf-8") as f:
            pjson = _json.load(f)
        ham_name = pjson["hamiltonian_name"]
        return bundle, ham_name, pjson, dataset_path

    def _get_axis(bundle, param):
        names = list(bundle["names"])
        if param not in names:
            raise ValueError(
                f"'{param}' not in bundle names {names}"
            )
        i = names.index(param)
        return i, bundle[f"axis_{i}_{param}"], list(bundle["shape"]), names

    def _field_series(bundle, sweep_axis, shape, q):
        n_params = len(shape)
        mid_idx = tuple(
            slice(None) if ax == sweep_axis else shape[ax] // 2
            for ax in range(n_params)
        )
        if q == "trace":
            return bundle["trace_grid"][mid_idx]
        elif q == "berry":
            return -2.0 * bundle["g_xy_imag_grid"][mid_idx]
        elif q == "imqxy":
            return bundle["g_xy_imag_grid"][mid_idx]
        raise ValueError(f"Unknown quantity '{q}'.")

    def _filter(param_values, field_series, pmin, pmax):
        mask = np.ones(len(param_values), dtype=bool)
        if pmin is not None: mask &= param_values >= pmin
        if pmax is not None: mask &= param_values <= pmax
        return param_values[mask], field_series[mask]

    # ---- load both datasets --------------------------------------------
    bundle_L, ham_L, pjson_L, dpath_L = _load(left_dataset_path)
    bundle_R, ham_R, pjson_R, dpath_R = _load(right_dataset_path)

    # k-grid check
    kx_L, ky_L = bundle_L["kx"], bundle_L["ky"]
    kx_R, ky_R = bundle_R["kx"], bundle_R["ky"]
    if kx_L.shape != kx_R.shape or not np.allclose(kx_L, kx_R):
        raise ValueError("kx grids differ between the two datasets; cannot join.")
    kx, ky = kx_L, ky_L

    q = quantity.lower()
    label_q = {"trace": "QGT Trace",
                "berry": "Berry Curvature \u03a9",
                "imqxy": "Im(Q_xy)"}[q]

    # ---- extract sweep axes & fields -----------------------------------
    ax_L, pv_L, shape_L, names_L = _get_axis(bundle_L, sweep_param)
    ax_R, pv_R, shape_R, names_R = _get_axis(bundle_R, sweep_param)

    fs_L = _field_series(bundle_L, ax_L, shape_L, q)
    fs_R = _field_series(bundle_R, ax_R, shape_R, q)

    pv_L, fs_L = _filter(pv_L, fs_L, param_min_left,  param_max_left)
    pv_R, fs_R = _filter(pv_R, fs_R, param_min_right, param_max_right)

    if len(pv_L) == 0 or len(pv_R) == 0:
        raise ValueError("No slices after filtering for one or both datasets.")

    # ---- join: left ascending, right descending ------------------------
    ord_L = np.argsort(pv_L);  pv_L = pv_L[ord_L];  fs_L = fs_L[ord_L]
    ord_R = np.argsort(pv_R);  pv_R_s = pv_R[ord_R]; fs_R_s = fs_R[ord_R]
    pv_R_rev = pv_R_s[::-1];   fs_R_rev = fs_R_s[::-1]

    n_left = len(pv_L)

    if drop_overlap and len(pv_R_s) and np.isclose(pv_L[-1], pv_R_s[-1], atol=tol, rtol=0):
        pv_R_rev = pv_R_rev[1:]
        fs_R_rev = fs_R_rev[1:]

    pv_join = np.concatenate([pv_L, pv_R_rev])
    fs_join = np.concatenate([fs_L, fs_R_rev], axis=0)   # (N_total, Ny, Nx)

    # ---- colour limits -------------------------------------------------
    if symmetric_cbar is None:
        symmetric_cbar = (q != "trace")
    finite = fs_join[np.isfinite(fs_join)]
    vmin_d, vmax_d = float(np.nanmin(finite)), float(np.nanmax(finite))
    if symmetric_cbar:
        amax = max(abs(vmin_d), abs(vmax_d))
        vmin, vmax = -amax, amax
    else:
        vmin, vmax = vmin_d, vmax_d

    # ---- build Plotly figure -------------------------------------------
    fig = go.Figure()

    for i, pval in enumerate(pv_join):
        src = label_left if i < n_left else label_right
        fig.add_trace(go.Heatmap(
            z=fs_join[i].tolist(),
            x=kx[0, :].tolist(),
            y=ky[:, 0].tolist(),
            colorscale=cmap,
            zmin=vmin, zmax=vmax,
            visible=(i == 0),
            name=f"{sweep_param}={pval:.4g} ({src})",
            colorbar=dict(title=label_q, thickness=18),
            hovertemplate="kx: %{x:.3f}<br>ky: %{y:.3f}<br>Value: %{z:.4g}<extra></extra>",
        ))

    steps = []
    for i, pval in enumerate(pv_join):
        src = label_left if i < n_left else label_right
        visible = [False] * len(pv_join)
        visible[i] = True
        steps.append(dict(
            method="update",
            args=[{"visible": visible},
                  {"title": f"{label_q}<br>"
                            f"{sweep_param} = {pval:.6g}  \u2502 {src}"}],
            label=f"{pval:.3g}",
        ))

    fig.update_layout(
        title=f"{label_q}<br>{sweep_param} = {pv_join[0]:.6g}  \u2502 {label_left}",
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

    # ---- sidebar (dual panels) -----------------------------------------
    def _sidebar_panel(pjson, title_label, title_color):
        def _fv(v):
            return f"{v:.6g}" if isinstance(v, float) else str(v)
        def _row(k, v):
            return f'<div class="row"><span class="key">{k}</span><span class="val">{v}</span></div>'
        def _sec(t, rows):
            return f'<div class="section"><div class="section-title">{t}</div>{rows}</div>'

        ham_rows = "".join(_row(k, _fv(v)) for k, v in sorted(pjson["parameters"].items()))

        scan_ranges  = pjson["scan_ranges"]
        scan_spacing = pjson["scan_spacing"]
        scan_rows = "".join(
            _row(n, f'{_fv(info["min"])} \u2192 {_fv(info["max"])}'
                 + f'  ({scan_spacing[n]["count"]} pts, {scan_spacing[n]["scale"]})')
            for n, info in scan_ranges.items()
        )

        kg = pjson["k_grid"]
        grid_rows = (
            _row("kx", f'{_fv(kg["kx_min"])} \u2192 {_fv(kg["kx_max"])}') +
            _row("ky", f'{_fv(kg["ky_min"])} \u2192 {_fv(kg["ky_max"])}') +
            _row("mesh", str(kg["mesh"]))
        )
        band_row = _row("band index", str(pjson["band_index"]))

        return (
            f'<div class="panel">'
            f'<div class="panel-head" style="background:{title_color}">'
            f'{pjson["hamiltonian_name"]} \u2014 {title_label}</div>'
            + _sec("Hamiltonian Parameters", ham_rows)
            + _sec("Sweep Range", scan_rows)
            + _sec("k-Grid", grid_rows + band_row)
            + '</div>'
        )

    sidebar_html = (
        '<div class="sidebar">'
        + _sidebar_panel(pjson_L, label_left,  "#3b4fa8")
        + _sidebar_panel(pjson_R, label_right, "#7b3fa8")
        + '</div>'
    )

    # ---- write HTML ----------------------------------------------------
    if output_html is None:
        output_html = os.path.join(
            _resolve(left_dataset_path),
            f"dynamic_2d_{q}_{sweep_param}_joined.html"
        )

    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{label_q} vs {sweep_param} (joined)</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #f0f2f7;
      margin: 0; padding: 24px;
      display: flex; justify-content: center; align-items: flex-start; gap: 20px;
    }}
    .plot-wrapper {{ flex: 0 0 auto; }}
    .sidebar {{
      flex: 0 0 270px;
      display: flex; flex-direction: column; gap: 14px;
      align-self: flex-start; margin-top: 8px;
    }}
    .panel {{
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.08);
      overflow: hidden;
    }}
    .panel-head {{
      color: #fff; font-size: 12.5px; font-weight: 700;
      letter-spacing: 0.03em; padding: 12px 16px; word-break: break-word;
    }}
    .section {{ padding: 8px 16px 4px 16px; }}
    .section-title {{
      font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.08em; color: #8892b0;
      margin-bottom: 5px; border-bottom: 1px solid #e8ecf4; padding-bottom: 3px;
    }}
    .row {{
      display: flex; justify-content: space-between; align-items: baseline;
      padding: 3px 0; font-size: 12px; border-bottom: 1px solid #f3f5fb;
    }}
    .row:last-child {{ border-bottom: none; }}
    .key {{ color: #4a5568; font-weight: 600; padding-right: 8px; white-space: nowrap; }}
    .val {{ color: #2d3748; text-align: right; font-variant-numeric: tabular-nums; word-break: break-all; }}
  </style>
</head>
<body>
  <div class="plot-wrapper">{plot_html}</div>
  {sidebar_html}
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Saved joined HTML to: {output_html}")
    return output_html


#@ 2D QGT ND — SquareLatticeHamiltonian parameter sweeps
# dataset1: left polarization,  omega in [0.1, 5.0],  mesh 150
# dataset2: right polarization, omega in [0.1, 50.0], mesh 100
# dynamic_2d_qgt_1d_sweep("results/2D_QGT_ND/SquareLatticeHamiltonian/dataset1", sweep_param="omega")

# dynamic_2d_qgt_1para_sweep_joined(
#     "results/2D_QGT_ND/SquareLatticeHamiltonian/dataset1",  # left
#     "results/2D_QGT_ND/SquareLatticeHamiltonian/dataset2",  # right
#     sweep_param="omega",
#     quantity="trace",
#     param_min_left=0.1,
#     param_min_right=0.5,
#     label_left="left polarization",
#     label_right="right polarization",
# )
