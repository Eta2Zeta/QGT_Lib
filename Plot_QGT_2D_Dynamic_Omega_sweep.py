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

def dynamic_2d_trace_hprime_eigs_vs_omega(
        dataset_path,
        omega_min=None,
        omega_max=None,
        output_html=None,
        trace_cmap="inferno",
        hprime_cmap="Viridis"):
    """
    Interactive HTML plot for the N-D QGT bundle produced by
    ``Calc_QGT_2D_nd_parameter_sweep.py``.

    Uses the ``omega`` axis as the slider, fixes all other parameter axes at
    their middle index, and shows:
      - eigenvalues vs kx at ky ~= 0
      - QGT trace heatmap
      - Re(H'00) heatmap
    """

    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.getcwd(), dataset_path)

    bundle_path = os.path.join(dataset_path, "qgt_nd_bundle.npz")
    json_path = os.path.join(dataset_path, "parameters.json")

    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"ND bundle not found: {bundle_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"parameters.json not found: {json_path}")

    bundle = np.load(bundle_path, allow_pickle=True)
    with open(json_path, "r", encoding="utf-8") as f:
        pjson = json.load(f)

    names = [str(n) for n in bundle["names"]]
    shape = [int(x) for x in bundle["shape"]]
    if "omega" not in names:
        raise ValueError(f"Parameter 'omega' not found in bundle. Available parameters: {names}")

    omega_axis = names.index("omega")
    omega_values_all = np.asarray(bundle[f"axis_{omega_axis}_omega"], dtype=float)

    indexer = tuple(
        slice(None) if ax == omega_axis else shape[ax] // 2
        for ax in range(len(names))
    )

    trace_series = np.asarray(bundle["trace_grid"][indexer])
    hprime_series = np.real(np.asarray(bundle["hamiltonian_prime_grid"][indexer])[..., 0, 0])
    eigen_series = np.asarray(bundle["eigenvalues_grid"][indexer])

    mask = np.ones(len(omega_values_all), dtype=bool)
    if omega_min is not None:
        mask &= omega_values_all >= omega_min
    if omega_max is not None:
        mask &= omega_values_all <= omega_max

    omega_values = omega_values_all[mask]
    trace_series = trace_series[mask]
    hprime_series = hprime_series[mask]
    eigen_series = eigen_series[mask]

    if len(omega_values) == 0:
        raise ValueError(f"No omega slices fall within [{omega_min}, {omega_max}].")

    kx = np.asarray(bundle["kx"])
    ky = np.asarray(bundle["ky"])
    ky_vals = ky[:, 0] if ky.ndim == 2 else ky
    ky_idx = int(np.argmin(np.abs(ky_vals)))
    kx_line = kx[ky_idx, :] if kx.ndim == 2 else kx
    sort_idx = np.argsort(kx_line)
    kx_line_sorted = kx_line[sort_idx]

    nbands = int(eigen_series.shape[-1])
    trace_vmin = float(np.nanmin(trace_series))
    trace_vmax = float(np.nanmax(trace_series))
    hprime_vmin = float(np.nanmin(hprime_series))
    hprime_vmax = float(np.nanmax(hprime_series))
    eig_vmin = float(np.nanmin(eigen_series))
    eig_vmax = float(np.nanmax(eigen_series))
    eig_pad = 0.05 * (eig_vmax - eig_vmin) if eig_vmax > eig_vmin else 1.0

    ham_name = pjson.get("hamiltonian_name", "Hamiltonian")
    fixed_labels = [
        f"{names[ax]}={bundle[f'axis_{ax}_{names[ax]}'][shape[ax] // 2]:.4g}"
        for ax in range(len(names))
        if ax != omega_axis
    ]
    fixed_str = ", ".join(fixed_labels)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        row_heights=[0.28, 0.72],
        subplot_titles=(
            "Eigenvalues at ky ~= 0",
            "QGT Trace",
            "Re H'00",
        ),
        vertical_spacing=0.13,
        horizontal_spacing=0.10,
    )

    traces_per_slice = nbands + 2
    for i, omega in enumerate(omega_values):
        visible = (i == 0)
        eig_line = eigen_series[i, ky_idx, :, :][sort_idx, :]

        for band in range(nbands):
            fig.add_trace(
                go.Scatter(
                    x=kx_line_sorted,
                    y=eig_line[:, band],
                    mode="lines",
                    line=dict(width=1.4),
                    name=f"band {band}",
                    legendgroup=f"band{band}",
                    showlegend=(i == 0),
                    visible=visible,
                    hovertemplate="kx: %{x:.4g}<br>E: %{y:.4g}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Heatmap(
                z=trace_series[i].tolist(),
                x=kx[0, :].tolist(),
                y=ky[:, 0].tolist(),
                colorscale=trace_cmap,
                zmin=trace_vmin,
                zmax=trace_vmax,
                colorbar=dict(title="Trace", x=0.46, len=0.55, y=0.28),
                visible=visible,
                name="Trace",
                hovertemplate="kx: %{x:.4g}<br>ky: %{y:.4g}<br>Trace: %{z:.4g}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=hprime_series[i].tolist(),
                x=kx[0, :].tolist(),
                y=ky[:, 0].tolist(),
                colorscale=hprime_cmap,
                zmin=hprime_vmin,
                zmax=hprime_vmax,
                colorbar=dict(title="Re H'00", x=1.02, len=0.55, y=0.28),
                visible=visible,
                name="Re H'00",
                hovertemplate="kx: %{x:.4g}<br>ky: %{y:.4g}<br>Re H'00: %{z:.4g}<extra></extra>",
            ),
            row=2,
            col=2,
        )

    steps = []
    total_traces = len(omega_values) * traces_per_slice
    for i, omega in enumerate(omega_values):
        visible = [False] * total_traces
        start = i * traces_per_slice
        for j in range(traces_per_slice):
            visible[start + j] = True
        title = f"{ham_name} — Trace, Re H'00, Eigenvalues<br>omega = {omega:.6g}"
        if fixed_str:
            title += f"  ({fixed_str})"
        steps.append(dict(
            method="update",
            args=[{"visible": visible}, {"title": title}],
            label=f"{omega:.3g}",
        ))

    initial_title = f"{ham_name} — Trace, Re H'00, Eigenvalues<br>omega = {omega_values[0]:.6g}"
    if fixed_str:
        initial_title += f"  ({fixed_str})"

    fig.update_layout(
        title=initial_title,
        width=1120,
        height=860,
        margin=dict(l=70, r=120, t=105, b=95),
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "omega: ", "font": {"size": 14}},
            pad={"t": 45},
            steps=steps,
        )],
    )
    fig.update_xaxes(title_text="kx", row=1, col=1)
    fig.update_yaxes(title_text="Eigenvalue", range=[eig_vmin - eig_pad, eig_vmax + eig_pad], row=1, col=1)
    fig.update_xaxes(title_text="kx", row=2, col=1)
    fig.update_yaxes(title_text="ky", scaleanchor="x2", scaleratio=1, row=2, col=1)
    fig.update_xaxes(title_text="kx", row=2, col=2)
    fig.update_yaxes(title_text="ky", scaleanchor="x3", scaleratio=1, row=2, col=2)

    def _fmt_val(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def _row(key, val):
        return f'<div class="row"><span class="key">{key}</span><span class="val">{val}</span></div>'

    def _section(title, rows_html):
        return f'<div class="section"><div class="section-title">{title}</div>{rows_html}</div>'

    ham_rows = "".join(
        _row(k, _fmt_val(v))
        for k, v in sorted(pjson.get("parameters", {}).items())
    )
    scan_rows = "".join(
        _row(
            name,
            f'{_fmt_val(info["min"])} to {_fmt_val(info["max"])}'
            + f' ({pjson["scan_spacing"][name]["count"]} pts, {pjson["scan_spacing"][name]["scale"]})'
        )
        for name, info in pjson.get("scan_ranges", {}).items()
    )
    kg = pjson.get("k_grid", {})
    kgrid_rows = (
        _row("kx", f'{_fmt_val(kg.get("kx_min"))} to {_fmt_val(kg.get("kx_max"))}') +
        _row("ky", f'{_fmt_val(kg.get("ky_min"))} to {_fmt_val(kg.get("ky_max"))}') +
        _row("mesh", str(kg.get("mesh")))
    )
    if "band_index" in pjson:
        kgrid_rows += _row("band index", str(pjson["band_index"]))

    sidebar_html = f"""
<div class="sidebar">
  <div class="sidebar-header">{ham_name}</div>
  {_section("Hamiltonian Parameters", ham_rows)}
  {_section("Sweep Range", scan_rows)}
  {_section("k-Grid", kgrid_rows)}
</div>"""

    if output_html is None:
        output_html = os.path.join(dataset_path, "dynamic_2d_trace_hprime_eigs_vs_omega.html")

    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{ham_name} — Trace Hprime Eigenvalues vs omega</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  font-family: 'Segoe UI', Arial, sans-serif;
  background: #f0f2f7;
  margin: 0;
  padding: 24px;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  gap: 24px;
}}
.plot-wrapper {{ flex: 0 0 auto; }}
.sidebar {{
  flex: 0 0 280px;
  background: #ffffff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  padding: 0 0 16px 0;
  overflow: hidden;
  align-self: flex-start;
  margin-top: 8px;
}}
.sidebar-header {{
  background: #34405f;
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
  color: #7b849c;
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


def dynamic_2d_trace_hprime_eigs_vs_omega_dual(
    folder_name_left, folder_name_right,
    *,
    omega_min=None, omega_max=None,
    label_left="numerical", label_right="analytical"
):
    """
    Compare TWO 2D QGT sweeps side-by-side.
    Layout (3 rows x 2 cols):
      Row 1: (slim) eigenvalues vs kx at ky≈0   [left | right]
      Row 2: Trace heatmap                      [left | right]
      Row 3: Re(H'00) heatmap                   [left | right]
    One slider (bottom) selects ω index in LEFT dataset; RIGHT snaps to nearest ω.
    """

    # ---- load both datasets ----
    entries_L, meta_L = load_qgt(folder_name_left)
    entries_R, meta_R = load_qgt(folder_name_right)

    # filter by ω range (independently)
    entries_L = filter_entries_by_omega(entries_L, omega_min, omega_max)
    entries_R = filter_entries_by_omega(entries_R, omega_min, omega_max)

    # extract k grids
    kx_L = np.asarray(meta_L["kx"]);  ky_L = np.asarray(meta_L["ky"])
    kx_R = np.asarray(meta_R["kx"]);  ky_R = np.asarray(meta_R["ky"])

    # collect ω arrays
    omegas_L = np.array([float(e["omega"]) for e in entries_L], dtype=float)
    omegas_R = np.array([float(e["omega"]) for e in entries_R], dtype=float)

    # choose ky index near 0 for each dataset
    ky_vals_L = ky_L[:, 0] if ky_L.ndim == 2 else ky_L
    ky_vals_R = ky_R[:, 0] if ky_R.ndim == 2 else ky_R
    ky_idx_L = int(np.argmin(np.abs(ky_vals_L - 0.0)))
    ky_idx_R = int(np.argmin(np.abs(ky_vals_R - 0.0)))

    # precompute Re(H'00) lists for global color limits
    h00_L_list = [np.real(np.asarray(e["hamiltonian_prime_array"])[..., 0, 0]) for e in entries_L]
    h00_R_list = [np.real(np.asarray(e["hamiltonian_prime_array"])[..., 0, 0]) for e in entries_R]

    # trace limits across both datasets
    trace_min = float(np.nanmin([np.nanmin(e["trace"]) for e in entries_L] +
                                [np.nanmin(e["trace"]) for e in entries_R]))
    trace_max = float(np.nanmax([np.nanmax(e["trace"]) for e in entries_L] +
                                [np.nanmax(e["trace"]) for e in entries_R]))

    # H'00 limits across both datasets
    h00_min = float(np.nanmin([np.nanmin(h) for h in h00_L_list] +
                              [np.nanmin(h) for h in h00_R_list]))
    h00_max = float(np.nanmax([np.nanmax(h) for h in h00_L_list] +
                              [np.nanmax(h) for h in h00_R_list]))

    # initial indices
    iL0 = 0
    # find nearest ω in R for initial left ω
    iR0 = int(np.argmin(np.abs(omegas_R - omegas_L[iL0])))

    # ---- prepare eigenvalue lines for each dataset (sorted by kx) ----
    def _prep_eig_line(entry, kx, ky, ky_idx):
        evals = np.asarray(entry["eigenvalues"])  # (Ny, Nx, Nb)
        kx_line = kx[ky_idx, :] if kx.ndim == 2 else kx
        order = np.argsort(kx_line)
        kx_sorted = kx_line[order]
        eig_line  = evals[ky_idx, :, :]   # (Nx, Nb)
        eig_sorted = eig_line[order, :]
        return kx_sorted, eig_sorted

    kxs_L0, eigs_L0 = _prep_eig_line(entries_L[iL0], kx_L, ky_L, ky_idx_L)
    kxs_R0, eigs_R0 = _prep_eig_line(entries_R[iR0], kx_R, ky_R, ky_idx_R)
    nb_L = eigs_L0.shape[1]
    nb_R = eigs_R0.shape[1]

    # ---- figure & layout ----
    fig = plt.figure(figsize=(16, 14))
    # leave some space at bottom for slider
    fig.subplots_adjust(top=0.96, bottom=0.10)

    # 3 rows, 2 cols; top row short, bottom two equal heights
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        height_ratios=[1, 5, 5], width_ratios=[1, 1],
        hspace=0.2, wspace=0.20
    )


    # Row 1: eigenvalues (left, right)
    ax_eigs_L = fig.add_subplot(gs[0, 0])
    ax_eigs_R = fig.add_subplot(gs[0, 1])

    # Row 2: trace (left, right)
    ax_trace_L = fig.add_subplot(gs[1, 0])
    ax_trace_R = fig.add_subplot(gs[1, 1])

    # Row 3: H'00 (left, right)
    ax_h_L = fig.add_subplot(gs[2, 0])
    ax_h_R = fig.add_subplot(gs[2, 1])

    # ---- draw initial eigenvalues ----
    lines_L = []
    for b in range(nb_L):
        ln, = ax_eigs_L.plot(kxs_L0, eigs_L0[:, b], lw=0.9)
        lines_L.append(ln)
    ax_eigs_L.set_title(f"{label_left}: Eigenvalues vs $k_x$ (slice at $k_y\\approx 0$)")
    ax_eigs_L.set_xlabel("$k_x$")
    ax_eigs_L.set_ylabel("Eigenvalue")
    ax_eigs_L.grid(True)

    lines_R = []
    for b in range(nb_R):
        ln, = ax_eigs_R.plot(kxs_R0, eigs_R0[:, b], lw=0.9)
        lines_R.append(ln)
    ax_eigs_R.set_title(f"{label_right}: Eigenvalues vs $k_x$ (slice at $k_y\\approx 0$)")
    ax_eigs_R.set_xlabel("$k_x$")
    ax_eigs_R.set_ylabel("Eigenvalue")
    ax_eigs_R.grid(True)

    # ---- draw initial trace heatmaps ----
    im_tr_L = ax_trace_L.imshow(
        np.asarray(entries_L[iL0]["trace"]),
        origin="lower",
        extent=[kx_L.min(), kx_L.max(), ky_L.min(), ky_L.max()],
        cmap="inferno", vmin=trace_min, vmax=trace_max, aspect="auto"
    )
    ax_trace_L.set_title(f"{label_left}: QGT Trace — $\\omega$ = {omegas_L[iL0]:.6f}")
    ax_trace_L.set_xlabel("$k_x$")
    ax_trace_L.set_ylabel("$k_y$")
    cbar_tr_L = fig.colorbar(im_tr_L, ax=ax_trace_L, fraction=0.047, pad=0.03)
    cbar_tr_L.set_label("Trace")
    std_txt_L = ax_trace_L.text(0.02, 0.98, f"std[Trace] = {np.nanstd(entries_L[iL0]['trace']):.4g}",
                                transform=ax_trace_L.transAxes, va="top", ha="left",
                                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    im_tr_R = ax_trace_R.imshow(
        np.asarray(entries_R[iR0]["trace"]),
        origin="lower",
        extent=[kx_R.min(), kx_R.max(), ky_R.min(), ky_R.max()],
        cmap="inferno", vmin=trace_min, vmax=trace_max, aspect="auto"
    )
    ax_trace_R.set_title(f"{label_right}: QGT Trace — $\\omega$ = {omegas_R[iR0]:.6f}")
    ax_trace_R.set_xlabel("$k_x$")
    ax_trace_R.set_ylabel("$k_y$")
    cbar_tr_R = fig.colorbar(im_tr_R, ax=ax_trace_R, fraction=0.047, pad=0.03)
    cbar_tr_R.set_label("Trace")
    std_txt_R = ax_trace_R.text(0.02, 0.98, f"std[Trace] = {np.nanstd(entries_R[iR0]['trace']):.4g}",
                                transform=ax_trace_R.transAxes, va="top", ha="left",
                                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    # ---- draw initial H'00 heatmaps ----
    im_h_L = ax_h_L.imshow(
        h00_L_list[iL0],
        origin="lower",
        extent=[kx_L.min(), kx_L.max(), ky_L.min(), ky_L.max()],
        cmap="viridis", vmin=h00_min, vmax=h00_max, aspect="auto"
    )
    ax_h_L.set_title(f"{label_left}: $\\Re\,H'_{ {0},{0} }$ — $\\omega$ = {omegas_L[iL0]:.6f}".replace(" {0}{0} ","{00}"))
    ax_h_L.set_xlabel("$k_x$")
    ax_h_L.set_ylabel("$k_y$")
    cbar_h_L = fig.colorbar(im_h_L, ax=ax_h_L, fraction=0.047, pad=0.03)
    cbar_h_L.set_label(r"$\Re\,H'_{00}$")

    im_h_R = ax_h_R.imshow(
        h00_R_list[iR0],
        origin="lower",
        extent=[kx_R.min(), kx_R.max(), ky_R.min(), ky_R.max()],
        cmap="viridis", vmin=h00_min, vmax=h00_max, aspect="auto"
    )
    ax_h_R.set_title(f"{label_right}: $\\Re\,H'_{ {0},{0} }$ — $\\omega$ = {omegas_R[iR0]:.6f}".replace(" {0}{0} ","{00}"))
    ax_h_R.set_xlabel("$k_x$")
    ax_h_R.set_ylabel("$k_y$")
    cbar_h_R = fig.colorbar(im_h_R, ax=ax_h_R, fraction=0.047, pad=0.03)
    cbar_h_R.set_label(r"$\Re\,H'_{00}$")

    # ---- slider (bottom) controls LEFT; RIGHT snaps to nearest ω ----
    ax_sl = plt.axes([0.16, 0.03, 0.68, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(ax_sl, r'left $\omega$ index', 0, len(omegas_L) - 1, valinit=iL0, valstep=1)

    # Pre-sort kx orderers for both datasets (so we don't recompute each update)
    order_L = np.argsort(kx_L[ky_idx_L, :] if kx_L.ndim == 2 else kx_L)
    order_R = np.argsort(kx_R[ky_idx_R, :] if kx_R.ndim == 2 else kx_R)
    kxs_L = (kx_L[ky_idx_L, :] if kx_L.ndim == 2 else kx_L)[order_L]
    kxs_R = (kx_R[ky_idx_R, :] if kx_R.ndim == 2 else kx_R)[order_R]

    def _update(_):
        iL = int(slider.val)
        # find closest ω in right dataset
        iR = int(np.argmin(np.abs(omegas_R - omegas_L[iL])))

        # eigenvalues (left)
        evL = np.asarray(entries_L[iL]["eigenvalues"])  # (Ny,Nx,Nb)
        lineL = evL[ky_idx_L, :, :][order_L, :]
        for b in range(min(nb_L, lineL.shape[1])):
            lines_L[b].set_data(kxs_L, lineL[:, b])

        # eigenvalues (right)
        evR = np.asarray(entries_R[iR]["eigenvalues"])
        lineR = evR[ky_idx_R, :, :][order_R, :]
        for b in range(min(nb_R, lineR.shape[1])):
            lines_R[b].set_data(kxs_R, lineR[:, b])

        # traces
        trL = np.asarray(entries_L[iL]["trace"])
        trR = np.asarray(entries_R[iR]["trace"])
        im_tr_L.set_data(trL)
        im_tr_R.set_data(trR)
        ax_trace_L.set_title(f"{label_left}: QGT Trace — $\\omega$ = {omegas_L[iL]:.6f}")
        ax_trace_R.set_title(f"{label_right}: QGT Trace — $\\omega$ = {omegas_R[iR]:.6f}")
        std_txt_L.set_text(f"std[Trace] = {np.nanstd(trL):.4g}")
        std_txt_R.set_text(f"std[Trace] = {np.nanstd(trR):.4g}")

        # H'00
        im_h_L.set_data(h00_L_list[iL])
        im_h_R.set_data(h00_R_list[iR])
        ax_h_L.set_title(f"{label_left}: $\\Re\,H'_{ {0},{0} }$ — $\\omega$ = {omegas_L[iL]:.6f}".replace(" {0}{0} ","{00}"))
        ax_h_R.set_title(f"{label_right}: $\\Re\,H'_{ {0},{0} }$ — $\\omega$ = {omegas_R[iR]:.6f}".replace(" {0}{0} ","{00}"))

        fig.canvas.draw_idle()

    slider.on_changed(_update)
    plt.show()



#@ 2D joined plots
def dynamic_2d_qgt_vs_omega_joined(
    left_folder_name,
    right_folder_name,
    *,
    quantity="trace",            # "trace" | "berry" | "imqxy"
    symmetric_cbar=None,         # None -> True for non-trace; False for trace
    omega_min_left=None,
    omega_max_left=None,
    omega_min_right=None,
    omega_max_right=None,
    tol=1e-9,
    drop_overlap=True,
    cmap='inferno'
):
    # ---- use your helpers ----
    # expects: load_qgt(folder_name) -> (entries, meta)
    #          filter_entries_by_omega(entries, omega_min, omega_max) -> filtered_entries
    entries_L, meta_L = load_qgt(left_folder_name)
    entries_R, meta_R = load_qgt(right_folder_name)

    # k-grid sanity
    kx_L, ky_L = np.asarray(meta_L["kx"]), np.asarray(meta_L["ky"])
    kx_R, ky_R = np.asarray(meta_R["kx"]), np.asarray(meta_R["ky"])
    if kx_L.shape != kx_R.shape or ky_L.shape != ky_R.shape \
       or not np.allclose(kx_L, kx_R) or not np.allclose(ky_L, ky_R):
        raise ValueError("kx/ky grids differ between folders; cannot join.")
    kx, ky = kx_L, ky_L

    # independent omega filters (value-based)
    filt_L = filter_entries_by_omega(entries_L, omega_min_left,  omega_max_left)
    filt_R = filter_entries_by_omega(entries_R, omega_min_right, omega_max_right)
    if len(filt_L) == 0 or len(filt_R) == 0:
        raise ValueError("No omega slices in range for one or both datasets.")

    # extract ω arrays
    l_om = np.array([float(e["omega"]) for e in filt_L], dtype=float)
    r_om = np.array([float(e["omega"]) for e in filt_R], dtype=float)

    # select field per entry
    q = quantity.lower()
    def _extract_field(entry):
        if q == "trace":
            return np.asarray(entry["trace"])
        if q in ("berry", "berry_curvature", "omega"):
            return -2.0 * np.asarray(entry["g_xy_imag"])
        if q in ("imqxy", "im(q_xy)", "im_qxy"):
            return np.asarray(entry["g_xy_imag"])
        raise ValueError(f"Unknown quantity '{quantity}'.")

    l_data = [ _extract_field(e) for e in filt_L ]
    r_data = [ _extract_field(e) for e in filt_R ]

    # left asc; right desc (reverse)
    l_ord = np.argsort(l_om); l_om = l_om[l_ord]; l_data = [l_data[i] for i in l_ord]
    r_ord = np.argsort(r_om); r_om_sorted = r_om[r_ord]; r_data_sorted = [r_data[i] for i in r_ord]
    r_om_rev = r_om_sorted[::-1]; r_data_rev = r_data_sorted[::-1]

    # number of left slices (for labeling)
    n_left = len(l_om)

    # optional de-dup at junction (match high-ω ends)
    if drop_overlap and r_om_sorted.size and np.isclose(l_om[-1], r_om_sorted[-1], atol=tol, rtol=0):
        r_om_rev   = r_om_rev[1:]
        r_data_rev = r_data_rev[1:]

    # join
    omegas_join = np.concatenate([l_om, r_om_rev])
    fields_join = l_data + r_data_rev

    # color limits
    if symmetric_cbar is None:
        symmetric_cbar = (q != "trace")
    if symmetric_cbar:
        vmax_abs = max(max(abs(np.nanmin(F)), abs(np.nanmax(F))) for F in fields_join)
        vmin, vmax = -vmax_abs, vmax_abs
    else:
        vmin = min(np.nanmin(F) for F in fields_join)
        vmax = max(np.nanmax(F) for F in fields_join)

    # plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(bottom=0.2)
    idx0 = 0
    im = ax.imshow(
        fields_join[idx0],
        origin='lower',
        extent=[kx.min(), kx.max(), ky.min(), ky.max()],
        cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto'
    )

    title_q = {"trace":"QGT Trace", "berry":"Berry Curvature Ω", "imqxy":"Im(Q_xy)"}[q]
    def _src(i): return "L" if i < n_left else "R"

    ax.set_title(f'{title_q} — ω = {omegas_join[idx0]:.6f}  (src: {_src(idx0)})')
    ax.set_xlabel("$k_x$"); ax.set_ylabel("$k_y$")
    cbar = plt.colorbar(im, ax=ax); cbar.set_label(title_q)

    # slider
    ax_sl = plt.axes([0.15, 0.06, 0.70, 0.03], facecolor='lightgoldenrodyellow')
    sl = Slider(ax_sl, '$\\omega$ idx', 0, len(omegas_join)-1, valinit=idx0, valstep=1)

    def _update(val):
        i = int(sl.val)
        im.set_data(fields_join[i])
        ax.set_title(f'{title_q} — ω = {omegas_join[i]:.6f}  (src: {_src(i)})')
        fig.canvas.draw_idle()

    sl.on_changed(_update)
    plt.show()

#@ Rhombohedral Graphene Hamiltonian

#! V = 30
# New 2D_QGT_ND structure:
# dynamic_2d_trace_hprime_eigs_vs_omega_nd(
#     "results/2D_QGT_ND/ChiralHamiltonianChiralBasisProjected/dataset1"
# )

# Old 2D_QGT_omega_sweep structure:
# dynamic_2d_trace_hprime_eigs_vs_omega("ChiralHamiltonianProjected/A0_0.10-V_30-analytic_magnus_True-magnus_order_1-n_5-omega_1000-polarization_right-t1_355.16-vF_542.10_kx-0.80_0.80_ky-0.80_0.80_mesh100_omega3.00e_01_5.00e_03_spacing_log_points14_band0_data_set1")

#@ Both numerical and analytical calcualtions
# dynamic_2d_trace_hprime_eigs_vs_omega_dual("RhombohedralGrapheneHamiltonian/A0_0.10-V_30-analytic_magnus_False-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10_kx-0.80_0.80_ky-0.80_0.80_mesh100_omega2.50e_01_5.00e_03_spacing_log_points84_data_set1",
#                                            "ChiralHamiltonianProjected/A0_0.10-V_30-analytic_magnus_True-magnus_order_1-n_5-omega_1000-polarization_right-t1_355.16-vF_542.10_kx-0.80_0.80_ky-0.80_0.80_mesh100_omega3.00e_01_5.00e_03_spacing_log_points32_band0_data_set3")


# dynamic_2d_qgt_vs_omega_joined("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  quantity="trace", omega_min_left=33, omega_min_right=50)

dynamic_2d_qgt_vs_omega_joined("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
                                 "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
                                 quantity="berry", omega_min_left=33, omega_min_right=50)

