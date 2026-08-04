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

from Library.plotting_qgt_2d import get_symmetric_plot_limits
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
        hprime_cmap="Viridis",
        trace_zmax_percentile=99.0):
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
    if trace_zmax_percentile is None:
        trace_vmax = float(np.nanmax(trace_series))
    else:
        finite_trace = trace_series[np.isfinite(trace_series)]
        trace_vmax = float(np.nanpercentile(finite_trace, trace_zmax_percentile))
        if trace_vmax <= trace_vmin:
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
    left_dataset_path,
    right_dataset_path,
    *,
    omega_min=None,
    omega_max=None,
    label_left="left",
    label_right="right",
    output_html=None,
    trace_cmap="inferno",
    hprime_cmap="Viridis",
    trace_zmax_percentile=99.0,
):
    """
    Plotly HTML dual comparison for two N-D QGT bundles with an omega axis.

    For each dataset, omega is used as the slider axis and all other parameter
    axes are fixed at their midpoint. The right dataset snaps to the nearest
    omega value selected from the left dataset.
    """
    sweep_param = "omega"

    def _resolve_path(dataset_path):
        return dataset_path if os.path.isabs(dataset_path) else os.path.join(os.getcwd(), dataset_path)

    def _load_nd_omega(dataset_path, pmin, pmax):
        dataset_path = _resolve_path(dataset_path)
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
        if sweep_param not in names:
            raise ValueError(f"Parameter 'omega' not found in bundle. Available parameters: {names}")

        omega_axis = names.index(sweep_param)
        omega_values_all = np.asarray(bundle[f"axis_{omega_axis}_{sweep_param}"], dtype=float)
        fixed_indices = {
            names[ax]: shape[ax] // 2
            for ax in range(len(names))
            if ax != omega_axis
        }
        indexer = tuple(
            slice(None) if ax == omega_axis else shape[ax] // 2
            for ax in range(len(names))
        )

        trace_series = np.asarray(bundle["trace_grid"][indexer])
        hprime_series = np.real(np.asarray(bundle["hamiltonian_prime_grid"][indexer])[..., 0, 0])
        eigen_series = np.asarray(bundle["eigenvalues_grid"][indexer])

        mask = np.ones(len(omega_values_all), dtype=bool)
        if pmin is not None:
            mask &= omega_values_all >= pmin
        if pmax is not None:
            mask &= omega_values_all <= pmax

        omega_values = omega_values_all[mask]
        trace_series = trace_series[mask]
        hprime_series = hprime_series[mask]
        eigen_series = eigen_series[mask]

        if len(omega_values) == 0:
            raise ValueError(f"No omega slices fall within [{pmin}, {pmax}] for {dataset_path}.")

        order = np.argsort(omega_values)
        omega_values = omega_values[order]
        trace_series = trace_series[order]
        hprime_series = hprime_series[order]
        eigen_series = eigen_series[order]

        kx = np.asarray(bundle["kx"])
        ky = np.asarray(bundle["ky"])
        ky_vals = ky[:, 0] if ky.ndim == 2 else ky
        ky_idx = int(np.argmin(np.abs(ky_vals)))
        kx_line = kx[ky_idx, :] if kx.ndim == 2 else kx
        sort_idx = np.argsort(kx_line)

        fixed_labels = []
        for name, idx in fixed_indices.items():
            ax = names.index(name)
            value = bundle[f"axis_{ax}_{name}"][idx]
            fixed_labels.append(f"{name}={float(value):.4g}")

        return {
            "dataset_path": dataset_path,
            "bundle": bundle,
            "pjson": pjson,
            "names": names,
            "shape": shape,
            "omega_values": omega_values,
            "trace": trace_series,
            "hprime": hprime_series,
            "eigen": eigen_series,
            "kx": kx,
            "ky": ky,
            "ky_idx": ky_idx,
            "kx_line_sorted": kx_line[sort_idx],
            "sort_idx": sort_idx,
            "fixed_str": ", ".join(fixed_labels),
            "nbands": int(eigen_series.shape[-1]),
            "ham_name": pjson.get("hamiltonian_name", "Hamiltonian"),
        }

    left = _load_nd_omega(left_dataset_path, omega_min, omega_max)
    right = _load_nd_omega(right_dataset_path, omega_min, omega_max)

    trace_finite = np.concatenate([
        left["trace"][np.isfinite(left["trace"])],
        right["trace"][np.isfinite(right["trace"])],
    ])
    trace_vmin = float(np.nanmin(trace_finite))
    if trace_zmax_percentile is None:
        trace_vmax = float(np.nanmax(trace_finite))
    else:
        trace_vmax = float(np.nanpercentile(trace_finite, trace_zmax_percentile))
        if trace_vmax <= trace_vmin:
            trace_vmax = float(np.nanmax(trace_finite))

    hprime_finite = np.concatenate([
        left["hprime"][np.isfinite(left["hprime"])],
        right["hprime"][np.isfinite(right["hprime"])],
    ])
    hprime_vmin = float(np.nanmin(hprime_finite))
    hprime_vmax = float(np.nanmax(hprime_finite))

    eig_finite = np.concatenate([
        left["eigen"][np.isfinite(left["eigen"])],
        right["eigen"][np.isfinite(right["eigen"])],
    ])
    eig_vmin = float(np.nanmin(eig_finite))
    eig_vmax = float(np.nanmax(eig_finite))
    eig_pad = 0.05 * (eig_vmax - eig_vmin) if eig_vmax > eig_vmin else 1.0

    right_indices = [
        int(np.argmin(np.abs(right["omega_values"] - omega)))
        for omega in left["omega_values"]
    ]

    subplot_titles = (
        f"{label_left}: eigenvalues at ky ~= 0",
        f"{label_right}: eigenvalues at ky ~= 0",
        f"{label_left}: QGT Trace",
        f"{label_right}: QGT Trace",
        f"{label_left}: Re H'00",
        f"{label_right}: Re H'00",
    )
    fig = make_subplots(
        rows=3,
        cols=2,
        row_heights=[0.22, 0.39, 0.39],
        subplot_titles=subplot_titles,
        vertical_spacing=0.09,
        horizontal_spacing=0.09,
    )

    traces_per_slice = left["nbands"] + right["nbands"] + 4
    for i, omega_left in enumerate(left["omega_values"]):
        visible = i == 0
        i_right = right_indices[i]
        omega_right = right["omega_values"][i_right]

        eig_left = left["eigen"][i, left["ky_idx"], :, :][left["sort_idx"], :]
        eig_right = right["eigen"][i_right, right["ky_idx"], :, :][right["sort_idx"], :]

        for band in range(left["nbands"]):
            fig.add_trace(
                go.Scatter(
                    x=left["kx_line_sorted"],
                    y=eig_left[:, band],
                    mode="lines",
                    line=dict(width=1.3),
                    name=f"{label_left} band {band}",
                    legendgroup=f"{label_left}-band-{band}",
                    showlegend=True,
                    visible=visible,
                    hovertemplate="kx: %{x:.4g}<br>E: %{y:.4g}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        for band in range(right["nbands"]):
            fig.add_trace(
                go.Scatter(
                    x=right["kx_line_sorted"],
                    y=eig_right[:, band],
                    mode="lines",
                    line=dict(width=1.3),
                    name=f"{label_right} band {band}",
                    legendgroup=f"{label_right}-band-{band}",
                    showlegend=True,
                    visible=visible,
                    hovertemplate="kx: %{x:.4g}<br>E: %{y:.4g}<extra></extra>",
                ),
                row=1,
                col=2,
            )

        fig.add_trace(
            go.Heatmap(
                z=left["trace"][i].tolist(),
                x=left["kx"][0, :].tolist(),
                y=left["ky"][:, 0].tolist(),
                colorscale=trace_cmap,
                zmin=trace_vmin,
                zmax=trace_vmax,
                showscale=False,
                visible=visible,
                name=f"{label_left} Trace",
                hovertemplate="kx: %{x:.4g}<br>ky: %{y:.4g}<br>Trace: %{z:.4g}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=right["trace"][i_right].tolist(),
                x=right["kx"][0, :].tolist(),
                y=right["ky"][:, 0].tolist(),
                colorscale=trace_cmap,
                zmin=trace_vmin,
                zmax=trace_vmax,
                colorbar=dict(title="Trace", x=1.02, len=0.30, y=0.57),
                visible=visible,
                name=f"{label_right} Trace",
                hovertemplate="kx: %{x:.4g}<br>ky: %{y:.4g}<br>Trace: %{z:.4g}<extra></extra>",
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Heatmap(
                z=left["hprime"][i].tolist(),
                x=left["kx"][0, :].tolist(),
                y=left["ky"][:, 0].tolist(),
                colorscale=hprime_cmap,
                zmin=hprime_vmin,
                zmax=hprime_vmax,
                showscale=False,
                visible=visible,
                name=f"{label_left} Re H'00",
                hovertemplate="kx: %{x:.4g}<br>ky: %{y:.4g}<br>Re H'00: %{z:.4g}<extra></extra>",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=right["hprime"][i_right].tolist(),
                x=right["kx"][0, :].tolist(),
                y=right["ky"][:, 0].tolist(),
                colorscale=hprime_cmap,
                zmin=hprime_vmin,
                zmax=hprime_vmax,
                colorbar=dict(title="Re H'00", x=1.02, len=0.30, y=0.16),
                visible=visible,
                name=f"{label_right} Re H'00",
                hovertemplate="kx: %{x:.4g}<br>ky: %{y:.4g}<br>Re H'00: %{z:.4g}<extra></extra>",
            ),
            row=3,
            col=2,
        )

    def _title(i):
        omega_left = left["omega_values"][i]
        omega_right = right["omega_values"][right_indices[i]]
        title = (
            "Trace, Re H'00, Eigenvalues<br>"
            f"{label_left} omega = {omega_left:.6g} | "
            f"{label_right} omega = {omega_right:.6g}"
        )
        fixed_bits = [s for s in (left["fixed_str"], right["fixed_str"]) if s]
        if fixed_bits:
            title += "<br>" + " | ".join(fixed_bits)
        return title

    steps = []
    total_traces = len(left["omega_values"]) * traces_per_slice
    for i, omega_left in enumerate(left["omega_values"]):
        visible = [False] * total_traces
        start = i * traces_per_slice
        for j in range(traces_per_slice):
            visible[start + j] = True
        steps.append(dict(
            method="update",
            args=[{"visible": visible}, {"title": _title(i)}],
            label=f"{omega_left:.3g}",
        ))

    fig.update_layout(
        title=_title(0),
        width=1320,
        height=1080,
        margin=dict(l=70, r=130, t=120, b=105),
        sliders=[dict(
            active=0,
            currentvalue={"prefix": f"{label_left} omega: ", "font": {"size": 14}},
            pad={"t": 50},
            steps=steps,
        )],
    )

    for col in (1, 2):
        fig.update_xaxes(title_text="kx", row=1, col=col)
        fig.update_yaxes(title_text="Eigenvalue", range=[eig_vmin - eig_pad, eig_vmax + eig_pad], row=1, col=col)
        fig.update_xaxes(title_text="kx", row=2, col=col)
        fig.update_yaxes(title_text="ky", scaleanchor=f"x{2 + col}", scaleratio=1, row=2, col=col)
        fig.update_xaxes(title_text="kx", row=3, col=col)
        fig.update_yaxes(title_text="ky", scaleanchor=f"x{4 + col}", scaleratio=1, row=3, col=col)

    def _fmt_val(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def _row(key, val):
        return f'<div class="row"><span class="key">{key}</span><span class="val">{val}</span></div>'

    def _section(title, rows_html):
        return f'<div class="section"><div class="section-title">{title}</div>{rows_html}</div>'

    def _panel(data, label, color):
        pjson = data["pjson"]
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
        if data["fixed_str"]:
            kgrid_rows += _row("fixed axes", data["fixed_str"])

        return f"""
<div class="panel">
  <div class="panel-header" style="background:{color};">{data["ham_name"]} - {label}</div>
  {_section("Hamiltonian Parameters", ham_rows)}
  {_section("Sweep Range", scan_rows)}
  {_section("k-Grid", kgrid_rows)}
</div>"""

    sidebar_html = f"""
<div class="sidebar">
  {_panel(left, label_left, "#34405f")}
  {_panel(right, label_right, "#5d486d")}
</div>"""

    if output_html is None:
        output_html = os.path.join(left["dataset_path"], "dynamic_2d_trace_hprime_eigs_vs_omega_dual.html")

    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Dual Trace Hprime Eigenvalues vs omega</title>
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
  flex: 0 0 300px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  align-self: flex-start;
  margin-top: 8px;
}}
.panel {{
  background: #ffffff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  overflow: hidden;
}}
.panel-header {{
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
  font-size: 12px;
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
    print(f"Saved dual omega HTML to: {output_html}")
    return output_html



#@ 2D QGT ND joined omega HTML plots
def dynamic_2d_qgt_vs_omega_jointed_html(
    left_dataset_path,
    right_dataset_path,
    *,
    quantity="trace",
    bands_to_plot=None,
    omega_min_left=None,
    omega_max_left=None,
    omega_min_right=None,
    omega_max_right=None,
    symmetric_cbar=None,
    drop_overlap=True,
    tol=1e-9,
    cmap="inferno",
    berry_zlim=None,
    berry_zlim_percentile=99.0,
    label_left="left",
    label_right="right",
    output_html=None,
):
    """
    HTML version of a joined omega sweep from two N-D QGT bundles.

    The left dataset is sorted ascending in omega; the right dataset is sorted
    descending and appended. This gives a single slider for left/right circular
    polarization comparisons across omega.
    """
    sweep_param = "omega"

    def _resolve(p):
        return p if os.path.isabs(p) else os.path.join(os.getcwd(), p)

    def _load(dataset_path):
        dataset_path = _resolve(dataset_path)
        bundle_path = os.path.join(dataset_path, "qgt_nd_bundle.npz")
        json_path = os.path.join(dataset_path, "parameters.json")
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"ND bundle not found: {bundle_path}")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"parameters.json not found: {json_path}")
        bundle = np.load(bundle_path, allow_pickle=True)
        with open(json_path, "r", encoding="utf-8") as f:
            pjson = json.load(f)
        return bundle, pjson, dataset_path

    def _get_omega_axis(bundle):
        names = [str(n) for n in bundle["names"]]
        if sweep_param not in names:
            raise ValueError(f"'omega' not in bundle names {names}")
        axis = names.index(sweep_param)
        return axis, np.asarray(bundle[f"axis_{axis}_{sweep_param}"], dtype=float), list(bundle["shape"])

    def _middle_indexer(sweep_axis, shape):
        return tuple(
            slice(None) if ax == sweep_axis else shape[ax] // 2
            for ax in range(len(shape))
        )

    def _field_series(bundle, sweep_axis, shape, q):
        mid_idx = tuple(
            slice(None) if ax == sweep_axis else shape[ax] // 2
            for ax in range(len(shape))
        )
        if q == "trace":
            return np.asarray(bundle["trace_grid"][mid_idx])
        if q == "berry":
            return -2.0 * np.asarray(bundle["g_xy_imag_grid"][mid_idx])
        if q == "imqxy":
            return np.asarray(bundle["g_xy_imag_grid"][mid_idx])
        raise ValueError(f"Unknown quantity '{q}'.")

    def _sym_series(bundle, sweep_axis, shape):
        if "eigenvalues_sym_grid" not in bundle:
            return None
        return np.asarray(bundle["eigenvalues_sym_grid"][_middle_indexer(sweep_axis, shape)])

    def _filter(param_values, field_series, sym_series, pmin, pmax):
        mask = np.ones(len(param_values), dtype=bool)
        if pmin is not None:
            mask &= param_values >= pmin
        if pmax is not None:
            mask &= param_values <= pmax
        sym_filtered = None if sym_series is None else sym_series[mask]
        return param_values[mask], field_series[mask], sym_filtered

    bundle_L, pjson_L, dpath_L = _load(left_dataset_path)
    bundle_R, pjson_R, _ = _load(right_dataset_path)

    kx_L, ky_L = bundle_L["kx"], bundle_L["ky"]
    kx_R, ky_R = bundle_R["kx"], bundle_R["ky"]
    if kx_L.shape != kx_R.shape or not np.allclose(kx_L, kx_R):
        raise ValueError("kx grids differ between the two datasets; cannot join.")
    if ky_L.shape != ky_R.shape or not np.allclose(ky_L, ky_R):
        raise ValueError("ky grids differ between the two datasets; cannot join.")
    kx, ky = kx_L, ky_L

    q = quantity.lower()
    label_q = {
        "trace": "QGT Trace",
        "berry": "Berry Curvature Ω<sub>xy</sub>",
        "imqxy": "Im(Q<sub>xy</sub>)",
    }[q]
    document_title = label_q.replace("<sub>", "_").replace("</sub>", "")

    ax_L, omega_L, shape_L = _get_omega_axis(bundle_L)
    ax_R, omega_R, shape_R = _get_omega_axis(bundle_R)
    field_L = _field_series(bundle_L, ax_L, shape_L, q)
    field_R = _field_series(bundle_R, ax_R, shape_R, q)
    sym_L = _sym_series(bundle_L, ax_L, shape_L)
    sym_R = _sym_series(bundle_R, ax_R, shape_R)

    omega_L, field_L, sym_L = _filter(omega_L, field_L, sym_L, omega_min_left, omega_max_left)
    omega_R, field_R, sym_R = _filter(omega_R, field_R, sym_R, omega_min_right, omega_max_right)
    if len(omega_L) == 0 or len(omega_R) == 0:
        raise ValueError("No omega slices after filtering for one or both datasets.")

    order_L = np.argsort(omega_L)
    omega_L = omega_L[order_L]
    field_L = field_L[order_L]
    if sym_L is not None:
        sym_L = sym_L[order_L]

    order_R = np.argsort(omega_R)
    omega_R_sorted = omega_R[order_R]
    field_R_sorted = field_R[order_R]
    sym_R_sorted = None if sym_R is None else sym_R[order_R]
    omega_R_rev = omega_R_sorted[::-1]
    field_R_rev = field_R_sorted[::-1]
    sym_R_rev = None if sym_R_sorted is None else sym_R_sorted[::-1]

    n_left = len(omega_L)
    if drop_overlap and len(omega_R_sorted) and np.isclose(omega_L[-1], omega_R_sorted[-1], atol=tol, rtol=0):
        omega_R_rev = omega_R_rev[1:]
        field_R_rev = field_R_rev[1:]
        if sym_R_rev is not None:
            sym_R_rev = sym_R_rev[1:]

    omega_join = np.concatenate([omega_L, omega_R_rev])
    field_join = np.concatenate([field_L, field_R_rev], axis=0)
    sym_join = None
    if sym_L is not None and sym_R_rev is not None:
        sym_join = np.concatenate([sym_L, sym_R_rev], axis=0)

    if symmetric_cbar is None:
        symmetric_cbar = q != "trace"
    finite = field_join[np.isfinite(field_join)]
    vmin_data, vmax_data = float(np.nanmin(finite)), float(np.nanmax(finite))
    if q == "berry":
        symmetric_limits = get_symmetric_plot_limits(
            finite,
            berry_zlim,
            berry_zlim_percentile,
        )
        if symmetric_cbar:
            vmin, vmax = symmetric_limits
        else:
            clipped_vmin = max(vmin_data, symmetric_limits[0])
            clipped_vmax = min(vmax_data, symmetric_limits[1])
            if clipped_vmin < clipped_vmax:
                vmin, vmax = clipped_vmin, clipped_vmax
            else:
                vmin, vmax = vmin_data, vmax_data
    elif symmetric_cbar:
        amax = max(abs(vmin_data), abs(vmax_data))
        vmin, vmax = -amax, amax
    else:
        vmin, vmax = vmin_data, vmax_data

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.28, 0.72],
        vertical_spacing=0.10,
        subplot_titles=("Symmetry-line eigenvalues", label_q),
    )

    if sym_join is not None:
        nbands_total = int(sym_join.shape[-1])
        if bands_to_plot is None:
            bands = list(range(nbands_total))
        else:
            bands = [int(b) for b in bands_to_plot if 0 <= int(b) < nbands_total]
            if not bands:
                raise ValueError(f"No valid bands_to_plot for {nbands_total} bands.")
        eig_finite = sym_join[..., bands]
        eig_finite = eig_finite[np.isfinite(eig_finite)]
        eig_vmin = float(np.nanmin(eig_finite)) if eig_finite.size else -1.0
        eig_vmax = float(np.nanmax(eig_finite)) if eig_finite.size else 1.0
        eig_pad = 0.05 * (eig_vmax - eig_vmin) if eig_vmax > eig_vmin else 1.0
    else:
        bands = []
        eig_vmin, eig_vmax, eig_pad = -1.0, 1.0, 1.0

    traces_per_slice = len(bands) + 1
    k_dist = np.asarray(bundle_L["k_dist"]) if "k_dist" in bundle_L else None
    for i, omega_val in enumerate(omega_join):
        src = label_left if i < n_left else label_right
        if sym_join is not None and k_dist is not None:
            for band in bands:
                fig.add_trace(
                    go.Scatter(
                        x=k_dist.tolist(),
                        y=sym_join[i, :, band].tolist(),
                        mode="lines",
                        line=dict(width=1.3),
                        name=f"band {band}",
                        legendgroup=f"band-{band}",
                        showlegend=True,
                        visible=(i == 0),
                        hovertemplate="k-path: %{x:.4g}<br>E: %{y:.4g}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        fig.add_trace(go.Heatmap(
            z=field_join[i].tolist(),
            x=kx[0, :].tolist(),
            y=ky[:, 0].tolist(),
            colorscale=cmap,
            zmin=vmin,
            zmax=vmax,
            visible=(i == 0),
            name=f"omega={omega_val:.4g} ({src})",
            colorbar=dict(title=label_q, thickness=18, len=0.60, y=0.32),
            hovertemplate="k<sub>x</sub>: %{x:.3f}<br>k<sub>y</sub>: %{y:.3f}<br>Value: %{z:.4g}<extra></extra>",
        ), row=2, col=1)

    steps = []
    for i, omega_val in enumerate(omega_join):
        src = label_left if i < n_left else label_right
        visible = [False] * (len(omega_join) * traces_per_slice)
        start = i * traces_per_slice
        for j in range(traces_per_slice):
            visible[start + j] = True
        steps.append(dict(
            method="update",
            args=[
                {"visible": visible},
                {"title": f"{label_q}<br>omega = {omega_val:.6g} | {src}"},
            ],
            label=f"{omega_val:.3g}",
        ))

    fig.update_layout(
        title=f"{label_q}<br>omega = {omega_join[0]:.6g} | {label_left}",
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "omega: ", "font": {"size": 14}},
            pad={"t": 50},
            steps=steps,
        )],
        width=780,
        height=860,
        margin=dict(l=70, r=95, t=105, b=90),
    )

    fig.update_yaxes(title_text="Eigenvalue", range=[eig_vmin - eig_pad, eig_vmax + eig_pad], row=1, col=1)
    if k_dist is not None and "node_indices" in bundle_L and "path_labels" in bundle_L:
        node_indices = np.asarray(bundle_L["node_indices"], dtype=int)
        labels = [str(x) for x in bundle_L["path_labels"]]
        tickvals = k_dist[node_indices]
        fig.update_xaxes(tickmode="array", tickvals=tickvals.tolist(), ticktext=labels, row=1, col=1)
        shapes = []
        for x in tickvals:
            shapes.append(dict(
                type="line",
                xref="x",
                yref="y domain",
                x0=float(x),
                x1=float(x),
                y0=0,
                y1=1,
                line=dict(color="rgba(80,80,80,0.35)", width=1, dash="dot"),
            ))
        fig.update_layout(shapes=shapes)
    else:
        fig.update_xaxes(title_text="k-path", row=1, col=1)

    fig.update_xaxes(title_text="k<sub>x</sub>", row=2, col=1)
    fig.update_yaxes(title_text="k<sub>y</sub>", scaleanchor="x2", scaleratio=1, row=2, col=1)

    def _sidebar_panel(pjson, title_label, title_color):
        def _fv(v):
            return f"{v:.6g}" if isinstance(v, float) else str(v)

        def _row(k, v):
            return f'<div class="row"><span class="key">{k}</span><span class="val">{v}</span></div>'

        def _sec(t, rows):
            return f'<div class="section"><div class="section-title">{t}</div>{rows}</div>'

        ham_rows = "".join(_row(k, _fv(v)) for k, v in sorted(pjson["parameters"].items()))
        scan_ranges = pjson["scan_ranges"]
        scan_spacing = pjson["scan_spacing"]
        scan_rows = "".join(
            _row(n, f'{_fv(info["min"])} to {_fv(info["max"])}'
                 + f' ({scan_spacing[n]["count"]} pts, {scan_spacing[n]["scale"]})')
            for n, info in scan_ranges.items()
        )
        kg = pjson["k_grid"]
        grid_rows = (
            _row("k<sub>x</sub>", f'{_fv(kg["kx_min"])} to {_fv(kg["kx_max"])}')
            + _row("k<sub>y</sub>", f'{_fv(kg["ky_min"])} to {_fv(kg["ky_max"])}')
            + _row("mesh", str(kg["mesh"]))
        )
        band_row = _row("band index", str(pjson.get("band_index", "unknown")))

        return (
            '<div class="panel">'
            f'<div class="panel-head" style="background:{title_color}">'
            f'{pjson["hamiltonian_name"]} - {title_label}</div>'
            + _sec("Hamiltonian Parameters", ham_rows)
            + _sec("Sweep Range", scan_rows)
            + _sec("k-Grid", grid_rows + band_row)
            + '</div>'
        )

    sidebar_html = (
        '<div class="sidebar">'
        + _sidebar_panel(pjson_L, label_left, "#3b4fa8")
        + _sidebar_panel(pjson_R, label_right, "#7b3fa8")
        + '</div>'
    )

    if output_html is None:
        output_html = os.path.join(dpath_L, f"dynamic_2d_{q}_vs_omega_jointed.html")

    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{document_title} vs omega jointed</title>
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
      border-radius: 8px;
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
    print(f"Saved jointed omega HTML to: {output_html}")
    return output_html


#@ Rhombohedral Graphene Hamiltonian

#! V = 30
# New 2D_QGT_ND structure:
# dynamic_2d_trace_hprime_eigs_vs_omega_nd(
#     "results/2D_QGT_ND/ChiralHamiltonianChiralBasisProjected/dataset1"
# )

#@ Full ChiralHamiltonian left/right joint omega HTML plot
dynamic_2d_qgt_vs_omega_jointed_html(
    "results/2D_QGT_ND/ChiralHamiltonian/dataset1",
    "results/2D_QGT_ND/ChiralHamiltonian/dataset3",
    quantity="trace",
    omega_min_left=50,
    omega_min_right=30,
    label_left="left drive",
    label_right="right drive",
    bands_to_plot=[3,4,5,6],
)

#@ New ND dual HTML: most recent ChiralBasisProjected omega sweeps
# dynamic_2d_trace_hprime_eigs_vs_omega_dual(
#     "results/2D_QGT_ND/ChiralHamiltonianChiralBasisProjected/dataset3",
#     "results/2D_QGT_ND/ChiralHamiltonianChiralBasisProjected/dataset5",
#     label_left="Numerical Direct Drive",
#     label_right="Analytic Direct Drive",
# )
