import json
import os
import pickle
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import Library.Hamiltonian.Hamiltonian
from Library.Hamiltonian.ChiralHamiltonian import ChiralHamiltonian
from Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected import (
    ChiralHamiltonianChiralBasisProjected,
)
from Library.plotting_utils import bz_wigner_seitz_from_bvecs, get_bvecs_from_meta


# Backward compatibility for old pickled Hamiltonians.
sys.modules["Library.Hamiltonian_v2"] = Library.Hamiltonian.Hamiltonian
Library.Hamiltonian.Hamiltonian.ChiralHamiltonian = ChiralHamiltonian
Library.Hamiltonian.Hamiltonian.RhombohedralGrapheneHamiltonian = (
    ChiralHamiltonianChiralBasisProjected
)


def _resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)


def _load_nd_bundle(root_dir):
    root_dir = _resolve_path(root_dir)
    bundle_path = os.path.join(root_dir, "qgt_nd_bundle.npz")
    meta_path = os.path.join(root_dir, "meta.pkl")
    json_path = os.path.join(root_dir, "parameters.json")

    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Cannot find bundle at: {bundle_path}")

    data = np.load(bundle_path, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    shape = tuple(int(x) for x in data["shape"])
    axes = [np.asarray(data[f"axis_{i}_{names[i]}"]) for i in range(len(names))]
    kx = np.asarray(data["kx"])
    ky = np.asarray(data["ky"])

    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

    params_json = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            params_json = json.load(f)

    return root_dir, data, names, axes, shape, kx, ky, meta, params_json


def _pick_field_grid(data, quantity="trace", *, convert_berry_from_imQ=True):
    q = quantity.lower()

    if q == "trace":
        return np.asarray(data["trace_grid"])

    if q in ("berry", "berry_curvature", "omega"):
        if "berry_grid" in data.files:
            return np.asarray(data["berry_grid"])
        gxyi = np.asarray(data["g_xy_imag_grid"])
        return (-2.0 * gxyi) if convert_berry_from_imQ else gxyi

    if q in ("imqxy", "im(q_xy)", "im_qxy"):
        return np.asarray(data["g_xy_imag_grid"])

    if q in ("trace_minus_berry", "trace_minus_omega"):
        trace = np.asarray(data["trace_grid"])
        if "berry_grid" in data.files:
            berry = np.asarray(data["berry_grid"])
        else:
            gxyi = np.asarray(data["g_xy_imag_grid"])
            berry = (-2.0 * gxyi) if convert_berry_from_imQ else gxyi
        return trace - berry

    raise ValueError(f"Unknown quantity '{quantity}'.")


def _label_from_quantity(quantity):
    return {
        "trace": "QGT Trace",
        "berry": "Berry Curvature Omega",
        "berry_curvature": "Berry Curvature Omega",
        "omega": "Berry Curvature Omega",
        "imqxy": "Im(Q_xy)",
        "im(q_xy)": "Im(Q_xy)",
        "im_qxy": "Im(Q_xy)",
        "trace_minus_berry": "Tr(g) - Omega",
        "trace_minus_omega": "Tr(g) - Omega",
    }.get(quantity.lower(), "Field")


def _json_script_value(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return str(v)


def _row(key, val):
    return f'<div class="row"><span class="key">{key}</span><span class="val">{val}</span></div>'


def _section(title, rows):
    return f'<div class="section"><div class="section-title">{title}</div>{rows}</div>'


def _sidebar_html(params_json, names, axes, band_index_text, band_cut_text):
    ham_name = params_json.get("hamiltonian_name", "ND QGT Bundle")

    ham_rows = "".join(
        _row(k, _fmt(v))
        for k, v in sorted(params_json.get("parameters", {}).items())
    )
    if not ham_rows:
        ham_rows = _row("source", "qgt_nd_bundle.npz")

    scan_ranges = params_json.get("scan_ranges", {})
    scan_spacing = params_json.get("scan_spacing", {})
    scan_rows = ""
    for name in names:
        if name in scan_ranges:
            info = scan_ranges[name]
            spacing = scan_spacing.get(name, {})
            scan_rows += _row(
                name,
                f'{_fmt(info.get("min"))} to {_fmt(info.get("max"))}'
                + f' ({spacing.get("count", len(axes[names.index(name)]))} pts, '
                + f'{spacing.get("scale", "unknown")})',
            )
        else:
            scan_rows += _row(name, f"{len(axes[names.index(name)])} pts")

    kg = params_json.get("k_grid", {})
    if kg:
        grid_rows = (
            _row("kx", f'{_fmt(kg.get("kx_min"))} to {_fmt(kg.get("kx_max"))}')
            + _row("ky", f'{_fmt(kg.get("ky_min"))} to {_fmt(kg.get("ky_max"))}')
            + _row("mesh", kg.get("mesh"))
        )
    else:
        grid_rows = _row("grid", "from bundle")

    return f"""
<aside class="sidebar">
  <div class="sidebar-header">{ham_name}</div>
  {_section("Hamiltonian Parameters", ham_rows)}
  {_section("Sweep Axes", scan_rows)}
  {_section("k-Grid", grid_rows + _row("band index", band_index_text) + _row("band cut", band_cut_text))}
  <div class="section">
    <div class="section-title">Controls</div>
    <div id="controls"></div>
  </div>
</aside>
"""


def dynamic_nd_field_with_bands_html(
    root_dir,
    *,
    quantity="trace",
    bands_to_plot=None,
    convert_berry_from_imQ=True,
    cmap="inferno",
    symmetric_cbar=None,
    title=None,
    show_integral=True,
    output_html=None,
    trace_zmax_percentile=99.5,
):
    """
    Write a Plotly/HTML version of the N-D multiparameter QGT viewer.

    This mirrors the Matplotlib ``dynamic_nd_field_with_bands`` workflow:
    a top symmetry-line eigenvalue panel when available, a bottom 2D field
    heatmap, BZ/path overlays when available, and one slider per parameter.
    """
    (
        root_dir,
        data,
        names,
        axes,
        shape,
        kx,
        ky,
        meta,
        params_json,
    ) = _load_nd_bundle(root_dir)

    field_grid = _pick_field_grid(
        data, quantity, convert_berry_from_imQ=convert_berry_from_imQ
    )
    q = quantity.lower()
    label_q = title or _label_from_quantity(quantity)

    dkx = float(data["dkx"]) if "dkx" in data.files else float(kx[0, 1] - kx[0, 0])
    dky = float(data["dky"]) if "dky" in data.files else float(ky[1, 0] - ky[0, 0])
    area_element = dkx * dky

    if symmetric_cbar is None:
        symmetric_cbar = q != "trace"

    finite = field_grid[np.isfinite(field_grid)]
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if q == "trace" and trace_zmax_percentile is not None:
        vmax = float(np.nanpercentile(finite, trace_zmax_percentile))
        if vmax <= vmin:
            vmax = float(np.nanmax(finite))
    if symmetric_cbar:
        amax = max(abs(vmin), abs(vmax))
        vmin, vmax = -amax, amax

    init_idx = tuple(ax.size // 2 for ax in axes)
    z0 = field_grid[init_idx + (slice(None), slice(None))]

    has_band_cut = all(
        key in data.files
        for key in ("eigenvalues_sym_grid", "k_dist", "node_indices", "path_labels")
    )
    if has_band_cut:
        eigenvalues_sym_grid = np.asarray(data["eigenvalues_sym_grid"])
        k_dist = np.asarray(data["k_dist"], dtype=float)
        node_indices = np.asarray(data["node_indices"], dtype=int)
        path_labels = [str(x) for x in np.asarray(data["path_labels"])]
        num_bands = int(eigenvalues_sym_grid.shape[-1])

        if bands_to_plot is None:
            bands = list(range(num_bands))
        elif isinstance(bands_to_plot, int):
            bands = [bands_to_plot]
        else:
            bands = list(bands_to_plot)
        bad = [b for b in bands if not (0 <= b < num_bands)]
        if bad:
            raise IndexError(f"Out-of-range bands {bad}; valid range is [0,{num_bands - 1}]")

        ev0 = eigenvalues_sym_grid[init_idx + (slice(None), slice(None))]
        ev_selected = eigenvalues_sym_grid[..., bands]
        ev_min = float(np.nanmin(ev_selected))
        ev_max = float(np.nanmax(ev_selected))
        ev_pad = 0.05 * (ev_max - ev_min) if ev_max > ev_min else 1.0
    else:
        eigenvalues_sym_grid = None
        k_dist = np.asarray([0.0, 1.0])
        node_indices = np.asarray([], dtype=int)
        path_labels = []
        bands = []
        ev0 = None
        ev_min, ev_max, ev_pad = -1.0, 1.0, 0.0

    has_path_overlay = "path_points" in data.files and "path_labels" in data.files
    path_points = np.asarray(data["path_points"], dtype=float) if has_path_overlay else None

    row_heights = [0.25, 0.75]
    vertical_spacing = 0.10
    heatmap_domain_height = (1.0 - vertical_spacing) * row_heights[1] / sum(row_heights)
    heatmap_domain_center = 0.5 * heatmap_domain_height

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=row_heights,
        vertical_spacing=vertical_spacing,
        subplot_titles=("Symmetry-line eigenvalues", label_q),
    )

    if has_band_cut:
        for b in bands:
            fig.add_trace(
                go.Scatter(
                    x=k_dist.tolist(),
                    y=ev0[:, b].tolist(),
                    mode="lines",
                    line=dict(width=1.4),
                    name=f"band {b}",
                    hovertemplate="k: %{x:.4g}<br>E: %{y:.4g}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        for idx in node_indices:
            fig.add_vline(x=float(k_dist[idx]), line_width=1, line_dash="dash", line_color="rgba(0,0,0,0.35)", row=1, col=1)
    else:
        fig.add_annotation(
            text="Symmetry-line eigenvalues unavailable",
            x=0.5,
            y=0.88,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="rgba(70,70,70,0.75)"),
        )

    heatmap_trace_index = len(fig.data)
    fig.add_trace(
        go.Heatmap(
            z=z0.tolist(),
            x=kx[0, :].tolist(),
            y=ky[:, 0].tolist(),
            colorscale=cmap,
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(
                title=label_q,
                thickness=18,
                len=heatmap_domain_height,
                y=heatmap_domain_center,
                yanchor="middle",
            ),
            hovertemplate="kx: %{x:.4g}<br>ky: %{y:.4g}<br>Value: %{z:.4g}<extra></extra>",
            name=label_q,
        ),
        row=2,
        col=1,
    )

    b1, b2 = get_bvecs_from_meta(meta)
    bx, by = bz_wigner_seitz_from_bvecs(b1, b2)
    fig.add_trace(
        go.Scatter(
            x=bx.tolist(),
            y=by.tolist(),
            mode="lines",
            line=dict(color="white", width=2),
            name="BZ",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    if has_path_overlay:
        fig.add_trace(
            go.Scatter(
                x=path_points[:, 0].tolist(),
                y=path_points[:, 1].tolist(),
                mode="lines",
                line=dict(color="red", width=1.5, dash="dash"),
                name="Sym Path",
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

        label_indices = node_indices if has_band_cut else np.arange(min(len(path_labels), len(path_points)))
        label_indices = label_indices[label_indices < len(path_points)]
        if len(label_indices) and path_labels:
            text = [path_labels[i] if i < len(path_labels) else "" for i in range(len(label_indices))]
            fig.add_trace(
                go.Scatter(
                    x=path_points[label_indices, 0].tolist(),
                    y=path_points[label_indices, 1].tolist(),
                    mode="markers+text",
                    marker=dict(color="red", size=7),
                    text=text,
                    textposition="top right",
                    name="Sym Points",
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )

    if show_integral and q in ("trace_minus_berry", "trace_minus_omega"):
        integral0 = float(np.nansum(z0) * area_element)
        extra0 = f" | integral={integral0:.6g}"
    else:
        extra0 = ""

    init_label = ", ".join(f"{names[i]}={axes[i][init_idx[i]]:.6g}" for i in range(len(names)))
    fig.update_layout(
        title=f"{label_q}<br>{init_label} | std={float(np.nanstd(z0)):.3e}{extra0}",
        width=900,
        height=900,
        margin=dict(l=70, r=110, t=100, b=80),
        showlegend=True,
    )
    fig.update_xaxes(title_text="k path", row=1, col=1)
    fig.update_yaxes(title_text="Eigenvalue", range=[ev_min - ev_pad, ev_max + ev_pad], row=1, col=1)
    if has_band_cut and len(node_indices):
        tick_vals = [float(k_dist[i]) for i in node_indices]
        tick_text = [path_labels[i] if i < len(path_labels) else "" for i in range(len(tick_vals))]
        fig.update_xaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_text, row=1, col=1)
    fig.update_xaxes(title_text="kx", row=2, col=1)
    fig.update_yaxes(title_text="ky", scaleanchor="x2", scaleratio=1, row=2, col=1)

    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="qgt-plot")

    axes_json = [axis.astype(float).tolist() for axis in axes]
    band_cut_text = "available" if has_band_cut else "unavailable"
    band_index = params_json.get("band_index")
    if band_index is None and isinstance(meta, dict):
        band_index = meta.get("band")
    band_index_text = _fmt(band_index) if band_index is not None else "unknown"
    sidebar = _sidebar_html(params_json, names, axes, band_index_text, band_cut_text)

    if output_html is None:
        output_html = os.path.join(root_dir, f"dynamic_nd_{q}_multipara.html")

    js_band_data = (
        _json_script_value(eigenvalues_sym_grid.tolist()) if has_band_cut else "null"
    )
    js_k_dist = _json_script_value(k_dist.astype(float).tolist())
    js_bands = _json_script_value(bands)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{label_q} multiparameter sweep</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  font-family: 'Segoe UI', Arial, sans-serif;
  background: #f0f2f7;
  margin: 0;
  padding: 24px;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  gap: 22px;
}}
.plot-wrapper {{ flex: 0 0 auto; }}
.sidebar {{
  flex: 0 0 300px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  overflow: hidden;
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
  gap: 10px;
  padding: 3px 0;
  font-size: 12px;
  border-bottom: 1px solid #f3f5fb;
}}
.row:last-child {{ border-bottom: none; }}
.key {{ color: #4a5568; font-weight: 600; white-space: nowrap; }}
.val {{ color: #2d3748; text-align: right; font-variant-numeric: tabular-nums; word-break: break-all; }}
.control {{ margin-bottom: 12px; }}
.control-head {{
  display: flex;
  justify-content: space-between;
  gap: 10px;
  font-size: 12px;
  margin-bottom: 4px;
}}
.control label {{ color: #4a5568; font-weight: 700; }}
.control output {{ color: #2d3748; font-variant-numeric: tabular-nums; text-align: right; }}
input[type=range] {{ width: 100%; }}
</style>
</head>
<body>
<div class="plot-wrapper">{plot_html}</div>
{sidebar}
<script>
const fieldData = {_json_script_value(field_grid.tolist())};
const eigenData = {js_band_data};
const names = {_json_script_value(names)};
const axes = {_json_script_value(axes_json)};
const bands = {js_bands};
const heatmapTraceIndex = {heatmap_trace_index};
const bandTraceStart = 0;
const hasBandCut = {str(has_band_cut).lower()};
const areaElement = {area_element:.17g};
const quantity = {_json_script_value(q)};
const labelQ = {_json_script_value(label_q)};

let current = axes.map(axis => Math.floor(axis.length / 2));

function nestedGet(arr, idxs) {{
  let out = arr;
  for (const idx of idxs) out = out[idx];
  return out;
}}

function formatValue(value) {{
  if (typeof value !== 'number') return String(value);
  return Number(value).toPrecision(6).replace(/\\.?0+$/, '');
}}

function calcStd(matrix) {{
  let n = 0, sum = 0, sum2 = 0;
  for (const row of matrix) {{
    for (const value of row) {{
      if (Number.isFinite(value)) {{
        n += 1;
        sum += value;
        sum2 += value * value;
      }}
    }}
  }}
  if (!n) return NaN;
  const mean = sum / n;
  return Math.sqrt(Math.max(0, sum2 / n - mean * mean));
}}

function calcIntegral(matrix) {{
  let sum = 0;
  for (const row of matrix) {{
    for (const value of row) {{
      if (Number.isFinite(value)) sum += value;
    }}
  }}
  return sum * areaElement;
}}

function titleFor(matrix) {{
  const params = names.map((name, i) => `${{name}}=${{formatValue(axes[i][current[i]])}}`).join(', ');
  let extra = `std=${{calcStd(matrix).toExponential(3)}}`;
  if (quantity === 'trace_minus_berry' || quantity === 'trace_minus_omega') {{
    extra += ` | integral=${{formatValue(calcIntegral(matrix))}}`;
  }}
  return `${{labelQ}}<br>${{params}} | ${{extra}}`;
}}

function updatePlot() {{
  const z = nestedGet(fieldData, current);
  const update = {{ z: [z] }};
  Plotly.restyle('qgt-plot', update, [heatmapTraceIndex]);

  if (hasBandCut && eigenData !== null) {{
    const ev = nestedGet(eigenData, current);
    bands.forEach((band, j) => {{
      const y = ev.map(row => row[band]);
      Plotly.restyle('qgt-plot', {{ y: [y] }}, [bandTraceStart + j]);
    }});
  }}

  Plotly.relayout('qgt-plot', {{ title: {{ text: titleFor(z) }} }});
}}

function buildControls() {{
  const root = document.getElementById('controls');
  names.forEach((name, i) => {{
    const wrapper = document.createElement('div');
    wrapper.className = 'control';

    const head = document.createElement('div');
    head.className = 'control-head';

    const label = document.createElement('label');
    label.textContent = name;

    const output = document.createElement('output');
    output.id = `control-output-${{i}}`;
    output.textContent = formatValue(axes[i][current[i]]);

    head.appendChild(label);
    head.appendChild(output);

    const input = document.createElement('input');
    input.type = 'range';
    input.min = 0;
    input.max = axes[i].length - 1;
    input.step = 1;
    input.value = current[i];
    input.addEventListener('input', () => {{
      current[i] = Number(input.value);
      output.textContent = formatValue(axes[i][current[i]]);
      updatePlot();
    }});

    wrapper.appendChild(head);
    wrapper.appendChild(input);
    root.appendChild(wrapper);
  }});
}}

buildControls();
</script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Saved interactive multiparameter HTML to: {output_html}")
    return output_html


if __name__ == "__main__":
    # Example:
    dynamic_nd_field_with_bands_html(
        "results/2D_QGT_ND/ChiralHamiltonian/dataset2",
        # "results/QGT_ND/LegacyWithN>1D/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10-RANGES_V_-10.000_50.000-omega_50.000_5000.000_-SPACING_V_48_linear-omega_32_linear_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1",
        quantity="trace",
        bands_to_plot=[3, 4, 5, 6],
    )
    pass
