import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle
from Library.utilities import in_range
from Library.plotting_utils import load_qgt, filter_entries_by_omega
import sys
import Library.Hamiltonian.Hamiltonian
from Library.Hamiltonian.ChiralHamiltonian import ChiralHamiltonian
from Library.Hamiltonian.SquareLatticeHamiltonian import SquareLatticeHamiltonian
import Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected
from Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected import ChiralHamiltonianChiralBasisProjected

# Patch for unpickling old data that references old module names
sys.modules["Library.Hamiltonian_v2"] = Library.Hamiltonian.Hamiltonian
sys.modules["Library.Hamiltonian.Chiral_Hamiltonian_Projected"] = Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected

# Backward-compat patches so pickle can find classes that were moved/renamed.
# Old .pkl files record the class path at save-time; these lines re-inject them.
Library.Hamiltonian.Hamiltonian.ChiralHamiltonian = ChiralHamiltonian
Library.Hamiltonian.Hamiltonian.SquareLatticeHamiltonian = SquareLatticeHamiltonian
Library.Hamiltonian.Hamiltonian.RhombohedralGrapheneHamiltonian = ChiralHamiltonianChiralBasisProjected
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


def dynamic_2d_qgt_1d_sweep_html(
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
    import plotly.graph_objects as go
    import pickle

    # ---------- resolve path ----------
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.getcwd(), dataset_path)

    bundle_path = os.path.join(dataset_path, "qgt_nd_bundle.npz")
    meta_path   = os.path.join(dataset_path, "meta.pkl")

    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"ND bundle not found: {bundle_path}")

    # ---------- load bundle ----------
    bundle = np.load(bundle_path, allow_pickle=True)

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

    # ---------- hamiltonian name (from meta.pkl) ----------
    ham_name = "Unknown Hamiltonian"
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        ht = meta.get("Hamiltonian_Template")
        ham_name = getattr(ht, "name", ham_name) if ht is not None else ham_name

    # Fixed-param annotation for the plot title (all axes except sweep)
    fixed_labels = [
        f"{names[ax]}={param_values[0]:.4g}"  # placeholder; value doesn't vary
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

    # ---------- load parameters.json for sidebar ----------
    import json
    json_path = os.path.join(dataset_path, "parameters.json")
    sidebar_html = ""
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            pjson = json.load(f)

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

        # --- Hamiltonian parameters ---
        ham_rows = "".join(
            _row(k, _fmt_val(v))
            for k, v in sorted(pjson.get("parameters", {}).items())
        )

        # --- Scan ranges ---
        scan_ranges = pjson.get("scan_ranges", {})
        scan_spacing = pjson.get("scan_spacing", {})
        # Handle both old list-of-lists format and new dict format
        if isinstance(scan_ranges, list):
            range_items = {entry[0]: {"min": entry[1], "max": entry[2]} for entry in scan_ranges}
        else:
            range_items = scan_ranges
        scan_rows = "".join(
            _row(
                name,
                f'{_fmt_val(info["min"])} → {_fmt_val(info["max"])}'
                + (f'  ({scan_spacing[name]["count"]} pts, {scan_spacing[name]["scale"]})'
                   if name in scan_spacing else "")
            )
            for name, info in range_items.items()
        )

        # --- k-grid ---
        kg = pjson.get("k_grid", {})
        kgrid_rows = (
            _row("kx", f'{_fmt_val(kg.get("kx_min","?"))} → {_fmt_val(kg.get("kx_max","?"))}') +
            _row("ky", f'{_fmt_val(kg.get("ky_min","?"))} → {_fmt_val(kg.get("ky_max","?"))}') +
            _row("mesh", str(kg.get("mesh", "?")))
        )

        # --- band ---
        band_row = _row("band index", str(pjson["band_index"])) if "band_index" in pjson else ""

        sidebar_html = f"""
<div class="sidebar">
  <div class="sidebar-header">{pjson.get("hamiltonian_name", ham_name)}</div>
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
    full_html = f"""<!DOCTYPE html>
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


def dynamic_2d_trace_hprime_eigs_vs_omega(folder_name, omega_min=None, omega_max=None):
    """
    Layout:
      ┌───────────────────────────────────────────────┐
      │   (slim) eigenvalues vs k_x at k_y≈0          │  <-- 1 row
      ├───────────────────────────┬───────────────────┤
      │       QGT Trace (2D)      │  Re[H'(0,0)] (2D) │  <-- big 2x1
      └───────────────────────────┴───────────────────┘
                     [  ω index slider at bottom  ]

    Assumes each saved entry has:
      - 'trace' : (Ny, Nx)
      - 'eigenvalues' : (Ny, Nx, Nb)
      - 'hamiltonian_prime_array' or 'H_prime_array' : (Ny, Nx, dim, dim)
    """
    # ---- load data ----
    results_path = os.path.join(os.getcwd(), "results", "2D_QGT_omega_sweep", folder_name)
    qgt_data_path = os.path.join(results_path, "QGT_2.npy")  # sometimes people use this name
    if not os.path.exists(qgt_data_path):
        qgt_data_path = os.path.join(results_path, "QGT_2D.npy")
    meta_path = os.path.join(results_path, "meta_info.pkl")

    if not os.path.exists(qgt_data_path):
        raise FileNotFoundError(f"QGT data not found in '{results_path}'.")
    with open(meta_path, "rb") as f:
        meta_info = pickle.load(f)
    entries = np.load(qgt_data_path, allow_pickle=True)

    # ---- omega filtering ----
    def _in_range(w):
        if (omega_min is not None) and (w < omega_min): return False
        if (omega_max is not None) and (w > omega_max): return False
        return True

    entries = [e for e in entries if _in_range(float(e["omega"]))]
    if len(entries) == 0:
        raise ValueError("No omega slices fall within the requested range.")

    kx = np.asarray(meta_info["kx"])
    ky = np.asarray(meta_info["ky"])
    omega_values = np.array([float(e["omega"]) for e in entries], dtype=float)

    # choose ky index closest to 0 for the 1D eigenvalue slice
    ky_vals_1d = ky[:, 0] if ky.ndim == 2 else ky
    ky_idx = int(np.argmin(np.abs(ky_vals_1d - 0.0)))

    # get H' array key and precompute Re H'00 for all ω (and global color limits)
    def _get_hprime(entry):
        if "hamiltonian_prime_array" in entry:
            return np.asarray(entry["hamiltonian_prime_array"])
        if "H_prime_array" in entry:
            return np.asarray(entry["H_prime_array"])
        raise KeyError("Entry lacks 'hamiltonian_prime_array' / 'H_prime_array'.")

    hprime00_list = [np.real(_get_hprime(e)[..., 0, 0]) for e in entries]
    hprime_vmin = float(np.nanmin([np.nanmin(h) for h in hprime00_list]))
    hprime_vmax = float(np.nanmax([np.nanmax(h) for h in hprime00_list]))

    trace_vmin = float(np.nanmin([np.nanmin(e["trace"]) for e in entries]))
    trace_vmax = float(np.nanmax([np.nanmax(e["trace"]) for e in entries]))

    # initial index and data
    i0 = 0
    trace0 = np.asarray(entries[i0]["trace"])
    h00_0  = hprime00_list[i0]
    evals0 = np.asarray(entries[i0]["eigenvalues"])  # (Ny, Nx, Nb)

    # 1D eigenvalues vs kx (take the ky≈0 row and sort by kx)
    kx_line = kx[ky_idx, :] if kx.ndim == 2 else kx
    sort_idx = np.argsort(kx_line)
    kx_line_sorted = kx_line[sort_idx]
    eig_line0 = evals0[ky_idx, :, :]         # (Nx, Nb)
    eig_line0_sorted = eig_line0[sort_idx, :]
    nbands = eig_line0_sorted.shape[1]

    # ---- figure layout ----
    fig = plt.figure(figsize=(25, 13))
    # leave room for slider at bottom
    fig.subplots_adjust(bottom=0.11)

    # slim top bar for eigenvalues
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 7], width_ratios=[1, 1], hspace=0.35, wspace=0.20)

    ax_eigs   = fig.add_subplot(gs[0, :])   # spans both columns
    ax_trace  = fig.add_subplot(gs[1, 0])   # bottom-left
    ax_hprime = fig.add_subplot(gs[1, 1])   # bottom-right

    # --- top: eigenvalues vs kx (all bands) ---
    eig_lines = []
    for b in range(nbands):
        ln, = ax_eigs.plot(kx_line_sorted, eig_line0_sorted[:, b], lw=0.9)
        eig_lines.append(ln)
    ax_eigs.set_title(r"Eigenvalues vs $k_x$ (slice at $k_y \approx 0$)")
    ax_eigs.set_xlabel(r"$k_x$")
    ax_eigs.set_ylabel("Eigenvalue")
    ax_eigs.grid(True)

    # --- bottom-left: trace heatmap (+ std readout) ---
    im_trace = ax_trace.imshow(
        trace0,
        origin="lower",
        extent=[kx.min(), kx.max(), ky.min(), ky.max()],
        cmap="inferno",
        vmin=trace_vmin, vmax=trace_vmax,
        aspect="auto"
    )
    ax_trace.set_title(f'QGT Trace — ω = {omega_values[i0]:.6f}')
    ax_trace.set_xlabel(r"$k_x$")
    ax_trace.set_ylabel(r"$k_y$")
    cbar_trace = fig.colorbar(im_trace, ax=ax_trace, fraction=0.047, pad=0.03)
    cbar_trace.set_label("Trace")
    std_text = ax_trace.text(
        0.02, 0.98, f"std[Trace] = {np.nanstd(trace0):.4g}",
        transform=ax_trace.transAxes, va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )

    # --- bottom-right: Re[H'(0,0)] heatmap ---
    im_hprime = ax_hprime.imshow(
        h00_0,
        origin="lower",
        extent=[kx.min(), kx.max(), ky.min(), ky.max()],
        cmap="viridis",
        vmin=hprime_vmin, vmax=hprime_vmax,
        aspect="auto"
    )
    ax_hprime.set_title(r"$\mathrm{Re}\,H'_{00}$" + f" — ω = {omega_values[i0]:.6f}")
    ax_hprime.set_xlabel(r"$k_x$")
    ax_hprime.set_ylabel(r"$k_y$")
    cbar_hprime = fig.colorbar(im_hprime, ax=ax_hprime, fraction=0.047, pad=0.03)
    cbar_hprime.set_label(r"$\mathrm{Re}\,H'_{00}$")

    # ---- slider (bottom) ----
    ax_slider = plt.axes([0.16, 0.04, 0.68, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(ax_slider, r'$\omega$ idx', 0, len(omega_values) - 1, valinit=i0, valstep=1)

    # ---- update handler ----
    def _update(_):
        idx = int(slider.val)

        # eigenvalues
        ev = np.asarray(entries[idx]["eigenvalues"])   # (Ny, Nx, Nb)
        line = ev[ky_idx, :, :]                        # (Nx, Nb)
        line_sorted = line[sort_idx, :]
        nb = min(nbands, line_sorted.shape[1])
        for b in range(nb):
            eig_lines[b].set_ydata(line_sorted[:, b])

        # trace
        tr = np.asarray(entries[idx]["trace"])
        im_trace.set_data(tr)
        ax_trace.set_title(f'QGT Trace — ω = {omega_values[idx]:.6f}')
        std_text.set_text(f"std[Trace] = {np.nanstd(tr):.4g}")

        # H'00
        im_hprime.set_data(hprime00_list[idx])
        ax_hprime.set_title(r"$\mathrm{Re}\,H'_{00}$" + f" — ω = {omega_values[idx]:.6f}")

        fig.canvas.draw_idle()

    slider.on_changed(_update)
    plt.show()



def _get_hprime_array(entry):
    if "hamiltonian_prime_array" in entry:
        return np.asarray(entry["hamiltonian_prime_array"])
    if "H_prime_array" in entry:
        return np.asarray(entry["H_prime_array"])
    raise KeyError("Entry lacks 'hamiltonian_prime_array' / 'H_prime_array'.")


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
    h00_L_list = [np.real(_get_hprime_array(e)[..., 0, 0]) for e in entries_L]
    h00_R_list = [np.real(_get_hprime_array(e)[..., 0, 0]) for e in entries_R]

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


def _wrap_periodic(vals, vmin, vmax):
    L = vmax - vmin
    return (vals - vmin) % L + vmin


def _bilinear_sample(grid, x_coords, y_coords, xq, yq, periodic=False, oob_value=None):
    """
    Bilinear sampling on rectilinear axes x_coords (Nx) and y_coords (Ny).
    If oob_value is not None and periodic is False, samples outside the box
    are set to oob_value (e.g., np.nan).
    """
    x = np.asarray(x_coords)
    y = np.asarray(y_coords)
    Xq = np.asarray(xq)
    Yq = np.asarray(yq)

    xmin, xmax = x[0], x[-1]
    ymin, ymax = y[0], y[-1]
    nx, ny = x.size, y.size

    if periodic:
        Xq = _wrap_periodic(Xq, xmin, xmax)
        Yq = _wrap_periodic(Yq, ymin, ymax)

    # cell indices
    ix = np.clip(np.searchsorted(x, Xq) - 1, 0, nx-2)
    iy = np.clip(np.searchsorted(y, Yq) - 1, 0, ny-2)

    x0 = x[ix]; x1 = x[ix+1]
    y0 = y[iy]; y1 = y[iy+1]

    tx = np.where(x1 > x0, (Xq - x0)/(x1 - x0), 0.0)
    ty = np.where(y1 > y0, (Yq - y0)/(y1 - y0), 0.0)

    f00 = grid[ix,   iy  ]
    f10 = grid[ix+1, iy  ]
    f01 = grid[ix,   iy+1]
    f11 = grid[ix+1, iy+1]

    f0 = f00*(1-tx) + f10*tx
    f1 = f01*(1-tx) + f11*tx
    f  = f0*(1-ty) + f1*ty

    if (oob_value is not None) and (not periodic):
        inside = (Xq >= xmin) & (Xq <= xmax) & (Yq >= ymin) & (Yq <= ymax)
        f = np.where(inside, f, oob_value)

    return f


def _line_segment_in_box(x0, y0, theta_rad, xmin, xmax, ymin, ymax, periodic=False, max_len=None):
    """
    If max_len is provided, we return a segment [-max_len/2, +max_len/2] along the unit
    direction (dx,dy), centered at (x0,y0), irrespective of the box (we'll NaN outside later).
    Otherwise, we compute the intersection with the box (old behavior).
    """
    dx, dy = np.cos(theta_rad), np.sin(theta_rad)

    if max_len is not None:
        half = 0.5*max_len
        return -half, +half, dx, dy

    if periodic:
        box_diag = np.hypot(xmax-xmin, ymax-ymin) if max_len is None else max_len
        tmin, tmax = -0.5*box_diag, 0.5*box_diag
        return tmin, tmax, dx, dy

    # old: intersect with box
    t_candidates = []
    if abs(dx) > 1e-15:
        t_candidates += [(xmin - x0)/dx, (xmax - x0)/dx]
    if abs(dy) > 1e-15:
        t_candidates += [(ymin - y0)/dy, (ymax - y0)/dy]

    pts = []
    for t in t_candidates:
        x = x0 + t*dx; y = y0 + t*dy
        if (xmin-1e-12) <= x <= (xmax+1e-12) and (ymin-1e-12) <= y <= (ymax+1e-12):
            pts.append(t)
    if len(pts) < 2:
        return 0.0, 0.0, dx, dy
    return min(pts), max(pts), dx, dy


def slice_field_along_line(field_2d, kx, ky, angle_deg, shift_x=0.0, shift_y=0.0,
                           n_samples=400, periodic=False, max_len=None):
    """
    If max_len is provided, generates a fixed-length segment (in k-units).
    Returns physical arc-length s (same units as k), not normalized.
    """
    kx = np.asarray(kx); ky = np.asarray(ky)
    xmin, xmax = kx.min(), kx.max()
    ymin, ymax = ky.min(), ky.max()
    theta = np.deg2rad(angle_deg)

    tmin, tmax, dx, dy = _line_segment_in_box(
        shift_x, shift_y, theta, xmin, xmax, ymin, ymax,
        periodic=periodic, max_len=max_len
    )
    if tmax <= tmin:
        return np.array([]), np.array([]), np.array([]), np.array([])

    t = np.linspace(tmin, tmax, n_samples)  # |(dx,dy)|=1, so t is arc length
    kx_line = shift_x + t*dx
    ky_line = shift_y + t*dy

    if periodic:
        kx_line = _wrap_periodic(kx_line, xmin, xmax)
        ky_line = _wrap_periodic(ky_line, ymin, ymax)

    s = t  # physical arc length (units of k)

    # Use NaN for out-of-bounds when NOT periodic
    vals = _bilinear_sample(
        field_2d, kx, ky, kx_line, ky_line,
        periodic=periodic, oob_value=(None if periodic else np.nan)
    )
    return s, kx_line, ky_line, vals


def dynamic_2d_trace_with_line(folder_name, omega_min=None, omega_max=None,
                               angle_deg=45.0, shift_x=0.0, shift_y=0.0,
                               n_samples=400, periodic=False, k_length=None):
    """
    Like dynamic_2d_trace_vs_omega, but also:
      - overlays a line (angle+shift)
      - plots the 1D trace along that line in a second axis
    """
    results_path = os.path.join(os.getcwd(), "results", "2D_QGT_omega_sweep", folder_name)
    qgt_data_path = os.path.join(results_path, "QGT_2D.npy")
    meta_path = os.path.join(results_path, "meta_info.pkl")

    if not os.path.exists(qgt_data_path):
        raise FileNotFoundError(f"QGT data not found in '{results_path}'.")

    with open(meta_path, "rb") as f:
        meta_info = pickle.load(f)
    qgt_data = np.load(qgt_data_path, allow_pickle=True)

    def _in_range(w):
        if (omega_min is not None) and (w < omega_min): return False
        if (omega_max is not None) and (w > omega_max): return False
        return True

    filtered = [entry for entry in qgt_data if _in_range(float(entry["omega"]))]
    if len(filtered) == 0:
        raise ValueError("No omega slices fall within the requested range.")

    kx_mesh = np.asarray(meta_info["kx"])
    ky_mesh = np.asarray(meta_info["ky"])

    # get unique 1D axes
    kx = kx_mesh[0, :]   # take the first row → unique kx values
    ky = ky_mesh[:, 0]   # take the first column → unique ky values
    omega_values = [float(entry["omega"]) for entry in filtered]

    flip_x = kx.size > 1 and kx[1] < kx[0]
    flip_y = ky.size > 1 and ky[1] < ky[0]
    if flip_x: kx = kx[::-1]
    if flip_y: ky = ky[::-1]

    def orient_field_yx(field_yx):
        """Return field with x/y flipped (if needed) so that kx, ky are increasing."""
        f = np.asarray(field_yx)
        if flip_x: f = f[:, ::-1]  # flip columns (x)
        if flip_y: f = f[::-1, :]  # flip rows (y)
        return f

    # Global color scale (for trace)
    vmin = min(np.nanmin(entry["trace"]) for entry in filtered)
    vmax = max(np.nanmax(entry["trace"]) for entry in filtered)

    # Figure with two rows: heatmap (top) and line slice (bottom)
    fig = plt.figure(figsize=(9, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[8, 1])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_line = fig.add_subplot(gs[1, 0], sharex=None)

    # Initial index
    idx0 = 0
    field_yx0 = orient_field_yx(filtered[idx0]["trace"])   # (Ny, Nx)
    field_xy0 = field_yx0.T      
    # imshow wants (rows=y, cols=x) → use (Ny, Nx)
    im = ax_img.imshow(
        field_yx0,
        origin='lower',
        extent=[kx.min(), kx.max(), ky.min(), ky.max()],
        cmap='inferno',
        vmin=vmin, vmax=vmax, aspect='auto'
    )
    ax_img.set_title(f'QGT Trace — $\\omega$ = {omega_values[idx0]:.6f}')
    ax_img.set_xlabel("$k_x$")
    ax_img.set_ylabel("$k_y$")
    cbar = plt.colorbar(
        im, ax=ax_img,
        location="right",   # "right" | "left" | "top" | "bottom"
        fraction=0.1,     # relative size of the colorbar
        pad=0.04,           # gap between axes and colorbar (in fraction of axes)
        shrink=1,         # shrink bar length
        aspect=30           # length:width ratio of the bar
    )

    cbar.set_label("Trace Amplitude")
    
    # Draw initial line overlay + 1D slice
    s0, kx_line0, ky_line0, vals0 = slice_field_along_line(
        field_2d=field_xy0, kx=kx, ky=ky,
        angle_deg=angle_deg, shift_x=shift_x, shift_y=shift_y,
        n_samples=n_samples, periodic=periodic, max_len=k_length
    )
    # For the overlay: break line outside the box when not periodic
    if not periodic:
        in_box0 = (kx_line0 >= kx.min()) & (kx_line0 <= kx.max()) & \
                  (ky_line0 >= ky.min()) & (ky_line0 <= ky.max())
        kx_plot0 = np.where(in_box0, kx_line0, np.nan)
        ky_plot0 = np.where(in_box0, ky_line0, np.nan)
    else:
        kx_plot0, ky_plot0 = kx_line0, ky_line0

    line_overlay, = ax_img.plot(kx_plot0, ky_plot0, lw=1.5, alpha=0.9, label="line slice")

    # Plot against PHYSICAL s (units of k)
    slice_plot, = ax_line.plot(s0, vals0, lw=1.5)
    ax_line.set_xlabel("arc length s in k-space")
    ax_line.set_ylabel("Trace on line")
    ax_line.grid(True)
    if s0.size: ax_line.set_xlim(s0.min(), s0.max())



    # --- slider geometry (figure coords) ---
    SL_BOTTOM = 0.05     # y for the slider row
    SL_HEIGHT = 0.03     # height of each slider
    SL_LEFT   = 0.12     # left margin
    SL_RIGHT  = 0.12     # right margin
    SL_GAP    = 0.1     # horizontal gap between sliders


    avail = 1.0 - SL_LEFT - SL_RIGHT           # total available width
    col_w = (avail - 2*SL_GAP) / 3.0           # width of each of 3 sliders

    x0 = SL_LEFT
    x1 = SL_LEFT + col_w + SL_GAP
    x2 = SL_LEFT + 2*(col_w + SL_GAP)
    right_edge = x2 + col_w

    OMEGA_Y = SL_BOTTOM + SL_HEIGHT + 0.01

    # Sliders: omega index, angle, kx-shift, ky-shift
    fig.subplots_adjust(bottom=0.18)
    ax_s_omega = plt.axes([x0, OMEGA_Y, right_edge - x0, 0.03], facecolor='lightgoldenrodyellow')
    s_omega = Slider(ax_s_omega, '$\\omega$ idx', 0, len(omega_values)-1, valinit=idx0, valstep=1)

    ax_s_angle = plt.axes([x0, SL_BOTTOM, col_w, SL_HEIGHT], facecolor='lightgoldenrodyellow')
    s_angle    = Slider(ax_s_angle, 'angle°', 0.0, 180.0, valinit=angle_deg, valstep=1.0)

    ax_s_kx = plt.axes([x1, SL_BOTTOM, col_w, SL_HEIGHT], facecolor='lightgoldenrodyellow')
    s_kx    = Slider(ax_s_kx, 'shift $k_x$', kx.min(), kx.max(), valinit=shift_x)

    ax_s_ky = plt.axes([x2, SL_BOTTOM, col_w, SL_HEIGHT], facecolor='lightgoldenrodyellow')
    s_ky    = Slider(ax_s_ky, 'shift $k_y$', ky.min(), ky.max(), valinit=shift_y)

    
    def _recompute_and_draw():
        idx = int(s_omega.val)
        ang = float(s_angle.val)
        sx  = float(s_kx.val)
        sy  = float(s_ky.val)

        field_yx = orient_field_yx(filtered[idx]["trace"])  # (Ny, Nx)
        field_xy = field_yx.T                               # (Nx, Ny) for sampler

        s, kx_line, ky_line, vals = slice_field_along_line(
            field_2d=field_xy, kx=kx, ky=ky,
            angle_deg=ang, shift_x=sx, shift_y=sy,
            n_samples=n_samples, periodic=periodic, max_len=k_length
        )

        im.set_data(field_yx)  # imshow wants (Ny, Nx)

        ax_img.set_title(f'QGT Trace — $\\omega$ = {omega_values[idx]:.6f}')

        # overlay line (break outside the box if not periodic)
        if not periodic:
            in_box = (kx_line >= kx.min()) & (kx_line <= kx.max()) & \
                     (ky_line >= ky.min()) & (ky_line <= ky.max())
            kx_plot = np.where(in_box, kx_line, np.nan)
            ky_plot = np.where(in_box, ky_line, np.nan)
        else:
            kx_plot, ky_plot = kx_line, ky_line

        # Update line overlay
        line_overlay.set_data(kx_line, ky_line)


        # 1D slice vs PHYSICAL s
        if s.size > 0:
            slice_plot.set_data(s, vals)
            ax_line.set_xlim(s.min(), s.max())
            ymin = np.nanmin(vals); ymax = np.nanmax(vals)
            if np.isfinite(ymin) and np.isfinite(ymax):
                pad = 0.05*(ymax - ymin + 1e-12)
                ax_line.set_ylim(ymin - pad, ymax + pad)


        fig.canvas.draw_idle()

    def _on_change(_):
        _recompute_and_draw()

    s_omega.on_changed(_on_change)
    s_angle.on_changed(_on_change)
    s_kx.on_changed(_on_change)
    s_ky.on_changed(_on_change)

    plt.show()

#@ 2D joined plots
def dynamic_2d_qgt_vs_omega_joined(
    left_folder_name,
    right_folder_name,
    *,
    quantity="trace",            # "trace" | "berry" | "imqxy"
    convert_berry_from_imQ=True, # if quantity="berry" and no "berry": use -2*Im(Q_xy)
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
            if "berry" in entry:
                return np.asarray(entry["berry"])
            if "g_xy_imag" in entry:
                return (-2.0 * np.asarray(entry["g_xy_imag"])) if convert_berry_from_imQ \
                       else np.asarray(entry["g_xy_imag"])
            raise KeyError("Entry lacks 'berry' and 'g_xy_imag'; cannot form Berry curvature.")
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

    title_q = {"trace":"QGT Trace", "berry":"Berry Curvature Ω", "imqxy":"Im(Q_xy)"} \
              .get(q, "Field")
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


def dynamic_2d_qgt_1para_sweep_joined_html(
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
    HTML version of ``dynamic_2d_qgt_vs_omega_joined``, generalised to any
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
    import pickle as _pickle
    import plotly.graph_objects as go

    # ---- helpers -------------------------------------------------------
    def _resolve(p):
        return p if os.path.isabs(p) else os.path.join(os.getcwd(), p)

    def _load(dataset_path):
        dataset_path = _resolve(dataset_path)
        bundle_path  = os.path.join(dataset_path, "qgt_nd_bundle.npz")
        meta_path    = os.path.join(dataset_path, "meta.pkl")
        json_path    = os.path.join(dataset_path, "parameters.json")
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"ND bundle not found: {bundle_path}")
        bundle   = np.load(bundle_path, allow_pickle=True)
        ham_name = "Unknown"
        pjson    = None
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                pjson = _json.load(f)
            ham_name = pjson.get("hamiltonian_name", ham_name)
        elif os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = _pickle.load(f)
            ht = meta.get("Hamiltonian_Template")
            ham_name = getattr(ht, "name", ham_name) if ht is not None else ham_name
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
                "imqxy": "Im(Q_xy)"}.get(q, quantity)

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
        if pjson is None:
            return f'<div class="panel"><div class="panel-head" style="background:{title_color}">{title_label}</div><p style="padding:10px;font-size:12px;color:#666">No parameters.json found.</p></div>'

        def _fv(v):
            return f"{v:.6g}" if isinstance(v, float) else str(v)
        def _row(k, v):
            return f'<div class="row"><span class="key">{k}</span><span class="val">{v}</span></div>'
        def _sec(t, rows):
            return f'<div class="section"><div class="section-title">{t}</div>{rows}</div>'

        ham_rows = "".join(_row(k, _fv(v)) for k, v in sorted(pjson.get("parameters", {}).items()))

        scan_ranges  = pjson.get("scan_ranges", {})
        scan_spacing = pjson.get("scan_spacing", {})
        if isinstance(scan_ranges, list):
            range_items = {e[0]: {"min": e[1], "max": e[2]} for e in scan_ranges}
        else:
            range_items = scan_ranges
        scan_rows = "".join(
            _row(n, f'{_fv(info["min"])} \u2192 {_fv(info["max"])}'
                 + (f'  ({scan_spacing[n]["count"]} pts, {scan_spacing[n]["scale"]})'
                    if n in scan_spacing else ""))
            for n, info in range_items.items()
        )

        kg = pjson.get("k_grid", {})
        grid_rows = (
            _row("kx", f'{_fv(kg.get("kx_min","?"))} \u2192 {_fv(kg.get("kx_max","?"))}') +
            _row("ky", f'{_fv(kg.get("ky_min","?"))} \u2192 {_fv(kg.get("ky_max","?"))}') +
            _row("mesh", str(kg.get("mesh", "?")))
        )
        band_row = _row("band index", str(pjson["band_index"])) if "band_index" in pjson else ""

        return (
            f'<div class="panel">'
            f'<div class="panel-head" style="background:{title_color}">'
            f'{pjson.get("hamiltonian_name","?")} \u2014 {title_label}</div>'
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


#@ Rhombohedral Graphene Hamiltonian

#! V = 30
# dynamic_2d_trace_hprime_eigs_vs_omega("ChiralHamiltonianProjected/A0_0.10-V_30-analytic_magnus_True-magnus_order_1-n_5-omega_1000-polarization_right-t1_355.16-vF_542.10_kx-0.80_0.80_ky-0.80_0.80_mesh100_omega3.00e_01_5.00e_03_spacing_log_points14_band0_data_set1")

# dynamic_2d_trace_hprime_eigs_vs_omega("RhombohedralGrapheneHamiltonian/A0_0.10-V_60-analytic_magnus_False-magnus_order_1-n_4-omega_1000-polarization_right-t1_355.16-vF_542.10_kx-0.80_0.80_ky-0.80_0.80_mesh100_omega3.00e_01_5.00e_03_spacing_log_points32_band0_data_set1", omega_min=50)

# dynamic_2d_trace_hprime_eigs_vs_omega("RhombohedralGrapheneHamiltonian/A0_0.10-V_60-analytic_magnus_False-magnus_order_1-n_3-omega_1000-polarization_right-t1_355.16-vF_542.10_kx-0.80_0.80_ky-0.80_0.80_mesh100_omega3.00e_01_5.00e_03_spacing_log_points32_band0_data_set1", omega_min=50)

#@ Both numerical and analytical calcualtions
# dynamic_2d_trace_hprime_eigs_vs_omega_dual("RhombohedralGrapheneHamiltonian/A0_0.10-V_30-analytic_magnus_False-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10_kx-0.80_0.80_ky-0.80_0.80_mesh100_omega2.50e_01_5.00e_03_spacing_log_points84_data_set1",
#                                            "RhombohedralGrapheneHamiltonian/A0_0.10-V_30-analytic_magnus_True-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10_kx-0.80_0.80_ky-0.80_0.80_mesh100_omega2.50e_01_5.00e_03_spacing_log_points12_band0_data_set3")


# dynamic_2d_trace_with_line(
#     "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1",
#     omega_min=33,
#     angle_deg=45.0,     # angle w.r.t +kx (in degrees)
#     shift_x=0.0,        # center/shift in kx
#     shift_y=0.0,        # center/shift in ky
#     n_samples=500,
#     k_length=1.2,
#     periodic=False      # set True if your k-grid is periodic (torus)
# )


# dynamic_2d_qgt_vs_omega_joined("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  quantity="trace", omega_min_left=33, omega_min_right=50)

# dynamic_2d_qgt_vs_omega_joined("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  quantity="berry", omega_min_left=33, omega_min_right=50)


#@ 2D QGT ND — SquareLatticeHamiltonian omega sweep
# dataset1: left polarization,  omega in [0.1, 5.0],  mesh 150
# dataset2: right polarization, omega in [0.1, 50.0], mesh 100
# dynamic_2d_qgt_1d_sweep_html("results/2D_QGT_ND/SquareLatticeHamiltonian/dataset2", sweep_param="omega")


dynamic_2d_qgt_1para_sweep_joined_html(
    "results/2D_QGT_ND/SquareLatticeHamiltonian/dataset1",  # left
    "results/2D_QGT_ND/SquareLatticeHamiltonian/dataset2",  # right
    sweep_param="omega",
    quantity="trace",
    param_min_left=0.1,
    param_min_right=0.5,
    label_left="left polarization",
    label_right="right polarization",
)
