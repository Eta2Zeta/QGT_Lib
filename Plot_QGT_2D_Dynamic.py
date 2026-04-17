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
# Patch for unpickling old data that references Library.Hamiltonian_v2 (direct module)
sys.modules["Library.Hamiltonian_v2"] = Library.Hamiltonian.Hamiltonian

# Also ensure ChiralHamiltonian is available in Hamiltonian_v2 for unpickling
try:
    Library.Hamiltonian.Hamiltonian.ChiralHamiltonian = ChiralHamiltonian
except ImportError:
    pass

mpl.rcParams.update({
    "font.size": 8,        # base font size
    "axes.titlesize": 8,   # ax.set_title
    "axes.labelsize": 8,   # ax.set_xlabel/set_ylabel
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 8, # fig.suptitle
})


def dynamic_with_eigenvalues(folder_name, bands=[0, 1]):
    """
    Visualize QGT trace, eigenvalues (arbitrary bands), and perturbation dynamically for different omega values.

    Parameters:
    - folder_name (str): The folder containing the QGT results.
    - bands (list): List of band indices to plot, e.g., [0, 1, 2, 3].
    """
    k_max = np.pi  # Maximum k value for the first Brillouin zone

    # Load the saved data
    result_folder_path = os.path.join(os.getcwd(), "results", "1D_QGT_results", folder_name)
    g_results_filepath = os.path.join(result_folder_path, "QGT_1D.npy")
    meta_filepath = os.path.join(result_folder_path, "meta_info.pkl")

    if not os.path.exists(g_results_filepath):
        raise FileNotFoundError(f"File '{g_results_filepath}' not found in the 'results' directory.")

    with open(meta_filepath, "rb") as meta_file:
        meta_info = pickle.load(meta_file)

    num_points = meta_info['num_k_points']
    g_results = np.load(g_results_filepath, allow_pickle=True)

    # Extract initial data
    initial_index = 0
    k_line = np.linspace(-k_max, k_max, num_points)
    initial_data = g_results[initial_index]

    # Compute global y-axis bounds
    y_min_trace = np.nanmin([np.nanmin(data['trace']) for data in g_results])
    y_max_trace = np.nanmax([np.nanmax(data['trace']) for data in g_results])
    y_min_perturb = np.nanmin([np.nanmin(data['perturbation']) for data in g_results])
    y_max_perturb = np.nanmax([np.nanmax(data['perturbation']) for data in g_results])

    # Compute eigenbounds across ONLY the selected bands for all data
    # (assuming data['eigenvalues'] is [N, nbands] or [N, 2] etc.)
    all_selected_eigs = []
    for data in g_results:
        evs = np.array(data['eigenvalues']) # Shape (N_k, N_bands) usually
        if evs.ndim == 2:
            # Check bands validity
            n_avail = evs.shape[1]
            for b in bands:
                 if b < 0 or b >= n_avail:
                     raise ValueError(f"Band {b} out of range [0, {n_avail-1}]")
            # Select specific bands
            sel = evs[:, bands]
            all_selected_eigs.append(sel)
    
    if len(all_selected_eigs) > 0:
        flat_sel = np.concatenate([x.flatten() for x in all_selected_eigs])
        y_min_eigen = np.nanmin(flat_sel)
        y_max_eigen = np.nanmax(flat_sel)
    else:
        y_min_eigen = 0
        y_max_eigen = 1

    y_min_magnus_operator_norm = np.nanmin([np.nanmin(data['magnus_operator_norm']) for data in g_results if 'magnus_operator_norm' in data])
    y_max_magnus_operator_norm = np.nanmax([np.nanmax(data['magnus_operator_norm']) for data in g_results if 'magnus_operator_norm' in data])

    eigen_buffer = 0.1 * (y_max_eigen - y_min_eigen) if (y_max_eigen != y_min_eigen) else 0.1

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(bottom=0.2, right=0.8)  # Leave room for third y-axis and slider

    ax2 = ax1.twinx()  # Second y-axis
    ax3 = ax1.twinx()  # Third y-axis
    ax3.spines['right'].set_position(('outward', 60))
    ax3.spines['right'].set_visible(True)
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 120))
    ax4.spines['right'].set_visible(True)

    eigenvalues = np.array(initial_data['eigenvalues']).T  # Transpose from (N, nbands) → (nbands, N)

    # Plot initial data for selected bands
    line_eigens = []
    # Use a color cycle or map
    colors = plt.cm.jet(np.linspace(0, 1, len(bands)))
    
    for i, b in enumerate(bands):
        # We can try to cycle distinct colors or just use the index
        ln, = ax1.plot(k_line, eigenvalues[b], label=f'Eigenvalue {b}', color=colors[i])
        line_eigens.append(ln)

    line_trace, = ax2.plot(k_line, initial_data['trace'], label='Trace', color='b', linestyle='--')
    line_perturb, = ax3.plot(k_line, initial_data['perturbation'], label='Perturbation', color='g', linestyle=':')

    line_magnus, = ax4.plot(k_line, initial_data['magnus_operator_norm'], label='Magnus op. norm', color='k', alpha=0.5)

    # Axis formatting
    ax1.set_ylabel('Eigenvalues', color='r')
    ax1.set_xlabel('k (along line)')
    ax1.set_ylim(y_min_eigen - eigen_buffer, y_max_eigen + eigen_buffer)
    ax1.tick_params(axis='y', labelcolor='r')

    ax2.set_ylabel('Trace Amplitude', color='b')
    ax2.set_ylim(y_min_trace, y_max_trace)
    ax2.tick_params(axis='y', labelcolor='b')

    ax3.set_ylabel('Perturbation', color='g')
    ax3.set_ylim(y_min_perturb, y_max_perturb)
    ax3.tick_params(axis='y', labelcolor='g')

    ax4.set_ylabel('Magnus op. norm', color='k')
    ax4.set_ylim(y_min_magnus_operator_norm, y_max_magnus_operator_norm)
    ax4.tick_params(axis='y', labelcolor='k')

    ax1.set_title('QGT Trace, Eigenvalues, and Perturbation for Different $\omega$')
    ax1.grid(True)

    # Slider setup
    ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, '$\omega$', 0, len(g_results) - 1, valinit=initial_index, valstep=1)

    # Update function
    def update(val):
        index = int(slider.val)
        data = g_results[index]

        eigenvalues = np.array(data['eigenvalues']).T
        # Update lines for each band
        for i, ln in enumerate(line_eigens):
            # bands[i] gives the band index
            b_idx = bands[i]
            ln.set_ydata(eigenvalues[b_idx])

        line_trace.set_ydata(data['trace'])
        line_perturb.set_ydata(data['perturbation'])
        line_magnus.set_ydata(data['magnus_operator_norm'])

        ax1.set_title(f'QGT Trace, Eigenvalues, Perturbation — $\omega$ = {data["omega"]:.6f}')

        # Legend might need refreshing if labels changed (they don't here)
        # Just gather handles
        lines = line_eigens + [line_trace, line_perturb, line_magnus]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper left")

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

def dynamic_with_eigenvalues_single_param(result_dir, band_index1=0, band_index2=1):
    """
    Visualize (vs k along the chosen line) for a single-parameter 1D sweep:
      - two eigenvalue branches (choose indices)
      - QGT trace
      - perturbation
      - Magnus operator norm (if present)
    """
    # Resolve directory
    if os.path.isdir(result_dir):
        folder_path = result_dir
    else:
        folder_path = os.path.join(os.getcwd(), "results", "1D_QGT_results", result_dir)

    g_results_path = os.path.join(folder_path, "QGT_1D.npy")
    meta_path      = os.path.join(folder_path, "meta_info.pkl")
    if not os.path.exists(g_results_path):
        raise FileNotFoundError(f"Missing '{g_results_path}'")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing '{meta_path}'")

    # Load metadata + results
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    param_name   = meta.get("param_name", "omega")
    swept_values = np.asarray(meta.get("values"))
    num_k_points = int(meta["num_k_points"])
    k_max        = float(meta["k_max"])

    k_line = np.linspace(-k_max, k_max, num_k_points)
    g_results = np.load(g_results_path, allow_pickle=True)

    # Which bands exist?
    sample_ev = np.asarray(g_results[0]["eigenvalues"])
    if sample_ev.ndim != 2:
        sample_ev = sample_ev.reshape(sample_ev.shape[0], -1)  # (Nk, nbands)
    nbands = sample_ev.shape[1]
    for b in (band_index1, band_index2):
        if b < 0 or b >= nbands:
            raise IndexError(f"Band index {b} out of range [0, {nbands-1}]")

    # Global y-limits computed ONLY from the two selected bands
    y_min_eval = np.inf; y_max_eval = -np.inf
    y_min_trace = np.inf; y_max_trace = -np.inf
    y_min_pert  = np.inf; y_max_pert  = -np.inf

    has_magnus = all("magnus_operator_norm" in d for d in g_results)
    y_min_mag = np.inf; y_max_mag = -np.inf

    for d in g_results:
        ev = np.asarray(d["eigenvalues"])
        if ev.ndim != 2:
            ev = ev.reshape(ev.shape[0], -1)    # (Nk, nbands)

        sel = np.column_stack((ev[:, band_index1], ev[:, band_index2]))  # (Nk, 2)
        y_min_eval = min(y_min_eval, np.nanmin(sel))
        y_max_eval = max(y_max_eval, np.nanmax(sel))

        y_min_trace = min(y_min_trace, np.nanmin(d["trace"]))
        y_max_trace = max(y_max_trace, np.nanmax(d["trace"]))
        y_min_pert  = min(y_min_pert,  np.nanmin(d["perturbation"]))
        y_max_pert  = max(y_max_pert,  np.nanmax(d["perturbation"]))
        if has_magnus:
            y_min_mag = min(y_min_mag, np.nanmin(d["magnus_operator_norm"]))
            y_max_mag = max(y_max_mag, np.nanmax(d["magnus_operator_norm"]))

    eval_buf = 0.1 * (y_max_eval - y_min_eval + 1e-12)

    # Figure / axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(bottom=0.2, right=0.84)

    ax2 = ax1.twinx()  # Trace
    ax3 = ax1.twinx()  # Perturbation
    ax3.spines['right'].set_position(('outward', 60))
    ax3.spines['right'].set_visible(True)

    ax4 = None
    if has_magnus:
        ax4 = ax1.twinx()
        ax4.spines['right'].set_position(('outward', 120))
        ax4.spines['right'].set_visible(True)

    # Initial slice
    i0 = 0
    d0 = g_results[i0]
    ev0 = np.asarray(d0["eigenvalues"])
    if ev0.ndim != 2:
        ev0 = ev0.reshape(ev0.shape[0], -1)  # (Nk, nbands)
    ev0_T = ev0.T  # (nbands, Nk)

    # Plots
    line_e1, = ax1.plot(k_line, ev0_T[band_index1], label=f"Eigen[{band_index1}]", color='r')
    line_e2, = ax1.plot(k_line, ev0_T[band_index2], label=f"Eigen[{band_index2}]", color='m')
    line_tr,   = ax2.plot(k_line, d0["trace"],        label="Trace",        color='b')
    line_pert, = ax3.plot(k_line, d0["perturbation"], label="Perturbation", color='g')
    if has_magnus:
        line_mag, = ax4.plot(k_line, d0["magnus_operator_norm"], label="Magnus ‖·‖", color='k')

    # Axis formatting (eigen y-lims use only the selected bands)
    ax1.set_xlabel("k (along line)")
    ax1.set_ylabel("Eigenvalues", color='r')
    ax1.set_ylim(y_min_eval - eval_buf, y_max_eval + eval_buf)
    ax1.tick_params(axis='y', labelcolor='r')

    ax2.set_ylabel("Trace", color='b')
    ax2.set_ylim(y_min_trace, y_max_trace)
    ax2.tick_params(axis='y', labelcolor='b')

    ax3.set_ylabel("Perturbation", color='g')
    ax3.set_ylim(y_min_pert, y_max_pert)
    ax3.tick_params(axis='y', labelcolor='g')

    if has_magnus:
        ax4.set_ylabel("Magnus ‖·‖", color='k')
        ax4.set_ylim(y_min_mag, y_max_mag)
        ax4.tick_params(axis='y', labelcolor='k')

    # Title / legend
    def _title_for(idx):
        pv = g_results[idx].get(param_name, swept_values[idx] if swept_values is not None else None)
        if pv is None:
            return f"QGT Trace, Eigenvalues, Perturbation — {param_name} [idx {idx}]"
        return f"QGT Trace, Eigenvalues, Perturbation — {param_name} = {pv:.6g}"

    lines = [line_e1, line_e2, line_tr, line_pert]
    if has_magnus: lines.append(line_mag)
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left")
    ax1.set_title(_title_for(i0))
    ax1.grid(True)

    # Slider
    ax_sl = plt.axes([0.15, 0.06, 0.66, 0.03], facecolor='lightgoldenrodyellow')
    s_idx = Slider(ax_sl, f"{param_name} idx", 0, len(g_results)-1, valinit=i0, valstep=1)

    # Update
    def _update(_):
        i = int(s_idx.val)
        d = g_results[i]

        ev = np.asarray(d["eigenvalues"])
        if ev.ndim != 2:
            ev = ev.reshape(ev.shape[0], -1)
        evT = ev.T

        line_e1.set_ydata(evT[band_index1])
        line_e2.set_ydata(evT[band_index2])
        line_tr.set_ydata(d["trace"])
        line_pert.set_ydata(d["perturbation"])
        if has_magnus:
            line_mag.set_ydata(d["magnus_operator_norm"])

        ax1.set_title(_title_for(i))
        fig.canvas.draw_idle()

    s_idx.on_changed(_update)
    plt.show()


#@ 2D QGT single-parameter omega sweep dynamic plotting
def dynamic_2d_trace_vs_omega(folder_name, omega_min=None, omega_max=None):
    """
    Dynamically visualize the QGT trace (2D heatmap) as a function of omega.

    Parameters:
        folder_name (str): Name of the subfolder in 'results/2D_QGT_omega_sweep/'.
        omega_min (float|None): Keep slices with omega >= omega_min.
        omega_max (float|None): Keep slices with omega <= omega_max.
    """
    results_path = os.path.join(os.getcwd(), "results", "2D_QGT_omega_sweep", folder_name)
    qgt_data_path = os.path.join(results_path, "QGT_2D.npy")
    meta_path = os.path.join(results_path, "meta_info.pkl")

    if not os.path.exists(qgt_data_path):
        raise FileNotFoundError(f"QGT data not found in '{results_path}'.")

    with open(meta_path, "rb") as f:
        meta_info = pickle.load(f)
    qgt_data = np.load(qgt_data_path, allow_pickle=True)

    # Filter by omega range (inclusive). If no bounds provided, keep all.
    def _in_range(w):
        if (omega_min is not None) and (w < omega_min): return False
        if (omega_max is not None) and (w > omega_max): return False
        return True

    filtered = [entry for entry in qgt_data if _in_range(float(entry["omega"]))]
    if len(filtered) == 0:
        raise ValueError("No omega slices fall within the requested range.")

    kx = meta_info["kx"]
    ky = meta_info["ky"]
    omega_values = [float(entry["omega"]) for entry in filtered]

    # Global color scaling on truncated data
    max_trace = max(np.nanmax(entry["trace"]) for entry in filtered)
    min_trace = min(np.nanmin(entry["trace"]) for entry in filtered)

    # Initial data
    initial_index = 0
    trace0 = filtered[initial_index]["trace"]

    # Setup figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(bottom=0.2)

    img = ax.imshow(
        trace0,
        origin='lower',
        extent=[kx.min(), kx.max(), ky.min(), ky.max()],
        cmap='inferno',
        vmin=min_trace,
        vmax=max_trace,
        aspect='auto'
    )

    ax.set_title(f'QGT Trace — $\\omega$ = {omega_values[initial_index]:.6f}')
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Trace Amplitude")

    # Slider for omega (over filtered indices)
    ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, '$\\omega$', 0, len(omega_values) - 1, valinit=initial_index, valstep=1)

    def update(val):
        index = int(slider.val)
        trace = filtered[index]["trace"]
        img.set_data(trace)
        ax.set_title(f'QGT Trace — $\\omega$ = {omega_values[index]:.6f}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


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

def dynamic_2d_berry_vs_omega(folder_name, omega_min=None, omega_max=None):
    """
    Dynamically visualize the Berry curvature (2D heatmap) as a function of omega.

    Parameters:
        folder_name (str): Name of the subfolder in 'results/2D_QGT_omega_sweep/'.
        omega_min (float|None): Keep slices with omega >= omega_min.
        omega_max (float|None): Keep slices with omega <= omega_max.
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

    kx = meta_info["kx"]
    ky = meta_info["ky"]
    omega_values = [float(entry["omega"]) for entry in filtered]

    # Global colorbar limits across truncated omega slices
    max_val = max(np.nanmax(-2 * entry["g_xy_imag"]) for entry in filtered)
    min_val = min(np.nanmin(-2 * entry["g_xy_imag"]) for entry in filtered)

    initial_index = 0
    berry0 = -2 * filtered[initial_index]["g_xy_imag"]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(bottom=0.2)

    img = ax.imshow(
        berry0,
        origin='lower',
        extent=[kx.min(), kx.max(), ky.min(), ky.max()],
        cmap='coolwarm',
        vmin=min_val,
        vmax=max_val,
        aspect='auto'
    )

    ax.set_title(f'Berry Curvature — $\\omega$ = {omega_values[initial_index]:.6f}')
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Berry Curvature (−2 Im[gₓᵧ])")

    ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, '$\\omega$', 0, len(omega_values) - 1, valinit=initial_index, valstep=1)

    def update(val):
        index = int(slider.val)
        berry_curvature = -2 * filtered[index]["g_xy_imag"]
        img.set_data(berry_curvature)
        ax.set_title(f'Berry Curvature — $\\omega$ = {omega_values[index]:.6f}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()



#@ 1D plots vs omega
def plot_metric_vs_omega(
    folder_name,
    metric="trace",                 # "trace" | "berry" | "trg-berry"
    omega_min=None,
    omega_max=None,
    *,
    use_precomputed=False,          # if entries already include 'berry'
    convert_from_imQ=True,          # if not precomputed: Ω = -2 * Im(Q_xy)
    return_data=False               # optionally return (omegas, values)
):
    """
    Unified plotting for:
      - metric="trace":  std_BZ[ Tr(g) ] vs ω
      - metric="berry":  std_BZ[ Ω ] vs ω  (Ω from entry['berry'] or -2*Im(Q_xy))
      - metric="trg-berry": ∫_BZ [Tr(g) - Ω] d^2k vs ω  (needs meta['dkx'], meta['dky'])

    Parameters
    ----------
    folder_name : str
        Path fed to load_qgt(folder_name).
    metric : {"trace", "berry", "trg-berry"}
        Quantity to aggregate/plot.
    omega_min, omega_max : float or None
        Optional frequency filter (inclusive).
    use_precomputed : bool
        If True and entries contain 'berry', use it; else derive Ω from g_xy_imag.
    convert_from_imQ : bool
        When deriving Ω from g_xy_imag, use Ω = -2 * Im(Q_xy) if True; else Ω = Im(Q_xy).
        (Sign choice does not affect std but matters for integrals.)
    return_data : bool
        If True, return (omegas_sorted, values_sorted) instead of just plotting.

    Notes
    -----
    Expects:
      entries[i] with keys: "omega", "trace", and either "berry" or "g_xy_imag"
      meta with keys: "dkx", "dky" (only required for metric="trg-berry").
    """
    # Load & filter
    entries, meta = load_qgt(folder_name)
    filtered = filter_entries_by_omega(entries, omega_min, omega_max)

    if len(filtered) == 0:
        raise ValueError("No entries after omega filtering; check omega_min/max or data.")

    # Helper to get Berry curvature array for an entry
    def _get_berry(entry):
        if use_precomputed and ("berry" in entry):
            return np.asarray(entry["berry"])
        gim = np.asarray(entry["g_xy_imag"])
        return (-2.0 * gim) if convert_from_imQ else gim

    omegas = np.array([float(e["omega"]) for e in filtered], dtype=float)

    if metric == "trace":
        # std over BZ of Tr(g)
        vals = np.array([np.nanstd(np.asarray(e["trace"])) for e in filtered], dtype=float)
        ylab = "Std. Dev over BZ"
        title = "Fluctuation of QGT Trace vs ω"
    elif metric == "berry":
        # std over BZ of Ω
        vals = np.array([np.nanstd(_get_berry(e)) for e in filtered], dtype=float)
        ylab = "Std. Dev over BZ"
        title = "Fluctuation of Berry Curvature vs ω"
    elif metric == "trg-berry":
        # integral over BZ of [Tr(g) − Ω]
        try:
            dkx = float(meta["dkx"]); dky = float(meta["dky"])
        except Exception as exc:
            raise KeyError("meta must contain 'dkx' and 'dky' to integrate over the BZ.") from exc
        area_element = dkx * dky
        vals = []
        for e in filtered:
            trace = np.asarray(e["trace"])
            berry = _get_berry(e)
            integrand = trace - berry
            vals.append(np.nansum(integrand) * area_element)
        vals = np.array(vals, dtype=float)
        ylab = r"$\int_{\mathrm{BZ}} [\mathrm{Tr}(g) - \Omega]\, d^2k$"
        title = r"Integrated $\mathrm{Tr}(g) - \Omega$ vs $\omega$"
    else:
        raise ValueError("metric must be one of: 'trace', 'berry', 'trg-berry'.")

    # Sort by omega for a clean line
    order = np.argsort(omegas)
    omegas_sorted = omegas[order]
    vals_sorted = vals[order]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(omegas_sorted, vals_sorted, marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel("Drive Frequency ω")
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if return_data:
        return omegas_sorted, vals_sorted

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

def load_qgt_entries(folder_name):
    """
    Load raw QGT entries and metadata from a results folder.
    Returns: (entries_list, kx, ky)
    """
    base = os.path.join(os.getcwd(), "results", "2D_QGT_omega_sweep", folder_name)
    qgt_path = os.path.join(base, "QGT_2D.npy")
    meta_path = os.path.join(base, "meta_info.pkl")
    if not os.path.exists(qgt_path):
        raise FileNotFoundError(f"QGT data not found in '{base}'.")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    entries = np.load(qgt_path, allow_pickle=True)
    kx = np.asarray(meta["kx"]); ky = np.asarray(meta["ky"])
    return list(entries), kx, ky

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

#! TwoOrbitalUnspinful
# 1D QGT

# 22.5 degrees
dynamic_with_eigenvalues("TwoOrbitalUnspinfulHamiltonian/A00.1_polarizationleft_magnus_order1_t1_mu0_zeta1.0_a1_angle0.4_kxshift0.00_kyshift0.00_points150_kmax4.44_omega1.00e-02_5.00e_01_spacing_log_points100_1")
dynamic_with_eigenvalues("TwoOrbitalUnspinfulHamiltonian/A00.1_polarizationleft_magnus_order1_t1_mu0_zeta1.0_a1_angle22.5_kxshift0.00_kyshift0.00_points150_kmax4.44_omega1.00e-02_5.00e-01_spacing_log_points100_1")

# 45 degrees
# dynamic_with_eigenvalues("TwoOrbitalUnspinfulHamiltonian/A00.1_polarizationleft_magnus_order1_t1_mu0_zeta1.0_a1_angle45.0_kxshift0.00_kyshift0.00_points150_kmax4.44_omega1.00e-02_5.00e_01_spacing_log_points100_1")

# dynamic_with_eigenvalues("TwoOrbitalUnspinfulHamiltonian/A00.1_polarizationlinear_x_magnus_order1_t1_mu0_zeta1.0_a1_angle45.0_kxshift0.00_kyshift0.00_points150_kmax4.44_omega1.00e-01_5.00e_01_spacing_log_points100_1")

# 0 degrees
# dynamic_with_eigenvalues("TwoOrbitalUnspinfulHamiltonian/A00.1_polarizationleft_magnus_order1_t1_mu0_zeta1.0_a1_angle0.0_kxshift0.00_kyshift0.00_points150_kmax4.44_omega1.00e-02_5.00e_01_spacing_log_points100_1")

#! Conclusion: There is practically no shift in the QGT trace. 


#! Sqaure Lattice
#@ t5=0
#~ 1D QGT
#* Centered around at (0, -pi/2)
#^ Along 45 degree line
#! Left Polarization
# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t50_angle45.0_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega5.00e-02_1.00e_01_spacing_log_points100_1")

#! Right Polarization
# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationright_magnus_order1_t11_t20.7071067811865475_t50_angle45.0_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega1.00e-01_5.00e_01_spacing_log_points100_1")

#! x Linear Polarization
# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationlinear_x_magnus_order1_t11_t20.7071067811865475_t50_angle45.0_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega1.00e-01_5.00e_01_spacing_log_points100_1")

#^ Along 0 degree line
# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t50_angle0.0_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega5.00e-02_5.00e_01_spacing_log_points100_1")

#~ 2D QGT
#! Left Polarization
# dynamic_2d_trace_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t50_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega1.00e-01_5.00e_01_spacing_log_points32_1")
# plot_trace_std_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t50_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega1.00e-01_5.00e_01_spacing_log_points32_1")
# plot_berry_std_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t50_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega1.00e-01_5.00e_01_spacing_log_points32_1")
# plot_integrated_trace_minus_berry("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t50_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega1.00e-01_5.00e_01_spacing_log_points32_1")

#! Right Polarization
# dynamic_2d_trace_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationright_magnus_order1_t11_t20.7071067811865475_t50_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega1.00e-01_5.00e_01_spacing_log_points32_1")

#@ t5=(1-np.sqrt(2))/4
#~ 1D QGT
#* Centered around at (0, -pi/2)
#^ Along 45 degree line
#! Left Polarization
# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_angle45.0_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega5.00e-02_5.00e_01_spacing_log_points100_1")
#! Right Polarization
# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationright_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_angle45.0_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega1.00e-01_5.00e_01_spacing_log_points100_1")
#! x Linear Polarization
# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationlinear_x_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_angle45.0_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega1.00e-01_5.00e_01_spacing_log_points100_1")

#^ Along 0 degree line 
#! Left Polarization
# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_angle0.0_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega1.00e-01_5.00e_01_spacing_log_points100_1")
#! Right Polarization
# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationright_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_angle0.0_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega1.00e-01_5.00e_01_spacing_log_points100_1")
#! x Linear Polarization
# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationlinear_x_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_angle0.0_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega1.00e-01_5.00e_01_spacing_log_points100_1")

#* Centered around at (0,0)
#^ Along 45 degree line 
#! Left Polarization
# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_angle45.0_kxshift0.00_kyshift0.00_points150_kmax4.44_omega5.00e-02_5.00e_01_spacing_log_points100_1")

#! Right Polarization
# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationright_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_angle45.0_kxshift0.00_kyshift0.00_points150_kmax4.44_omega5.00e-02_5.00e_01_spacing_log_points100_1")

#~ 2D QGT
#! Left Polarization
# dynamic_2d_trace_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega5.00e-02_5.00e_01_spacing_log_points32_1")
# plot_trace_std_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega5.00e-02_5.00e_01_spacing_log_points32_1")
# plot_berry_std_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega5.00e-02_5.00e_01_spacing_log_points32_1")
# plot_integrated_trace_minus_berry("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega5.00e-02_5.00e_01_spacing_log_points32_1")

#! Right Polarization
# dynamic_2d_trace_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationright_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega1.00e-01_5.00e_01_spacing_log_points32_1")
# plot_trace_std_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationright_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega1.00e-01_5.00e_01_spacing_log_points32_1")
# plot_berry_std_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationright_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega1.00e-01_5.00e_01_spacing_log_points32_1")
# plot_integrated_trace_minus_berry("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationright_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega1.00e-01_5.00e_01_spacing_log_points32_1")


#@ Rhombohedral Graphene Hamiltonian

#! V = 30
# dynamic_2d_trace_hprime_eigs_vs_omega("ChiralHamiltonianProjected/A0_0.10-V_30-analytic_magnus_True-magnus_order_1-n_5-omega_1000-polarization_right-t1_355.16-vF_542.10_kx-0.80_0.80_ky-0.80_0.80_mesh100_omega3.00e_01_5.00e_03_spacing_log_points14_band0_data_set1")

# dynamic_2d_trace_hprime_eigs_vs_omega("RhombohedralGrapheneHamiltonian/A0_0.10-V_60-analytic_magnus_False-magnus_order_1-n_4-omega_1000-polarization_right-t1_355.16-vF_542.10_kx-0.80_0.80_ky-0.80_0.80_mesh100_omega3.00e_01_5.00e_03_spacing_log_points32_band0_data_set1", omega_min=50)

# dynamic_2d_trace_hprime_eigs_vs_omega("RhombohedralGrapheneHamiltonian/A0_0.10-V_60-analytic_magnus_False-magnus_order_1-n_3-omega_1000-polarization_right-t1_355.16-vF_542.10_kx-0.80_0.80_ky-0.80_0.80_mesh100_omega3.00e_01_5.00e_03_spacing_log_points32_band0_data_set1", omega_min=50)

#@ Both numerical and analytical calcualtions
# dynamic_2d_trace_hprime_eigs_vs_omega_dual("RhombohedralGrapheneHamiltonian/A0_0.10-V_30-analytic_magnus_False-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10_kx-0.80_0.80_ky-0.80_0.80_mesh100_omega2.50e_01_5.00e_03_spacing_log_points84_data_set1",
#                                            "RhombohedralGrapheneHamiltonian/A0_0.10-V_30-analytic_magnus_True-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10_kx-0.80_0.80_ky-0.80_0.80_mesh100_omega2.50e_01_5.00e_03_spacing_log_points12_band0_data_set3")


#! Full Chiral Hamiltonian
#~ 1D QGT
# dynamic_with_eigenvalues("ChiralHamiltonian/A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_angle0.0_kxshift0.00_kyshift0.00_points100_kmax1.57_omega5.00e_00_5.00e_03_spacing_log_points30_2", bands=[0,1,2,3,4, 5,6,7,8,9])
# dynamic_with_eigenvalues("ChiralHamiltonian/A0_0.10-V_30.00-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-polarization_right-t1_355.16-vF_542.10_angle0.0_kxshift0.00_kyshift0.00_points100_kmax1.57_omega5.00e_00_5.00e_03_spacing_log_points30_1", bands=[4, 5])

# dynamic_with_eigenvalues_single_param("ChiralHamiltonian/A0_0-V_20.00-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-polarization_left-t1_355.16-vF_542.10_angle0.0_kx0.00_ky0.00_kmax1.57_param_V_5_50_spacing_linear_N20_kN100_data_set1", band_index1=4, band_index2=5)

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


# & Left Polarization
# dynamic_2d_trace_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", omega_min=33)
# dynamic_2d_berry_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", omega_min=33)
# plot_metric_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", omega_min=33, metric="berry")

# & Right Polarization
# dynamic_2d_trace_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", omega_min=50)
# dynamic_2d_berry_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", omega_min=50)
# plot_trace_std_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", omega_min=50)
# plot_berry_std_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", omega_min=50)
# plot_integrated_trace_minus_berry("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", omega_min=50)

# dynamic_2d_qgt_vs_omega_joined("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  quantity="trace", omega_min_left=33, omega_min_right=50)

# dynamic_2d_qgt_vs_omega_joined("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  quantity="berry", omega_min_left=33, omega_min_right=50)


