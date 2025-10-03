import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle
from Library.utilities import in_range

mpl.rcParams.update({
    "font.size": 8,        # base font size
    "axes.titlesize": 8,   # ax.set_title
    "axes.labelsize": 8,   # ax.set_xlabel/set_ylabel
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 8, # fig.suptitle
})


def dynamic_THF():
    k_max = 1 * (np.pi)  # Maximum k value for the first Brillouin zone
    # Load the saved data
    g_results = np.load("g_results_log.npy", allow_pickle=True)

    # Extract the initial data for visualization
    initial_index = 0
    k_line = np.linspace(-k_max, k_max, 300)  # Example: same k_line as before
    initial_data = g_results[initial_index]

    # Calculate global y-axis bounds
    y_min = min(np.min(data['trace']) for data in g_results)
    y_max = max(np.max(data['trace']) for data in g_results)

    # Plot setup
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    line, = ax.plot(k_line, initial_data['trace'], label=f'Trace (G={initial_data["G"]:.6f})')
    ax.set_title('QGT Trace Along Line for Different G Values')
    ax.set_xlabel('k (along line)')
    ax.set_ylabel('Trace ($g_{xx} + g_{yy}$)')
    ax.legend()
    ax.grid(True)

    # Set y-axis limits dynamically
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max)

    # Slider setup
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'G', 0, len(g_results) - 1, valinit=initial_index, valstep=1)

    # Slider update function
    def update(val):
        index = int(slider.val)
        data = g_results[index]
        line.set_ydata(data['trace'])
        line.set_label(f'Trace (G={data["G"]:.6f})')
        ax.legend()
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

# dynamic_THF()

def dynamic(folder_name):
    k_max = 1 * (np.pi)  # Maximum k value for the first Brillouin zone
    # Load the saved data
    g_results_filepath = os.path.join(os.getcwd(), "results", "1D_QGT_results", folder_name, "QGT_1D.npy")


    # Check if the file exists
    if not os.path.exists(g_results_filepath):
        raise FileNotFoundError(f"File '{g_results_filepath}' not found in the 'results' directory.")


    g_results = np.load(g_results_filepath, allow_pickle=True)

    # Extract the initial data for visualization
    initial_index = 0
    k_line = np.linspace(-k_max, k_max, 100)  # Example: same k_line as before
    initial_data = g_results[initial_index]

    # Calculate global y-axis bounds
    y_min = min(np.min(data['trace']) for data in g_results)
    y_max = max(np.max(data['trace']) for data in g_results)

    # Plot setup
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    line, = ax.plot(k_line, initial_data['trace'], label=f'Trace ($\omega$={initial_data["omega"]:.6f})')
    ax.set_title('QGT Trace Along Line for D          ifferent $\omega$ Values')
    ax.set_xlabel('k (along line)')
    ax.set_ylabel('Trace ($g_{xx} + g_{yy}$)')
    ax.legend()
    ax.grid(True)

    # Set y-axis limits dynamically
    ax.set_yscale('linear')
    ax.set_ylim(y_min, y_max)

    # Slider setup
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, '$\omega$', 0, len(g_results) - 1, valinit=initial_index, valstep=1)

    # Slider update function
    def update(val):
        index = int(slider.val)
        data = g_results[index]
        line.set_ydata(data['trace'])
        line.set_label(f'Trace ($\omega$={data["omega"]:.6f})')
        ax.legend()
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

def dynamic_with_eigenvalue(folder_name):
    """
    Visualize QGT trace and eigenvalues dynamically for different omega values.
    The Eigenvalues are just the eigenvalue of that one band specified in the calculation. 

    Parameters:
    - folder_name (str): The folder containing the QGT results.
    """
    k_max = 1 * (np.pi)  # Maximum k value for the first Brillouin zone
    
    # Load the saved data
    result_folder_path = os.path.join(os.getcwd(), "results", "1D_QGT_results", folder_name)
    g_results_filepath = os.path.join(result_folder_path, "QGT_1D.npy")
    meta_filepath = os.path.join(result_folder_path, "meta_info.pkl")
    with open(meta_filepath, "rb") as meta_file:
            meta_info = pickle.load(meta_file)

    num_k_points = meta_info['num_k_points']


    # Check if the file exists
    if not os.path.exists(g_results_filepath):
        raise FileNotFoundError(f"File '{g_results_filepath}' not found in the 'results' directory.")

    g_results = np.load(g_results_filepath, allow_pickle=True)

    # Extract the initial data for visualization
    initial_index = 0
    k_line = np.linspace(-k_max, k_max, num_k_points)  # Same k_line as before
    initial_data = g_results[initial_index]

    # Compute global y-axis bounds
    y_min_trace = min(np.min(data['trace']) for data in g_results)
    y_max_trace = max(np.max(data['trace']) for data in g_results)
    y_min_perturb = min(np.min(data['perturbation']) for data in g_results)
    y_max_perturb = max(np.max(data['perturbation']) for data in g_results)
    y_min_eigen = min(np.min(data['eigenvalues']) for data in g_results)
    y_max_eigen = max(np.max(data['eigenvalues']) for data in g_results)

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Make the figure larger
    fig.subplots_adjust(bottom=0.2, right=0.8)  # Increase as needed
    ax2 = ax1.twinx()  # second y-axis
    ax3 = ax1.twinx()  # third y-axis
    ax3.spines['right'].set_position(('outward', 60))  # offset it
    ax3.spines['right'].set_visible(True)




    # Plot initial data
    line_eigen, = ax1.plot(k_line, initial_data['eigenvalues'], label='Eigenvalues', color='r')
    line_trace, = ax2.plot(k_line, initial_data['trace'], label='Trace', color='b')
    line_perturb, = ax3.plot(k_line, initial_data['perturbation'], label='Perturbation', color='g')

    # Label axes
    ax1.set_ylabel('Eigenvalues', color='r')
    ax1.set_xlabel('k (along line)')
    ax1.set_ylim(y_min_eigen, y_max_eigen)
    ax1.tick_params(axis='y', labelcolor='r')

    ax2.set_ylabel('Trace Amplitude', color='b')
    ax2.set_ylim(y_min_trace, y_max_trace)
    ax2.tick_params(axis='y', labelcolor='b')

    ax3.set_ylabel('Perturbation', color='g')
    ax3.set_ylim(y_min_perturb, y_max_perturb)
    ax3.tick_params(axis='y', labelcolor='g')


    ax1.set_title('QGT Trace, Eigenvalues, and Perturbation for Different $\omega$')
    ax1.grid(True)

    # Slider
    ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, '$\omega$', 0, len(g_results) - 1, valinit=initial_index, valstep=1)

    # Slider update function
    def update(val):
        index = int(slider.val)
        data = g_results[index]
        line_eigen.set_ydata(data['eigenvalues'])
        line_trace.set_ydata(data['trace'])
        line_perturb.set_ydata(data['perturbation'])

        # Optional: update title or labels
        ax1.set_title(f'QGT Trace, Eigenvalues, Perturbation — $\omega$ = {data["omega"]:.6f}')

        # Combined legend
        lines = [line_eigen, line_trace, line_perturb]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper left")

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()



def dynamic_with_eigenvalues(folder_name, band_index1=0, band_index2=1):
    """
    Visualize QGT trace, eigenvalues (2 bands), and perturbation dynamically for different omega values.

    Parameters:
    - folder_name (str): The folder containing the QGT results.
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
    y_min_eigen = np.nanmin([np.nanmin(data['eigenvalues']) for data in g_results])
    y_max_eigen = np.nanmax([np.nanmax(data['eigenvalues']) for data in g_results])
    y_min_magnus_operator_norm = np.nanmin([np.nanmin(data['magnus_operator_norm']) for data in g_results if 'magnus_operator_norm' in data])
    y_max_magnus_operator_norm = np.nanmax([np.nanmax(data['magnus_operator_norm']) for data in g_results if 'magnus_operator_norm' in data])

    eigen_buffer = 0.1 * (y_max_eigen - y_min_eigen)  # Buffer for eigenvalues

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


    eigenvalues = np.array(initial_data['eigenvalues']).T  # Transpose from (N, 2) → (2, N)


    # Plot initial data (eigenvalues now has shape [2, N])
    line_eigen1, = ax1.plot(k_line, eigenvalues[band_index1], label='Eigenvalue 1', color='r')
    line_eigen2, = ax1.plot(k_line, eigenvalues[band_index2], label='Eigenvalue 2', color='m')

    line_trace, = ax2.plot(k_line, initial_data['trace'], label='Trace', color='b')
    line_perturb, = ax3.plot(k_line, initial_data['perturbation'], label='Perturbation', color='g')

    line_magnus, = ax4.plot(k_line, initial_data['magnus_operator_norm'], label='Magnus op. norm', color='k')

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
        line_eigen1.set_ydata(eigenvalues[band_index1])
        line_eigen2.set_ydata(eigenvalues[band_index2])

        line_trace.set_ydata(data['trace'])
        line_perturb.set_ydata(data['perturbation'])
        line_magnus.set_ydata(data['magnus_operator_norm'])

        ax1.set_title(f'QGT Trace, Eigenvalues, Perturbation — $\omega$ = {data["omega"]:.6f}')

        lines = [line_eigen1, line_eigen2, line_trace, line_perturb, line_magnus]
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

def load_qgt(folder_name):
    """Load QGT entries (np object array) and meta dict from a sweep folder."""
    base = os.path.join(os.getcwd(), "results", "2D_QGT_omega_sweep", folder_name)
    qgt_path  = os.path.join(base, "QGT_2D.npy")
    meta_path = os.path.join(base, "meta_info.pkl")
    if not os.path.exists(qgt_path):
        raise FileNotFoundError(f"QGT data not found in '{base}'.")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    entries = np.load(qgt_path, allow_pickle=True)
    return entries, meta

def filter_entries_by_omega(entries, omega_min=None, omega_max=None):
    """Return a list of entries whose float(entry['omega']) lies in [omega_min, omega_max]."""
    def _in_range(w):
        if (omega_min is not None) and (w < omega_min): return False
        if (omega_max is not None) and (w > omega_max): return False
        return True
    filtered = [e for e in entries if _in_range(float(e["omega"]))]
    if len(filtered) == 0:
        raise ValueError("No omega slices fall within the requested range.")
    return filtered


# --------------------- plotting routines ---------------------

def plot_trace_std_vs_omega(folder_name, omega_min=None, omega_max=None):
    """
    Plot the standard deviation of QGT trace over the Brillouin zone for each omega.
    """
    entries, _ = load_qgt(folder_name)
    filtered = filter_entries_by_omega(entries, omega_min, omega_max)

    # collect values
    omegas = np.array([float(e["omega"]) for e in filtered], dtype=float)
    trace_std = np.array([np.nanstd(e["trace"]) for e in filtered], dtype=float)

    # sort by omega for a clean line
    order = np.argsort(omegas)
    omegas, trace_std = omegas[order], trace_std[order]

    plt.figure(figsize=(8, 5))
    plt.plot(omegas, trace_std, marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel("Drive Frequency ω")
    plt.ylabel("Std. Dev of QGT Trace over BZ")
    plt.title("Fluctuation of QGT Trace vs ω")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_berry_std_vs_omega(folder_name, omega_min=None, omega_max=None, *,
                            use_precomputed=False,  # set True if entries include 'berry'
                            convert_from_imQ=True   # if not precomputed: Ω = -2 * Im(Q_xy)
):
    """
    Plot the standard deviation of Berry curvature over the Brillouin zone for each omega.
    """
    entries, _ = load_qgt(folder_name)
    filtered = filter_entries_by_omega(entries, omega_min, omega_max)

    omegas = np.array([float(e["omega"]) for e in filtered], dtype=float)
    berry_std = []
    for e in filtered:
        if use_precomputed and ("berry" in e):
            berry = np.asarray(e["berry"])
        else:
            gim = np.asarray(e["g_xy_imag"])
            berry = (-2.0 * gim) if convert_from_imQ else gim  # sign choice doesn't affect std
        berry_std.append(np.nanstd(berry))
    berry_std = np.array(berry_std, dtype=float)

    order = np.argsort(omegas)
    omegas, berry_std = omegas[order], berry_std[order]

    plt.figure(figsize=(8, 5))
    plt.plot(omegas, berry_std, marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel("Drive Frequency ω")
    plt.ylabel("Std. Dev of Berry Curvature over BZ")
    plt.title("Fluctuation of Berry Curvature vs ω")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_integrated_trace_minus_berry(folder_name, omega_min=None, omega_max=None, *,
                                      use_precomputed=False,  # set True if entries include 'berry'
                                      convert_from_imQ=True   # if not precomputed: Ω = -2 * Im(Q_xy)
):
    """
    Plot ∫_{BZ} [Tr(g) − Ω] d^2k versus ω. Uses dkx,dky from meta for the area element.
    """
    entries, meta = load_qgt(folder_name)
    filtered = filter_entries_by_omega(entries, omega_min, omega_max)

    dkx = float(meta["dkx"]); dky = float(meta["dky"])
    area_element = dkx * dky

    omega_vals = []
    integrated_vals = []

    for e in filtered:
        omega_vals.append(float(e["omega"]))
        trace = np.asarray(e["trace"])
        if use_precomputed and ("berry" in e):
            berry = np.asarray(e["berry"])
        else:
            gim = np.asarray(e["g_xy_imag"])
            berry = (-2.0 * gim) if convert_from_imQ else gim
        integrand = trace - berry
        total = np.nansum(integrand) * area_element
        integrated_vals.append(total)

    omega_vals = np.array(omega_vals, dtype=float)
    integrated_vals = np.array(integrated_vals, dtype=float)

    order = np.argsort(omega_vals)
    omega_vals, integrated_vals = omega_vals[order], integrated_vals[order]

    plt.figure(figsize=(8, 5))
    plt.plot(omega_vals, integrated_vals, marker='o', linestyle='-')

    plt.xscale('log')
    plt.xlabel("Drive Frequency ω")
    plt.ylabel(r"$\int_{\mathrm{BZ}} \left[\mathrm{Tr}(g) - \Omega\right]\, d^2k$")
    plt.title("Integrated Tr(g) − Ω vs ω")
    plt.grid(True)
    plt.tight_layout()
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

# TwoOrbitalUnspinful

# 1D QGT

# 22.5 degrees
# dynamic_with_eigenvalues("TwoOrbitalUnspinfulHamiltonian/A00.1_polarizationleft_magnus_order1_t1_mu0_zeta1.0_a1_angle0.4_kxshift0.00_kyshift0.00_points150_kmax4.44_omega1.00e-02_5.00e_01_spacing_log_points100_1")
# dynamic_with_eigenvalues("TwoOrbitalUnspinfulHamiltonian/A00.1_polarizationleft_magnus_order1_t1_mu0_zeta1.0_a1_angle22.5_kxshift0.00_kyshift0.00_points150_kmax4.44_omega1.00e-02_5.00e-01_spacing_log_points100_1")

# 45 degrees
# dynamic_with_eigenvalues("TwoOrbitalUnspinfulHamiltonian/A00.1_polarizationleft_magnus_order1_t1_mu0_zeta1.0_a1_angle45.0_kxshift0.00_kyshift0.00_points150_kmax4.44_omega1.00e-02_5.00e_01_spacing_log_points100_1")

# dynamic_with_eigenvalues("TwoOrbitalUnspinfulHamiltonian/A00.1_polarizationlinear_x_magnus_order1_t1_mu0_zeta1.0_a1_angle45.0_kxshift0.00_kyshift0.00_points150_kmax4.44_omega1.00e-01_5.00e_01_spacing_log_points100_1")

# 0 degrees
# dynamic_with_eigenvalues("TwoOrbitalUnspinfulHamiltonian/A00.1_polarizationleft_magnus_order1_t1_mu0_zeta1.0_a1_angle0.0_kxshift0.00_kyshift0.00_points150_kmax4.44_omega1.00e-02_5.00e_01_spacing_log_points100_1")

#! Conclusion: There is practically no shift in the QGT trace. 


# Sqaure Lattice
#& t5=0
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

#! Trace std
# plot_trace_std_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t50_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega1.00e-01_5.00e_01_spacing_log_points32_1")

#! Berry std
# plot_berry_std_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t50_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega1.00e-01_5.00e_01_spacing_log_points32_1")

#! Integrated trace - Berry
# plot_integrated_trace_minus_berry("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t50_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega1.00e-01_5.00e_01_spacing_log_points32_1")


#! Right Polarization
# dynamic_2d_trace_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationright_magnus_order1_t11_t20.7071067811865475_t50_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega1.00e-01_5.00e_01_spacing_log_points32_1")



#& t5=(1-np.sqrt(2))/4
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

#! V = 6
#~ 1D QGT
# dynamic_with_eigenvalues("RhombohedralGrapheneHamiltonian/A00.1_polarizationleft_magnus_order1_vF542.1_t1355.16_V6_n5_angle0.0_kxshift0.00_kyshift0.00_points100_kmax1.57_omega5.00e_00_5.00e_03_spacing_log_points30_1")

# dynamic_with_eigenvalues("RhombohedralGrapheneHamiltonian/A00.1_polarizationright_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_angle0.0_kxshift0.00_kyshift0.00_points100_kmax1.57_omega5.00e_00_5.00e_03_spacing_log_points30_1")


#~ 2D QGT
#! Left Polarization

# Analytic Magnus Expansion

# dynamic_2d_trace_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusTrue_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")

# Valid k 

# dynamic_2d_trace_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# dynamic_2d_berry_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# plot_trace_std_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# plot_berry_std_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# plot_integrated_trace_minus_berry("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")

# square k range
# dynamic_2d_trace_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# dynamic_2d_berry_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# plot_trace_std_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# plot_berry_std_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# plot_integrated_trace_minus_berry("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")


#! Right Polarization

# Valid k 

# dynamic_2d_trace_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# dynamic_2d_berry_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# plot_trace_std_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# plot_berry_std_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")     
# plot_integrated_trace_minus_berry("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")


# Higher z limit

# dynamic_2d_trace_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_2")
# dynamic_2d_berry_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_2")
# plot_trace_std_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_2")
# plot_berry_std_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_2")
# plot_integrated_trace_minus_berry("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_2")

# square k range
# dynamic_2d_trace_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points40_1")
# dynamic_2d_berry_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points40_1")
# plot_trace_std_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points40_1")
# plot_berry_std_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points40_1")
# plot_integrated_trace_minus_berry("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_vF542.1_t1355.16_V6_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points40_1")


#! V = 30

dynamic_2d_trace_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V30_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points6_1")
# dynamic_2d_berry_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V30_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points6_1")

#! Same as above but with correct? band

# dynamic_2d_trace_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V30_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points14_1")
# dynamic_2d_berry_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_vF542.1_t1355.16_V30_n5_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points14_1")

#! V = 0
# & Left Polarization
# dynamic_2d_trace_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_vF542.1_t1355.16_V0_n5_kx-0.66_0.66_ky-0.66_0.66_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# dynamic_2d_berry_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_vF542.1_t1355.16_V0_n5_kx-0.66_0.66_ky-0.66_0.66_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# plot_trace_std_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_vF542.1_t1355.16_V0_n5_kx-0.66_0.66_ky-0.66_0.66_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# plot_berry_std_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_vF542.1_t1355.16_V0_n5_kx-0.66_0.66_ky-0.66_0.66_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")
# plot_integrated_trace_minus_berry("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_vF542.1_t1355.16_V0_n5_kx-0.66_0.66_ky-0.66_0.66_mesh150_omega5.00e_00_5.00e_03_spacing_log_points32_1")

# & Right Polarization
# dynamic_2d_trace_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_vF542.1_t1355.16_V0_n5_kx-0.66_0.66_ky-0.66_0.66_mesh150_omega5.00e_00_5.00e_03_spacing_log_points40_1")
# dynamic_2d_berry_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_vF542.1_t1355.16_V0_n5_kx-0.66_0.66_ky-0.66_0.66_mesh150_omega5.00e_00_5.00e_03_spacing_log_points40_1")
# plot_trace_std_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_vF542.1_t1355.16_V0_n5_kx-0.66_0.66_ky-0.66_0.66_mesh150_omega5.00e_00_5.00e_03_spacing_log_points40_1")
# plot_berry_std_vs_omega("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_vF542.1_t1355.16_V0_n5_kx-0.66_0.66_ky-0.66_0.66_mesh150_omega5.00e_00_5.00e_03_spacing_log_points40_1")
# plot_integrated_trace_minus_berry("RhombohedralGrapheneHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_vF542.1_t1355.16_V0_n5_kx-0.66_0.66_ky-0.66_0.66_mesh150_omega5.00e_00_5.00e_03_spacing_log_points40_1")


# Full Chiral Hamiltonian
#~ 1D QGT
# dynamic_with_eigenvalues("ChiralHamiltonian/A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_angle0.0_kxshift0.00_kyshift0.00_points100_kmax1.57_omega5.00e_00_5.00e_03_spacing_log_points30_2", band_index1=4, band_index2=5)
# dynamic_with_eigenvalues("ChiralHamiltonian/A0_0.10-V_30.00-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-polarization_right-t1_355.16-vF_542.10_angle0.0_kxshift0.00_kyshift0.00_points100_kmax1.57_omega5.00e_00_5.00e_03_spacing_log_points30_1", band_index1=4, band_index2=5)

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


# & Trial run with coarse omega grid
# dynamic_2d_trace_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points16_1")
# dynamic_2d_berry_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points16_1")
# plot_trace_std_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points16_1")
# plot_berry_std_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points16_1")
# plot_integrated_trace_minus_berry("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points16_1")

# & Left Polarization
# dynamic_2d_trace_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", omega_min=33)
# dynamic_2d_berry_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", omega_min=33)
# plot_trace_std_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", omega_min=33)
# plot_berry_std_vs_omega("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", omega_min=33)
# plot_integrated_trace_minus_berry("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", omega_min=33)

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
