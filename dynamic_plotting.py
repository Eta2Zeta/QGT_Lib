import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle

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



def dynamic_with_eigenvalues(folder_name):
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
    y_min_trace = min(np.min(data['trace']) for data in g_results)
    y_max_trace = max(np.max(data['trace']) for data in g_results)
    y_min_perturb = min(np.min(data['perturbation']) for data in g_results)
    y_max_perturb = max(np.max(data['perturbation']) for data in g_results)
    y_min_eigen = min(np.min(data['eigenvalues']) for data in g_results)
    y_max_eigen = max(np.max(data['eigenvalues']) for data in g_results)

    eigen_buffer = 0.1 * (y_max_eigen - y_min_eigen)  # Buffer for eigenvalues

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.2, right=0.8)  # Leave room for third y-axis and slider

    ax2 = ax1.twinx()  # Second y-axis
    ax3 = ax1.twinx()  # Third y-axis
    ax3.spines['right'].set_position(('outward', 60))
    ax3.spines['right'].set_visible(True)

    eigenvalues = np.array(initial_data['eigenvalues']).T  # Transpose from (N, 2) → (2, N)


    # Plot initial data (eigenvalues now has shape [2, N])
    line_eigen1, = ax1.plot(k_line, eigenvalues[0], label='Eigenvalue 1', color='r')
    line_eigen2, = ax1.plot(k_line, eigenvalues[1], label='Eigenvalue 2', color='m')

    line_trace, = ax2.plot(k_line, initial_data['trace'], label='Trace', color='b')
    line_perturb, = ax3.plot(k_line, initial_data['perturbation'], label='Perturbation', color='g')

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
        line_eigen1.set_ydata(eigenvalues[0])
        line_eigen2.set_ydata(eigenvalues[1])

        line_trace.set_ydata(data['trace'])
        line_perturb.set_ydata(data['perturbation'])

        ax1.set_title(f'QGT Trace, Eigenvalues, Perturbation — $\omega$ = {data["omega"]:.6f}')

        lines = [line_eigen1, line_eigen2, line_trace, line_perturb]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper left")

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


def dynamic_2d_trace_vs_omega(folder_name):
    """
    Dynamically visualize the QGT trace (2D heatmap) as a function of omega.

    Parameters:
        folder_name (str): Name of the subfolder in 'results/2D_QGT_omega_sweep/' containing the results.
    """
    results_path = os.path.join(os.getcwd(), "results", "2D_QGT_omega_sweep", folder_name)
    qgt_data_path = os.path.join(results_path, "QGT_2D.npy")
    meta_path = os.path.join(results_path, "meta_info.pkl")

    if not os.path.exists(qgt_data_path):
        raise FileNotFoundError(f"QGT data not found in '{results_path}'.")

    # Load data
    with open(meta_path, "rb") as f:
        meta_info = pickle.load(f)

    qgt_data = np.load(qgt_data_path, allow_pickle=True)

    kx = meta_info["kx"]
    ky = meta_info["ky"]
    omega_values = [entry["omega"] for entry in qgt_data]

    # Compute min/max for fixed color scaling
    min_trace = min(np.min(entry["trace"]) for entry in qgt_data)
    max_trace = max(np.max(entry["trace"]) for entry in qgt_data)

    # Initial data
    initial_index = 0
    trace = qgt_data[initial_index]["trace"]

    # Setup figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(bottom=0.2)

    img = ax.imshow(
        trace,
        origin='lower',
        extent=[kx.min(), kx.max(), ky.min(), ky.max()],
        cmap='inferno',
        vmin=min_trace,
        vmax=max_trace,
        aspect='auto'
    )

    ax.set_title(f'QGT Trace — $\omega$ = {omega_values[initial_index]:.6f}')
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Trace Amplitude")

    # Slider for omega
    ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, '$\omega$', 0, len(omega_values) - 1, valinit=initial_index, valstep=1)

    def update(val):
        index = int(slider.val)
        trace = qgt_data[index]["trace"]

        img.set_data(trace)
        ax.set_title(f'QGT Trace — $\omega$ = {omega_values[index]:.6f}')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

# 1D QGT

# TwoOrbitalUnspinful
# 22.5 degrees
# dynamic_with_eigenvalues("TwoOrbitalUnspinfulHamiltonian/A00.1_polarizationleft_magnus_order1_t1_mu0_zeta1.0_a1_angle0.4_kxshift0.00_kyshift0.00_points150_kmax4.44_omega1.00e-02_5.00e_01_spacing_log_points100_1")
# dynamic_with_eigenvalues("TwoOrbitalUnspinfulHamiltonian/A00.1_polarizationleft_magnus_order1_t1_mu0_zeta1.0_a1_angle22.5_kxshift0.00_kyshift0.00_points150_kmax4.44_omega1.00e-02_5.00e-01_spacing_log_points100_1")

# 45 degrees
# dynamic_with_eigenvalues("TwoOrbitalUnspinfulHamiltonian/A00.1_polarizationleft_magnus_order1_t1_mu0_zeta1.0_a1_angle45.0_kxshift0.00_kyshift0.00_points150_kmax4.44_omega1.00e-02_5.00e_01_spacing_log_points100_1")

# 0 degrees
# dynamic_with_eigenvalues("TwoOrbitalUnspinfulHamiltonian/A00.1_polarizationleft_magnus_order1_t1_mu0_zeta1.0_a1_angle0.0_kxshift0.00_kyshift0.00_points150_kmax4.44_omega1.00e-02_5.00e_01_spacing_log_points100_1")

# Conclusion: There is practically no shift in the QGT trace. 

# # Sqaure Lattice
# t5=0
# Along the shifted 45 degree line
# dynamic_with_eigenvalues("1D_QGT_SquareLatticeHamiltonian_dim2_omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t50_angle0.8_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega5.00e-01_5.00e_01_spacing_log_points100_2")

# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t50_angle45.0_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega1.00e-02_1.00e_01_spacing_log_points100_1")

# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t50_angle45.0_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega5.00e-02_1.00e_01_spacing_log_points100_1")

# Along the shifted 0 degree line
# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t50_angle0.0_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega5.00e-02_5.00e_01_spacing_log_points100_1")


# t5=(1-np.sqrt(2))/4
# dynamic_with_eigenvalues("1D_QGT_SquareLatticeHamiltonian_dim2_omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_angle0.8_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega5.00e-01_5.00e_01_spacing_log_points100_1")

# dynamic_with_eigenvalues("SquareLatticeHamiltonian/A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_angle45.0_kxshift0.00_kyshift-1.57_points150_kmax4.44_omega5.00e-02_5.00e_01_spacing_log_points100_1")


# 2D QGT
# dynamic_2d_trace_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega5.00e-02_5.00e_01_spacing_log_points4_2")
dynamic_2d_trace_vs_omega("SquareLatticeHamiltonian/omega5.0_A00.1_polarizationleft_magnus_order1_t11_t20.7071067811865475_t5-0.10355339059327379_kx-3.14_3.14_ky-3.14_3.14_mesh150_omega5.00e-02_5.00e_01_spacing_log_points32_1")