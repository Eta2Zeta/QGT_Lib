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

# def dynamic_with_eigenvalues(folder_name):
#     """
#     Visualize QGT trace and eigenvalues dynamically for different omega values.

#     Parameters:
#     """
#     k_max = 1 * (np.pi)  # Maximum k value for the first Brillouin zone
#     # Load the saved data
#     g_results_filepath = os.path.join(os.getcwd(), "results", "1D_QGT_results",  folder_name, "QGT_1D.npy")
#     print(g_results_filepath)

#     # Check if the file exists
#     if not os.path.exists(g_results_filepath):
#         raise FileNotFoundError(f"File '{g_results_filepath}' not found in the 'results' directory.")


#     g_results = np.load(g_results_filepath, allow_pickle=True)

#     # Extract the initial data for visualization
#     initial_index = 0
#     k_line = np.linspace(-k_max, k_max, 100)  # Example: same k_line as before
#     initial_data = g_results[initial_index]

#     # Calculate global y-axis bounds for both trace and eigenvalues
#     y_min_trace = min(np.min(data['trace']) for data in g_results)
#     y_max_trace = max(np.max(data['trace']) for data in g_results)
#     y_min_eigen = min(np.min(data['eigenvalues']) for data in g_results)
#     y_max_eigen = max(np.max(data['eigenvalues']) for data in g_results)

#     y_min = min(y_min_trace, y_min_eigen)
#     y_max = max(y_max_trace, y_max_eigen)

#     # Plot setup
#     fig, ax = plt.subplots()
#     plt.subplots_adjust(bottom=0.25)

#     # Create the trace and eigenvalue lines
#     line_trace, = ax.plot(k_line, initial_data['trace'], label=f'Trace ($\omega$={initial_data["omega"]:.6f})', color='b')
#     line_eigen, = ax.plot(k_line, initial_data['eigenvalues'], label='Eigenvalues', color='r')

#     ax.set_title('QGT Trace and Eigenvalues Along Line for Different $\omega$ Values')
#     ax.set_xlabel('k (along line)')
#     ax.set_ylabel('Value')
#     ax.legend()
#     ax.grid(True)

#     # Set y-axis limits dynamically
#     ax.set_yscale('linear')
#     ax.set_ylim(y_min, y_max)

#     # Slider setup
#     ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
#     slider = Slider(ax_slider, '$\omega$', 0, len(g_results) - 1, valinit=initial_index, valstep=1)

#     # Slider update function
#     def update(val):
#         index = int(slider.val)
#         data = g_results[index]
#         line_trace.set_ydata(data['trace'])
#         line_trace.set_label(f'Trace ($\omega$={data["omega"]:.6f})')
#         line_eigen.set_ydata(data['eigenvalues'])
#         ax.legend()
#         fig.canvas.draw_idle()

#     slider.on_changed(update)

#     plt.show()


def dynamic_with_eigenvalues(folder_name):
    """
    Visualize QGT trace and eigenvalues dynamically for different omega values.

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

    num_points = meta_info['num_points']


    # Check if the file exists
    if not os.path.exists(g_results_filepath):
        raise FileNotFoundError(f"File '{g_results_filepath}' not found in the 'results' directory.")

    g_results = np.load(g_results_filepath, allow_pickle=True)

    # Extract the initial data for visualization
    initial_index = 0
    k_line = np.linspace(-k_max, k_max, num_points)  # Same k_line as before
    initial_data = g_results[initial_index]

    # Calculate global y-axis bounds for trace and eigenvalues separately
    y_min_trace = min(np.min(data['trace']) for data in g_results)
    y_max_trace = max(np.max(data['trace']) for data in g_results)
    y_min_eigen = min(np.min(data['eigenvalues']) for data in g_results)
    y_max_eigen = max(np.max(data['eigenvalues']) for data in g_results)

    # Plot setup
    fig, ax1 = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Create the secondary y-axis
    ax2 = ax1.twinx()

    # Create the trace and eigenvalue lines
    line_eigen, = ax1.plot(k_line, initial_data['eigenvalues'], label='Eigenvalues', color='r')
    line_trace, = ax2.plot(k_line, initial_data['trace'], label=f'Trace ($\omega$={initial_data["omega"]:.6f})', color='b')

    # Formatting primary y-axis (left, for eigenvalues)
    ax1.set_ylabel('Eigenvalues', color='r')
    ax1.set_xlabel('k (along line)')
    ax1.set_ylim(y_min_eigen, y_max_eigen)
    ax1.tick_params(axis='y', labelcolor='r')

    # Formatting secondary y-axis (right, for trace)
    ax2.set_ylabel('Trace Amplitude', color='b')
    ax2.set_ylim(y_min_trace, y_max_trace)
    ax2.tick_params(axis='y', labelcolor='b')

    ax1.set_title('QGT Trace and Eigenvalues Along Line for Different $\omega$ Values')
    ax1.grid(True)

    # Slider setup
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, '$\omega$', 0, len(g_results) - 1, valinit=initial_index, valstep=1)

    # Slider update function
    def update(val):
        index = int(slider.val)
        data = g_results[index]
        line_eigen.set_ydata(data['eigenvalues'])
        line_trace.set_ydata(data['trace'])
        line_trace.set_label(f'Trace ($\omega$={data["omega"]:.6f})')
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()



# dynamic(g_results_filename="square_lattice_45deg_0x_npio2y_100p_min1en1_max1e1_A1_band1_g_results_linear.npy")
# dynamic_with_eigenvalues(g_results_filename="square_lattice_45deg_0x_npio2y_100p_min1en2_max1en1_A1_band1_g_results_linear.npy")
dynamic_with_eigenvalues("1D_QGT_TwoOrbitalUnspinfulHamiltonian_dim2_omega10.0_A00.1_polarizationleft_magnus_order1_t1_mu0_zeta1.0_a1_angle0.4_kxshift0.00_kyshift0.00_points50_kmax4.44_omega1.00e-02_1.00e-01_spacing_linear_1")
# dynamic_with_eigenvalues("1D_QGT_TwoOrbitalUnspinfulHamiltonian_dim2_omega10.0_A00.1_polarizationleft_magnus_order1_t1_mu0_zeta1.0_a1_angle0.8_kxshift0.00_kyshift-1.57_points100_kmax4.44_omega0.01_0.10_spacing_linear_1")
