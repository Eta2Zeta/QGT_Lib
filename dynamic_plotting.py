import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

def dynamic(g_results_filename):
    k_max = 1 * (np.pi)  # Maximum k value for the first Brillouin zone
    # Load the saved data
    g_results = np.load(g_results_filename, allow_pickle=True)

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

def dynamic_with_eigenvalues(g_results_filename):
    """
    Visualize QGT trace and eigenvalues dynamically for different omega values.

    Parameters:
    - g_results_filename: Path to the file containing the QGT results.
    """
    k_max = 1 * (np.pi)  # Maximum k value for the first Brillouin zone
    # Load the saved data
    g_results = np.load(g_results_filename, allow_pickle=True)

    # Extract the initial data for visualization
    initial_index = 0
    k_line = np.linspace(-k_max, k_max, 100)  # Example: same k_line as before
    initial_data = g_results[initial_index]

    # Calculate global y-axis bounds for both trace and eigenvalues
    y_min_trace = min(np.min(data['trace']) for data in g_results)
    y_max_trace = max(np.max(data['trace']) for data in g_results)
    y_min_eigen = min(np.min(data['eigenvalues']) for data in g_results)
    y_max_eigen = max(np.max(data['eigenvalues']) for data in g_results)

    y_min = min(y_min_trace, y_min_eigen)
    y_max = max(y_max_trace, y_max_eigen)

    # Plot setup
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Create the trace and eigenvalue lines
    line_trace, = ax.plot(k_line, initial_data['trace'], label=f'Trace ($\omega$={initial_data["omega"]:.6f})', color='b')
    line_eigen, = ax.plot(k_line, initial_data['eigenvalues'], label='Eigenvalues', color='r')

    ax.set_title('QGT Trace and Eigenvalues Along Line for Different $\omega$ Values')
    ax.set_xlabel('k (along line)')
    ax.set_ylabel('Value')
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
        line_trace.set_ydata(data['trace'])
        line_trace.set_label(f'Trace ($\omega$={data["omega"]:.6f})')
        line_eigen.set_ydata(data['eigenvalues'])
        ax.legend()
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

dynamic(g_results_filename="g_results_linear.npy")
# dynamic_with_eigenvalues(g_results_filename="g_results_linear.npy")