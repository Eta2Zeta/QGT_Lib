import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


k_max = 1 * (np.pi)  # Maximum k value for the first Brillouin zone
# Load the saved data
g_results = np.load("g_results_linear.npy", allow_pickle=True)

# Extract the initial data for visualization
initial_index = 0
k_line = np.linspace(-k_max, k_max, 500)  # Example: same k_line as before
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
