import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eigenvalue_calc_lib import spiral_eigenvalues_eigenfunctions
from QMT_lib import *
from Hamiltonian_v1 import *
from plotting_lib import *



# Define parameters
mesh_spacing = 200
dim = 6
band = 2 # Which band to calculate your QMT on
k_max = 1 * (np.pi)  # Maximum k value for the first Brillouin zone
delta_k = k_max / mesh_spacing # Small step for numerical differentiation
z_cutoff = 1e1 #where to cutoff the plot for the z axis when singularties occur


# File paths for loading the data
eigenvalues_file = "eigenvalues.npy"
eigenfunctions_file = "eigenfunctions.npy"

# Load the eigenvalues and eigenfunctions from files
if os.path.exists(eigenvalues_file) and os.path.exists(eigenfunctions_file):
    eigenvalues = np.load(eigenvalues_file)
    eigenfunctions = np.load(eigenfunctions_file)
    print("Loaded eigenvalues and eigenfunctions from files.")
else:
    print("Eigenvalues or eigenfunctions files not found. Please ensure they are available at the specified paths.")
    sys.exit(1)


# Create kx and ky arrays
kx = np.linspace(-k_max, k_max, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)

# Calculate QGT components
# g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array = QGT_grid(
#     kx, ky, eigenvalues, eigenfunctions, quantum_geometric_tensor_num, 
#     H_THF, delta_k, dim, band_index=band, z_cutoff=z_cutoff
# )




# # plot_g_components_3d(kx, ky, g_xx_array, g_yy_array, trace_array)
# plot_g_components_2d(kx, ky, g_xx_array, g_yy_array, trace_array)

# plot_trace_w_eigenvalue(kx, ky, g_xx_array, g_yy_array, eigenvalues, trace_array, eigenvalue_band=band)

# Define the line parameters
angle_deg = 90  # Line angle in degrees
angle_rad = np.deg2rad(angle_deg)
num_points = 500  # Number of points along the line
k_line = np.linspace(-k_max, k_max, num_points)


def range_of_G(spacing='log', G_min=1e-6, G_max=0.02, num_points=100):
    """
    Calculate QGT for a range of G values and save the results to a file.
    The output file name is dynamically set based on the spacing type.

    Parameters:
    - spacing: Type of spacing for G values ('log' or 'linear').
    - G_min: Minimum value of G.
    - G_max: Maximum value of G.
    - num_points: Number of G values.

    Returns:
    - None: Saves the results to a file.
    """
    # Generate G values based on the specified spacing
    if spacing == 'log':
        G_values = np.logspace(np.log10(G_max), np.log10(G_min), num_points)
        file_name = "g_results_log.npy"
    elif spacing == 'linear':
        G_values = np.linspace(G_max, G_min, num_points)
        file_name = "g_results_linear.npy"
    else:
        raise ValueError("Invalid spacing. Choose 'log' or 'linear'.")

    # Initialize a list to store results for each G
    g_results = []

    # Loop over the G values
    for G in G_values:
        # Create the Hamiltonian for the current G
        H_THF_current = H_THF_factory(G)
        
        # Calculate QGT along the line
        g_xx, g_xy_real, g_xy_imag, g_yy, trace = QGT_line(
            H_THF_current, k_line, angle_rad, delta_k, dim, band_index=band
        )
        
        # Store the results as a dictionary for this G
        g_results.append({
            'G': G,
            'g_xx': g_xx,
            'g_xy_real': g_xy_real,
            'g_xy_imag': g_xy_imag,
            'g_yy': g_yy,
            'trace': trace,
        })

    # Save the results to an .npy file with the dynamically set name
    np.save(file_name, g_results)
    print(f"Results saved to {file_name}")


range_of_G(spacing="linear")

exit()
