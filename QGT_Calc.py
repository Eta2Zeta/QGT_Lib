import sys
import os
import numpy as np
import pickle
from tqdm import tqdm  # Import tqdm for progress bar


# from Library import * 
from Library.plotting_lib import *
from Library.Hamiltonian_v1 import *
from Library.Hamiltonian_v2 import * 
from Library.eigenvalue_calc_lib import *
from Library.QGT_lib import *
from Library.topology import *


# Define parameters
band = 1 # Which band to calculate your QMT on, starting from 0
z_cutoff = 1e1 #where to cutoff the plot for the z axis when singularties occur


# Define the temp directory for storing .npy files
temp_dir = os.path.join(os.getcwd(), "temp")

# File paths for loading the data
eigenvalues_file = os.path.join(temp_dir, "eigenvalues.npy")
eigenfunctions_file = os.path.join(temp_dir, "eigenfunctions.npy")
meta_info_file = os.path.join(temp_dir, "meta_info.npy")  # New file for meta information

# Load the eigenvalues and eigenfunctions from files
if os.path.exists(eigenvalues_file) and os.path.exists(eigenfunctions_file) and os.path.exists(meta_info_file):
    eigenvalues = np.load(eigenvalues_file)
    eigenfunctions = np.load(eigenfunctions_file)
    with open(meta_info_file, "rb") as meta_file:
        meta_info = pickle.load(meta_file)
        kx = meta_info["kx"]
        ky = meta_info["ky"]
        dkx = meta_info["dkx"]
        dky = meta_info["dky"]
        mesh_spacing = meta_info["mesh_spacing"]
        Hamiltonian_Obj = meta_info["Hamiltonian_Obj"]
    print("Loaded eigenvalues, eigenfunctions, and meta information from files.")
else:
    print("Eigenvalues or eigenfunctions files not found. Please ensure they are available at the specified paths.")
    sys.exit(1)

def max_grid_spacing(kx, ky):
    # Calculate the largest difference in kx and ky to determine k_max
    kx_diff = np.max(np.abs(np.diff(kx, axis=1)))  # Differences along kx
    ky_diff = np.max(np.abs(np.diff(ky, axis=0)))  # Differences along ky
    return max(kx_diff, ky_diff)  # Take the maximum difference

delta_k = max_grid_spacing(kx, ky)




# Calculate QGT components
# g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array = QGT_grid(
#     kx, ky, eigenvalues, eigenfunctions, quantum_geometric_tensor_num, 
#     Hamiltonian_Obj, delta_k, band_index=band, z_cutoff=z_cutoff
# )


# chern_number = compute_chern_number(g_xy_imag_array, dkx, dky)

# print("Chern number is: ", chern_number)

# plot_QGT_components_3d(kx, ky, g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array)

# plot_QMT_wtrace_3d(kx, ky, g_xx_array, g_yy_array, trace_array)

# # plot_g_components_2d(g_xx_array, g_yy_array, trace_array, k_max=k_max)

# plot_trace_w_eigenvalue(kx, ky, g_xx_array, g_yy_array, eigenvalues, trace_array, eigenvalue_band=band)


# Define the line parameters
angle_deg = 0  # Line angle in degrees
k_angle = np.deg2rad(angle_deg) # Convert into Radians
kx_shift = 0
ky_shift = 0
num_points = 300  # Number of points along the line
k_max = 1 * (np.pi)
k_line = np.linspace(-k_max, k_max, num_points)
line_kx = k_line * np.cos(k_angle) + kx_shift
line_ky = k_line * np.sin(k_angle) + ky_shift

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
            H_THF_current, line_kx, line_ky, delta_k, band_index=band
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

# range_of_G()


# Define the line parameters
angle_deg = 45  # Line angle in degrees
k_angle = np.deg2rad(angle_deg) # Convert into Radians
kx_shift = 0
ky_shift = -np.pi/2
# ky_shift = 0
num_points = 100  # Number of points along the line
# k_max = 1 * (np.pi)
k_max = np.sqrt(2) * (np.pi)
k_line = np.linspace(-k_max, k_max, num_points)
line_kx = k_line * np.cos(k_angle) + kx_shift
line_ky = k_line * np.sin(k_angle) + ky_shift



def range_of_omega(spacing='log', omega_min=1e-2, omega_max=1e-1, num_points=100):
    """
    Calculate QGT for a range of omega values and save the results to a file.
    The output file name is dynamically set based on the spacing type.

    Parameters:
    - spacing: Type of spacing for omega values ('log' or 'linear').
    - omega: Minimum value of omega.
    - omega: Maximum value of omega.
    - num_points: Number of omega values.

    Returns:
    - None: Saves the results to a file.
    """
    # Give some amplitude to the light
    Hamiltonian_Obj.A0 = 1

    # Generate omega values based on the specified spacing
    if spacing == 'log':
        omega_values = np.logspace(np.log10(omega_max), np.log10(omega_min), num_points)
        file_name = "g_results_log.npy"
    elif spacing == 'linear':
        omega_values = np.linspace(omega_max, omega_min, num_points)
        file_name = "g_results_linear.npy"
    else:
        raise ValueError("Invalid spacing. Choose 'log' or 'linear'.")

    # Initialize a list to store results for each G
    g_results = []

    # Use tqdm to create a progress bar for the loop
    for omega in tqdm(omega_values, desc="Processing omega values", unit="omega"):
        # Create the Hamiltonian for the current G
        Hamiltonian_Obj.omega = omega
        
        # Calculate QGT along the line
        eigenvalues, g_xx, g_xy_real, g_xy_imag, g_yy, trace = QGT_line(
            Hamiltonian_Obj, line_kx, line_ky, delta_k, band_index=band
        )
        
        # Store the results as a dictionary for this G
        g_results.append({
            'omega': omega,
            'g_xx': g_xx,
            'g_xy_real': g_xy_real,
            'g_xy_imag': g_xy_imag,
            'g_yy': g_yy,
            'trace': trace,
            'eigenvalues': eigenvalues
        })

    # Save the results to an .npy file with the dynamically set name
    np.save(file_name, g_results)
    print(f"Results saved to {file_name}")




range_of_omega(spacing="linear")

exit()
