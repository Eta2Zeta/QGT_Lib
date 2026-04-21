import sys
import os
import numpy as np
import pickle
from tqdm import tqdm  # Import tqdm for progress bar
import copy
from multiprocessing import Pool, cpu_count
from functools import partial


# from Library import * 
from Library.Hamiltonian_v1 import *
from Library.Hamiltonian.Hamiltonian import * 
from Library.eigenvalue_calc_lib import *
from Library.QGT_lib import *
from Library.topology import *
from Library.utilities import *
from Library.data_management_utils_2d import *
from Library.plotting_lib_2d import *


# Define parameters
band = 0 # Which band to calculate your QMT on, starting from 0
z_cutoff = 1e3 #where to cutoff the plot for the z axis when singularties occur

# Define the temp directory for storing .npy files
temp_dir = os.path.join(os.getcwd(), "temp")

# File paths for loading the data
eigenvalues_file = os.path.join(temp_dir, "eigenvalues.npy")
eigenfunctions_file = os.path.join(temp_dir, "eigenfunctions.npy")
meta_info_file = os.path.join(temp_dir, "meta_info.pkl")  # New file for meta information

# Load the eigenvalues and eigenfunctions from files
if os.path.exists(eigenvalues_file) and os.path.exists(eigenfunctions_file) and os.path.exists(meta_info_file):
    eigenvalues = np.load(eigenvalues_file)
    eigenfunctions = np.load(eigenfunctions_file)
    with open(meta_info_file, "rb") as meta_file:
        meta_info = pickle.load(meta_file)
        Hamiltonian_Obj = meta_info["Hamiltonian_Obj"] # ALWAYS REQUIRED
        kx = meta_info["kx"] # Required for 1D
        ky = meta_info["ky"] # Required for 1D
        dkx = meta_info["dkx"]
        dky = meta_info["dky"]
        mesh_spacing = meta_info["mesh_spacing"]
        kx_range = meta_info["kx_range"]
        ky_range = meta_info["ky_range"]
    print("Loaded eigenvalues, eigenfunctions, and meta information from files.")
    print(f"Current Hamiltonian: {Hamiltonian_Obj.name}")
else:
    print("Eigenvalues or eigenfunctions files not found. Please ensure they are available at the specified paths.")
    sys.exit(1)

delta_k = min(dkx, dky)





def get_eigenstates_for_omega(hamiltonian, kx, ky, mesh_spacing):
    """
    Computes eigenvalues, eigenfunctions, and Hamiltonian matrices for a given
    Hamiltonian and k-grid.
    Simplified version of calculation_2d for internal use.
    """
    # Run the grid calculation (now returns 4 arrays)
    eigenvalues, eigenfunctions, H_array, H_prime_array = grid_eigenvalues_eigenfunctions(
        hamiltonian, kx, ky, mesh_spacing, dim=hamiltonian.dim
    )
    
    return eigenvalues, eigenfunctions, H_array, H_prime_array


def compute_qgt_for_omega(
    omega,
    hamiltonian_template,
    kx, ky,
    delta_k,
    band,
    z_cutoff
):
    """
    Compute QGT grid for a single omega value.

    Returns a dictionary with:
      - 'omega'
      - 'g_xx', 'g_xy_real', 'g_xy_imag', 'g_yy', 'trace'
      - 'eigenvalues', 'eigenfunctions'
      - 'hamiltonian_array', 'hamiltonian_prime_array'
    """

    # Copy the template Hamiltonian and set ω
    hamiltonian = copy.deepcopy(hamiltonian_template)
    hamiltonian.omega = omega

    print(f"  > Worker computing for omega = {omega:.2e}")

    # Compute eigenvalues, eigenfunctions, and Hamiltonian matrices on the grid
    eigenvalues, eigenfunctions, hamiltonian_array, hamiltonian_prime_array = get_eigenstates_for_omega(
        hamiltonian, kx, ky, mesh_spacing
    )
    print(f"  > Eigenstates computed for omega = {omega:.2e}")

    # Compute QGT components for the chosen band
    g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array = QGT_grid_num(
        kx, ky, eigenvalues, eigenfunctions, quantum_geometric_tensor_num,
        hamiltonian, delta_k, band_index=band, z_cutoff=z_cutoff
    )

    return {
        "omega": omega,
        "g_xx": g_xx_array,
        "g_xy_real": g_xy_real_array,
        "g_xy_imag": g_xy_imag_array,
        "g_yy": g_yy_array,
        "trace": trace_array,
        "eigenvalues": eigenvalues,
        "eigenfunctions": eigenfunctions,
        "hamiltonian_array": hamiltonian_array,
        "hamiltonian_prime_array": hamiltonian_prime_array,
    }


def range_of_omega_2d_par(spacing='log', omega_min=5e0, omega_max=5e3, num_omega_points=64, band = 1, force_new = False):
    """
    Parallelized: Calculate the full 2D QGT grid for a range of omega values and save the results.
    """

    # Generate omega values
    if spacing == 'log':
        omega_values = np.logspace(np.log10(omega_max), np.log10(omega_min), num_omega_points)
    elif spacing == 'linear':
        omega_values = np.linspace(omega_max, omega_min, num_omega_points)
    else:
        raise ValueError("Invalid spacing.Use 'log' or 'linear'.")

    # Ensure A0 is set
    Hamiltonian_Obj.A0 = 0.1
    Hamiltonian_Obj.polarization = "right"
    Hamiltonian_Obj.analytic_magnus = True

    # Setup directory
    file_paths, use_existing, results_subdir = setup_QGT_results_directory_2D_omega_range(
        Hamiltonian_Obj, kx_range, ky_range, mesh_spacing,
        omega_min, omega_max, num_omega_points, spacing,
        band=band,                      # <— add this
        force_new=force_new
    )


    if use_existing:
        QGT_2D = np.load(file_paths["QGT_2D"], allow_pickle=True)
        with open(file_paths["meta_info"], "rb") as meta_file:
            meta_info = pickle.load(meta_file)
        print(f"Loaded existing 2D QGT omega sweep from '{results_subdir}'")
        return

    # Prepare the partial function with fixed args
    compute_func = partial(
        compute_qgt_for_omega,
        hamiltonian_template=Hamiltonian_Obj,
        kx=kx,
        ky=ky,
        delta_k=delta_k,
        band=band,
        z_cutoff=z_cutoff
    )

    num_processes = min(cpu_count(), len(omega_values))

    # Run in parallel
    omega_qgt_results = []
    print(f"Launching parallel QGT computation on {cpu_count()} cores...")
    with Pool(processes=num_processes) as pool:
        omega_qgt_results = list(tqdm(pool.imap(compute_func, omega_values), total=len(omega_values)))



    # Save results
    np.save(file_paths["QGT_2D"], omega_qgt_results)

    meta_info = {
        "kx": kx,
        "ky": ky,
        "dkx": dkx,
        "dky": dky,
        "mesh_spacing": mesh_spacing,
        "kx_range": kx_range,
        "ky_range": ky_range,
        "omega_min": omega_min,
        "omega_max": omega_max,
        "num_omega_points": num_omega_points,
        "spacing": spacing,
        "band": int(band),                 # <— add this
        "Hamiltonian_Obj": Hamiltonian_Obj,
    }

    with open(file_paths["meta_info"], "wb") as meta_file:
        pickle.dump(meta_info, meta_file)

    print(f"✅ Saved full 2D QGT omega sweep to '{results_subdir}'")


if __name__ == '__main__':
    # Default parameters for 2D omega sweep
    # range_of_omega_2d(spacing="log")
    # range_of_omega_2d_par(spacing="log", omega_min=30, omega_max=5e3, num_omega_points=0, band = 0, force_new = True)
    # range_of_omega_2d_par(spacing="log", omega_min=30, omega_max=5e3, num_omega_points=14, band = 0, force_new = True)
    pass

