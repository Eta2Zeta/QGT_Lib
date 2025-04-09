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
from Library.Hamiltonian_v2 import * 
from Library.eigenvalue_calc_lib import *
from Library.QGT_lib import *
from Library.topology import *
from Library.utilities import *
from Library.plotting_lib import *


# Define parameters
band = 1 # Which band to calculate your QMT on, starting from 0
z_cutoff = 1e1 #where to cutoff the plot for the z axis when singularties occur


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
        kx = meta_info["kx"]
        ky = meta_info["ky"]
        dkx = meta_info["dkx"]
        dky = meta_info["dky"]
        mesh_spacing = meta_info["mesh_spacing"]
        Hamiltonian_Obj = meta_info["Hamiltonian_Obj"]
        kx_range = meta_info["kx_range"]
        ky_range = meta_info["ky_range"]
        mesh_spacing = meta_info["mesh_spacing"]
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

def calculate_2d():
    file_paths, use_existing, results_subdir = setup_QGT_results_directory(Hamiltonian_Obj, kx_range, ky_range, mesh_spacing)
    if use_existing:
        # Load existing QGT data
        g_xx_array = np.load(file_paths["g_xx"])
        g_xy_real_array = np.load(file_paths["g_xy_real"])
        g_xy_imag_array = np.load(file_paths["g_xy_imag"])
        g_yy_array = np.load(file_paths["g_yy"])
        trace_array = np.load(file_paths["trace"])

        with open(file_paths["meta_info"], "rb") as meta_file:
            qgt_meta_info = pickle.load(meta_file)

        print("Loaded QGT data from existing files.")


    else:
        # Compute QGT if data does not exist
        g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array = QGT_grid(
            kx, ky, eigenvalues, eigenfunctions, quantum_geometric_tensor_num, 
            Hamiltonian_Obj, delta_k, band_index=band, z_cutoff=z_cutoff
        )

        # Save QGT results
        for key, array in {
            "g_xx": g_xx_array,
            "g_xy_real": g_xy_real_array,
            "g_xy_imag": g_xy_imag_array,
            "g_yy": g_yy_array,
            "trace": trace_array
        }.items():
            np.save(file_paths[key], array)
            np.save(os.path.join(temp_dir, os.path.basename(file_paths[key])), array)  # Save to temp directory

        # Save QGT metadata
        qgt_meta_info = {
            "kx": kx,
            "ky": ky,
            "dkx": dkx, 
            "dky": dky,
            "mesh_spacing": mesh_spacing,
            "Hamiltonian_Obj": Hamiltonian_Obj  
        }

        with open(file_paths["meta_info"], "wb") as meta_file:
            pickle.dump(qgt_meta_info, meta_file)
        with open(os.path.join(temp_dir, "qgt_meta_info.pkl"), "wb") as meta_file:
            pickle.dump(qgt_meta_info, meta_file)  # Save to temp directory

        print(f"Saved QGT results to '{results_subdir}' and copied to temp directory: {temp_dir}")




    # chern_number = compute_chern_number(g_xy_imag_array, dkx, dky)

    # print("Chern number is: ", chern_number)

    plot_QGT_components_3d(kx, ky, g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array)

    # plot_g_components_2d(g_xx_array, g_yy_array, trace_array, k_max=k_max)

    plot_trace_w_eigenvalue(kx, ky, g_xx_array, g_yy_array, eigenvalues, trace_array, eigenvalue_band=band)




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
    

    # Generate G values based on the specified spacing
    if spacing == 'log':
        G_values = np.logspace(np.log10(G_max), np.log10(G_min), num_points)
    elif spacing == 'linear':
        G_values = np.linspace(G_max, G_min, num_points)
    else:
        raise ValueError("Invalid spacing. Choose 'log' or 'linear'.")
    
    # Create the results directory TODO: Correct this
    file_paths, use_existing, results_subdir = setup_QGT_results_directory(Hamiltonian_Obj, kx_range, ky_range, num_points, k_max, G_min, G_max, spacing)

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




def range_of_omega(spacing='linear', omega_min=5e-2, omega_max=5e1, num_k_points=150, num_omega_points=100):
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

    # Define the line parameters
    # angle_deg = 0  # Line angle in degrees
    angle_deg = 45  # Line angle in degrees
    # angle_deg = 22.5  # Line angle in degrees
    k_angle = np.deg2rad(angle_deg) # Convert into Radians
    kx_shift = 0
    ky_shift = -np.pi/2
    # ky_shift = 0
    # k_max = 1 * (np.pi) # Choose this when you do 0 or 90 degrees
    k_max = np.sqrt(2) * (np.pi) # Choose this when you do 45 degrees
    k_line = np.linspace(-k_max, k_max, num_k_points)
    line_kx = k_line * np.cos(k_angle) + kx_shift
    line_ky = k_line * np.sin(k_angle) + ky_shift


    # Give some amplitude to the light
    Hamiltonian_Obj.A0 = 0.1

    # Generate omega values based on the specified spacing
    if spacing == 'log':
        omega_values = np.logspace(np.log10(omega_max), np.log10(omega_min), num_omega_points)
    elif spacing == 'linear':
        omega_values = np.linspace(omega_max, omega_min, num_omega_points)
    else:
        raise ValueError("Invalid spacing. Choose 'log' or 'linear'.")
    

    # Create the results directory TODO: Correct this
    file_paths, use_existing, results_subdir = setup_QGT_results_directory_1D(Hamiltonian_Obj, angle_deg, kx_shift, ky_shift, num_k_points, num_omega_points, k_max, omega_min, omega_max, spacing, force_new= True)

    if use_existing:
        # Load existing QGT results
        QGT_1D = np.load(file_paths["QGT_1D"], allow_pickle=True)

        with open(file_paths["meta_info"], "rb") as meta_file:
            meta_info = pickle.load(meta_file)

        print(f"Loaded existing QGT 1D results from '{results_subdir}'.")
        return

    # Initialize list to store QGT results
    g_results = []

    # Use tqdm to track progress over omega values
    for omega in tqdm(omega_values, desc="Processing omega values", unit="omega"):
        # Update the Hamiltonian with the current omega
        Hamiltonian_Obj.omega = omega
        
        # Calculate QGT along the line
        eigenvalues, permutations, g_xx, g_xy_real, g_xy_imag, g_yy, trace = QGT_line(
            Hamiltonian_Obj, line_kx, line_ky, delta_k, band_index=band
        )
        
        # Store the results
        g_results.append({
            'omega': omega,
            'g_xx': g_xx,
            'g_xy_real': g_xy_real,
            'g_xy_imag': g_xy_imag,
            'g_yy': g_yy,
            'trace': trace,
            'eigenvalues': eigenvalues,
            'perturbation': permutations
        })

    # Save the results
    np.save(file_paths["QGT_1D"], g_results)

    # Save metadata
    meta_info = {
        "k_angle": k_angle,
        "kx_shift": kx_shift,
        "ky_shift": ky_shift,
        "num_k_points": num_k_points,
        "num_omega_points": num_omega_points,
        "k_max": k_max,
        "omega_min": omega_min,
        "omega_max": omega_max,
        "spacing": spacing,
        "Hamiltonian_Obj": Hamiltonian_Obj  
    }
    
    with open(file_paths["meta_info"], "wb") as meta_file:
        pickle.dump(meta_info, meta_file)

    print(f"Saved QGT 1D results to '{results_subdir}'.")


def range_of_omega_2d(spacing='linear', omega_min=5e-2, omega_max=5e1, num_omega_points=4):
    """
    Calculate the full 2D QGT grid for a range of omega values and save the results.

    Parameters:
    - spacing: 'log' or 'linear'
    - omega_min: Minimum omega value
    - omega_max: Maximum omega value
    - num_omega_points: Number of omega samples

    Returns:
    - None (saves results and metadata to file)
    """
    # Use either logarithmic or linear spacing
    if spacing == 'log':
        omega_values = np.logspace(np.log10(omega_max), np.log10(omega_min), num_omega_points)
    elif spacing == 'linear':
        omega_values = np.linspace(omega_max, omega_min, num_omega_points)
    else:
        raise ValueError("Invalid spacing. Use 'log' or 'linear'.")

    # Set some default amplitude
    Hamiltonian_Obj.A0 = 0.1

    # Directory setup
    file_paths, use_existing, results_subdir = setup_QGT_results_directory_2D_omega_range(
        Hamiltonian_Obj, kx_range, ky_range, mesh_spacing, omega_min, omega_max, num_omega_points, spacing, force_new=True
    )

    if use_existing:
        QGT_2D = np.load(file_paths["QGT_2D"], allow_pickle=True)
        with open(file_paths["meta_info"], "rb") as meta_file:
            meta_info = pickle.load(meta_file)
        print(f"Loaded existing 2D QGT omega sweep from '{results_subdir}'")
        return

    # Accumulate results for each omega
    omega_qgt_results = []

    for omega in tqdm(omega_values, desc="Processing omega values (2D)", unit="omega"):
        # Set omega and compute QGT
        Hamiltonian_Obj.omega = omega

        g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array = QGT_grid(
            kx, ky, eigenvalues, eigenfunctions, quantum_geometric_tensor_num,
            Hamiltonian_Obj, delta_k, band_index=band, z_cutoff=z_cutoff
        )

        omega_qgt_results.append({
            "omega": omega,
            "g_xx": g_xx_array,
            "g_xy_real": g_xy_real_array,
            "g_xy_imag": g_xy_imag_array,
            "g_yy": g_yy_array,
            "trace": trace_array
        })

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
        "Hamiltonian_Obj": Hamiltonian_Obj,
    }

    with open(file_paths["meta_info"], "wb") as meta_file:
        pickle.dump(meta_info, meta_file)

    print(f"Saved full 2D QGT omega sweep to '{results_subdir}'")



def compute_qgt_for_omega(omega, hamiltonian_template, kx, ky, delta_k, eigenvalues, eigenfunctions, band, z_cutoff):
    """
    Compute QGT grid for a single omega value.

    Returns:
        Dictionary with QGT components and omega.
    """

    hamiltonian = copy.deepcopy(hamiltonian_template)
    hamiltonian.omega = omega

    g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array = QGT_grid(
        kx, ky, eigenvalues, eigenfunctions, quantum_geometric_tensor_num,
        hamiltonian, delta_k, band_index=band, z_cutoff=z_cutoff
    )

    return {
        "omega": omega,
        "g_xx": g_xx_array,
        "g_xy_real": g_xy_real_array,
        "g_xy_imag": g_xy_imag_array,
        "g_yy": g_yy_array,
        "trace": trace_array
    }






def range_of_omega_2d_par(spacing='linear', omega_min=5e-2, omega_max=5e1, num_omega_points=32):
    """
    Parallelized: Calculate the full 2D QGT grid for a range of omega values and save the results.
    """

    # Generate omega values
    if spacing == 'log':
        omega_values = np.logspace(np.log10(omega_max), np.log10(omega_min), num_omega_points)
    elif spacing == 'linear':
        omega_values = np.linspace(omega_max, omega_min, num_omega_points)
    else:
        raise ValueError("Invalid spacing. Use 'log' or 'linear'.")

    # Ensure A0 is set
    Hamiltonian_Obj.A0 = 0.1

    # Setup directory
    file_paths, use_existing, results_subdir = setup_QGT_results_directory_2D_omega_range(
        Hamiltonian_Obj, kx_range, ky_range, mesh_spacing, omega_min, omega_max, num_omega_points, spacing, force_new=True
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
        eigenvalues=eigenvalues,
        eigenfunctions=eigenfunctions,
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
        "Hamiltonian_Obj": Hamiltonian_Obj,
    }

    with open(file_paths["meta_info"], "wb") as meta_file:
        pickle.dump(meta_info, meta_file)

    print(f"✅ Saved full 2D QGT omega sweep to '{results_subdir}'")

if __name__ == '__main__':

    # calculate_2d()
    # range_of_omega(spacing="log")
    # range_of_omega_2d(spacing="log")
    range_of_omega_2d_par(spacing="log")
