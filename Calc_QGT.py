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
from Library.Hamiltonian.Hamiltonian_v2 import * 
from Library.eigenvalue_calc_lib import *
from Library.QGT_lib import *
from Library.topology import *
from Library.utilities import *
from Library.plotting_lib import *


# Define parameters
band = 1 # Which band to calculate your QMT on, starting from 0
z_cutoff = 20 #where to cutoff the plot for the z axis when singularties occur

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

def calculate_2d():
    # Define method name for directory naming ("analytic", "numerical", etc.)
    method_name = "numerical"
    file_paths, use_existing, results_subdir = setup_QGT_results_directory(
        Hamiltonian_Obj, kx_range, ky_range, mesh_spacing, 
        force_new=True, method_name=method_name
    )
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
        Hamiltonian_Obj.A0 = 0.0
        Hamiltonian_Obj.omega = 5e3

        # Select QGT calculation method
        if method_name == "numerical":
            print("Using Numerical QGT Calculation (Projector Method)...")
            g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array = QGT_grid_num(
                kx, ky, eigenvalues, eigenfunctions, quantum_geometric_tensor_num_eigenvector_ordered, 
                Hamiltonian_Obj, delta_k=1e-5, band_index=band, z_cutoff=z_cutoff
            )
            # g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array = QGT_grid_num(
            #     kx, ky, eigenvalues, eigenfunctions, quantum_geometric_tensor_num, 
            #     Hamiltonian_Obj, delta_k=1e-5, band_index=band, z_cutoff=z_cutoff
            # )
        
        elif method_name == "array_analytic":
            print("Using Analytic QGT Calculation (Block Diagonalization)...")
            g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array = Hamiltonian_Obj.compute_qgt_analytic(
                kx, ky, band_index=band
            )
            
            if z_cutoff is not None:
                g_xx_array = np.clip(g_xx_array, -z_cutoff, z_cutoff)
                g_xy_real_array = np.clip(g_xy_real_array, -z_cutoff, z_cutoff)
                g_xy_imag_array = np.clip(g_xy_imag_array, -z_cutoff, z_cutoff)
                g_yy_array = np.clip(g_yy_array, -z_cutoff, z_cutoff)
                trace_array = np.clip(trace_array, -z_cutoff, z_cutoff)

        elif method_name == "analytic":
            print("Using Array Analytic QGT Calculation...")
            # Assuming QGT_grid_analytic is available in imports
            g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array = QGT_grid_analytic(
                kx, ky, quantum_geometric_tensor_analytic, 
                Hamiltonian_Obj, z_cutoff=z_cutoff
            )

        elif method_name == "semi_numerical":
            print("Using Semi-Numerical QGT Calculation...")
            g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array = QGT_grid_semi_num(
                kx, ky,
                quantum_geometric_tensor_semi_num,
                hamiltonian=Hamiltonian_Obj,
                delta_k=dkx,
                band_index=band,
                z_cutoff=z_cutoff
            )
        
        else:
            raise ValueError(f"Unknown QGT calculation method: {method_name}")


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




    # b1, b2 = Hamiltonian_Obj.b1, Hamiltonian_Obj.b2
    # chern_number = compute_chern_number(
    #     g_xy_imag_array,
    #     dkx, dky,
    #     kx, ky,
    #     b1, b2
    # )
    # print("Chern number is: ", chern_number)


    # plot_QGT_components_3d(kx, ky, g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array)

    # plot_g_components_2d(g_xx_array, g_yy_array, trace_array, k_max=k_max)

    # plot_trace_w_eigenvalue(kx, ky, g_xx_array, g_yy_array, eigenvalues, trace_array, eigenvalue_band=band)


    # --- FHS Method (Commented out effectively by not using its result) ---
    # flux_field = berry_flux_FHS(eigenfunctions, dim_band=band)
    # berry_curvature_fhs = flux_field / (dkx * dky)
    # g_xy_imag_fhs = -0.5 * berry_curvature_fhs

    # Plot using the Projector Method Result (g_xy_imag_array)
    # This was populated earlier by compute_QGT_projector call
    
    print("Plotting QGT components (Projector Method)...")
    plot_qmt_eig_berry_trace_3d(
        kx, ky, eigenvalues, g_xy_imag_array, trace_array,
        eigenvalue_band=band,
        zlims=(None, (-z_cutoff, z_cutoff), (-z_cutoff, z_cutoff))
    )


def calculate_3d(hamiltonian=None):
    if hamiltonian is None:
        hamiltonian = Hamiltonian_Obj # Fallback to loaded object
        
    print(f"Starting 3D QGT Calculation for {hamiltonian.name}...")
    
    # Define 3D grid parameters (matching RuO2_Hamiltonian logic typically)
    # Using specific range for 3D
    mesh_size = 20 # Reduced for 3D speed, or reuse meta_info["mesh_spacing"] if appropriate
    k_range = np.pi 
    
    kx_vals = np.linspace(-k_range, k_range, mesh_size)
    ky_vals = np.linspace(-k_range, k_range, mesh_size)
    kz_vals = np.linspace(-k_range, k_range, mesh_size)
    
    dkx = kx_vals[1] - kx_vals[0]
    # dky = ky_vals[1] - ky_vals[0]
    # dkz = kz_vals[1] - kz_vals[0]
    delta_k = dkx # Uniform assumption

    # Setup directories using the new utility function
    file_paths, use_existing, results_dir = setup_3D_QGT_results_directory(
        hamiltonian, 
        [kx_vals[0], kx_vals[-1]], 
        [ky_vals[0], ky_vals[-1]], 
        [kz_vals[0], kz_vals[-1]], 
        mesh_size, 
        force_new=True
    )
    
    # 1. Load or Compute Eigenvalues/Eigenvectors
    # We still use temp_dir for intermediate heavy files if needed, or save to results_dir
    # Let's save to results_dir to be self-contained
    eig_file = os.path.join(results_dir, "eigenvalues_3d.npy")
    vec_file = os.path.join(results_dir, "eigenvectors_3d.npy")
    
    if os.path.exists(eig_file) and os.path.exists(vec_file) and use_existing:
        print("Loading 3D eigenvalues and eigenvectors...")
        eigenvalues_3d = np.load(eig_file)
        eigenvectors_3d = np.load(vec_file)
    else:
        print("Computing 3D eigenvalues and eigenvectors...")
        eigenvalues_3d, eigenvectors_3d = compute_eigenvalues_3d(hamiltonian, kx_vals, ky_vals, kz_vals)
        np.save(eig_file, eigenvalues_3d)
        np.save(vec_file, eigenvectors_3d)

    # 2. Compute QGT
    if use_existing and all(os.path.exists(p) for p in file_paths.values()):
         print("Loading existing QGT results...")
         g_xx_arr = np.load(file_paths["g_xx"])
         g_yy_arr = np.load(file_paths["g_yy"])
         g_zz_arr = np.load(file_paths["g_zz"])
         g_xy_real_arr = np.load(file_paths["g_xy_real"])
         g_xy_imag_arr = np.load(file_paths["g_xy_imag"])
         g_xz_real_arr = np.load(file_paths["g_xz_real"])
         g_xz_imag_arr = np.load(file_paths["g_xz_imag"])
         g_yz_real_arr = np.load(file_paths["g_yz_real"])
         g_yz_imag_arr = np.load(file_paths["g_yz_imag"])
    else:
        # We use QGT_grid_3d_num
        print("Computing 3D QGT Grid...")
        results = QGT_grid_3d_num(
            kx_vals, ky_vals, kz_vals, eigenvalues_3d, eigenvectors_3d, 
            quantum_geometric_tensor_3d_num_eigenvector_ordered, # Use ordered version
            hamiltonian, delta_k=delta_k, band_index=band, z_cutoff=z_cutoff
        )
        
        (g_xx_arr, g_yy_arr, g_zz_arr, 
         g_xy_real_arr, g_xy_imag_arr, 
         g_xz_real_arr, g_xz_imag_arr, 
         g_yz_real_arr, g_yz_imag_arr) = results
     
        # 3. Save Results
        print("Saving 3D QGT Results...")
        save_dict = {
            "g_xx": g_xx_arr, "g_yy": g_yy_arr, "g_zz": g_zz_arr,
            "g_xy_real": g_xy_real_arr, "g_xy_imag": g_xy_imag_arr,
            "g_xz_real": g_xz_real_arr, "g_xz_imag": g_xz_imag_arr,
            "g_yz_real": g_yz_real_arr, "g_yz_imag": g_yz_imag_arr
        }
        
        for key, val in save_dict.items():
            np.save(file_paths[key], val)
            
        # Save QGT metadata
        qgt_meta_info = {
            "kx_vals": kx_vals,
            "ky_vals": ky_vals,
            "kz_vals": kz_vals,
            "mesh_spacing": mesh_size,
            "Hamiltonian_Obj": hamiltonian  
        }
        with open(file_paths["meta_info"], "wb") as meta_file:
            pickle.dump(qgt_meta_info, meta_file)
        
    print(f"3D QGT calculation complete. Results saved to {results_dir}")


if __name__ == '__main__':

    # calculate_2d()
    run_hamiltonian = RuO2Hamiltonian()
    calculate_3d(hamiltonian=run_hamiltonian)