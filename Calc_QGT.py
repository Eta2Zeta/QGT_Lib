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
from Library.plotting_lib_2d import *
from Library.Hamiltonian.RuO2Hamiltonian import *
from Library.Hamiltonian.gWaveAltermagnetHamiltonian import *
from Library.data_management_utils import setup_3D_QGT_results_directory



def calculate_2d(Force_new=True):
    # Define parameters
    band = 1 # Which band to calculate your QMT on, starting from 0
    z_cutoff = 1 #where to cutoff the plot for the z axis when singularties occur

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
            kz = meta_info["kz"]
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

    # Define method name for directory naming ("analytic", "numerical", etc.)
    method_name = "numerical"
    
    # NEW SETUP
    from Library.data_management_utils_2D import setup_2D_QGT_results_directory
    
    Hamiltonian_name = getattr(Hamiltonian_Obj, "name", "Hamiltonian")
    
    # Construct metadata parameters dictionary with ALL info
    meta_params = {
        "hamiltonian_name": Hamiltonian_name,
        "kz": kz,
        "kx_range": kx_range,
        "ky_range": ky_range,
        "mesh_spacing": mesh_spacing,
        "method_name": method_name,
        "include_endpoints": True,
        # Objects to include in pickle but exclude from JSON
        "Hamiltonian_Obj": Hamiltonian_Obj,
        "kx": kx,
        "ky": ky,
        "dkx": dkx,
        "dky": dky
    }
    
    file_paths, use_existing, results_subdir, meta_target = setup_2D_QGT_results_directory(
        meta_params=meta_params,
        force_new=Force_new
    )
    
    if use_existing:
        # Load existing QGT data
        g_xx_array = np.load(file_paths["g_xx"])
        g_yy_array = np.load(file_paths["g_yy"])
        g_zz_array = np.load(file_paths["g_zz"])
        g_xy_real_array = np.load(file_paths["g_xy_real"])
        g_xy_imag_array = np.load(file_paths["g_xy_imag"])
        g_xz_real_array = np.load(file_paths["g_xz_real"])
        g_xz_imag_array = np.load(file_paths["g_xz_imag"])
        g_yz_real_array = np.load(file_paths["g_yz_real"])
        g_yz_imag_array = np.load(file_paths["g_yz_imag"])
        trace_array = np.load(file_paths["trace"])

        with open(file_paths["meta_pkl"], "rb") as meta_file:
            qgt_meta_info = pickle.load(meta_file)

        print("Loaded QGT data from existing files.")


    else:
        Hamiltonian_Obj.A0 = 0.0
        Hamiltonian_Obj.omega = 5e3

        # Select QGT calculation method
        if method_name == "numerical":
            # Note: QGT_grid_num now returns 9 components
            # We must use a 3D-compatible function for the inner loop, e.g. quantum_geometric_tensor_3d_num_eigenvector_ordered
            
            (g_xx_array, g_yy_array, g_zz_array, 
             g_xy_real_array, g_xy_imag_array, 
             g_xz_real_array, g_xz_imag_array, 
             g_yz_real_array, g_yz_imag_array) = QGT_grid_num(
                kx, ky, eigenvalues, eigenfunctions, quantum_geometric_tensor_3d_num_eigenvector_ordered, 
                Hamiltonian_Obj, delta_k=1e-5, band_index=band, z_cutoff=z_cutoff, kk=kz
            )
            
            # Trace is typically sum of diagonal elements g_ii
            trace_array = g_xx_array + g_yy_array + g_zz_array
        
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
        # We now have 9 components + trace
        save_dict = {
            "g_xx": g_xx_array,
            "g_yy": g_yy_array,
            "g_zz": g_zz_array,
            "g_xy_real": g_xy_real_array,
            "g_xy_imag": g_xy_imag_array,
            "g_xz_real": g_xz_real_array,
            "g_xz_imag": g_xz_imag_array,
            "g_yz_real": g_yz_real_array,
            "g_yz_imag": g_yz_imag_array,
            "trace": trace_array
        }

        for key, array in save_dict.items():
            # If the file path is not in file_paths (e.g. g_zz, g_xz...), we might need to define it or just skip
            # The setup_2D_QGT_results_directory might not have returned paths for 3D components if it wasn't updated.
            # However, we can construct the path if needed or just rely on what is in file_paths.
            # Checking if key is in file_paths:
            if key in file_paths:
                np.save(file_paths[key], array)
                np.save(os.path.join(temp_dir, os.path.basename(file_paths[key])), array)
            else:
                # If key not in file_paths (likely for new 3D components in 2D struct), 
                # we can save them to temp_dir or define a new path convention.
                # For now, let's save to temp_dir at least.
                np.save(os.path.join(temp_dir, f"{key}.npy"), array)

        # Save QGT metadata (JSON)
        # Clean up meta_target for JSON saving (remove objects not serializable)
        meta_info_json = meta_target.copy()
        
        keys_to_remove = ["Hamiltonian_Obj", "kx", "ky"]
        for key in keys_to_remove:
            if key in meta_info_json:
                del meta_info_json[key]
                
        with open(file_paths["meta_json"], "w") as f:
            json.dump(meta_info_json, f, indent=2, sort_keys=True)

        # Save QGT metadata (Pickle)
        # qgt_meta_info IS proper meta_target (which includes everything)
        qgt_meta_info = meta_target

        with open(file_paths["meta_pkl"], "wb") as meta_file:
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


    # plot_QGT_components_3d(kx, ky, g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, stride_size=1)

    # plot_g_components_2d(g_xx_array, g_yy_array, trace_array, k_max=k_max)

    # plot_trace_w_eigenvalue(kx, ky, g_xx_array, g_yy_array, eigenvalues, trace_array, eigenvalue_band=band)


    # --- FHS Method (Commented out effectively by not using its result) ---
    # flux_field = berry_flux_FHS(eigenfunctions, dim_band=band)
    # berry_curvature_fhs = flux_field / (dkx * dky)
    # g_xy_imag_fhs = -0.5 * berry_curvature_fhs


    # plot_qmt_eig_berry_trace_3d(
    #     kx, ky, eigenvalues, g_xy_imag_array, trace_array,
    #     eigenvalue_band=band,
    #     zlims=(None, (-z_cutoff, z_cutoff), (-z_cutoff, z_cutoff))
    # )

    plot_qmt_eig_berry_trace_2d(
        kx, ky, eigenvalues, g_xy_imag_array, trace_array,
        eigenvalue_band=band,
        zlims=(None, (-z_cutoff, z_cutoff), (-z_cutoff, z_cutoff)),
        components="xy"
    )

    plot_qmt_eig_berry_trace_2d(
        kx, ky, eigenvalues, g_xz_imag_array, trace_array,
        eigenvalue_band=band,
        zlims=(None, (-z_cutoff, z_cutoff), (-z_cutoff, z_cutoff)),
        components="xz"
    )

    plot_qmt_eig_berry_trace_2d(
        kx, ky, eigenvalues, g_yz_imag_array, trace_array,
        eigenvalue_band=band,
        zlims=(None, (-z_cutoff, z_cutoff), (-z_cutoff, z_cutoff)),
        components="yz"
    )


def calculate_3d():
    print(f"Starting 3D QGT Calculation...")
    band = 0
    z_cutoff = 20
    # --- Check Temp Directory First (User Request) ---
    temp_dir = os.path.join(os.getcwd(), "temp")
    temp_eig = os.path.join(temp_dir, "eigenvalues_3d.npy")
    temp_vec = os.path.join(temp_dir, "eigenvectors_3d.npy")
    temp_meta = os.path.join(temp_dir, "meta_info.pkl")
    
    
    if os.path.exists(temp_eig) and os.path.exists(temp_vec) and os.path.exists(temp_meta):
        print("Found results in temp directory. Attempting to load from there...")
        try:
            with open(temp_meta, "rb") as f:
                temp_info = pickle.load(f)
            
            # Check if it looks like what we expect
            # We trust the user wants to use this
            hamiltonian = temp_info["Hamiltonian_Obj"]
            kx_vals = temp_info["kx_vals"]
            ky_vals = temp_info["ky_vals"]
            kz_vals = temp_info["kz_vals"]
            mesh_shape = temp_info["mesh_shape"]
            kvals_mode = temp_info["kvals_mode"]
            include_endpoints = temp_info["include_endpoints"]
            eigenvalues_3d = np.load(temp_eig)
            eigenvectors_3d = np.load(temp_vec)
            print(f"Successfully loaded data from temp. Hamiltonian: {hamiltonian.name}, Mesh: {mesh_size}")
            
        except Exception as e:
            print(f"Failed to load from temp: {e}. Falling back to standard calculation.")


    delta_k = 1e-4 # Use small step for numerical derivative, matching 2D calculation
    
    qgt_file_paths, qgt_use_existing, qgt_results_dir, qgt_meta_target = setup_3D_QGT_results_directory(
        hamiltonian,
        [kx_vals[0], kx_vals[-1]],
        [ky_vals[0], ky_vals[-1]],
        [kz_vals[0], kz_vals[-1]],
        mesh_shape=mesh_shape,
        include_endpoints=include_endpoints,
        force_new=True,
        kvals_mode=kvals_mode,  # if your function supports it
    )

    if qgt_use_existing and all(os.path.exists(p) for p in qgt_file_paths.values()):
         print("Loading existing QGT results...")
         g_xx_arr = np.load(qgt_file_paths["g_xx"])
         g_yy_arr = np.load(qgt_file_paths["g_yy"])
         g_zz_arr = np.load(qgt_file_paths["g_zz"])
         g_xy_real_arr = np.load(qgt_file_paths["g_xy_real"])
         g_xy_imag_arr = np.load(qgt_file_paths["g_xy_imag"])
         g_xz_real_arr = np.load(qgt_file_paths["g_xz_real"])
         g_xz_imag_arr = np.load(qgt_file_paths["g_xz_imag"])
         g_yz_real_arr = np.load(qgt_file_paths["g_yz_real"])
         g_yz_imag_arr = np.load(qgt_file_paths["g_yz_imag"])
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
            np.save(qgt_file_paths[key], val)
            
        # JSON
        qgt_meta = dict(qgt_meta_target)
        qgt_meta.update({"dk": delta_k, "band_index": band, "z_cutoff": z_cutoff})
        with open(qgt_file_paths["meta_json"], "w") as f:
            json.dump(qgt_meta, f, indent=2, sort_keys=True)

        # Pickle (loadable)
        qgt_meta_pkl = {
            "kx_vals": kx_vals, "ky_vals": ky_vals, "kz_vals": kz_vals,
            "mesh_shape": mesh_shape,
            "dk": delta_k,
            "include_endpoints": include_endpoints,
            "kvals_mode": kvals_mode,
            "Hamiltonian_Obj": hamiltonian,
        }
        with open(qgt_file_paths["meta_pkl"], "wb") as f:
            pickle.dump(qgt_meta_pkl, f)

        
    print(f"3D QGT calculation complete. Results saved to {qgt_results_dir}")


if __name__ == '__main__':

    calculate_2d(Force_new=True)
    # calculate_3d()