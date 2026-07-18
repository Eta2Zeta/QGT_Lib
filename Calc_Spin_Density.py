import sys
import os
import numpy as np
import pickle
import json
import shutil

from Library.data_management_utils_2d import setup_Spin_Density_results_directory
from Library.output_utils import print_calculation_complete
from Plot_Spin_Density import plot_all_spin_components

def calculate_spin_density_all_bands(Force_new=True):
    # Retrieve pre-computed vectors from the temp directory
    temp_dir = os.path.join(os.getcwd(), "temp")
    
    eigenvalues_file = os.path.join(temp_dir, "eigenvalues.npy")
    eigenfunctions_file = os.path.join(temp_dir, "eigenfunctions.npy")
    meta_info_file = os.path.join(temp_dir, "meta_info.pkl")

    if not (os.path.exists(eigenvalues_file) and os.path.exists(eigenfunctions_file) and os.path.exists(meta_info_file)):
        raise FileNotFoundError("Missing temp eigenvalues/eigenfunctions/meta_info.pkl. Please ensure a 2D eigenvalue calc ran.")

    print("Loading eigenfunctions and meta_info...")
    eigenfunctions = np.load(eigenfunctions_file)

    with open(meta_info_file, "rb") as meta_file:
        meta_info = pickle.load(meta_file)
        Hamiltonian_Obj = meta_info["Hamiltonian_Obj"]
        ki = meta_info["ki"]
        kj = meta_info["kj"]
        
        # safely extract variables just to keep structure similar
        kk = meta_info.get("kk", 0.0)
        dki = meta_info.get("dki", 0.0)
        dkj = meta_info.get("dkj", 0.0)
        mesh_spacing = meta_info.get("mesh_spacing", 0.0)
        ki_range = meta_info.get("ki_range", None)
        kj_range = meta_info.get("kj_range", None)
        include_endpoints = meta_info.get("include_endpoints", True)
        order = meta_info.get("order", 2)

    n_bands = Hamiltonian_Obj.dim
    print(f"Detected {n_bands} bands. Computing spin density across entire Grid...")

    # Set up the Spin Density Operator directly from the Hamiltonian object encapsulation
    tau_0_sigma_x = Hamiltonian_Obj.get_spin_operator('x')
    tau_0_sigma_y = Hamiltonian_Obj.get_spin_operator('y')
    tau_0_sigma_z = Hamiltonian_Obj.get_spin_operator('z')
    
    # Pre-allocate results array (kx, ky, n_bands)
    # The eigenfunctions array has shape (Nx, Ny, bands, dim)
    grid_shape = eigenfunctions.shape[:-2] 
    spin_x_all = np.zeros((*grid_shape, n_bands), dtype=float)
    spin_y_all = np.zeros((*grid_shape, n_bands), dtype=float)
    spin_z_all = np.zeros((*grid_shape, n_bands), dtype=float)

    # Calculate natively across the entire space concurrently per band via Einsum projection
    for b in range(n_bands):
        # Extract purely the eigenvector component (shape: Nx, Ny, 4)
        u_nk = eigenfunctions[..., b, :]
        
        # Create column (ket) and row (bra) arrays for batch matrix multiplication
        u_ket = u_nk[..., np.newaxis]           # Shape: (..., 4, 1)
        u_bra = u_nk.conj()[..., np.newaxis, :] # Shape: (..., 1, 4)

        # Standard Matrix Product directly mirroring Dirac notation: <u | operator | u>
        spin_x_all[..., b] = (u_bra @ tau_0_sigma_x @ u_ket).squeeze().real
        spin_y_all[..., b] = (u_bra @ tau_0_sigma_y @ u_ket).squeeze().real
        spin_z_all[..., b] = (u_bra @ tau_0_sigma_z @ u_ket).squeeze().real

    # Organize Metadata
    Hamiltonian_name = getattr(Hamiltonian_Obj, "name", "Hamiltonian")
    hamiltonian_params = Hamiltonian_Obj.get_parameters_dict(parameter="2D")
    meta_params = {
        "hamiltonian_name": Hamiltonian_name,
        "kk": kk,
        "ki_range": ki_range,
        "kj_range": kj_range,
        "mesh_spacing": mesh_spacing,
        "include_endpoints": bool(include_endpoints),
        "n_bands": int(n_bands),
        "hamiltonian_params": hamiltonian_params,
        "dki": dki, 
        "dkj": dkj,
        "order": order
    }

    # Use the spin-specific results folder constructor from plotting_lib limits
    file_paths, use_existing, results_subdir, meta_target_json = setup_Spin_Density_results_directory(
        meta_params=meta_params,
        force_new=Force_new
    )
    
    if use_existing and os.path.exists(file_paths["spin_x"]):
        print("Existing spin arrays successfully identified and extracted. Overriding explicitly...")

    # Dump the numpy projection arrays externally
    np.save(file_paths["spin_x"], spin_x_all)
    np.save(file_paths["spin_y"], spin_y_all)
    np.save(file_paths["spin_z"], spin_z_all)

    with open(file_paths["meta_json"], "w") as f:
        json.dump(meta_target_json, f, indent=2, sort_keys=True)

    # Keep the runtime pickle separate and untouched: it contains Hamiltonian_Obj, ki, kj, etc.
    shutil.copy2(meta_info_file, file_paths["meta_pkl"])

    plot_all_spin_components(results_subdir)

    print_calculation_complete("Spin Expectations", results_subdir)

if __name__ == '__main__':
    calculate_spin_density_all_bands(Force_new=True)
