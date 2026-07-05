import os
import json
import pickle
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from Library.data_management_utils_common import pick_or_create_result_dir_simple, meta_matcher_all_fields, dump_metadata

def setup_2D_Eigen_results_directory(
    meta_params: Dict[str, Any],
    force_new: bool = False,
):
    """
    Sets up the results directory for 2D eigenvalue calculations using metadata matching.
    
    Returns:
        file_paths (dict): Paths for eigenvalues, eigenfunctions, meta_json, meta_pkl.
        use_existing (bool): Whether an existing directory was reused.
        results_subdir (str): The path to the results directory.
        meta_target (dict): The metadata dictionary used for matching.
    """
    Hamiltonian_name = meta_params["hamiltonian_name"]
    base_root = os.path.join(os.getcwd(), "results", "2D_Eigen_results", Hamiltonian_name)

    required_files = ["eigenvalues.npy", "eigenfunctions.npy", "meta.json", "meta_info.pkl"]


    dir_path, used = pick_or_create_result_dir_simple(
        base_root=base_root,
        base_name="dataset_",
        required_params=meta_params,
        force_new=force_new,
        required_files=required_files
    )
    
    file_paths = {
        "eigenvalues": os.path.join(dir_path, "eigenvalues.npy"),
        "eigenfunctions": os.path.join(dir_path, "eigenfunctions.npy"),
        "meta_json": os.path.join(dir_path, "meta.json"),
        "meta_pkl": os.path.join(dir_path, "meta_info.pkl"),
    }


    print(("Using existing 2D Eigen results directory: " if used else "Created new 2D Eigen results directory: ") + dir_path)
    
    full_meta = meta_params.copy()
    
    return file_paths, used, dir_path, full_meta

def setup_2D_QGT_results_directory(
    meta_params: Dict[str, Any],
    force_new: bool = False,
):
    """
    Sets up the results directory for 2D QGT calculations using metadata matching.
    
    Parameters:
    - hamiltonian: The Hamiltonian object.
    - meta_params: Dictionary containing metadata fields (kz, kx_range, ky_range, mesh_spacing, method_name, include_endpoints).
    - force_new: Whether to force creating a new directory.
    """

    # Define base_root using hamiltonian_name
    Hamiltonian_name = meta_params.get("hamiltonian_name", "Unknown_Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "2D_QGT_results", Hamiltonian_name)
    
    json_meta = {k: v for k, v in meta_params.items() if not k.endswith("_Obj")}

    required_files = [
        "g_xx.npy", 
        "g_xy_real.npy", 
        "g_xy_imag.npy", 
        "g_yy.npy", 
        "g_zz.npy",
        "g_xz_real.npy",
        "g_xz_imag.npy",
        "g_yz_real.npy",
        "g_yz_imag.npy",
        "trace.npy", 
        "meta.json", 
        "meta_info.pkl"
    ]

    dir_path, used = pick_or_create_result_dir_simple(
        base_root=base_root,
        base_name="dataset_",
        required_params=json_meta,
        force_new=force_new,
        required_files=required_files
    )
    
    if not used:
        dump_metadata(json_meta, os.path.join(dir_path, "meta.json"))

    file_paths = {
        "g_xx": os.path.join(dir_path, "g_xx.npy"),
        "g_yy": os.path.join(dir_path, "g_yy.npy"),
        "g_zz": os.path.join(dir_path, "g_zz.npy"),
        "g_xy_real": os.path.join(dir_path, "g_xy_real.npy"),
        "g_xy_imag": os.path.join(dir_path, "g_xy_imag.npy"),
        "g_xz_real": os.path.join(dir_path, "g_xz_real.npy"),
        "g_xz_imag": os.path.join(dir_path, "g_xz_imag.npy"),
        "g_yz_real": os.path.join(dir_path, "g_yz_real.npy"),
        "g_yz_imag": os.path.join(dir_path, "g_yz_imag.npy"),
        "trace": os.path.join(dir_path, "trace.npy"),
        "meta_json": os.path.join(dir_path, "meta.json"),
        "meta_pkl": os.path.join(dir_path, "meta_info.pkl"),
    }

    print(("Using existing 2D QGT results directory: " if used else "Created new 2D QGT results directory: ") + dir_path)
    
    # Return the FULL meta_params as meta_target for the caller to work with
    # This ensures qgt_meta_info (pickle) contains everything passed in
    full_meta = meta_params.copy()
    full_meta["hamiltonian_name"] = Hamiltonian_name # ensure this is set
    
    return file_paths, used, dir_path, full_meta

def setup_Spin_Density_results_directory(
    meta_params: Dict[str, Any],
    force_new: bool = False,
):
    """
    Sets up the results directory for computationally extracted Spin Density using metadata matching.
    """
    Hamiltonian_name = meta_params.get("hamiltonian_name", "Unknown_Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "Spin_Density_results", Hamiltonian_name)
    
    # Extract Hamiltonian parameters
    hamiltonian = meta_params.get("Hamiltonian_Obj", None)
    if hamiltonian is not None and hasattr(hamiltonian, "get_parameters_dict"):
        ham_params = hamiltonian.get_parameters_dict(parameter="2D")
    else:
        ham_params = {}

    meta_params["hamiltonian_params"] = ham_params

    required_files = [
        "spin_x.npy", 
        "spin_y.npy", 
        "spin_z.npy", 
        "meta.json", 
        "meta_info.pkl"
    ]

    dir_path, used = pick_or_create_result_dir_simple(
        base_root=base_root,
        base_name="dataset_",
        required_params=meta_params,
        force_new=force_new,
        required_files=required_files
    )
    
    if not used:
        # Strip object references when dumping JSON
        json_meta = {k: v for k, v in meta_params.items() if not k.endswith("_Obj")}
        with open(os.path.join(dir_path, "parameters.json"), "w") as f:
            json.dump(json_meta, f, indent=4)

    file_paths = {
        "spin_x": os.path.join(dir_path, "spin_x.npy"),
        "spin_y": os.path.join(dir_path, "spin_y.npy"),
        "spin_z": os.path.join(dir_path, "spin_z.npy"),
        "meta_json": os.path.join(dir_path, "meta.json"),
        "meta_pkl": os.path.join(dir_path, "meta_info.pkl"),
    }

    print(("Using existing Spin Density results directory: " if used else "Created new Spin Density results directory: ") + dir_path)
    
    full_meta = meta_params.copy()
    full_meta["hamiltonian_name"] = Hamiltonian_name
    
    return file_paths, used, dir_path, full_meta
