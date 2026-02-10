import os
import pickle
from typing import Dict, Any, Tuple
from .data_management_utils import pick_or_create_dataset_dir, meta_matcher_all_fields

def setup_2D_Eigen_results_directory(
    hamiltonian,
    kx_range, ky_range,
    mesh_spacing,
    include_endpoints: bool = True,
    force_new: bool = False,
    kvals_mode: str = "endpoints",
):
    """
    Sets up the results directory for 2D eigenvalue calculations using metadata matching.
    
    Returns:
        file_paths (dict): Paths for eigenvalues, eigenfunctions, meta_json, meta_pkl.
        use_existing (bool): Whether an existing directory was reused.
        results_subdir (str): The path to the results directory.
        meta_target (dict): The metadata dictionary used for matching.
    """
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "2D_Eigen_results", Hamiltonian_name)

    required_files = ["eigenvalues.npy", "eigenfunctions.npy", "meta.json", "meta_info.pkl"]

    # Match target (what defines this dataset)
    meta_target = {
        "hamiltonian_name": Hamiltonian_name,
        "include_endpoints": bool(include_endpoints),
        "mesh_spacing": int(mesh_spacing),
        "kvals_mode": str(kvals_mode),
        "kx_range": [float(kx_range[0]), float(kx_range[1])],
        "ky_range": [float(ky_range[0]), float(ky_range[1])],
    }

    dir_path, used = pick_or_create_dataset_dir(
        base_root,
        meta_target=meta_target,
        required_files=required_files,
        meta_matcher=meta_matcher_all_fields,
        force_new=force_new,
        prefix="data_set_",
        start_index=1,
    )

    file_paths = {
        "eigenvalues": os.path.join(dir_path, "eigenvalues.npy"),
        "eigenfunctions": os.path.join(dir_path, "eigenfunctions.npy"),
        "meta_json": os.path.join(dir_path, "meta.json"),
        "meta_pkl": os.path.join(dir_path, "meta_info.pkl"),
    }


    print(("Using existing 2D Eigen results directory: " if used else "Created new 2D Eigen results directory: ") + dir_path)
    return file_paths, used, dir_path, meta_target

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
    # Construct match_target for directory matching (JSON serializable fields only)
    # We explicitly select fields that define the dataset identity
    match_target = {
        "hamiltonian_name": meta_params.get("hamiltonian_name"),
        "kz": meta_params.get("kz"),
        "include_endpoints": meta_params.get("include_endpoints", True),
        "mesh_spacing": meta_params.get("mesh_spacing"),
        "method_name": meta_params.get("method_name"),
    }
    
    # Handle ranges carefully (lists of floats)
    if "kx_range" in meta_params:
        match_target["kx_range"] = [float(x) for x in meta_params["kx_range"]]
    if "ky_range" in meta_params:
        match_target["ky_range"] = [float(x) for x in meta_params["ky_range"]]

    # Ensure types for consistent matching
    if match_target.get("mesh_spacing") is not None:
        match_target["mesh_spacing"] = int(match_target["mesh_spacing"])
        
    # Define base_root using hamiltonian_name
    Hamiltonian_name = meta_params.get("hamiltonian_name", "Unknown_Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "2D_QGT_results", Hamiltonian_name)

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

    dir_path, used = pick_or_create_dataset_dir(
        base_root,
        meta_target=match_target,  # Use filtered target for matching
        required_files=required_files,
        meta_matcher=meta_matcher_all_fields,
        force_new=force_new,
        prefix="data_set_",
        start_index=1,
    )

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
