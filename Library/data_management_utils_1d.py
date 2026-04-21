import os
import re
import json
import pickle
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from Library.data_management_utils_common import pick_or_create_result_dir_simple, dump_metadata

def setup_results_directory_1d(hamiltonian, k_angle, kx_shift, ky_shift, num_points, k_max, *, force_new=False):
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "1D_Eigen_results", Hamiltonian_name)
    
    if hasattr(hamiltonian, "get_parameters_dict"):
        ham_params = hamiltonian.get_parameters_dict(parameter="1D")
    else:
        ham_params = getattr(hamiltonian, "__dict__", {})
        
    meta_target = {
        "hamiltonian_name": Hamiltonian_name,
        "hamiltonian_params": ham_params,
        "k_angle": float(k_angle),
        "kx_shift": float(kx_shift),
        "ky_shift": float(ky_shift),
        "num_points": int(num_points),
        "k_max": float(k_max),
    }

    dir_path, used = pick_or_create_result_dir_simple(
        base_root=base_root,
        base_name="dataset_",
        required_params=meta_target,
        force_new=force_new
    )
    
    if not used:
        dump_metadata(meta_target, os.path.join(dir_path, "parameters.json"))

    file_paths = {
        "eigenvalues": os.path.join(dir_path, "eigenvalues.npy"),
        "eigenfunctions": os.path.join(dir_path, "eigenfunctions.npy"),
        "meta_info": os.path.join(dir_path, "meta_info.pkl"),
    }

    print(("Using existing results directory: " if used else "Created new results directory: ") + dir_path)
    return file_paths, used, dir_path

def setup_QGT_results_directory_1D(
    hamiltonian,
    k_angle,
    kx_shift,
    ky_shift,
    num_k_points,
    num_omega_points,
    k_max,
    omega_min,
    omega_max,
    spacing,
    force_new=False,
):
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "1D_QGT_results", Hamiltonian_name)

    if hasattr(hamiltonian, "get_parameters_dict"):
        ham_params = hamiltonian.get_parameters_dict(parameter="1D")
    else:
        ham_params = getattr(hamiltonian, "__dict__", {})

    meta_target = {
        "hamiltonian_name": Hamiltonian_name,
        "hamiltonian_params": ham_params,
        "k_angle": float(k_angle),
        "kx_shift": float(kx_shift),
        "ky_shift": float(ky_shift),
        "num_k_points": int(num_k_points),
        "num_omega_points": int(num_omega_points),
        "k_max": float(k_max),
        "omega_min": float(omega_min),
        "omega_max": float(omega_max),
        "spacing": str(spacing)
    }

    dir_path, used = pick_or_create_result_dir_simple(
        base_root=base_root,
        base_name="dataset_",
        required_params=meta_target,
        force_new=force_new
    )
    
    if not used:
        dump_metadata(meta_target, os.path.join(dir_path, "parameters.json"))

    file_paths = {
        "QGT_1D": os.path.join(dir_path, "QGT_1D.npy"),
        "meta_info": os.path.join(dir_path, "meta_info.pkl"),
    }

    print(("Using existing QGT results directory: " if used else "Created new QGT results directory: ") + dir_path)
    return file_paths, used, dir_path

def setup_QGT_results_directory_1D_single_param(
    hamiltonian,
    *,
    param_name: str,
    vmin: float,
    vmax: float,
    spacing: str,
    num_param_points: int,
    num_k_points: int,
    angle_deg: float,
    kx_shift: float,
    ky_shift: float,
    k_max: float,
    force_new: bool = False,
) -> Tuple[dict, bool, str]:
    """
    Create/reuse a results directory for a 1D sweep of ONE Hamiltonian parameter.
    Returns (file_paths_dict, used_existing, out_dir).
    """
    # Top-level group by Hamiltonian name
    Hname = getattr(hamiltonian, "name", "Hamiltonian")
    
    def _sanitize(name: str) -> str:
        return re.sub(r"[^\w.\-]", "_", str(name))
        
    base_root = os.path.join(os.getcwd(), "results", "1D_QGT_results", _sanitize(Hname))
    os.makedirs(base_root, exist_ok=True)

    if hasattr(hamiltonian, "get_parameters_dict"):
        ham_params = hamiltonian.get_parameters_dict(parameter="1D")
    else:
        ham_params = getattr(hamiltonian, "__dict__", {})

    meta_target = {
        "hamiltonian_name": Hname,
        "hamiltonian_params": ham_params,
        "param_name": str(param_name),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "spacing": str(spacing),
        "num_param_points": int(num_param_points),
        "num_k_points": int(num_k_points),
        "angle_deg": float(angle_deg),
        "kx_shift": float(kx_shift),
        "ky_shift": float(ky_shift),
        "k_max": float(k_max),
    }

    out_dir, used_existing = pick_or_create_result_dir_simple(
        base_root=base_root,
        base_name="dataset_",
        required_params=meta_target,
        force_new=force_new
    )
    
    if not used_existing:
        dump_metadata(meta_target, os.path.join(out_dir, "parameters.json"))

    file_paths = {
        "QGT_1D":   os.path.join(out_dir, "QGT_1D.npy"),
        "meta_info":os.path.join(out_dir, "meta_info.pkl"),
    }
    return file_paths, used_existing, out_dir

def setup_sym_points_results_directory(
    hamiltonian,
    path_points,
    path_labels,
    num_points_per_segment,
    force_new=False
):
    """
    Create or reuse a results directory for symmetry-path band structure calculations.
    Works for both 2D and 3D paths — the dimensionality is inferred from path_points.
    """
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "Sym_Points_results", Hamiltonian_name)

    path_points_list = [np.array(p).tolist() for p in path_points]

    if hasattr(hamiltonian, "get_parameters_dict"):
        ham_params = hamiltonian.get_parameters_dict(parameter="sym")
    else:
        ham_params = getattr(hamiltonian, "__dict__", {})

    meta_target = {
        "hamiltonian_name": Hamiltonian_name,
        "hamiltonian_params": ham_params,
        "path_labels": path_labels,
        "num_points_per_segment": int(num_points_per_segment),
        "path_points": path_points_list,
    }

    dir_path, used = pick_or_create_result_dir_simple(
        base_root=base_root,
        base_name="dataset_",
        required_params=meta_target,
        force_new=force_new
    )
    
    if not used:
        dump_metadata(meta_target, os.path.join(dir_path, "parameters.json"))

    file_paths = {
        "eigenvalues": os.path.join(dir_path, "eigenvalues.npy"),
        "meta_json":   os.path.join(dir_path, "meta.json"),
        "meta_pkl":    os.path.join(dir_path, "meta_info.pkl"),
    }

    print(("Using existing Sym Points results directory: " if used else "Created new Sym Points results directory: ") + dir_path)
    return file_paths, used, dir_path, meta_target

def setup_1D_angles_results_directory(
    hamiltonian,
    k_max,
    num_angles,
    num_points_per_line,
    force_new=False
):
    """
    Create or reuse a results directory for 1D angled line band structure calculations.
    lives under 1D_Angles_results.
    """
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "1D_Angles_results", Hamiltonian_name)

    if hasattr(hamiltonian, "get_parameters_dict"):
        ham_params = hamiltonian.get_parameters_dict(parameter="1D_Angles")
    else:
        ham_params = getattr(hamiltonian, "__dict__", {})

    meta_target = {
        "hamiltonian_name": Hamiltonian_name,
        "hamiltonian_params": ham_params,
        "k_max": float(k_max),
        "num_angles": int(num_angles),
        "num_points_per_line": int(num_points_per_line),
    }

    dir_path, used = pick_or_create_result_dir_simple(
        base_root=base_root,
        base_name="dataset_",
        required_params=meta_target,
        force_new=force_new
    )

    if not used:
        with open(os.path.join(dir_path, "parameters.json"), "w") as f:
            json.dump(meta_target, f, indent=4)

    file_paths = {
        "eigenvalues": os.path.join(dir_path, "eigenvalues.npy"),
        "meta_json":   os.path.join(dir_path, "meta.json"),
        "meta_pkl":    os.path.join(dir_path, "meta_info.pkl"),
    }

    print(("Using existing 1D Angles results directory: " if used else "Created new 1D Angles results directory: ") + dir_path)
    return file_paths, used, dir_path, meta_target

