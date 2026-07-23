import os
import json
import pickle
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from Library.data_management_utils_common import pick_or_create_result_dir_simple, meta_matcher_all_fields, dump_metadata

def setup_3D_Eigen_results_directory(
    hamiltonian, kx_range, ky_range, kz_range,
    mesh_shape, include_endpoints=True, force_new=False,
    kvals_mode="endpoints",
):
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "3D_Eigen_results", Hamiltonian_name)

    nx, ny, nz = mesh_shape
    
    if hasattr(hamiltonian, "get_parameters_dict"):
        ham_params = hamiltonian.get_parameters_dict(parameter="3D")
    else:
        ham_params = getattr(hamiltonian, "__dict__", {})

    meta_target = {
        "hamiltonian_name": Hamiltonian_name,
        "hamiltonian_params": ham_params,
        "kx_range": [float(kx_range[0]), float(kx_range[1])],
        "ky_range": [float(ky_range[0]), float(ky_range[1])],
        "kz_range": [float(kz_range[0]), float(kz_range[1])],
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "include_endpoints": bool(include_endpoints),
        "kvals_mode": str(kvals_mode),
    }

    required_files = [
        "eigenvalues_3d.npy",
        "eigenvectors_3d.npy",
        "meta.json",
        "meta_info.pkl",
    ]

    dir_path, used = pick_or_create_result_dir_simple(
        base_root=base_root,
        base_name="dataset_",
        required_params=meta_target,
        force_new=force_new,
        required_files=required_files,
    )

    file_paths = {k: os.path.join(dir_path, fname) for k, fname in {
        "eigenvalues": "eigenvalues_3d.npy",
        "eigenfunctions": "eigenvectors_3d.npy",
        "meta_json": "meta.json",
        "meta_pkl": "meta_info.pkl",
    }.items()}

    print(("Using existing 3D Eigen results directory: " if used else "Created new 3D Eigen results directory: ") + dir_path)
    return file_paths, used, dir_path, meta_target

def setup_3D_QGT_results_directory(
    hamiltonian,
    kx_range, ky_range, kz_range,
    mesh_shape,
    include_endpoints=True,
    force_new=False,
    kvals_mode: str = "endpoints",
    *,
    # NEW: include these in meta-matching so ALL-bands runs don't collide with single-band runs
    method_name: str = "numerical",
    band_index="ALL",          # int or "ALL"
    n_bands=None,              # required if band_index == "ALL"
):
    """
    Creates (or reuses) a results directory for 3D QGT computations.

    Supports two modes:
      - band_index is an int: single-band results saved (still as .npy arrays).
      - band_index == "ALL": stacked results saved with shape (n_bands, nx, ny, nz).

    Returns:
      file_paths: dict of output file paths
      used_existing: bool
      dir_path: str
      meta_target: dict used for matching/saving
    """
    nx, ny, nz = map(int, mesh_shape)

    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "3D_QGT_results", Hamiltonian_name)

    # Get Hamiltonian parameters natively as a dictionary
    if hasattr(hamiltonian, "get_parameters_dict"):
        ham_params = hamiltonian.get_parameters_dict(parameter="3D")
    else:
        ham_params = {}

    # Normalize band_index
    band_key = band_index
    if isinstance(band_key, str):
        band_key = band_key.upper()
    is_all = (band_key == "ALL")

    if is_all:
        if n_bands is None:
            raise ValueError("setup_3D_QGT_results_directory: n_bands must be provided when band_index='ALL'")
        n_bands = int(n_bands)
    else:
        # Single band: store as int
        band_index = int(band_index)

    # Match target (what defines this dataset)
    meta_target = {
        "hamiltonian_name": str(Hamiltonian_name),
        "hamiltonian_params": ham_params,
        "mesh_shape": [nx, ny, nz],
        "include_endpoints": bool(include_endpoints),
        "kvals_mode": str(kvals_mode),
        "kx_range": [float(kx_range[0]), float(kx_range[1])],
        "ky_range": [float(ky_range[0]), float(ky_range[1])],
        "kz_range": [float(kz_range[0]), float(kz_range[1])],

        # NEW: make meta matching robust across methods and band modes
        "method_name": str(method_name),
        "band_index": ("ALL" if is_all else int(band_index)),
        "n_bands": (int(n_bands) if is_all else None),
    }

    # If your meta_matcher treats None fields strictly, it might fail matches between
    # (single band) and (ALL bands). That's GOOD — we *want* them separated.
    dir_path, used = pick_or_create_result_dir_simple(
        base_root=base_root,
        base_name="dataset_",
        required_params=meta_target,
        force_new=force_new
    )
    
    if not used:
        dump_metadata(meta_target, os.path.join(dir_path, "parameters.json"))

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
        "trace": os.path.join(dir_path, "trace.npy"),          # NEW
        "meta_json": os.path.join(dir_path, "meta.json"),
        "meta_pkl": os.path.join(dir_path, "meta_info.pkl"),
    }

    print(("Using existing 3D QGT results directory: " if used else "Created new 3D QGT results directory: ") + dir_path)
    return file_paths, used, dir_path, meta_target
