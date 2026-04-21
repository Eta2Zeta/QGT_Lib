import os
import re
import json
import pickle
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from Library.data_management_utils_common import pick_or_create_result_dir_simple, dump_metadata

def setup_phase_diagram_results_general(
    hamiltonian_template,
    param_ranges,
    parameter_spacing=None,
    decimals=2,
    force_new_range=False
):
    Hname = getattr(hamiltonian_template, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "phase_diagram", re.sub(r'[^\w.-]','_',Hname))

    meta_target = {
        "hamiltonian_name": Hname,
        "param_ranges": param_ranges,
        "parameter_spacing": parameter_spacing
    }

    dir_path, used = pick_or_create_result_dir_simple(
        base_root=base_root,
        base_name="dataset_",
        required_params=meta_target,
        force_new=force_new_range
    )
    
    if not used:
        dump_metadata(meta_target, os.path.join(dir_path, "parameters.json"))

    print(("Using existing phase-diagram range directory: " if used else "Created new phase-diagram range directory: ") + dir_path)
    return dir_path, used

def setup_phase_point_directory_general(range_root_dir, param_values: dict, decimals=2, force_new_point=False):
    meta_target = {"param_values": param_values}

    dir_path, used = pick_or_create_result_dir_simple(
        base_root=range_root_dir,
        base_name="point_",
        required_params=meta_target,
        force_new=force_new_point
    )
    
    if not used:
        dump_metadata(meta_target, os.path.join(dir_path, "parameters.json"))

    # Build paths
    fps = {k: os.path.join(dir_path, fname) for k, fname in {
        "eigenvalues": "eigenvalues.npy",
        "eigenfunctions": "eigenfunctions.npy",
        "g_xx": "g_xx.npy",
        "g_xy_real": "g_xy_real.npy",
        "g_xy_imag": "g_xy_imag.npy",
        "g_yy": "g_yy.npy",
        "trace": "trace.npy",
        "chern": "chern.npy",
        "meta_info": "meta_info.pkl",
    }.items()}

    print(("Using existing phase-point directory: " if used else "Created phase-point directory: ") + dir_path)
    return fps, used, dir_path

def setup_qgt_nd_results_dir(
    hamiltonian_template,
    param_ranges,
    parameter_spacing,
    kx_range,
    ky_range,
    mesh_spacing,
    band_index=None,
    decimals=3,
    force_new=False
):
    """
    New version of setup that uses 'datasetN' folders and 'parameters.json'.
    Includes band_index in metadata check!
    """
    Hname = getattr(hamiltonian_template, "name", "Hamiltonian")
    
    def _sanitize(name: str) -> str:
        return re.sub(r"[^\w.\-]", "_", str(name))
        
    base_root = os.path.join(os.getcwd(), "results", "QGT_ND", _sanitize(Hname))
    
    # 1. Normalize ranges
    def _norm_ranges_list(ranges):
         if isinstance(ranges, dict):
            items = sorted(ranges.items(), key=lambda kv: kv[0])
            return [[k, float(v[0]), float(v[1])] for k, v in items]
         items = sorted([[n, float(a), float(b)] for (n, a, b) in ranges], key=lambda x: x[0])
         return items
    
    range_list = _norm_ranges_list(param_ranges)
    
    # 2. Normalize spacing
    def _parse_spacing(spec):
        if isinstance(spec, int): return int(spec), "linear"
        if isinstance(spec, dict):
            c = int(spec.get("count", spec.get("n", spec.get("points", 1))))
            s = str(spec.get("scale", spec.get("spacing", "linear"))).lower().strip()
            return c, s
        return 1, "linear"

    spacing_dict = {}
    if isinstance(parameter_spacing, int):
         spacing_dict = {n: {"count": int(parameter_spacing), "scale": "linear"} for (n, _, _) in range_list}
    elif isinstance(parameter_spacing, dict):
        for (n, _, _) in range_list:
             spec = parameter_spacing.get(n, 1)
             cnt, scl = _parse_spacing(spec)
             spacing_dict[n] = {"count": cnt, "scale": scl}
            

    # Collect all public, simple attributes AND properties
    params = {}
    for k in dir(hamiltonian_template):
        if k.startswith('_') or k in ('name', 'dim', 'get_filename'):
            continue
        try:
            val = getattr(hamiltonian_template, k)
            if not callable(val) and isinstance(val, (int, float, str, bool)):
                params[k] = val
        except Exception:
            pass

    metadata = {
        "hamiltonian_name": Hname,
        "parameters": params,
        "scan_ranges": range_list, 
        "scan_spacing": spacing_dict,
        "k_grid": {
            "kx_min": float(kx_range[0]), "kx_max": float(kx_range[1]),
            "ky_min": float(ky_range[0]), "ky_max": float(ky_range[1]),
            "mesh": int(mesh_spacing)
        }
    }
    
    if band_index is not None:
        metadata["band_index"] = int(band_index)
        
    # Attempt to find or create
    dir_path, used = pick_or_create_result_dir_simple(
        base_root=base_root,
        base_name="dataset",
        required_params=metadata,
        force_new=force_new
    )
    
    if not used:
        dump_metadata(metadata, os.path.join(dir_path, "parameters.json"))
            
    print(("Using existing (JSON) QGT directory: " if used else "Created new (JSON) QGT directory: ") + dir_path)
    return dir_path, used

