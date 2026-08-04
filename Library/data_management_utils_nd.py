import os
import re
import json
import pickle
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from Library.data_management_utils_common import pick_or_create_result_dir_simple, dump_metadata


def build_parameter_points(
    param_ranges,
    parameter_spacing,
    *,
    ensure_increasing_axes=True
):
    """
    Build an N-D parameter grid.

    Parameters
    ----------
    param_ranges :
        Dict {name: (min, max)} OR iterable of (name, min, max).
    parameter_spacing :
        int, dict {name: int}, dict {name: (n, scale)},
        or dict {name: {"n": n, "scale": ..., "inverse": bool}}.
    ensure_increasing_axes :
        If True (default), each axis is returned in ascending order.

    Returns
    -------
    points_with_idx, parameter_names, axes_values, grid_shape
    """

    # -------- normalize ranges --------
    if isinstance(param_ranges, dict):
        items = sorted(param_ranges.items(), key=lambda kv: str(kv[0]))
        ranges_by_name = {str(k): (float(v[0]), float(v[1])) for k, v in items}
        parameter_names = [str(k) for k, _ in items]
    else:
        items = sorted([(str(n), (float(a), float(b))) for (n, a, b) in param_ranges],
                       key=lambda x: x[0])
        ranges_by_name = {k: (a, b) for (k, (a, b)) in items}
        parameter_names = [k for (k, _) in items]

    # -------- parse spacing spec --------
    def _parse_one(spec):
        if isinstance(spec, int):
            return int(spec), "linear", False
        if isinstance(spec, (tuple, list)):
            n = int(spec[0])
            scale = str(spec[1]).lower() if len(spec) >= 2 else "linear"
            return n, scale, False
        if isinstance(spec, dict):
            n = int(spec.get("n", spec.get("count", 1)))
            scale = str(spec.get("scale", "linear")).lower()
            inverse = bool(spec.get("inverse", False))
            return n, scale, inverse
        raise ValueError(f"Unrecognized spacing spec: {spec}")

    if isinstance(parameter_spacing, int):
        per_param = {name: (int(parameter_spacing), "linear", False) for name in parameter_names}
    elif isinstance(parameter_spacing, dict):
        per_param = {name: _parse_one(parameter_spacing.get(name, 1)) for name in parameter_names}
    else:
        raise ValueError("parameter_spacing must be int or dict")

    # -------- helpers --------
    def _space_inclusive(a, b, count, *, scale="linear"):
        a = float(a); b = float(b); count = int(count)
        if count < 2:
            return np.array([a], dtype=float)
        if scale == "linear":
            return np.linspace(a, b, count, dtype=float)
        if scale == "log":
            if a <= 0 or b <= 0:
                raise ValueError(f"log spacing requires positive endpoints; got [{a}, {b}]")
            return np.logspace(np.log10(a), np.log10(b), count, dtype=float)
        raise ValueError("scale must be 'linear' or 'log'")

    # -------- build axes --------
    axes_values = []
    for name in parameter_names:
        pmin, pmax = ranges_by_name[name]
        count, scale, inverse_flag = per_param[name]

        if not inverse_flag:
            axis = _space_inclusive(pmin, pmax, count, scale=scale)
        else:
            inv_min = 1.0 / float(pmax)
            inv_max = 1.0 / float(pmin)
            inv_axis = _space_inclusive(inv_min, inv_max, count, scale=scale)
            axis = 1.0 / inv_axis
            if ensure_increasing_axes and axis.size > 1 and axis[0] > axis[-1]:
                axis = axis[::-1]

        axes_values.append(axis)

    # -------- mesh + enumerate points --------
    mesh_arrays = np.meshgrid(*axes_values, indexing="ij")
    grid_shape = tuple(len(ax) for ax in axes_values)

    points_with_idx = []
    for idx_tuple in np.ndindex(*grid_shape):
        point = {parameter_names[i]: float(mesh_arrays[i][idx_tuple]) for i in range(len(parameter_names))}
        points_with_idx.append((point, idx_tuple))

    return points_with_idx, parameter_names, axes_values, grid_shape

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
    grid_info,
    mesh_spacing,
    kk=0.0,
    band_index=None,
    floquet_max_l=None,
    decimals=3,
    force_new=False
):
    """
    New version of setup that uses 'datasetN' folders and 'parameters.json'.
    """
    Hname = getattr(hamiltonian_template, "name", "Hamiltonian")
    
    def _sanitize(name: str) -> str:
        return re.sub(r"[^\w.\-]", "_", str(name))
        
    base_root = os.path.join(os.getcwd(), "results", "2D_QGT_ND", _sanitize(Hname))
    
    # 1. Normalize ranges → dict {name: {"min": ..., "max": ...}}
    def _norm_ranges_dict(ranges):
        if isinstance(ranges, dict):
            items = sorted(ranges.items(), key=lambda kv: kv[0])
            return {k: {"min": float(v[0]), "max": float(v[1])} for k, v in items}
        items = sorted([(str(n), float(a), float(b)) for (n, a, b) in ranges], key=lambda x: x[0])
        return {n: {"min": a, "max": b} for (n, a, b) in items}

    range_dict = _norm_ranges_dict(param_ranges)
    param_names_sorted = list(range_dict.keys())   # sorted alphabetically

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
        spacing_dict = {n: {"count": int(parameter_spacing), "scale": "linear"} for n in param_names_sorted}
    elif isinstance(parameter_spacing, dict):
        for n in param_names_sorted:
            spec = parameter_spacing.get(n, 1)
            cnt, scl = _parse_spacing(spec)
            spacing_dict[n] = {"count": cnt, "scale": scl}
            

    if not hasattr(hamiltonian_template, "get_parameters_dict"):
        raise TypeError(
            "hamiltonian_template must provide get_parameters_dict(parameter='2D')"
        )
    params = hamiltonian_template.get_parameters_dict(parameter="2D")

    k_grid = dict(grid_info)
    k_grid["mesh"] = int(mesh_spacing)
    k_grid["fixed_coordinate"] = float(kk)

    metadata = {
        "hamiltonian_name": Hname,
        "parameters": params,
        "scan_ranges": range_dict,
        "scan_spacing": spacing_dict,
        "k_grid": k_grid,
    }
    
    if band_index is not None:
        metadata["band_index"] = int(band_index)
    if floquet_max_l is not None:
        metadata["floquet_diagnostic"] = {
            "max_l": int(floquet_max_l),
            "band_basis": "zero_fourier_harmonic_energy_order",
            "index_order": ["coupled_band", "photon_index_l"],
            "includes_same_band": False,
        }
        
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
