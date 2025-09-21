import sys
import os
import numpy as np
import pickle
from tqdm import tqdm  # Import tqdm for progress bar
import copy
import multiprocessing as mp
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


# ----- PARAM GRID -----

def _linspace_inclusive(a, b, n):
    if n == 1: return np.array([0.5*(a+b)], dtype=float)
    return np.linspace(a, b, int(n), dtype=float)

def build_parameter_points(param_ranges, parameter_spacing):
    """
    param_ranges: dict {name: (min, max)}  OR iterable [(name, min, max), ...]
    parameter_spacing: int  OR dict {name: n_points}
    returns: list of dicts [{name: value, ...}, ...] in a stable key order
    """
    if isinstance(param_ranges, dict):
        items = sorted(param_ranges.items(), key=lambda kv: kv[0])
        rng = {k: tuple(v) for k, v in items}
        names = [k for k, _ in items]
    else:
        items = sorted([(n, (a, b)) for (n, a, b) in param_ranges], key=lambda x: x[0])
        rng = {k: (a, b) for (k, (a, b)) in items}
        names = [k for (k, _) in items]

    if isinstance(parameter_spacing, int):
        counts = {k: int(parameter_spacing) for k in names}
    else:
        counts = {k: int(parameter_spacing.get(k, 1)) for k in names}

    axes  = [ _linspace_inclusive(*rng[k], counts[k]) for k in names ]
    mesh  = np.meshgrid(*axes, indexing="ij")
    shape = tuple(len(ax) for ax in axes)

    points_with_idx = []
    for idx in np.ndindex(*shape):
        d = { names[i]: float(mesh[i][idx]) for i in range(len(names)) }
        points_with_idx.append((d, idx))

    return points_with_idx, names, axes, shape


# ----- STEP 1: EIGEN (per-point worker) -----

def _worker_save_eigen_point(arg,
                             h_template,
                             kx, ky, mesh_spacing, kx_range, ky_range,
                             setup_range_dir_fn, setup_point_dir_fn,
                             range_root_dir,
                             dim, temp_dir=None):
    """
    Compute & save eigenvalues/eigenfunctions for a single parameter point.
    """
    param_values, idx = arg           # <-- unpack (dict, idx_tuple)
    param_values = dict(param_values) # make a copy so we can stash idx (optional)
    param_values["_idx"] = idx

    H = copy.deepcopy(h_template)
    # set parameters on the Hamiltonian instance
    for k, v in param_values.items():
        setattr(H, k, v)

    # per-point dir (reuse if complete)
    file_paths, used_existing, point_dir = setup_point_dir_fn(range_root_dir, param_values)
    if used_existing and os.path.exists(file_paths["eigenvalues"]) and os.path.exists(file_paths["eigenfunctions"]):
        return point_dir, idx  # already done

    # allocate & compute
    eigenfunctions = np.full((mesh_spacing, mesh_spacing, dim, dim), np.nan, dtype=complex)
    eigenvalues    = np.full((mesh_spacing, mesh_spacing, dim), np.nan, dtype=float)

    # your existing routine:
    ev, ef, pf, neigh_pf, m1, m2 = spiral_eigenvalues_eigenfunctions(
        H, kx, ky, mesh_spacing, dim=dim, phase_correction=False
    )
    eigenvalues, eigenfunctions = ev, ef

    # save
    np.save(file_paths["eigenvalues"], eigenvalues)
    np.save(file_paths["eigenfunctions"], eigenfunctions)

    meta_info = {
        "Hamiltonian_Obj": H,
        "param_values": param_values,
        "kx": kx, "ky": ky,
        "dkx": abs(kx[0,1] - kx[0,0]),
        "dky": abs(ky[1,0] - ky[0,0]),
        "mesh_spacing": mesh_spacing,
        "kx_range": kx_range,
        "ky_range": ky_range
    }
    with open(file_paths["meta_info"], "wb") as f:
        pickle.dump(meta_info, f)

    # optional temp copies
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        for key in ["eigenvalues", "eigenfunctions", "meta_info"]:
            src = file_paths[key] if key in file_paths else os.path.join(point_dir, f"{key}.npy")
            if os.path.exists(src):
                dst = os.path.join(temp_dir, os.path.basename(src))
                try:
                    import shutil
                    shutil.copy(src, dst)
                except Exception:
                    pass

    return point_dir, idx


# ----- STEP 2: QGT + CHERN (per-point worker) -----

def _worker_qgt_chern_point(arg, band):
    """
    Load eigen data in 'point_dir', compute QGT on full grid, then Chern# (masked to 1st BZ if available).
    Saves g_xx, g_xy_real, g_xy_imag, g_yy, trace, chern.npy to the same dir.
    """
    point_dir, idx = arg   # <-- unpack
    # load meta + eigen
    with open(os.path.join(point_dir, "meta_info.pkl"), "rb") as f:
        meta = pickle.load(f)

    H   = meta["Hamiltonian_Obj"]
    kx  = meta["kx"]
    ky  = meta["ky"]
    dkx = meta["dkx"]
    dky = meta["dky"]

    eigenvalues    = np.load(os.path.join(point_dir, "eigenvalues.npy"))
    eigenfunctions = np.load(os.path.join(point_dir, "eigenfunctions.npy"))

    delta_k = min(dkx, dky)
    z_cutoff = 1e2  # or pass as arg if you prefer

    g_xx, g_xy_r, g_xy_i, g_yy, trace = QGT_grid_num(
        kx, ky, eigenvalues, eigenfunctions, quantum_geometric_tensor_num,
        H, delta_k, band_index=band, z_cutoff=z_cutoff
    )

    # save QGT components
    np.save(os.path.join(point_dir, "g_xx.npy"), g_xx)
    np.save(os.path.join(point_dir, "g_xy_real.npy"), g_xy_r)
    np.save(os.path.join(point_dir, "g_xy_imag.npy"), g_xy_i)
    np.save(os.path.join(point_dir, "g_yy.npy"), g_yy)
    np.save(os.path.join(point_dir, "trace.npy"), trace)

    # Chern number (use hex mask if b1/b2 exist)
    if hasattr(H, "b1") and hasattr(H, "b2"):
        ch = compute_chern_number(g_xy_i, dkx, dky, kx, ky, H.b1, H.b2)
    else:
        # fallback: integrate over entire rectangular grid
        Berry = -2 * g_xy_i
        integral = np.trapz(np.trapz(Berry, dx=dky, axis=1), dx=dkx, axis=0)
        ch = integral / (2*np.pi)

    np.save(os.path.join(point_dir, "chern.npy"), np.array([ch], dtype=float))
    return idx, ch


# ----- MASTER DRIVER (two stages, both parallel) -----

def compute_phase_diagram_chern_parallel(hamiltonian_template,
                                         param_ranges,
                                         parameter_spacing,
                                         kx_range, ky_range, mesh_spacing,
                                         setup_range_dir_fn,    # setup_phase_diagram_results_general
                                         setup_point_dir_fn,    # setup_phase_point_directory_general
                                         band=1,
                                         processes=None,
                                         temp_dir=None):
    """
    1) Build parameter grid from param_ranges & parameter_spacing
    2) Create range root dir
    3) Parallel: for each point -> eigenvalues/eigenfunctions saved to its dir
    4) Parallel: for each point dir -> QGT + Chern saved to its dir
    Returns: (range_root_dir, list_of_point_dirs)
    """
    # k-grid
    kx_min, kx_max = kx_range
    ky_min, ky_max = ky_range
    kx = np.linspace(kx_min, kx_max, mesh_spacing)
    ky = np.linspace(ky_min, ky_max, mesh_spacing)
    kx, ky = np.meshgrid(kx, ky)
    dim = int(hamiltonian_template.dim)

    # range root
    range_root, _ = setup_range_dir_fn(hamiltonian_template, param_ranges, parameter_spacing, decimals=2)

    # grid of parameter points
    points_with_idx, names, axes, shape = build_parameter_points(param_ranges, parameter_spacing)
    
    # ---- Stage 1: eigen ----
    worker1 = partial(
        _worker_save_eigen_point,
        h_template=hamiltonian_template,
        kx=kx, ky=ky, mesh_spacing=mesh_spacing,
        kx_range=kx_range, ky_range=ky_range,
        setup_range_dir_fn=setup_range_dir_fn,
        setup_point_dir_fn=setup_point_dir_fn,
        range_root_dir=range_root,
        dim=dim,
        temp_dir=temp_dir
    )
    point_dirs_with_idx = []
    procs = processes or min(cpu_count(), max(1, len(points_with_idx)))
    with Pool(processes=procs) as pool:
        for tup in tqdm(pool.imap(worker1, points_with_idx),
                        total=len(points_with_idx),
                        desc="Eigen (2D) per point"):
            if isinstance(tup, tuple) and len(tup) == 2:
                point_dirs_with_idx.append(tup)            # (point_dir, idx)
            else:
                # make worker1 ALWAYS return (point_dir, idx).
                raise RuntimeError("Stage-1 worker returned an unexpected shape.")


    chern_grid = np.full(shape, np.nan, dtype=float)

    worker2 = partial(_worker_qgt_chern_point, band=band)
    with Pool(processes=procs) as pool:
        for idx, ch in tqdm(pool.imap(worker2, point_dirs_with_idx),
                            total=len(point_dirs_with_idx),
                            desc="QGT+Chern per point"):
            chern_grid[idx] = ch

    # Save one bundle at the sweep root
    np.save(os.path.join(range_root, "chern_grid.npy"), chern_grid)
    np.savez(os.path.join(range_root, "param_axes.npz"),
            **{f"axis_{i}_{names[i]}": axes[i] for i in range(len(names))},
            names=np.array(names, dtype=object))
    # Build a simple list of directories (same order as point grid)
    point_dirs = [pd for (pd, _) in point_dirs_with_idx]
    return range_root, point_dirs


# --- Haldane template ---
t2 = 1.0/3.0
H_template = HaldaneHamiltonian(t1=-1.0, t2=t2, M=0.0, psi=np.pi/2, a=1.0, omega=2*np.pi, A0=0.0)
# H_template = ChiralHamiltonian(n=5)
# (Optional) ensure b-vectors exist on the template
if not hasattr(H_template, "b1") or not hasattr(H_template, "b2"):
    a = getattr(H_template, "a", 1.0)
    H_template.b1 = (2*np.pi/(3*a)) * np.array([1.0,  np.sqrt(3.0)])
    H_template.b2 = (2*np.pi/(3*a)) * np.array([1.0, -np.sqrt(3.0)])
    

#^ Haldane Calculations
# # --- parameter ranges (dict: name -> (min, max)) ---
# param_ranges = {
#     "M":   (-2*np.pi*t2, 2*np.pi*t2),
#     "psi": (-np.pi, np.pi),
# }
# # --- spacing (dict: name -> #points) or a single int ---
# parameter_spacing = {
#     "M":   32,    # 5 values from -0.6..0.6
#     "psi": 32,
# }

# --- parameter ranges (dict: name -> (min, max)) ---
param_ranges = {
    "M":   (-2*np.pi*t2, 2*np.pi*t2)
}
# --- spacing (dict: name -> #points) or a single int ---
parameter_spacing = {
    "M":   100,
}


#^ Rhombohedral Graphene
# # --- parameter ranges (dict: name -> (min, max)) ---
# param_ranges = {
#     "V":   (-10, 10),
# }
# # --- spacing (dict: name -> #points) or a single int ---
# parameter_spacing = {
#     "V":   100,
# }

# --- k-grid settings ---
kx_range = (-np.pi, np.pi)
ky_range = (-np.pi, np.pi)
mesh_spacing = 64  # quick – bump to 128/256 for production


def main():
    # build H_template, param_ranges, parameter_spacing, kx/ky ranges, etc.
    range_root, point_dirs = compute_phase_diagram_chern_parallel(
        hamiltonian_template=H_template,
        param_ranges=param_ranges,
        parameter_spacing=parameter_spacing,
        kx_range=kx_range,
        ky_range=ky_range,
        mesh_spacing=mesh_spacing,
        setup_range_dir_fn=setup_phase_diagram_results_general,
        setup_point_dir_fn=setup_phase_point_directory_general,
        band=1,
        processes=None,
        temp_dir=os.path.join(os.getcwd(), "temp"),
    )

if __name__ == "__main__":
    mp.freeze_support()          # safe on mac/win; no-op on linux
    # Optional, explicit (mac uses spawn by default):
    # mp.set_start_method("spawn", force=True)
    main()
