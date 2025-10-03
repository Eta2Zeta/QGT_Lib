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

def _space_inclusive(a, b, n, scale="linear"):
    n = int(n)
    if n <= 0:
        raise ValueError("n must be >= 1")
    if scale == "linear":
        if n == 1: 
            return np.array([0.5*(a+b)], dtype=float)
        return np.linspace(a, b, n, dtype=float)
    elif scale == "log":
        if a <= 0 or b <= 0:
            raise ValueError("log spacing requires positive endpoints.")
        if n == 1:
            # geometric mean puts the single point “in the middle” on a log scale
            return np.array([np.sqrt(a*b)], dtype=float)
        return np.logspace(np.log10(a), np.log10(b), n, dtype=float)
    else:
        raise ValueError(f"Unknown scale '{scale}' (use 'linear' or 'log').")

def build_parameter_points(param_ranges, parameter_spacing):
    """
    param_ranges:
        dict {name: (min, max)}  OR iterable [(name, min, max), ...]

    parameter_spacing (per-parameter choices):
        - int: same count for all, default 'linear'
        - dict {name: int}: per-parameter counts, default 'linear'
        - dict {name: (n, scale)} e.g. {"omega": (40,"log"), "V": (2,"linear")}
        - dict {name: {"n": n, "scale": "log"|"linear"}}  (more explicit)

    returns:
        (points_with_idx, names, axes, shape)
        points_with_idx = [({name: value, ...}, idx_tuple), ...]
    """
    # normalize ranges
    if isinstance(param_ranges, dict):
        items = sorted(param_ranges.items(), key=lambda kv: kv[0])
        rng = {k: tuple(v) for k, v in items}
        names = [k for k, _ in items]
    else:
        items = sorted([(n, (a, b)) for (n, a, b) in param_ranges], key=lambda x: x[0])
        rng = {k: (a, b) for (k, (a, b)) in items}
        names = [k for (k, _) in items]

    # parse spacing specs
    def _parse_spec(spec):
        # returns (n, scale)
        if isinstance(spec, int):
            return int(spec), "linear"
        if isinstance(spec, (tuple, list)):
            if len(spec) < 1:
                raise ValueError("spacing tuple must be (n, [scale])")
            n = int(spec[0])
            scale = str(spec[1]).lower() if len(spec) >= 2 else "linear"
            return n, scale
        if isinstance(spec, dict):
            n = int(spec.get("n", spec.get("count", 1)))
            scale = str(spec.get("scale", "linear")).lower()
            return n, scale
        # fallback
        raise ValueError(f"Unrecognized spacing spec: {spec}")

    if isinstance(parameter_spacing, int):
        counts_scales = {k: (int(parameter_spacing), "linear") for k in names}
    elif isinstance(parameter_spacing, dict):
        counts_scales = {k: _parse_spec(parameter_spacing.get(k, 1)) for k in names}
    else:
        raise ValueError("parameter_spacing must be int or dict")

    # build axes
    axes  = []
    for k in names:
        a, b = rng[k]
        n, scale = counts_scales[k]
        axes.append(_space_inclusive(a, b, n, scale=scale))

    mesh  = np.meshgrid(*axes, indexing="ij")
    shape = tuple(len(ax) for ax in axes)

    points_with_idx = []
    for idx in np.ndindex(*shape):
        d = { names[i]: float(mesh[i][idx]) for i in range(len(names)) }
        points_with_idx.append((d, idx))

    return points_with_idx, names, axes, shape

# ---------- per-point worker ----------
def _worker_qgt_point(arg, h_template, kx, ky, mesh_spacing, band, z_cutoff):
    """
    arg is (param_values_dict, idx_tuple)
    Returns: (idx_tuple, g_xx, g_xy_r, g_xy_i, g_yy, trace, chern)
    """
    param_values, idx = arg
    H = copy.deepcopy(h_template)
    for k, v in param_values.items():
        setattr(H, k, v)

    # eigen
    ev, ef, *_ = spiral_eigenvalues_eigenfunctions(
        H, kx, ky, mesh_spacing, dim=int(H.dim), phase_correction=False
    )

    # QGT fields
    dkx = abs(kx[0,1] - kx[0,0])
    dky = abs(ky[1,0] - ky[0,0])
    delta_k = min(dkx, dky)

    g_xx, g_xy_r, g_xy_i, g_yy, trace = QGT_grid_num(
        kx, ky, ev, ef, quantum_geometric_tensor_num,
        H, delta_k, band_index=band, z_cutoff=z_cutoff
    )

    # Chern
    if hasattr(H, "b1") and hasattr(H, "b2"):
        ch = compute_chern_number(g_xy_i, dkx, dky, kx, ky, H.b1, H.b2)
    else:
        Berry = -2.0 * g_xy_i
        integral = np.trapz(np.trapz(Berry, dx=dky, axis=1), dx=dkx, axis=0)
        ch = integral / (2*np.pi)

    return idx, g_xx, g_xy_r, g_xy_i, g_yy, trace, float(ch)


# ---------- master driver: compute & save one N-D bundle ----------
def compute_qgt_nd_parallel(hamiltonian_template,
                            param_ranges, parameter_spacing,
                            kx_range, ky_range, mesh_spacing,
                            band=0, z_cutoff=1e2, processes=None,
                            force_new_dir=False, float_dtype=np.float64):
    """
    Builds an N-D parameter grid, computes QGT per point (in parallel),
    then assembles *contiguous N-D arrays*:
        g_xx_grid, g_xy_real_grid, g_xy_imag_grid, g_yy_grid, trace_grid
        with shape (*param_shape, Ny, Nx)
        and chern_grid with shape (*param_shape,)
    Saves a single npz bundle + a small meta.pkl in a dedicated directory.
    Returns: (root_dir, npz_path)
    """
    # k-grid
    kx_lin = np.linspace(kx_range[0], kx_range[1], mesh_spacing)
    ky_lin = np.linspace(ky_range[0], ky_range[1], mesh_spacing)
    kx, ky = np.meshgrid(kx_lin, ky_lin)       # shapes (Ny,Nx)
    Ny, Nx = ky.shape

    # ensure reciprocal b-vectors exist (for hex BZ masks / chern)
    H_template = copy.deepcopy(hamiltonian_template)
    if (not hasattr(H_template, "b1")) or (not hasattr(H_template, "b2")):
        a = getattr(H_template, "a", 1.0)
        H_template.b1 = (2*np.pi/(3*a)) * np.array([1.0,  np.sqrt(3.0)])
        H_template.b2 = (2*np.pi/(3*a)) * np.array([1.0, -np.sqrt(3.0)])

    # parameter grid
    points_with_idx, names, axes, shape = build_parameter_points(param_ranges, parameter_spacing)

    # output arrays
    out_shape_fields = tuple(shape) + (Ny, Nx)
    g_xx_grid      = np.empty(out_shape_fields, dtype=float_dtype)
    g_xy_real_grid = np.empty(out_shape_fields, dtype=float_dtype)
    g_xy_imag_grid = np.empty(out_shape_fields, dtype=float_dtype)
    g_yy_grid      = np.empty(out_shape_fields, dtype=float_dtype)
    trace_grid     = np.empty(out_shape_fields, dtype=float_dtype)
    chern_grid     = np.empty(shape,            dtype=float_dtype)

    # directory
    root, used = setup_qgt_nd_results_dir(H_template, param_ranges, parameter_spacing,
                                          kx_range, ky_range, mesh_spacing,
                                          force_new=force_new_dir)
    bundle_path = os.path.join(root, "qgt_nd_bundle.npz")
    meta_path   = os.path.join(root, "meta.pkl")
    if (not force_new_dir) and os.path.exists(bundle_path):
        print(f"Bundle already exists: {bundle_path}")
        return root, bundle_path

    # worker
    worker = partial(
        _worker_qgt_point,
        h_template=H_template,
        kx=kx, ky=ky, mesh_spacing=mesh_spacing,
        band=band, z_cutoff=z_cutoff
    )

    procs = processes or min(cpu_count(), max(1, len(points_with_idx)))
    # procs = 1
    print(f"Launching QGT N-D sweep on {procs} processes over {len(points_with_idx)} points ...")

    with Pool(processes=procs) as pool:
        for (idx, gxx, gxyr, gxyi, gyy, tr, ch) in tqdm(
            pool.imap(worker, points_with_idx),
            total=len(points_with_idx),
            desc="QGT per parameter point"
        ):
            # idx is an N-D index into the param grid
            g_xx_grid[idx]      = gxx
            g_xy_real_grid[idx] = gxyr
            g_xy_imag_grid[idx] = gxyi
            g_yy_grid[idx]      = gyy
            trace_grid[idx]     = tr
            chern_grid[idx]     = ch

    # save bundle
    np.savez_compressed(
        bundle_path,
        names=np.array(names, dtype=object),
        shape=np.array(shape, dtype=int),
        kx=kx, ky=ky,
        dkx=abs(kx_lin[1]-kx_lin[0]) if Nx>1 else np.nan,
        dky=abs(ky_lin[1]-ky_lin[0]) if Ny>1 else np.nan,
        mesh_spacing=np.int32(mesh_spacing),
        **{f"axis_{i}_{names[i]}": axes[i] for i in range(len(names))},
        g_xx_grid=g_xx_grid,
        g_xy_real_grid=g_xy_real_grid,
        g_xy_imag_grid=g_xy_imag_grid,
        g_yy_grid=g_yy_grid,
        trace_grid=trace_grid,
        chern_grid=chern_grid
    )

    # save minimal meta (full H template, ranges, spacing, etc.)
    meta = {
        "Hamiltonian_Template": H_template,
        "param_ranges": param_ranges,
        "parameter_spacing": parameter_spacing,
        "kx_range": tuple(kx_range),
        "ky_range": tuple(ky_range),
        "mesh_spacing": int(mesh_spacing),
        "band": int(band),
        "z_cutoff": float(z_cutoff),
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"✅ Saved N-D QGT bundle to {bundle_path}")
    return root, bundle_path


# --- Build Hamiltonian template ---
H_template = ChiralHamiltonian(A0=0.1, n=5)
H_template.polarization = "right"

# ensure b-vectors exist
if not hasattr(H_template, "b1") or not hasattr(H_template, "b2"):
    a = getattr(H_template, "a", 1.0)
    H_template.b1 = (2*np.pi/(3*a)) * np.array([1.0,  np.sqrt(3.0)])
    H_template.b2 = (2*np.pi/(3*a)) * np.array([1.0, -np.sqrt(3.0)])

# --- parameter ranges ---
param_ranges = {
    "omega": (50, 5e3),   # drive frequency
    "V":     (10, 50),      # onsite potential or whatever V means in your H
}

parameter_spacing = {
    "omega": (16, "log"),   # number of ω points
    "V":     16,   # number of V points
}

# --- k-grid ---
k = 0.9
kx_range = (-k, k)
ky_range = (-k, k)
mesh_spacing = 100   # bump up for production

def main():
    root, bundle_path = compute_qgt_nd_parallel(
        hamiltonian_template=H_template,
        param_ranges=param_ranges,
        parameter_spacing=parameter_spacing,
        kx_range=kx_range,
        ky_range=ky_range,
        mesh_spacing=mesh_spacing,
        band=5,            # which band to evaluate
        z_cutoff=1e2,
        processes=None,    # auto-choose CPU count
        force_new_dir=False,
        float_dtype=np.float32
    )
    print(f"Results saved in {bundle_path}")

if __name__ == "__main__":
    mp.freeze_support()
    main()