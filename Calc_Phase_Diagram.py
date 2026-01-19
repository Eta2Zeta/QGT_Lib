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
from Hamiltonian.Hamiltonian_v2 import * 
from Library.eigenvalue_calc_lib import *
from Library.QGT_lib import *
from Library.topology import *
from Library.utilities import *
from Library.plotting_lib import *

def build_parameter_points(
    param_ranges,
    parameter_spacing,
    *,
    ensure_increasing_axes=True   # keep each axis ascending for plotting/meshing
):
    """
    Build an N-D parameter grid.

    Parameters
    ----------
    param_ranges :
        Dict {name: (min, max)} OR iterable of (name, min, max).
        Example:
            {"omega": (50.0, 5000.0), "V": (10.0, 50.0)}
        or:
            [("V", 10.0, 50.0), ("omega", 50.0, 5000.0)]

    parameter_spacing :
        One of the following (mix-and-match per parameter when using dict):
          1) int
             - Same count for all parameters, linear spacing.
                e.g. 32
          2) dict {name: int}
             - Per-parameter counts, linear spacing.
                e.g. {"omega": 16, "V": 16}
          3) dict {name: (n, scale)}
             - Per-parameter (count, scale), scale ∈ {"linear","log"}.
                e.g. {"omega": (16,"log"), "V": (16,"linear")}
          4) dict {name: {"n": n, "scale": "linear"|"log", "inverse": bool}}
             - Fully explicit; set inverse=True to space uniformly in 1/param.
               (Example: for omega, linear spacing in 1/omega from 1/omega_max → 1/omega_min,
                then flipped so omega itself is increasing.)
                e.g. {"omega": {"n": 16, "scale": "linear", "inverse": True}}

    ensure_increasing_axes :
        If True (default), each returned axis is non-decreasing (friendlier for plotting).

    Returns
    -------
    points_with_idx, parameter_names, axes_values, grid_shape
      points_with_idx : [({name: value, ...}, idx_tuple), ...]
      parameter_names : [str, ...]  (stable sorted)
      axes_values     : [np.ndarray, ...] (values per parameter)
      grid_shape      : tuple[int, ...]
    """
    import numpy as np

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

    # -------- parse spacing spec → (count, scale, inverse) per parameter --------
    def _parse_one(spec):
        # returns (n, scale, inverse)
        if isinstance(spec, int):
            return int(spec), "linear", False
        if isinstance(spec, (tuple, list)):
            if len(spec) < 1:
                raise ValueError("spacing tuple must be (n, [scale])")
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
            # Space uniformly in the inverse domain: 1/p from 1/pmax → 1/pmin
            inv_min = 1.0 / float(pmax)
            inv_max = 1.0 / float(pmin)
            inv_axis = _space_inclusive(inv_min, inv_max, count, scale=scale)
            axis = 1.0 / inv_axis
            # Make sure axis is increasing if desired
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


# ---------- per-point worker ----------
def _worker_qgt_point(arg, h_template, kx, ky, mesh_spacing, band, z_cutoff):
    """
    arg is (param_values_dict, idx_tuple)  OR  (param_values_dict, idx_tuple, progress_label)
    """
    import copy

    # --- unpack ---
    if len(arg) == 3:
        param_values, idx, progress_label = arg
    else:
        param_values, idx = arg
        progress_label = None

    H = copy.deepcopy(h_template)
    for k, v in param_values.items():
        setattr(H, k, v)

    # Full k-grid eigenproblem + Hamiltonians
    eigenvalues, eigenfunctions, hamiltonian_array, hamiltonian_prime_array = grid_eigenvalues_eigenfunctions(
        H, kx, ky, mesh_spacing, dim=int(H.dim)
    )

    # QGT fields
    dkx = abs(kx[0, 1] - kx[0, 0])
    dky = abs(ky[1, 0] - ky[0, 0])
    delta_k = min(dkx, dky)

    g_xx, g_xy_r, g_xy_i, g_yy, trace = QGT_grid_num(
        kx, ky, eigenvalues, eigenfunctions, quantum_geometric_tensor_num,
        H, delta_k, band_index=band, z_cutoff=z_cutoff,
        progress_label=progress_label   # <— NEW
    )

    # Chern (unchanged) ...
    if hasattr(H, "b1") and hasattr(H, "b2"):
        chern = compute_chern_number(g_xy_i, dkx, dky, kx, ky, H.b1, H.b2)
    else:
        berry = -2.0 * g_xy_i
        integral = np.trapz(np.trapz(berry, dx=dky, axis=1), dx=dkx, axis=0)
        chern = integral / (2*np.pi)

    return (
        idx,
        g_xx, g_xy_r, g_xy_i, g_yy, trace, float(chern),
        eigenvalues, eigenfunctions,
        hamiltonian_array, hamiltonian_prime_array,
    )



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
    dim = int(getattr(hamiltonian_template, "dim", 0))


    # ensure reciprocal b-vectors exist (for hex BZ masks / chern)
    H_template = copy.deepcopy(hamiltonian_template)
    if (not hasattr(H_template, "b1")) or (not hasattr(H_template, "b2")):
        a = getattr(H_template, "a", 1.0)
        H_template.b1 = (2*np.pi/(3*a)) * np.array([1.0,  np.sqrt(3.0)])
        H_template.b2 = (2*np.pi/(3*a)) * np.array([1.0, -np.sqrt(3.0)])

    # parameter grid
    points_with_idx, names, axes, shape = build_parameter_points(param_ranges, parameter_spacing)

    total_param_points = int(np.prod(shape))

    def _label_for_idx(idx_tuple):
        # flat index 0..N-1 -> show as 1..N
        flat = np.ravel_multi_index(idx_tuple, shape)
        coord = ", ".join(f"{names[i]}={axes[i][idx_tuple[i]]:.6g}" for i in range(len(names)))
        return f"{flat+1}/{total_param_points} | {coord}"

    # attach label to each point
    points_labeled = [(d, idx, _label_for_idx(idx)) for (d, idx) in points_with_idx]

    # output arrays
    out_shape_fields = tuple(shape) + (Ny, Nx)
    g_xx_grid      = np.empty(out_shape_fields, dtype=float_dtype)
    g_xy_real_grid = np.empty(out_shape_fields, dtype=float_dtype)
    g_xy_imag_grid = np.empty(out_shape_fields, dtype=float_dtype)
    g_yy_grid      = np.empty(out_shape_fields, dtype=float_dtype)
    trace_grid     = np.empty(out_shape_fields, dtype=float_dtype)
    chern_grid     = np.empty(shape,            dtype=float_dtype)

    eigenvalues_grid         = np.empty(out_shape_fields + (dim,),           dtype=float_dtype)
    eigenfunctions_grid      = np.empty(out_shape_fields + (dim, dim),       dtype=np.complex128)
    hamiltonian_grid         = np.empty(out_shape_fields + (dim, dim),       dtype=np.complex128)
    hamiltonian_prime_grid   = np.empty(out_shape_fields + (dim, dim),       dtype=np.complex128)



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
        # master tqdm for overall parameter sweep
        with tqdm(total=len(points_labeled), desc="QGT per parameter point", unit="pt") as pbar:
            for result in pool.imap(worker, points_labeled):
                (idx,
                gxx, gxyr, gxyi, gyy, tr, ch,
                eigenvalues, eigenfunctions,
                hamiltonian_array, hamiltonian_prime_array) = result

                # idx is an N-D index into the param grid
                g_xx_grid[idx]      = gxx
                g_xy_real_grid[idx] = gxyr
                g_xy_imag_grid[idx] = gxyi
                g_yy_grid[idx]      = gyy
                trace_grid[idx]     = tr
                chern_grid[idx]     = ch

                eigenvalues_grid[idx]       = eigenvalues                       # (Ny, Nx, dim)
                eigenfunctions_grid[idx]    = eigenfunctions                    # (Ny, Nx, dim, dim)
                hamiltonian_grid[idx]       = hamiltonian_array                  # (Ny, Nx, dim, dim)
                hamiltonian_prime_grid[idx] = hamiltonian_prime_array            # (Ny, Nx, dim, dim)

                # show which param point just finished on the master bar
                pbar.set_postfix_str(_label_for_idx(idx))
                pbar.update(1)



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
        chern_grid=chern_grid,
        # NEW:
        eigenvalues_grid=eigenvalues_grid,
        eigenfunctions_grid=eigenfunctions_grid,
        hamiltonian_grid=hamiltonian_grid,
        hamiltonian_prime_grid=hamiltonian_prime_grid,
    )

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
# H_template = ChiralHamiltonian(A0=0.1, n=5)
H_template = RhombohedralGrapheneHamiltonian(n=5, V=30, A0=0.1)
H_template.polarization = "left"

# ensure b-vectors exist
if not hasattr(H_template, "b1") or not hasattr(H_template, "b2"):
    a = getattr(H_template, "a", 1.0)
    H_template.b1 = (2*np.pi/(3*a)) * np.array([1.0,  np.sqrt(3.0)])
    H_template.b2 = (2*np.pi/(3*a)) * np.array([1.0, -np.sqrt(3.0)])

# --- parameter ranges ---
param_ranges = {
    "omega": (20, 5e3),   # drive frequency
    "V":     (-10, 50),      # onsite potential or whatever V means in your H
}

parameter_spacing = {
    "omega": {"n": 48, "scale": "linear", "inverse": True},
    "V":     {"n": 32, "scale": "linear"}
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
        band=1,            # which band to evaluate
        z_cutoff=1e2,
        processes=None,    # auto-choose CPU count
        force_new_dir=False,
        float_dtype=np.float32
    )
    print(f"Results saved in {bundle_path}")

if __name__ == "__main__":
    mp.freeze_support()
    main()