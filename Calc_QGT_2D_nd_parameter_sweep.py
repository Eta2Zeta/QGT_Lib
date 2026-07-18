import sys
import os
import numpy as np
import pickle
from tqdm import tqdm
import copy
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial

from Library.Hamiltonian.Hamiltonian import *
from Library.Hamiltonian.ChiralHamiltonian import ChiralHamiltonian
from Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected import ChiralHamiltonianChiralBasisProjected
from Library.Hamiltonian.SquareLatticeHamiltonian import SquareLatticeHamiltonian
from Library.Hamiltonian.gWaveAltermagnetHamiltonian import gWaveAltermagnetHamiltonian
from Library.eigenvalue_calc_lib import *
from Library.QGT_lib import *
from Library.topology import *
from Library.data_management_utils_nd import (
    build_parameter_points,
    setup_qgt_nd_results_dir,
)
from Library.utilities import generate_2d_sym_lines
from Library.Hamiltonian_helper import get_Hamiltonian
from Library.eigenvalue_calc_lib_1d import eigenvalues_along_path
from Library.output_utils import print_calculation_complete


# ---------- per-point worker ----------
def _worker_qgt_point(arg, h_template, kx, ky, mesh_spacing, band, z_cutoff, k_path):
    """
    arg: (param_values_dict, idx_tuple) OR (param_values_dict, idx_tuple, progress_label)
    """
    if len(arg) == 3:
        param_values, idx, progress_label = arg
    else:
        param_values, idx = arg
        progress_label = None

    H = copy.deepcopy(h_template)
    for k, v in param_values.items():
        setattr(H, k, v)

    # Full k-grid eigenproblem + Hamiltonians
    eigenvalues, eigenfunctions, hamiltonian_array, hamiltonian_prime_array = \
        grid_eigenvalues_eigenfunctions(H, kx, ky, mesh_spacing, dim=int(H.dim))

    # QGT fields
    dkx = abs(kx[0, 1] - kx[0, 0])
    dky = abs(ky[1, 0] - ky[0, 0])
    delta_k = min(dkx, dky)

    (g_xx, g_yy, g_zz,
     g_xy_r, g_xy_i,
     g_xz_r, g_xz_i,
     g_yz_r, g_yz_i) = QGT_grid_num(
        kx, ky, eigenvalues, eigenfunctions, quantum_geometric_tensor_3d_num_phase_corrected,
        H, delta_k, band_index=band,
        progress_label=progress_label,
    )
    trace = g_xx + g_yy

    # Chern number
    if hasattr(H, "b1") and hasattr(H, "b2"):
        chern = compute_chern_number(g_xy_i, dkx, dky, kx, ky, H.b1, H.b2)
    else:
        berry = -2.0 * g_xy_i
        integral = np.trapz(np.trapz(berry, dx=dky, axis=1), dx=dkx, axis=0)
        chern = integral / (2 * np.pi)

    # Calculate eigenvalues along symmetry path
    eigenvalues_sym, _ = eigenvalues_along_path(H, k_path)

    return (
        idx,
        g_xx, g_xy_r, g_xy_i, g_yy, trace, float(chern),
        eigenvalues, eigenfunctions,
        hamiltonian_array, hamiltonian_prime_array,
        eigenvalues_sym,
    )


# ---------- master driver ----------
def compute_qgt_nd_parallel(
    hamiltonian_template,
    param_ranges,
    parameter_spacing,
    kx_range,
    ky_range,
    mesh_spacing,
    band=0,
    z_cutoff=1e2,
    num_points_per_segment=100,
    processes=None,
    force_new_dir=False,
    float_dtype=np.float64,
):
    """
    Builds an N-D parameter grid, computes QGT per point in parallel, then
    assembles contiguous N-D arrays:
        g_xx_grid, g_xy_real_grid, g_xy_imag_grid, g_yy_grid, trace_grid
        shape: (*param_shape, Ny, Nx)
        chern_grid shape: (*param_shape,)
    Saves a single .npz bundle + meta.pkl in a dedicated directory.
    Returns: (root_dir, npz_path)
    """
    # k-grid
    kx_lin = np.linspace(kx_range[0], kx_range[1], mesh_spacing)
    ky_lin = np.linspace(ky_range[0], ky_range[1], mesh_spacing)
    kx, ky = np.meshgrid(kx_lin, ky_lin)   # (Ny, Nx)

    Ny, Nx = ky.shape
    dim = int(getattr(hamiltonian_template, "dim", 0))
    band = int(band)
    if dim <= 0:
        raise ValueError(
            f"Invalid Hamiltonian dimension dim={dim}. "
            "The Hamiltonian template must define a positive .dim attribute."
        )
    if not 0 <= band < dim:
        raise ValueError(
            f"Invalid band index band={band} for {hamiltonian_template.__class__.__name__} "
            f"with dim={dim}. Valid band indices are 0 to {dim - 1}."
        )

    # Ensure reciprocal b-vectors exist
    H_template = copy.deepcopy(hamiltonian_template)
    if not hasattr(H_template, "b1") or not hasattr(H_template, "b2"):
        a = getattr(H_template, "a", 1.0)
        H_template.b1 = (2 * np.pi / (3 * a)) * np.array([1.0,  np.sqrt(3.0)])
        H_template.b2 = (2 * np.pi / (3 * a)) * np.array([1.0, -np.sqrt(3.0)])

    # Generate symmetry line k-points
    k_path, k_dist, node_indices, path_labels, path_points = \
        generate_2d_sym_lines(H_template, num_points_per_segment)
    num_k_points = len(k_path)

    # Parameter grid
    points_with_idx, names, axes, shape = build_parameter_points(param_ranges, parameter_spacing)
    total_param_points = int(np.prod(shape))

    def _label_for_idx(idx_tuple):
        flat = np.ravel_multi_index(idx_tuple, shape)
        coord = ", ".join(f"{names[i]}={axes[i][idx_tuple[i]]:.6g}" for i in range(len(names)))
        return f"{flat + 1}/{total_param_points} | {coord}"

    points_labeled = [(d, idx, _label_for_idx(idx)) for (d, idx) in points_with_idx]

    # Output arrays
    out_shape_fields = tuple(shape) + (Ny, Nx)
    g_xx_grid            = np.empty(out_shape_fields,              dtype=float_dtype)
    g_xy_real_grid       = np.empty(out_shape_fields,              dtype=float_dtype)
    g_xy_imag_grid       = np.empty(out_shape_fields,              dtype=float_dtype)
    g_yy_grid            = np.empty(out_shape_fields,              dtype=float_dtype)
    trace_grid           = np.empty(out_shape_fields,              dtype=float_dtype)
    chern_grid           = np.empty(shape,                         dtype=float_dtype)
    eigenvalues_grid     = np.empty(out_shape_fields + (dim,),     dtype=float_dtype)
    eigenfunctions_grid  = np.empty(out_shape_fields + (dim, dim), dtype=np.complex128)
    hamiltonian_grid     = np.empty(out_shape_fields + (dim, dim), dtype=np.complex128)
    hamiltonian_prime_grid = np.empty(out_shape_fields + (dim, dim), dtype=np.complex128)
    eigenvalues_sym_grid = np.empty(tuple(shape) + (num_k_points, dim), dtype=float_dtype)

    # Directory
    root, used = setup_qgt_nd_results_dir(
        H_template, param_ranges, parameter_spacing,
        kx_range, ky_range, mesh_spacing,
        band_index=band, force_new=force_new_dir,
    )
    bundle_path = os.path.join(root, "qgt_nd_bundle.npz")
    meta_path   = os.path.join(root, "meta.pkl")

    if (not force_new_dir) and os.path.exists(bundle_path):
        print(f"Bundle already exists: {bundle_path}")
        return root, bundle_path

    # Worker
    worker = partial(
        _worker_qgt_point,
        h_template=H_template,
        kx=kx, ky=ky, mesh_spacing=mesh_spacing,
        band=band, z_cutoff=z_cutoff,
        k_path=k_path,
    )

    procs = processes or min(max(1, cpu_count() - 1), max(1, len(points_with_idx)))
    print(f"Launching QGT N-D sweep on {procs} processes over {len(points_with_idx)} points ...")

    with Pool(processes=procs) as pool:
        with tqdm(total=len(points_labeled), desc="QGT per parameter point", unit="pt") as pbar:
            for result in pool.imap(worker, points_labeled):
                (idx,
                 gxx, gxyr, gxyi, gyy, tr, ch,
                 eigenvalues, eigenfunctions,
                 hamiltonian_array, hamiltonian_prime_array,
                 eigenvalues_sym) = result

                g_xx_grid[idx]            = gxx
                g_xy_real_grid[idx]       = gxyr
                g_xy_imag_grid[idx]       = gxyi
                g_yy_grid[idx]            = gyy
                trace_grid[idx]           = tr
                chern_grid[idx]           = ch
                eigenvalues_grid[idx]     = eigenvalues
                eigenfunctions_grid[idx]  = eigenfunctions
                hamiltonian_grid[idx]     = hamiltonian_array
                hamiltonian_prime_grid[idx] = hamiltonian_prime_array
                eigenvalues_sym_grid[idx] = eigenvalues_sym

                pbar.set_postfix_str(_label_for_idx(idx))
                pbar.update(1)

    # Save bundle
    np.savez_compressed(
        bundle_path,
        names=np.array(names, dtype=object),
        shape=np.array(shape, dtype=int),
        kx=kx, ky=ky,
        dkx=abs(kx_lin[1] - kx_lin[0]) if Nx > 1 else np.nan,
        dky=abs(ky_lin[1] - ky_lin[0]) if Ny > 1 else np.nan,
        mesh_spacing=np.int32(mesh_spacing),
        **{f"axis_{i}_{names[i]}": axes[i] for i in range(len(names))},
        g_xx_grid=g_xx_grid,
        g_xy_real_grid=g_xy_real_grid,
        g_xy_imag_grid=g_xy_imag_grid,
        g_yy_grid=g_yy_grid,
        trace_grid=trace_grid,
        chern_grid=chern_grid,
        eigenvalues_grid=eigenvalues_grid,
        eigenfunctions_grid=eigenfunctions_grid,
        hamiltonian_grid=hamiltonian_grid,
        hamiltonian_prime_grid=hamiltonian_prime_grid,
        k_path=k_path,
        k_dist=k_dist,
        node_indices=np.array(node_indices, dtype=int),
        path_labels=np.array(path_labels, dtype=object),
        path_points=path_points,
        eigenvalues_sym_grid=eigenvalues_sym_grid,
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

    print_calculation_complete("N-D QGT", bundle_path, artifact="Bundle")
    return root, bundle_path


# =============================================================================
# Configuration — edit below to set up your sweep
# =============================================================================

# H_template = gWaveAltermagnetHamiltonian(
#     t1=0.3, t2=0.3, t3=0.3, t4=0.3, mu=0,
#     Jx=0.2, Jy=0.0, Jz=0.0, lamb=0.1, lamb_z=0.1,
# )

# param_ranges = {
#     "Jx":     (0.0, 1),
#     "Jy":     (0.0, 1),
#     "Jz":     (0.0, 1),
#     "lamb":   (0.0, 0.3),
#     "lamb_z": (0.0, 0.3),
#     "t1":     (0.0, 0.3),
#     "t2":     (0.0, 0.3),
#     "t3":     (0.0, 0.3),
#     "t4":     (0.0, 0.3),
# }

# _n = 3
# parameter_spacing = {
#     "Jx":     {"n": 2 * _n, "scale": "linear"},
#     "Jy":     {"n": 2 * _n, "scale": "linear"},
#     "Jz":     {"n": 2 * _n, "scale": "linear"},
#     "lamb":   {"n": _n,     "scale": "linear"},
#     "lamb_z": {"n": _n,     "scale": "linear"},
#     "t1":     {"n": _n,     "scale": "linear"},
#     "t2":     {"n": _n,     "scale": "linear"},
#     "t3":     {"n": _n,     "scale": "linear"},
#     "t4":     {"n": _n,     "scale": "linear"},
# }

# H_template = SquareLatticeHamiltonian(A0=0.1, omega=5e0, t1=1, t2=1/np.sqrt(2), t5=0)
# H_template.polarization = 'left'

# param_ranges = {
#     "omega": (0.1, 50)
# }

# parameter_spacing = {
#     "omega": {"n": 6, "scale": "log"}
# }
# k = 2.5

# H_template = ChiralHamiltonianChiralBasisProjected(
#     n=5,
#     vF=542.10,
#     t1=355.16,
#     V=30.0,
#     omega=1000.0,
#     A0=0.10,
#     polarization="right",
#     magnus_order=1,
#     analytic_magnus=True,
#     magnus_first_term_mode="direct_drive",
# )

# param_ranges = {
#     "omega": (30.0, 5000.0),
# }

# parameter_spacing = {
#     "omega": {"n": 42, "scale": "log"},
# }
# k = 0.8


H_template = ChiralHamiltonian(
    n=5,
    a=1.0,
    vF=542.10,
    t1=355.16,
    V=30.0,
    valley='K',
    omega=2 * np.pi,
    A0=0.10,
    polarization='right',
    magnus_order=1,
    analytic_magnus=False,
)

param_ranges = {
    "omega": (5.0, 5000.0),
}

parameter_spacing = {
    "omega": {"n": 32, "scale": "log"},
}
k = 0.82
kx_range = (-k, k)
ky_range = (-k, k)
mesh_spacing = 150


def main():
    root, bundle_path = compute_qgt_nd_parallel(
        hamiltonian_template=H_template,
        param_ranges=param_ranges,
        parameter_spacing=parameter_spacing,
        kx_range=kx_range,
        ky_range=ky_range,
        mesh_spacing=mesh_spacing,
        band=4,
        z_cutoff=1e2,
        num_points_per_segment=200,
        processes=None,
        force_new_dir=False,
        float_dtype=np.float32,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
