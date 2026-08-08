import sys
import os
import shutil
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
from Library.Hamiltonian.THF_Hamiltonian import THF_Hamiltonian
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
from Library.dimension_lib import (
    CARTESIAN_ORDERS,
    create_2d_coordinate_grid_from_ranges,
    cylindrical_order_axes,
    is_cylindrical_order,
    map_k_by_order,
    normalize_coordinate_order,
)


def _metric_trace_on_sampled_plane(g_xx, g_yy, g_zz, order):
    """Return the metric trace in the two-dimensional sampled Cartesian plane."""
    order = normalize_coordinate_order(order)
    diagonal = {"x": g_xx, "y": g_yy, "z": g_zz}
    if order in CARTESIAN_ORDERS:
        plane_axes = order[:2]
    else:
        _, _, _, fixed_axis = cylindrical_order_axes(order)
        plane_axes = tuple(axis for axis in "xyz" if axis != fixed_axis)
    return diagonal[plane_axes[0]] + diagonal[plane_axes[1]]


def _open_nd_output_memmap(memmap_dir, name, shape, dtype):
    """Create one temporary disk-backed N-D output array."""
    return np.lib.format.open_memmap(
        os.path.join(memmap_dir, f"{name}.npy"),
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


def _flush_and_close_memmaps(memmaps):
    for array in memmaps:
        array.flush()
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _discard_nd_output_memmaps(memmaps, memmap_dir):
    """Close temporary output arrays and remove their backing directory."""
    _flush_and_close_memmaps(memmaps)
    shutil.rmtree(memmap_dir, ignore_errors=True)


# ---------- per-point worker ----------
def _worker_qgt_point(
    arg,
    h_template,
    ki,
    kj,
    mesh_spacing,
    band,
    k_path,
    max_l,
    kk,
    order,
    delta_k,
    phi_periodic,
):
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

    # Full k-grid eigenproblem. Matrix grids are not retained by N-D sweeps.
    (
        eigenvalues,
        eigenfunctions,
        _,
        _,
        floquet_max_ratio_array,
        floquet_max_ratio_indices_array,
    ) = grid_eigenvalues_eigenfunctions(
        H,
        ki,
        kj,
        mesh_spacing,
        dim=int(H.dim),
        kk=kk,
        order=order,
        max_l=max_l,
        store_hamiltonians=False,
    )

    # QGT fields
    (g_xx, g_yy, g_zz,
     g_xy_r, g_xy_i,
     g_xz_r, g_xz_i,
     g_yz_r, g_yz_i) = QGT_grid_num(
        ki, kj, eigenvalues, eigenfunctions, quantum_geometric_tensor_3d_num_phase_corrected,
        H, delta_k, band_index=band,
        progress_label=progress_label,
        kk=kk,
        order=order,
    )
    del eigenfunctions
    trace = _metric_trace_on_sampled_plane(g_xx, g_yy, g_zz, order)

    # Chern number
    omega_x, omega_y, omega_z = berry_curvature_components_from_qgt(
        g_xy_i,
        g_xz_i,
        g_yz_i,
    )
    if order == "xyz" and hasattr(H, "b1") and hasattr(H, "b2"):
        dki = abs(ki[0, 1] - ki[0, 0])
        dkj = abs(kj[1, 0] - kj[0, 0])
        chern = compute_chern_number(g_xy_i, dki, dkj, ki, kj, H.b1, H.b2)
    else:
        berry_flux = integrate_berry_flux_2d(
            omega_x,
            omega_y,
            omega_z,
            ki,
            kj,
            order=order,
            phi_periodic=phi_periodic,
        )
        chern = berry_flux / (2.0 * np.pi)

    winding = None
    if is_cylindrical_order(order) and phi_periodic:
        reference_axis, tangent_axis, tangent_sign, _ = (
            cylindrical_order_axes(order)
        )
        omega_by_axis = {"x": omega_x, "y": omega_y, "z": omega_z}
        _, winding = winding_numbers_vs_radius(
            ki,
            kj,
            omega_by_axis[reference_axis],
            tangent_sign * omega_by_axis[tangent_axis],
        )

    # Calculate eigenvalues along symmetry path
    eigenvalues_sym, _ = eigenvalues_along_path(H, k_path)

    return (
        idx,
        g_xx, g_yy, g_zz,
        g_xy_r, g_xy_i,
        g_xz_r, g_xz_i,
        g_yz_r, g_yz_i,
        trace, float(chern), winding,
        eigenvalues,
        floquet_max_ratio_array, floquet_max_ratio_indices_array,
        eigenvalues_sym,
    )


# ---------- master driver ----------
def compute_qgt_nd_parallel(
    hamiltonian_template,
    param_ranges,
    parameter_spacing,
    ki_range,
    kj_range,
    mesh_spacing,
    band=0,
    num_points_per_segment=100,
    processes=None,
    force_new_dir=False,
    float_dtype=np.float64,
    max_l=10,
    kk=0.0,
    order="xyz",
    include_endpoints=True,
    delta_k=1e-5,
):
    """
    Build an N-D parameter sweep on a Cartesian or cylindrical momentum grid.

    ``ki_range`` and ``kj_range`` describe the first two input coordinates for
    ``order``. For example, ``order='xpz'`` means ``ki=r``, ``kj=phi``, and
    ``kk=kz``. All nine QGT components are retained.

    The disk-backed field arrays have shapes:
        QGT fields: (*param_shape, N_kj, N_ki)
        chern_grid shape: (*param_shape,)
        winding_grid shape: (*param_shape, N_ki), for full polar circles
        floquet_max_ratio_grid shape: (*param_shape, Ny, Nx, dim)
        floquet_max_ratio_indices_grid shape: (*param_shape, Ny, Nx, dim, 2)
    Temporary .npy memmaps keep completed parameter points out of RAM. They
    are packed into the existing .npz bundle and removed after completion.
    Saves a single .npz bundle + meta.pkl in a dedicated directory.
    Returns: (root_dir, npz_path)
    """
    if isinstance(max_l, (bool, np.bool_)) or not isinstance(
        max_l,
        (int, np.integer),
    ):
        raise TypeError("max_l must be a positive integer")
    max_l = int(max_l)
    if max_l < 1:
        raise ValueError("max_l must be at least 1")

    order = normalize_coordinate_order(order)
    kk = float(kk)
    delta_k = float(delta_k)
    if not np.isfinite(kk):
        raise ValueError("kk must be finite")
    if not np.isfinite(delta_k) or delta_k <= 0.0:
        raise ValueError("delta_k must be a finite positive Cartesian step")

    # k-grid
    ki, kj, grid_info = create_2d_coordinate_grid_from_ranges(
        ki_range,
        kj_range,
        mesh_spacing,
        order=order,
        include_endpoints=include_endpoints,
    )
    kx, ky, kz = map_k_by_order(ki, kj, kk, order)
    grid_info.update(
        {
            "kk": kk,
            "kx_min": float(np.min(kx)),
            "kx_max": float(np.max(kx)),
            "ky_min": float(np.min(ky)),
            "ky_max": float(np.max(ky)),
            "kz_min": float(np.min(kz)),
            "kz_max": float(np.max(kz)),
        }
    )

    Ny, Nx = ki.shape
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

    # Directory
    root, used = setup_qgt_nd_results_dir(
        H_template, param_ranges, parameter_spacing,
        grid_info, mesh_spacing,
        kk=kk,
        band_index=band,
        floquet_max_l=max_l,
        force_new=force_new_dir,
    )
    bundle_path = os.path.join(root, "qgt_nd_bundle.npz")
    meta_path = os.path.join(root, "meta.pkl")
    memmap_dir = os.path.join(root, ".qgt_nd_memmap")

    if (not force_new_dir) and os.path.exists(bundle_path):
        if os.path.isdir(memmap_dir):
            shutil.rmtree(memmap_dir)
        print(f"Bundle already exists: {bundle_path}")
        return root, bundle_path

    if os.path.isdir(memmap_dir):
        shutil.rmtree(memmap_dir)
    os.makedirs(memmap_dir)

    # Disk-backed output arrays
    out_shape_fields = tuple(shape) + (Ny, Nx)
    g_xx_grid = _open_nd_output_memmap(
        memmap_dir, "g_xx_grid", out_shape_fields, float_dtype
    )
    g_yy_grid = _open_nd_output_memmap(
        memmap_dir, "g_yy_grid", out_shape_fields, float_dtype
    )
    g_zz_grid = _open_nd_output_memmap(
        memmap_dir, "g_zz_grid", out_shape_fields, float_dtype
    )
    g_xy_real_grid = _open_nd_output_memmap(
        memmap_dir, "g_xy_real_grid", out_shape_fields, float_dtype
    )
    g_xy_imag_grid = _open_nd_output_memmap(
        memmap_dir, "g_xy_imag_grid", out_shape_fields, float_dtype
    )
    g_xz_real_grid = _open_nd_output_memmap(
        memmap_dir, "g_xz_real_grid", out_shape_fields, float_dtype
    )
    g_xz_imag_grid = _open_nd_output_memmap(
        memmap_dir, "g_xz_imag_grid", out_shape_fields, float_dtype
    )
    g_yz_real_grid = _open_nd_output_memmap(
        memmap_dir, "g_yz_real_grid", out_shape_fields, float_dtype
    )
    g_yz_imag_grid = _open_nd_output_memmap(
        memmap_dir, "g_yz_imag_grid", out_shape_fields, float_dtype
    )
    trace_grid = _open_nd_output_memmap(
        memmap_dir, "trace_grid", out_shape_fields, float_dtype
    )
    chern_grid = _open_nd_output_memmap(
        memmap_dir, "chern_grid", tuple(shape), float_dtype
    )
    winding_grid = (
        _open_nd_output_memmap(
            memmap_dir,
            "winding_grid",
            tuple(shape) + (Nx,),
            float_dtype,
        )
        if is_cylindrical_order(order) and grid_info["phi_periodic"]
        else None
    )
    eigenvalues_grid = _open_nd_output_memmap(
        memmap_dir,
        "eigenvalues_grid",
        out_shape_fields + (dim,),
        float_dtype,
    )
    floquet_max_ratio_grid = _open_nd_output_memmap(
        memmap_dir,
        "floquet_max_ratio_grid",
        out_shape_fields + (dim,),
        np.float64,
    )
    floquet_max_ratio_indices_grid = _open_nd_output_memmap(
        memmap_dir,
        "floquet_max_ratio_indices_grid",
        out_shape_fields + (dim, 2),
        np.int32,
    )
    eigenvalues_sym_grid = _open_nd_output_memmap(
        memmap_dir,
        "eigenvalues_sym_grid",
        tuple(shape) + (num_k_points, dim),
        float_dtype,
    )

    output_memmaps = [
        g_xx_grid,
        g_yy_grid,
        g_zz_grid,
        g_xy_real_grid,
        g_xy_imag_grid,
        g_xz_real_grid,
        g_xz_imag_grid,
        g_yz_real_grid,
        g_yz_imag_grid,
        trace_grid,
        chern_grid,
        eigenvalues_grid,
        floquet_max_ratio_grid,
        floquet_max_ratio_indices_grid,
        eigenvalues_sym_grid,
    ]
    if winding_grid is not None:
        output_memmaps.append(winding_grid)

    print(f"Writing sweep arrays incrementally to: {memmap_dir}")

    # Worker
    worker = partial(
        _worker_qgt_point,
        h_template=H_template,
        ki=ki, kj=kj, mesh_spacing=mesh_spacing,
        band=band,
        k_path=k_path,
        max_l=max_l,
        kk=kk,
        order=order,
        delta_k=delta_k,
        phi_periodic=bool(grid_info["phi_periodic"]),
    )

    procs = processes or min(max(1, cpu_count() - 1), max(1, len(points_with_idx)))
    print(f"Launching QGT N-D sweep on {procs} processes over {len(points_with_idx)} points ...")

    def _store_result(result):
        (idx,
         gxx, gyy, gzz,
         gxyr, gxyi,
         gxzr, gxzi,
         gyzr, gyzi,
         tr, ch, winding,
         eigenvalues,
         floquet_max_ratio_array, floquet_max_ratio_indices_array,
         eigenvalues_sym) = result

        g_xx_grid[idx] = gxx
        g_yy_grid[idx] = gyy
        g_zz_grid[idx] = gzz
        g_xy_real_grid[idx] = gxyr
        g_xy_imag_grid[idx] = gxyi
        g_xz_real_grid[idx] = gxzr
        g_xz_imag_grid[idx] = gxzi
        g_yz_real_grid[idx] = gyzr
        g_yz_imag_grid[idx] = gyzi
        trace_grid[idx] = tr
        chern_grid[idx] = ch
        if winding_grid is not None:
            winding_grid[idx] = winding
        eigenvalues_grid[idx] = eigenvalues
        floquet_max_ratio_grid[idx] = floquet_max_ratio_array
        floquet_max_ratio_indices_grid[idx] = floquet_max_ratio_indices_array
        eigenvalues_sym_grid[idx] = eigenvalues_sym
        return idx

    try:
        if procs == 1:
            result_iterator = map(worker, points_labeled)
            with tqdm(
                total=len(points_labeled),
                desc="QGT per parameter point",
                unit="pt",
            ) as pbar:
                for result in result_iterator:
                    idx = _store_result(result)
                    pbar.set_postfix_str(_label_for_idx(idx))
                    pbar.update(1)
        else:
            with Pool(processes=procs) as pool:
                with tqdm(
                    total=len(points_labeled),
                    desc="QGT per parameter point",
                    unit="pt",
                ) as pbar:
                    for result in pool.imap(worker, points_labeled):
                        idx = _store_result(result)
                        pbar.set_postfix_str(_label_for_idx(idx))
                        pbar.update(1)
    except BaseException:
        _discard_nd_output_memmaps(output_memmaps, memmap_dir)
        raise

    # Save bundle
    bundle_data = dict(
        names=np.array(names, dtype=object),
        shape=np.array(shape, dtype=int),
        ki=ki,
        kj=kj,
        kx=kx,
        ky=ky,
        kz=kz,
        order=np.array(order),
        kk=np.float64(kk),
        coordinate_labels=np.array(grid_info["coordinate_labels"], dtype=object),
        coordinate_system=np.array(grid_info["coordinate_system"]),
        phi_periodic=np.bool_(grid_info["phi_periodic"]),
        dki=np.float64(grid_info["dki"]),
        dkj=np.float64(grid_info["dkj"]),
        delta_k=np.float64(delta_k),
        mesh_spacing=np.int32(mesh_spacing),
        **{f"axis_{i}_{names[i]}": axes[i] for i in range(len(names))},
        g_xx_grid=g_xx_grid,
        g_yy_grid=g_yy_grid,
        g_zz_grid=g_zz_grid,
        g_xy_real_grid=g_xy_real_grid,
        g_xy_imag_grid=g_xy_imag_grid,
        g_xz_real_grid=g_xz_real_grid,
        g_xz_imag_grid=g_xz_imag_grid,
        g_yz_real_grid=g_yz_real_grid,
        g_yz_imag_grid=g_yz_imag_grid,
        trace_grid=trace_grid,
        chern_grid=chern_grid,
        eigenvalues_grid=eigenvalues_grid,
        floquet_max_l=np.int32(max_l),
        floquet_max_ratio_grid=floquet_max_ratio_grid,
        floquet_max_ratio_indices_grid=floquet_max_ratio_indices_grid,
        k_path=k_path,
        k_dist=k_dist,
        node_indices=np.array(node_indices, dtype=int),
        path_labels=np.array(path_labels, dtype=object),
        path_points=path_points,
        eigenvalues_sym_grid=eigenvalues_sym_grid,
    )
    if order == "xyz":
        bundle_data["dkx"] = np.float64(grid_info["dki"])
        bundle_data["dky"] = np.float64(grid_info["dkj"])
    if winding_grid is not None:
        bundle_data["winding_radius"] = np.asarray(ki[0, :], dtype=float)
        bundle_data["winding_grid"] = winding_grid
    for array in output_memmaps:
        array.flush()
    try:
        np.savez_compressed(bundle_path, **bundle_data)
    except BaseException:
        if os.path.exists(bundle_path):
            os.remove(bundle_path)
        raise
    finally:
        _discard_nd_output_memmaps(output_memmaps, memmap_dir)

    meta = {
        "Hamiltonian_Template": H_template,
        "param_ranges": param_ranges,
        "parameter_spacing": parameter_spacing,
        "grid_info": grid_info,
        "ki_range": tuple(ki_range),
        "kj_range": tuple(kj_range),
        "kk": kk,
        "order": order,
        "include_endpoints": bool(include_endpoints),
        "delta_k": delta_k,
        "mesh_spacing": int(mesh_spacing),
        "band": int(band),
        "floquet_max_l": int(max_l),
        "floquet_ratio_band_basis": "zero_fourier_harmonic_energy_order",
        "floquet_ratio_index_order": ("coupled_band", "photon_index_l"),
        "floquet_ratio_includes_same_band": False,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print_calculation_complete("N-D QGT", bundle_path, artifact="Bundle")
    return root, bundle_path


# =============================================================================
# Configuration — edit below to set up your sweep
# =============================================================================


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


# H_template = ChiralHamiltonian(
#     n=5,
#     a=1.0,
#     vF=542.10,
#     t1=355.16,
#     V=30.0,
#     valley='K',
#     omega=2 * np.pi,
#     A0=0.10,
#     polarization='right',
#     magnus_order=1,
#     analytic_magnus=False,
# )
#
# param_ranges = {
#     "omega": (5.0, 5000.0),
# }
#
# parameter_spacing = {
#     "omega": {"n": 32, "scale": "log"},
# }
# k = 0.82
# ki_range = (-k, k)
# kj_range = (-k, k)
# mesh_spacing = 150


def thf_flat_band_matching_omega(hamiltonian, potential_strength):
    """Return omega for which the Gamma-point drive splitting equals |V|.

    For the central flat-band basis pair (components 3 and 4), the first
    circular-drive Magnus term has magnitude

        A0^2 * (v_star^2 - g(A0)^2 * v_star_double_prime^2) / omega.

    Right polarization has the same sign pattern as the positive-V term;
    left polarization reverses it. The returned value matches magnitudes.
    """
    potential_strength = abs(float(potential_strength))
    if potential_strength == 0:
        raise ValueError("potential_strength must be nonzero")
    if hamiltonian.A0 == 0:
        raise ValueError("The matching frequency is undefined when A0=0")

    drive_form_factor = hamiltonian.gaussian_form_factor(hamiltonian.A0, 0.0)
    flat_band_coefficient = (
        hamiltonian.v_star**2
        - (drive_form_factor * hamiltonian.v_star_double_prime) ** 2
    )
    if flat_band_coefficient <= 0:
        raise ValueError("The flat-band drive coefficient must be positive")
    return hamiltonian.A0**2 * flat_band_coefficient / potential_strength


THF_A0 = 0.10
THF_V_MAX = 10.0  # meV; deliberately smaller than |gamma| = 24.75 meV

# H_template = THF_Hamiltonian(
#     A0=THF_A0,
#     V=0.0,
#     omega=1.0,  # Replaced below by the analytically matched sweep endpoint.
#     polarization="right",
#     magnus_order=1,
#     analytic_magnus=False,
# )

# if not THF_V_MAX < abs(H_template.gamma):
#     raise ValueError("THF_V_MAX must be smaller than |gamma|")

# # THF_OMEGA_MIN = thf_flat_band_matching_omega(H_template, THF_V_MAX)
# THF_OMEGA_MIN = 5000
# THF_OMEGA_MAX = 100000
# H_template.omega = THF_OMEGA_MIN


H_template = gWaveAltermagnetHamiltonian(
    t1=0.3, t2=0.3, t3=0.3, t4=0.3, mu=0,
    Jx=0.0, Jy=0.0, Jz=0.2, lamb=0.1, lamb_z=0.1,
)

param_ranges = {
    "t3": (-0.5, 0.5),
    "t4": (-0.5, 0.5),
    "lamb_z":   (-0.5, 0.5),
}

_n = 30
parameter_spacing = {
    "t3": {"n": _n,     "scale": "log"},
    "t4": {"n": 2,     "scale": "linear"},
    "lamb_z": {"n": _n,     "scale": "linear"}
}

# param_ranges = {
#     "V": (0.0, THF_V_MAX),
#     "omega": (THF_OMEGA_MIN, THF_OMEGA_MAX),
# }

# parameter_spacing = {
#     "V": {"n": 5, "scale": "linear"},
#     "omega": {"n": 16, "scale": "log"},
# }


# order = "xyz"
kk = 0.0
include_endpoints = True
# ki_range = (-H_template.k_theta, H_template.k_theta)
# kj_range = (-H_template.k_theta, H_template.k_theta)

# Polar example for any supported cylindrical order:
order = "xpz"  # Also: ypz, xpy, zpy, ypx, zpx
ki_range = (0.0, 1.5 * np.pi) # radius
kj_range = (0.0, 2.0 * np.pi)        # phi
mesh_spacing = 100
flat_band_index = 2
num_points_per_segment = 100
sweep_processes = 10
floquet_max_l = 10


def main():
    root, bundle_path = compute_qgt_nd_parallel(
        hamiltonian_template=H_template,
        param_ranges=param_ranges,
        parameter_spacing=parameter_spacing,
        ki_range=ki_range,
        kj_range=kj_range,
        mesh_spacing=mesh_spacing,
        band=flat_band_index,
        num_points_per_segment=num_points_per_segment,
        processes=sweep_processes,
        force_new_dir=False,
        float_dtype=np.float32,
        max_l=floquet_max_l,
        kk=kk,
        order=order,
        include_endpoints=include_endpoints,
        delta_k=1e-5,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
