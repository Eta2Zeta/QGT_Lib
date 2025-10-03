import os, copy, pickle
import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# ---------- utility: create one sweep root for N-D QGT ----------
def _sanitize(name: str) -> str:
    import re
    return re.sub(r'[^\w.\-]', '_', str(name))

def setup_qgt_nd_results_dir(
    hamiltonian_template,
    param_ranges,
    parameter_spacing,
    kx_range,
    ky_range,
    mesh_spacing,
    decimals=3,
    force_new=False
):
    """
    Create (or reuse) the root dir to hold a single N-D QGT npz bundle.
    Reuse only when 'qgt_nd_bundle.npz' is present; otherwise make a new numbered dir.
    Returns: (root_dir, used_existing)
    """
    Hname = getattr(hamiltonian_template, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "QGT_ND", _sanitize(Hname))

    # ---- label for parameter ranges (deterministic, sorted by name) ----
    if isinstance(param_ranges, dict):
        items = sorted(param_ranges.items(), key=lambda kv: kv[0])  # [(name,(min,max)),...]
    else:
        # iterable of (name, min, max) -> normalize to (name,(min,max)) then sort
        items = sorted([(n, (a, b)) for (n, a, b) in param_ranges], key=lambda x: x[0])

    parts = [
        f"{name}_{float(vmin):.{decimals}f}_{float(vmax):.{decimals}f}"
        for name, (vmin, vmax) in items
    ]

    # ---- label for per-parameter spacing ----
    if isinstance(parameter_spacing, int):
        spacing_parts = [f"{name}_{int(parameter_spacing)}" for (name, _) in items]
    else:
        spacing_parts = [f"{name}_{int(parameter_spacing.get(name, 1))}" for (name, _) in items]

    label  = "RANGES[" + "-".join(parts) + "]__SPACING[" + "-".join(spacing_parts) + "]"
    klabel = f"kx{kx_range[0]:.2f}_{kx_range[1]:.2f}__ky{ky_range[0]:.2f}_{ky_range[1]:.2f}__mesh{mesh_spacing}"
    base_name = _sanitize(f"{label}__{klabel}")

    # ---- reuse/create via the shared helper ----
    required_files = ["qgt_nd_bundle.npz"]  # only reuse if the bundle is present
    dir_path, used = pick_or_create_result_dir(
        base_root=base_root,
        base_name=base_name,
        required_files=required_files,
        validator=None,
        force_new=force_new,
        suffix_template="_data_set{n}",
        start_index=1
    )

    print(("Using existing QGT N-D sweep directory: " if used else "Created QGT N-D sweep directory: ") + dir_path)
    return dir_path, used

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
