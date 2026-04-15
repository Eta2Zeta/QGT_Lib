import numpy as np
import matplotlib.pyplot as plt
import os
from Library.plotting_utils import load_qgt, filter_entries_by_omega
from Library.topology import compute_chern_number
import Library.Hamiltonian.Hamiltonian
import Library.Hamiltonian.ChiralHamiltonian
import sys
import pickle
sys.modules["Library.Hamiltonian_v2"] = Library.Hamiltonian.Hamiltonian_v2
sys.modules["Library.Hamiltonian_v2.ChiralHamiltonian"] = Library.Hamiltonian.ChiralHamiltonian
# Fix pickle loading error by patching the module directly
# We need to assign the CLASS, not the module.
Library.Hamiltonian.Hamiltonian_v2.ChiralHamiltonian = Library.Hamiltonian.ChiralHamiltonian.ChiralHamiltonian


def _join_by_signed_inverse_omega(omegas_left, vals_left, omegas_right, vals_right,
                                  *, drop_overlap=True, tol=1e-9,
                                  left_sign=-1.0, right_sign=+1.0):
    """
    Build a single x-axis using signed 1/omega:
      left:  x = left_sign * (1/omega_left_sorted)   [default: negative]
      right: x = right_sign * (1/omega_right_sorted_reversed)  [default: positive]

    Steps:
      - Sort left omegas low->high and keep matching vals.
      - Sort right omegas low->high, then reverse to high->low and keep vals.
      - Optionally drop the duplicate junction if max(omega_left)==max(omega_right).
      - Return concatenated x and y arrays.
    """
    # Left: sort by increasing omega
    idxL = np.argsort(omegas_left)
    l_om = np.asarray(omegas_left)[idxL]
    l_v  = [vals_left[i] for i in idxL]
    x_left = left_sign * (1.0 / l_om)

    # Right: sort by increasing omega, then reverse (high->low)
    idxR = np.argsort(omegas_right)
    r_om_sorted = np.asarray(omegas_right)[idxR]
    r_v_sorted  = [vals_right[i] for i in idxR]
    r_om_rev    = r_om_sorted[::-1]
    r_v_rev     = r_v_sorted[::-1]
    x_right     = right_sign * (1.0 / r_om_rev)

    # If both runs include the exact same highest omega, drop the first right point
    if drop_overlap and r_om_sorted.size and np.isclose(l_om[-1], r_om_sorted[-1], atol=tol, rtol=0):
        r_om_rev = r_om_rev[1:]
        r_v_rev  = r_v_rev[1:]
        x_right  = x_right[1:]

    x = np.concatenate([x_left, x_right])
    y = np.array(l_v + r_v_rev, dtype=float)
    return x, y


# ---------------- joined plots (two datasets) ----------------

def plot_trace_std_vs_signed_invomega_joined(
    left_folder_name,
    right_folder_name,
    *,
    omega_min_left=None,
    omega_max_left=None,
    omega_min_right=None,
    omega_max_right=None,
    drop_overlap=True,
    tol=1e-9,
    left_sign=-1.0,
    right_sign=+1.0
):
    """Std of Tr[g] over BZ vs signed 1/omega, joined left|right."""
    entries_L, _ = load_qgt(left_folder_name)
    entries_R, _ = load_qgt(right_folder_name)

    filt_L = filter_entries_by_omega(entries_L, omega_min_left,  omega_max_left)
    filt_R = filter_entries_by_omega(entries_R, omega_min_right, omega_max_right)
    if len(filt_L) == 0 or len(filt_R) == 0:
        raise ValueError("No omega slices in range for one or both datasets.")

    omegas_L = np.array([float(e["omega"]) for e in filt_L], dtype=float)
    vals_L   = [np.nanstd(e["trace"]) for e in filt_L]

    omegas_R = np.array([float(e["omega"]) for e in filt_R], dtype=float)
    vals_R   = [np.nanstd(e["trace"]) for e in filt_R]

    x, y = _join_by_signed_inverse_omega(
        omegas_L, vals_L, omegas_R, vals_R,
        drop_overlap=drop_overlap, tol=tol, left_sign=left_sign, right_sign=right_sign
    )

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.axvline(0.0, color='k', lw=1, ls='--', alpha=0.5)  # seam at 0
    plt.xlabel("signed 1 / ω   (left < 0, right > 0)")
    plt.ylabel("Std. Dev of QGT Trace over BZ")
    plt.title("Fluctuation of QGT Trace vs signed 1/ω (joined left | right-reversed)")
    plt.grid(True, axis='both', alpha=0.35)
    plt.tight_layout()
    plt.show()


def plot_berry_std_vs_signed_invomega_joined(
    left_folder_name,
    right_folder_name,
    *,
    omega_min_left=None,
    omega_max_left=None,
    omega_min_right=None,
    omega_max_right=None,
    use_precomputed=False,   # True if entries include 'berry'
    convert_from_imQ=True,   # if not precomputed: Ω = -2 * Im(Q_xy)
    drop_overlap=True,
    tol=1e-9,
    left_sign=-1.0,
    right_sign=+1.0
):
    """Std of Berry curvature over BZ vs signed 1/omega, joined left|right."""
    entries_L, _ = load_qgt(left_folder_name)
    entries_R, _ = load_qgt(right_folder_name)

    filt_L = filter_entries_by_omega(entries_L, omega_min_left,  omega_max_left)
    filt_R = filter_entries_by_omega(entries_R, omega_min_right, omega_max_right)
    if len(filt_L) == 0 or len(filt_R) == 0:
        raise ValueError("No omega slices in range for one or both datasets.")

    omegas_L = np.array([float(e["omega"]) for e in filt_L], dtype=float)
    vals_L = []
    for e in filt_L:
        if use_precomputed and ("berry" in e):
            berry = np.asarray(e["berry"])
        else:
            gim = np.asarray(e["g_xy_imag"])
            berry = (-2.0 * gim) if convert_from_imQ else gim
        vals_L.append(np.nanstd(berry))

    omegas_R = np.array([float(e["omega"]) for e in filt_R], dtype=float)
    vals_R = []
    for e in filt_R:
        if use_precomputed and ("berry" in e):
            berry = np.asarray(e["berry"])
        else:
            gim = np.asarray(e["g_xy_imag"])
            berry = (-2.0 * gim) if convert_from_imQ else gim
        vals_R.append(np.nanstd(berry))

    x, y = _join_by_signed_inverse_omega(
        omegas_L, vals_L, omegas_R, vals_R,
        drop_overlap=drop_overlap, tol=tol, left_sign=left_sign, right_sign=right_sign
    )

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.axvline(0.0, color='k', lw=1, ls='--', alpha=0.5)
    plt.xlabel("signed 1 / ω   (left < 0, right > 0)")
    plt.ylabel("Std. Dev of Berry Curvature over BZ")
    plt.title("Fluctuation of Berry Curvature vs signed 1/ω (joined left | right-reversed)")
    plt.grid(True, axis='both', alpha=0.35)
    plt.tight_layout()
    plt.show()


def plot_integrated_trace_minus_berry_signed_invomega_joined(
    left_folder_name,
    right_folder_name,
    *,
    omega_min_left=None,
    omega_max_left=None,
    omega_min_right=None,
    omega_max_right=None,
    use_precomputed=False,   # True if entries include 'berry'
    convert_from_imQ=True,   # if not precomputed: Ω = -2 * Im(Q_xy)
    drop_overlap=True,
    tol=1e-9,
    left_sign=-1.0,
    right_sign=+1.0
):
    r"""Plot  \int_{BZ} [ Tr(g) - Ω ] d^2k  vs signed 1/ω, joined left|right."""
    entries_L, meta_L = load_qgt(left_folder_name)
    entries_R, meta_R = load_qgt(right_folder_name)

    # sanity: same dkx/dky so the integral is comparable
    dkx_L, dky_L = float(meta_L["dkx"]), float(meta_L["dky"])
    dkx_R, dky_R = float(meta_R["dkx"]), float(meta_R["dky"])
    if not (np.isclose(dkx_L, dkx_R) and np.isclose(dky_L, dky_R)):
        raise ValueError("dkx/dky differ between datasets; cannot join safely.")
    area = dkx_L * dky_L

    filt_L = filter_entries_by_omega(entries_L, omega_min_left,  omega_max_left)
    filt_R = filter_entries_by_omega(entries_R, omega_min_right, omega_max_right)
    if len(filt_L) == 0 or len(filt_R) == 0:
        raise ValueError("No omega slices in range for one or both datasets.")

    omegas_L, vals_L = [], []
    for e in filt_L:
        w = float(e["omega"])
        trace = np.asarray(e["trace"])
        if use_precomputed and ("berry" in e):
            berry = np.asarray(e["berry"])
        else:
            gim = np.asarray(e["g_xy_imag"])
            berry = (-2.0 * gim) if convert_from_imQ else gim
        vals_L.append(np.nansum(trace - berry) * area)
        omegas_L.append(w)

    omegas_R, vals_R = [], []
    for e in filt_R:
        w = float(e["omega"])
        trace = np.asarray(e["trace"])
        if use_precomputed and ("berry" in e):
            berry = np.asarray(e["berry"])
        else:
            gim = np.asarray(e["g_xy_imag"])
            berry = (-2.0 * gim) if convert_from_imQ else gim
        vals_R.append(np.nansum(trace - berry) * area)
        omegas_R.append(w)

    x, y = _join_by_signed_inverse_omega(
        np.array(omegas_L, dtype=float), vals_L,
        np.array(omegas_R, dtype=float), vals_R,
        drop_overlap=drop_overlap, tol=tol, left_sign=left_sign, right_sign=right_sign
    )

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.axvline(0.0, color='k', lw=1, ls='--', alpha=0.5)
    plt.xlabel("signed 1 / ω   (left < 0, right > 0)")
    plt.ylabel(r"$\int_{\mathrm{BZ}} [\,\mathrm{Tr}(g) - \Omega\,]\, d^2k$")
    plt.title(r"Integrated Tr(g) − Ω vs signed 1/ω (joined left | right-reversed)")
    plt.grid(True, axis='both', alpha=0.35)
    plt.tight_layout()
    plt.show()
    
# plot_trace_std_vs_signed_invomega_joined("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  omega_min_left=33, omega_min_right=50)


# plot_berry_std_vs_signed_invomega_joined("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  omega_min_left=33, omega_min_right=50)


# plot_integrated_trace_minus_berry_signed_invomega_joined("ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1", 
#                                  omega_min_left=33, omega_min_right=50)

# 2D, single data set not joined
def plot_trace_std_param2d_inverse_omega(result_dir,
                                         *,
                                         x_param,              # e.g. "omega" or "V"
                                         y_param,              # the other one
                                         cmap="inferno",
                                         symmetric_cbar=False,
                                         save_path=None,
                                         show=True):

    # Resolve bundle path
    bundle_path = (os.path.join(result_dir, "qgt_nd_bundle.npz")
                   if os.path.isdir(result_dir) else result_dir)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    bundle = np.load(bundle_path, allow_pickle=True)

    # --- parameter names and axes ---
    names = [str(n) for n in bundle["names"]]
    if len(names) != 2:
        raise ValueError(f"Expected exactly 2 parameters in bundle, found {len(names)}: {names}")
    if x_param not in names or y_param not in names:
        raise KeyError(f"Bundle has parameters {names}, but requested x={x_param}, y={y_param}.")

    name_to_idx = {n: i for i, n in enumerate(names)}
    ix = name_to_idx[x_param]
    iy = name_to_idx[y_param]
    if ix == iy:
        raise ValueError("x_param and y_param must be different.")

    def _axis_for(i, name):
        key = f"axis_{i}_{name}"
        if key not in bundle:
            raise KeyError(f"Missing axis array in bundle: '{key}'")
        return np.asarray(bundle[key], dtype=float)

    x_values = _axis_for(ix, x_param)
    y_values = _axis_for(iy, y_param)

    # --- handle 1/omega transformation ---
    if x_param.lower() == "omega":
        x_values = 1.0 / x_values
        x_label = "1 / omega"
    else:
        x_label = x_param

    # --- compute std over BZ ---
    trace_grid = np.asarray(bundle["trace_grid"])  # (N0, N1, Ny, Nx)
    std_bz = np.nanstd(trace_grid, axis=(-2, -1))  # (N0, N1) in bundle order (names[0], names[1])

    # --- reorder for plotting ---
    if (ix, iy) == (0, 1):
        Z = std_bz.T  # (N1, N0) = (len(y), len(x))
    elif (ix, iy) == (1, 0):
        Z = std_bz
    else:
        raise RuntimeError("Unexpected parameter indexing logic.")

    # --- color limits ---
    if symmetric_cbar:
        vmax_abs = float(np.nanmax(np.abs(Z)))
        vmin, vmax = -vmax_abs, vmax_abs
    else:
        vmin = float(np.nanmin(Z))
        vmax = float(np.nanmax(Z))

    # --- plot ---
    X, Y = np.meshgrid(x_values, y_values, indexing="xy")
    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_param)
    ax.set_title("Std of QGT Trace over Brillouin Zone")

    # Always use linear for 1/omega plot
    ax.set_xscale("linear")
    ax.set_yscale("linear")

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Std[Tr(g)]")

    # Debug prints
    print(f"[debug] x axis: {x_label} {x_values[0]:.6g} -> {x_values[-1]:.6g}")
    print(f"[debug] y axis: {y_param} {y_values[0]:.6g} -> {y_values[-1]:.6g}")
    print(f"[debug] Z shape: {Z.shape} (should be len(y) x len(x) = {len(y_values)} x {len(x_values)})")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)



# plot_trace_std_param2d_inverse_omega(
#     "results/QGT_ND/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_left-t1_355.16-vF_542.10-RANGES_V_10.000_50.000-omega_50.000_5000.000_-SPACING_V_16_linear-omega_16_log_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1",
#     x_param="omega", 
#     y_param="V",
# )
# plot_trace_std_param2d_inverse_omega(
#     "results/QGT_ND/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10-RANGES_V_10.000_50.000-omega_50.000_5000.000_-SPACING_V_16_linear-omega_16_log_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1",
#     x_param="omega", 
#     y_param="V",
# )

# plot_trace_std_param2d_inverse_omega(
#     "results/QGT_ND/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10-RANGES_V_10.000_50.000-omega_50.000_5000.000_-SPACING_V_32_linear-omega_32_linear_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1",
#     x_param="omega", 
#     y_param="V",
# )

# ------------------------------------------------------------------
# Helpers for joined signed 1/omega plotting (left=negative, right=positive)
# ------------------------------------------------------------------

def _load_bundle(path_like):
    bundle_path = (os.path.join(path_like, "qgt_nd_bundle.npz")
                   if os.path.isdir(path_like) else path_like)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    return np.load(bundle_path, allow_pickle=True)

def _get_b_vectors(path_like):
    """Load b1, b2 from meta.pkl in the given directory or bundle parent."""
    
    # Try to find meta.pkl
    if os.path.isdir(path_like):
        meta_path = os.path.join(path_like, "meta.pkl")
    else:
        # assume path_like is the bundle file, so parent is dir
        meta_path = os.path.join(os.path.dirname(path_like), "meta.pkl")
        
    if not os.path.exists(meta_path):
        print(f"Warning: meta.pkl not found at {meta_path}. Cannot load b-vectors.")
        return None, None
        
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        
    H = meta.get("Hamiltonian_Template")
    if H and hasattr(H, "b1") and hasattr(H, "b2"):
        return H.b1, H.b2
    else:
        print("Warning: Hamiltonian template in meta.pkl missing b1/b2.")
        return None, None

def _axis_for(bundle, i, name):
    key = f"axis_{i}_{name}"
    if key not in bundle:
        raise KeyError(f"Missing axis array in bundle: '{key}'")
    return np.asarray(bundle[key], dtype=float)

def _pick_field_grid(bundle, quantity, convert_berry_from_imQ):
    q = quantity.lower()
    if q == "trace":
        return np.asarray(bundle["trace_grid"])
    # Berry
    if q in ("berry", "berry_curvature", "omega"):
        if "berry_grid" in bundle.files:
            return np.asarray(bundle["berry_grid"])
        gxyi = np.asarray(bundle["g_xy_imag_grid"])
        return (-2.0 * gxyi) if convert_berry_from_imQ else gxyi
    # Trace - Berry
    if q in ("trace_minus_berry", "trace_minus_omega"):
        trace = np.asarray(bundle["trace_grid"])
        if "berry_grid" in bundle.files:
            berry = np.asarray(bundle["berry_grid"])
        else:
            gxyi = np.asarray(bundle["g_xy_imag_grid"])
            berry = (-2.0 * gxyi) if convert_berry_from_imQ else gxyi
        return trace - berry
    raise ValueError(f"Unknown quantity '{quantity}'.")

def prepare_joined_signed_invomega_data(
    left_result_dir,
    right_result_dir,
    *,
    y_param,
    quantity="trace",
    convert_berry_from_imQ=True,
    drop_overlap=True,
    tol=1e-9
):
    """
    Helper to load and mesh two datasets for signed 1/omega plotting.
    Returns: (x, y_values, Z, metadata_dict)
    """
    
    def _calc_field(bundle, result_dir):
        if quantity.lower() == "chern":
            # Load b-vectors and z_cutoff
            if os.path.isdir(result_dir):
                mpath = os.path.join(result_dir, "meta.pkl")
            else:
                mpath = os.path.join(os.path.dirname(result_dir), "meta.pkl")
            
            b1, b2 = None, None
            z_cutoff = 1e9 # default high if not found
            
            if os.path.exists(mpath):
                with open(mpath, "rb") as f:
                    meta_dict = pickle.load(f)
                    H = meta_dict.get("Hamiltonian_Template")
                    if H and hasattr(H, "b1") and hasattr(H, "b2"):
                        b1, b2 = H.b1, H.b2
                    if "z_cutoff" in meta_dict:
                        z_cutoff = float(meta_dict["z_cutoff"])
            
            if b1 is None or b2 is None:
                raise ValueError(f"Cannot calculate Chern for {result_dir}: missing b1/b2 in meta.pkl")
            
            # Load G_xy_imag
            g_xy_i = np.asarray(bundle["g_xy_imag_grid"]) # (N0, N1, Ny, Nx)
            kx_grid = np.asarray(bundle["kx"])
            ky_grid = np.asarray(bundle["ky"])
            
            dkx = float(bundle["dkx"]) if "dkx" in bundle else abs(kx_grid[0,1]-kx_grid[0,0])
            dky = float(bundle["dky"]) if "dky" in bundle else abs(ky_grid[1,0]-ky_grid[0,0])
            
            # Compute Chern for each parameter point
            dims = g_xy_i.shape[:-2] # (N0, N1)
            chern_grid = np.zeros(dims)
            
            # Saturation threshold
            sat_thresh = z_cutoff * 0.999
            
            for idx in np.ndindex(dims):
                 g_slice = g_xy_i[idx] 
                 if np.max(np.abs(g_slice)) >= sat_thresh:
                     chern_grid[idx] = np.nan
                 else:
                     ch = compute_chern_number(g_slice, dkx, dky, kx_grid, ky_grid, b1, b2)
                     chern_grid[idx] = ch
                 
            return chern_grid # already reduced
            
        else:
            # Standard fields (Trace, Berry, etc.) -> (N0, N1, Ny, Nx)
            f_grid = _pick_field_grid(bundle, quantity, convert_berry_from_imQ)
            return np.nanstd(f_grid, axis=(-2, -1))

    # ----- load left bundle -----
    L = _load_bundle(left_result_dir)
    names_L = [str(n) for n in L["names"]]
    
    # Handle single-sided (left only) case
    if right_result_dir is None:
        if len(names_L) != 2:
            raise ValueError(f"Expected 2 parameters, got {len(names_L)}")
        if "omega" not in names_L or y_param not in names_L:
             raise KeyError("Bundle must have 'omega' and y_param")
             
        idxL = {n: i for i, n in enumerate(names_L)}
        iωL, iyL = idxL["omega"], idxL[y_param]
        
        ωL = _axis_for(L, iωL, "omega")
        yL = _axis_for(L, iyL, y_param)
        y_values = yL
        
        stdL = _calc_field(L, left_result_dir)
        
        def _to_y_by_omega(std, iω, iy):
            if (iω, iy) == (0, 1): return std.T
            return std
            
        ZL = _to_y_by_omega(stdL, iωL, iyL)
        
        idxL_sort = np.argsort(ωL)
        ωL_sorted = ωL[idxL_sort]
        ZL_sorted = ZL[:, idxL_sort]
        
        x = -1.0 / ωL_sorted
        Z = ZL_sorted
        
        metadata = {
            "omega_left_sorted": ωL_sorted,
            "omega_right_sorted": np.array([]),
            "x_left": x,
            "x_right": np.array([])
        }
        return x, y_values, Z, metadata

    R = _load_bundle(right_result_dir)

    # ----- basic checks -----
    names_L = [str(n) for n in L["names"]]
    names_R = [str(n) for n in R["names"]]
    if len(names_L) != 2 or len(names_R) != 2:
        raise ValueError(f"Expected 2 parameters in each bundle, got {len(names_L)} and {len(names_R)}.")

    if "omega" not in names_L or "omega" not in names_R:
        raise KeyError("Both bundles must include a parameter named 'omega'.")

    if y_param not in names_L or y_param not in names_R:
        raise KeyError(f"Both bundles must include y_param='{y_param}'. Found {names_L} and {names_R}.")

    # Identify which index is ω and which is y_param in each bundle
    idxL = {n: i for i, n in enumerate(names_L)}
    idxR = {n: i for i, n in enumerate(names_R)}
    iωL, iyL = idxL["omega"], idxL[y_param]
    iωR, iyR = idxR["omega"], idxR[y_param]
    if iωL == iyL or iωR == iyR:
        raise RuntimeError("omega and y_param must be different axes in each bundle.")

    # k-grid compatibility (shapes and steps)
    kx_L, ky_L = np.asarray(L["kx"]), np.asarray(L["ky"])
    kx_R, ky_R = np.asarray(R["kx"]), np.asarray(R["ky"])
    if kx_L.shape != kx_R.shape or ky_L.shape != ky_R.shape:
        raise ValueError("kx/ky grid shapes differ between bundles.")
    if ("dkx" in L.files and "dkx" in R.files and "dky" in L.files and "dky" in R.files):
        if not (np.isclose(float(L["dkx"]), float(R["dkx"])) and
                np.isclose(float(L["dky"]), float(R["dky"]))):
            raise ValueError("dkx/dky differ between bundles.")

    # ----- get axes -----
    ωL = _axis_for(L, iωL, "omega")
    yL = _axis_for(L, iyL, y_param)
    ωR = _axis_for(R, iωR, "omega")
    yR = _axis_for(R, iyR, y_param)

    # y must match (within tol)
    if len(yL) != len(yR) or not np.allclose(yL, yR, atol=tol, rtol=0):
        # Allow slight mismatches if shapes match?
        if len(yL) == len(yR) and np.allclose(yL, yR, atol=1e-5):
             yR = yL 
        else:
             raise ValueError("The non-ω parameter axis (y) differs between bundles; cannot join.")
    y_values = yL  # shared



    stdL = _calc_field(L, left_result_dir)
    stdR = _calc_field(R, right_result_dir)

    # Reorder to Z(y, ω) so columns correspond to ω, rows correspond to y
    def _to_y_by_omega(std, iω, iy):
        if (iω, iy) == (0, 1):
            return std.T  # (Ny, Nω)
        elif (iω, iy) == (1, 0):
            return std    # already (Ny, Nω)
        else:
            raise RuntimeError("Unexpected parameter indexing logic.")
    ZL = _to_y_by_omega(stdL, iωL, iyL)   # (Ny, Nω_L)
    ZR = _to_y_by_omega(stdR, iωR, iyR)   # (Ny, Nω_R)

    # ----- sort ω and reorder columns accordingly -----
    idxL_sort = np.argsort(ωL)            # low -> high
    ωL_sorted = ωL[idxL_sort]
    ZL_sorted = ZL[:, idxL_sort]

    idxR_sort = np.argsort(ωR)            # low -> high
    ωR_sorted = ωR[idxR_sort]
    ZR_sorted = ZR[:, idxR_sort]

    # Right side: reverse (high -> low) so it approaches the seam from the right
    ωR_rev = ωR_sorted[::-1]
    ZR_rev = ZR_sorted[:, ::-1]

    # Drop duplicate junction column if both share the same highest ω
    if drop_overlap and ωL_sorted.size and ωR_sorted.size and np.isclose(ωL_sorted[-1], ωR_sorted[-1], atol=tol, rtol=0):
        ωR_rev = ωR_rev[1:]
        ZR_rev = ZR_rev[:, 1:]

    # ----- build signed 1/ω x-axis and join along columns -----
    x_left  = -1.0 / ωL_sorted      # negative side (left)
    x_right = +1.0 / ωR_rev         # positive side (right, reversed)

    x = np.concatenate([x_left, x_right])                # (Nω_L + Nω_R')
    Z = np.concatenate([ZL_sorted, ZR_rev], axis=1)      # (Ny, Nω_L + Nω_R')
    
    metadata = {
        "omega_left_sorted": ωL_sorted,
        "omega_right_sorted": ωR_sorted,
        "x_left": x_left,
        "x_right": x_right
    }

    return x, y_values, Z, metadata


def plot_qgt_std_param2d_signed_invomega_joined(
    left_result_dir,
    right_result_dir,
    *,
    y_param,                 # e.g. "V" (shared across both runs)
    quantity="trace",        # "trace" | "berry" | "trace_minus_berry"
    convert_berry_from_imQ=True,
    drop_overlap=True,       # drop the duplicated highest-ω column if both include it
    tol=1e-9,
    cmap="inferno",
    symmetric_cbar=False,     # None -> True for non-trace (berry / trace_minus_berry), False for trace
    save_path=None,
    show=True
):
    """
    Join two 2D parameter sweeps along ω by plotting signed 1/ω on x:
      left bundle -> x = - 1/ω   (ω sorted low->high)
      right bundle -> x = + 1/ω  (ω sorted low->high, then reversed to high->low)

    y-axis is the shared second parameter (e.g. V).  Color shows the *std over BZ*
    of the selected quantity at each (y, ω) point.

    quantity options:
      - "trace"            : std over BZ of Tr(g)
      - "berry"            : std over BZ of Ω  (Berry curvature)
      - "trace_minus_berry": std over BZ of [Tr(g) − Ω]

    Ω is read from 'berry_grid' if present; otherwise derived as -2 * Im(Q_xy).
    """

    x, y_values, Z, meta = prepare_joined_signed_invomega_data(
        left_result_dir, right_result_dir,
        y_param=y_param, quantity=quantity,
        convert_berry_from_imQ=convert_berry_from_imQ,
        drop_overlap=drop_overlap, tol=tol
    )

    # ----- color limits -----
    if symmetric_cbar is None:
        symmetric_cbar = (quantity.lower() != "trace")
    if symmetric_cbar:
        vmax_abs = float(np.nanmax(np.abs(Z)))
        vmin, vmax = -vmax_abs, vmax_abs
    else:
        vmin = float(np.nanmin(Z))
        vmax = float(np.nanmax(Z))

    # ----- plot -----
    X, Y = np.meshgrid(x, y_values, indexing="xy")
    fig, ax = plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.axvline(0.0, color='k', lw=1, ls='--', alpha=0.5)
    ax.set_xlabel("signed 1 / ω   (left < 0, right > 0)")
    ax.set_ylabel(y_param)

    title_map = {
        "trace": "Std over BZ of Tr(g)",
        "berry": "Std over BZ of Ω",
        "trace_minus_berry": "Std over BZ of [Tr(g) − Ω]",
        "chern": "Chern Number (calculated over BZ)"
    }
    ttl = title_map.get(quantity.lower(), "Std over BZ")
    ax.set_title(f"{ttl} vs signed 1/ω (joined left | right)")

    ax.set_xscale("linear")  # we already did 1/ω
    ax.set_yscale("linear")

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label({
        "trace": "Std[Tr(g)]",
        "berry": "Std[Ω]",
        "trace_minus_berry": "Std[Tr(g) − Ω]",
        "chern": "Chern Number"
    }.get(quantity.lower(), "Std[Field]"))

    # Debug prints
    print(f"[debug] y axis '{y_param}': {y_values[0]:.6g} -> {y_values[-1]:.6g} (Ny={len(y_values)})")
    if meta['omega_left_sorted'].size:
        print(f"[debug] left ω range:  {meta['omega_left_sorted'][0]:.6g} -> {meta['omega_left_sorted'][-1]:.6g}")
    if meta['omega_right_sorted'].size:
        print(f"[debug] right ω range: {meta['omega_right_sorted'][0]:.6g} -> {meta['omega_right_sorted'][-1]:.6g}")
    print(f"[debug] Z shape for pcolormesh: {Z.shape} (Ny x Nx_joined = {len(y_values)} x {len(x)})")

    # Define cursor format to show Z value
    def format_coord(x_pt, y_pt):
        # find nearest index in x-array (columns)
        col = np.abs(x - x_pt).argmin()
        # find nearest index in y_values (rows)
        row = np.abs(y_values - y_pt).argmin()
        
        if 0 <= col < Z.shape[1] and 0 <= row < Z.shape[0]:
            z_val = Z[row, col]
            return f"x={x_pt:.4g}, y={y_pt:.4g}, z={z_val:.4g}"
        else:
            return f"x={x_pt:.4g}, y={y_pt:.4g}"
            
    ax.format_coord = format_coord


    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_slices_qgt_std_param2d_signed_invomega_joined(
    left_result_dir,
    right_result_dir,
    *,
    # Slice targets
    slice_y_val,            # The 'V' value to slice at
    slice_omega_val,        # The 'omega' value to slice at (will be converted to x= +/- 1/omega and we pick closest)
    
    y_param,                # e.g. "V"
    quantity="trace",
    convert_berry_from_imQ=True,
    drop_overlap=True,
    tol=1e-9,
    save_path=None,
    export_slice_path=None,  # New option to save slice data
    show=True
):
    """
    Plot 1D slices from the joined 2D dataset:
    1. Horizontal slice: Fix 'y_param' approx slice_y_val, plot vs signed 1/omega.
    2. Vertical slice: Fix 'omega' approx slice_omega_val, plot vs y_param.
       (Note: we check both left (negative x) and right (positive x) for the closest match to +/- 1/slice_omega_val).
    """

    # 1. Prepare data
    x, y_values, Z, meta = prepare_joined_signed_invomega_data(
        left_result_dir, right_result_dir,
        y_param=y_param, quantity=quantity,
        convert_berry_from_imQ=convert_berry_from_imQ,
        drop_overlap=drop_overlap, tol=tol
    )

    # 2. Find indices
    # -- y index
    idx_y = int(np.argmin(np.abs(y_values - slice_y_val)))
    actual_y = y_values[idx_y]

    # -- x (omega) index
    # We look for the x that corresponds to +/- 1/slice_omega_val and pick the closest one in the entire array x.
    target_x_candidates = [1.0/slice_omega_val, -1.0/slice_omega_val]
    best_idx_x = -1
    best_dist = np.inf
    
    for cand in target_x_candidates:
        dist = np.abs(x - cand)
        min_d = np.min(dist)
        if min_d < best_dist:
            best_dist = min_d
            best_idx_x = int(np.argmin(dist))
            
    actual_x = x[best_idx_x]
    # Recover approximate omega from x
    actual_omega_approx = 1.0 / np.abs(actual_x) if abs(actual_x) > 1e-12 else np.inf
    sign_str = " (right)" if actual_x > 0 else " (left)"

    # 3. Extract slices
    # Slice 1: Horizontal -> Z[idx_y, :] vs x
    slice_horz = Z[idx_y, :]
    
    # Slice 2: Vertical -> Z[:, best_idx_x] vs y_values
    slice_vert = Z[:, best_idx_x]

    # --- Export if requested ---
    if export_slice_path:
        os.makedirs(os.path.dirname(export_slice_path) or ".", exist_ok=True)
        # We save two arrays for horizontal (x, y) and two for vertical (x, y)
        # Using np.savez
        np.savez(
            export_slice_path,
            # Horizontal slice (vs signed 1/omega)
            horz_x=x,
            horz_y=slice_horz,
            horz_param_val=actual_y,
            horz_param_name=y_param,
            # Vertical slice (vs y_param)
            vert_x=y_values,
            vert_y=slice_vert,
            vert_param_val=actual_x,
            vert_param_name="signed_inv_omega",
            vert_omega_approx=actual_omega_approx,
            quantity=quantity
        )
        print(f"Saved slice data to {export_slice_path}")

    # 4. Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # -- Ax1: Horizontal Slice (vs 1/omega)
    ax1.plot(x, slice_horz, 'b-o', markersize=3, label=f"Slice at {y_param} ≈ {actual_y:.4g}")
    ax1.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(actual_x, color='r', linestyle=':', alpha=0.5, label=f"Cut x ≈ {actual_x:.4g}")
    ax1.set_xlabel("signed 1 / ω")
    ax1.set_ylabel(f"Std[{quantity}]")
    ax1.set_title(f"Horizontal Slice at {y_param} ≈ {actual_y:.4g}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -- Ax2: Vertical Slice (vs V)
    ax2.plot(y_values, slice_vert, 'g-o', markersize=3, label=f"Slice at ω ≈ {actual_omega_approx:.4g}{sign_str}")
    ax2.axvline(actual_y, color='r', linestyle=':', alpha=0.5, label=f"Cut {y_param} ≈ {actual_y:.4g}")
    ax2.set_xlabel(y_param)
    ax2.set_ylabel(f"Std[{quantity}]")
    ax2.set_title(f"Vertical Slice at x ≈ {actual_x:.4g}\n(ω ≈ {actual_omega_approx:.4g})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Slices of {quantity} Std", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved slices to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

#! Full Rhombohedral Graphene Hamiltonian
full_Chiral_Hamiltonian_left_drive_dir = "results/QGT_ND/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_left-t1_355.16-vF_542.10-RANGES_V_-10.000_50.000-omega_50.000_5000.000_-SPACING_V_48_linear-omega_32_linear_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1"
full_Chiral_Hamiltonian_right_drive_dir = "results/QGT_ND/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10-RANGES_V_-10.000_50.000-omega_50.000_5000.000_-SPACING_V_48_linear-omega_32_linear_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1"

plot_qgt_std_param2d_signed_invomega_joined(
    left_result_dir  ="results/QGT_ND/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_left-t1_355.16-vF_542.10-RANGES_V_-10.000_50.000-omega_50.000_5000.000_-SPACING_V_48_linear-omega_32_linear_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1",
    right_result_dir ="results/QGT_ND/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10-RANGES_V_-10.000_50.000-omega_50.000_5000.000_-SPACING_V_48_linear-omega_32_linear_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1",
    y_param="V",            # the shared second parameter
    quantity="trace",      # "trace" | "berry" | "trace_minus_berry"
    drop_overlap=True,      # typical when both include the same max ω
)

plot_qgt_std_param2d_signed_invomega_joined(
    left_result_dir  = full_Chiral_Hamiltonian_left_drive_dir,
    right_result_dir = full_Chiral_Hamiltonian_right_drive_dir,
    y_param="V",            # the shared second parameter
    quantity="berry",      # "trace" | "berry" | "trace_minus_berry"
    drop_overlap=True,      # typical when both include the same max ω
    show=True
)


plot_qgt_std_param2d_signed_invomega_joined(
    left_result_dir  = full_Chiral_Hamiltonian_left_drive_dir,
    right_result_dir = full_Chiral_Hamiltonian_right_drive_dir,
    y_param="V",            # the shared second parameter
    quantity="trace_minus_berry",      # "trace" | "berry" | "trace_minus_berry"
    drop_overlap=True,      # typical when both include the same max ω
    show=True
)

plot_qgt_std_param2d_signed_invomega_joined(
    left_result_dir  = full_Chiral_Hamiltonian_left_drive_dir,
    right_result_dir = full_Chiral_Hamiltonian_right_drive_dir,
    y_param="V",            # the shared second parameter
    quantity="chern",      # "trace" | "berry" | "trace_minus_berry" | "chern"
    drop_overlap=True,      # typical when both include the same max ω
    show=True
)

#! Chiral Projected Hamiltonian 
right_drive_strong = "results/QGT_ND/RhombohedralGrapheneHamiltonian/A0_0.10-V_30-analytic_magnus_False-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10-RANGES_V_-10.000_50.000-omega_20.000_5000.000_-SPACING_V_32_linear-omega_48_linear_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1"
left_drive_strong = "results/QGT_ND/RhombohedralGrapheneHamiltonian/A0_0.10-V_30-analytic_magnus_False-magnus_order_1-n_5-omega_6.28-polarization_left-t1_355.16-vF_542.10-RANGES_V_-10.000_50.000-omega_20.000_5000.000_-SPACING_V_32_linear-omega_48_linear_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1"

# plot_qgt_std_param2d_signed_invomega_joined(
#     left_result_dir  = left_drive_strong,
#     right_result_dir = right_drive_strong,
#     y_param="V",            # the shared second parameter
#     quantity="trace",      # "trace" | "berry" | "trace_minus_berry"
#     drop_overlap=True,      # typical when both include the same max ω
#     show=False
# )

# plot_qgt_std_param2d_signed_invomega_joined(
#     left_result_dir   = left_drive_strong,
#     right_result_dir = right_drive_strong,
#     y_param="V",            # the shared second parameter
#     quantity="berry",      # "trace" | "berry" | "trace_minus_berry"
#     drop_overlap=True,      # typical when both include the same max ω
#     show=False
# )


# plot_qgt_std_param2d_signed_invomega_joined(
#     left_result_dir   = left_drive_strong,
#     right_result_dir = right_drive_strong,
#     y_param="V",            # the shared second parameter
#     quantity="trace_minus_berry",      # "trace" | "berry" | "trace_minus_berry"
#     drop_overlap=True,      # typical when both include the same max ω
#     show=False
# )

# Example usage for slices (commented out or active as needed):
# plot_slices_qgt_std_param2d_signed_invomega_joined(
#     left_result_dir=full_Chiral_Hamiltonian_left_drive_dir,
#     right_result_dir=full_Chiral_Hamiltonian_right_drive_dir,
#     slice_y_val=20.0,
#     slice_omega_val=100.0,
#     y_param="V",
#     quantity="berry",
#     export_slice_path="results/1d_idealness_resutls/slice_data_export.npz",
#     show=False
# )

large_k_left = "results/QGT_ND/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_left-t1_355.16-vF_542.10-RANGES_V_-10.000_50.000-omega_20.000_5000.000_-SPACING_V_3_linear-omega_3_linear_-kx-2.50_2.50__ky-2.50_2.50__mesh300_data_set1"

# plot_qgt_std_param2d_signed_invomega_joined(
#     left_result_dir  = large_k_left,
#     right_result_dir = None,
#     y_param="V",            # the shared second parameter
#     quantity="chern",      # "trace" | "berry" | "trace_minus_berry" | "chern"
#     drop_overlap=True,      # typical when both include the same max ω
#     show=True
# )
