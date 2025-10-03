import numpy as np
import matplotlib.pyplot as plt
import os
from Library.plotting_utils import load_qgt, filter_entries_by_omega
# ---------- small internal join helper ----------

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


def plot_trace_std_param2d(result_dir,
                           *,
                           x_param,              # e.g. "omega" or "V"
                           y_param,              # the other one
                           xscale="linear",      # "linear" or "log"
                           yscale="linear",      # "linear" or "log"
                           cmap="inferno",
                           symmetric_cbar=False,
                           save_path=None,
                           show=True):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

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

    # --- compute std over BZ ---
    trace_grid = np.asarray(bundle["trace_grid"])  # (N0, N1, Ny, Nx)
    std_bz = np.nanstd(trace_grid, axis=(-2, -1))  # (N0, N1) in bundle order (names[0], names[1])

    # --- reorder for plotting ---
    # pcolormesh with indexing="xy" expects Z shape (len(y), len(x)).
    # If x=names[0], y=names[1]  => std_bz is (N0, N1) -> must transpose.
    # If x=names[1], y=names[0]  => std_bz.T is (N1, N0) -> already matches, so use std_bz (no T).
    if (ix, iy) == (0, 1):
        Z = std_bz.T  # (N1, N0) = (len(y), len(x))
    elif (ix, iy) == (1, 0):
        Z = std_bz    # already (len(y), len(x))
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

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title("Std of QGT Trace over Brillouin Zone")

    if xscale not in ("linear", "log") or yscale not in ("linear", "log"):
        raise ValueError("xscale/yscale must be 'linear' or 'log'.")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Std[Tr(g)]")

    # quick sanity print so you can confirm monotonicity directions
    print(f"[debug] x axis '{x_param}': {x_values[0]:.6g} -> {x_values[-1]:.6g} ({'increasing' if x_values[-1]>x_values[0] else 'decreasing'})")
    print(f"[debug] y axis '{y_param}': {y_values[0]:.6g} -> {y_values[-1]:.6g} ({'increasing' if y_values[-1]>y_values[0] else 'decreasing'})")
    print(f"[debug] Z shape for pcolormesh: {Z.shape} (should be len(y) x len(x) = {len(y_values)} x {len(x_values)})")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


plot_trace_std_param2d(
    "results/QGT_ND/ChiralHamiltonian/RANGES_V_10.000_50.000-omega_50.000_5000.000___SPACING_V_16_linear-omega_16_log___kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1",
    x_param="omega", xscale="log",
    y_param="V",     yscale="linear"
)
