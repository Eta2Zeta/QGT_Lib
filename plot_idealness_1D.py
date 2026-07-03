import numpy as np
import matplotlib.pyplot as plt
from Library.plotting_utils import load_qgt, filter_entries_by_omega
from Plot_QGT_2D_Dynamic_ND_sweep import _load_nd_bundle, _pick_field_grid, _label_from_quantity


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


def plot_qgt_metric_vs_signed_invomega_joined(
    left_folder_name,
    right_folder_name,
    *,
    quantity="trace",        # "trace" | "berry" | "trace_minus_berry"
    omega_min_left=None,
    omega_max_left=None,
    omega_min_right=None,
    omega_max_right=None,
    use_precomputed=False,   # True if entries include 'berry'
    convert_from_imQ=True,   # if not precomputed: Omega = -2 * Im(Q_xy)
    drop_overlap=True,
    tol=1e-9,
    left_sign=-1.0,
    right_sign=+1.0
):
    """
    Plots a QGT metric (trace, berry, or integrated trace_minus_berry) vs signed 1/omega,
    joined left|right.
    """
    entries_L, meta_L = load_qgt(left_folder_name)
    entries_R, meta_R = load_qgt(right_folder_name)

    filt_L = filter_entries_by_omega(entries_L, omega_min_left,  omega_max_left)
    filt_R = filter_entries_by_omega(entries_R, omega_min_right, omega_max_right)
    if len(filt_L) == 0 or len(filt_R) == 0:
        raise ValueError("No omega slices in range for one or both datasets.")

    def _get_berry(e):
        if use_precomputed and ("berry" in e):
            return np.asarray(e["berry"])
        gim = np.asarray(e["g_xy_imag"])
        return (-2.0 * gim) if convert_from_imQ else gim

    area = 1.0
    if quantity == "trace_minus_berry":
        dkx_L, dky_L = float(meta_L["dkx"]), float(meta_L["dky"])
        dkx_R, dky_R = float(meta_R["dkx"]), float(meta_R["dky"])
        if not (np.isclose(dkx_L, dkx_R) and np.isclose(dky_L, dky_R)):
            raise ValueError("dkx/dky differ between datasets; cannot join safely.")
        area = dkx_L * dky_L

    omegas_L, vals_L = [], []
    for e in filt_L:
        w = float(e["omega"])
        if quantity == "trace":
            val = np.nanstd(e["trace"])
        elif quantity == "berry":
            val = np.nanstd(_get_berry(e))
        elif quantity == "trace_minus_berry":
            val = np.nansum(np.asarray(e["trace"]) - _get_berry(e)) * area
        else:
            raise ValueError(f"Unknown quantity: {quantity}")
        vals_L.append(val)
        omegas_L.append(w)

    omegas_R, vals_R = [], []
    for e in filt_R:
        w = float(e["omega"])
        if quantity == "trace":
            val = np.nanstd(e["trace"])
        elif quantity == "berry":
            val = np.nanstd(_get_berry(e))
        elif quantity == "trace_minus_berry":
            val = np.nansum(np.asarray(e["trace"]) - _get_berry(e)) * area
        vals_R.append(val)
        omegas_R.append(w)

    x, y = _join_by_signed_inverse_omega(
        np.array(omegas_L, dtype=float), vals_L,
        np.array(omegas_R, dtype=float), vals_R,
        drop_overlap=drop_overlap, tol=tol, left_sign=left_sign, right_sign=right_sign
    )

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.axvline(0.0, color='k', lw=1, ls='--', alpha=0.5)
    plt.xlabel("signed 1 / omega   (left < 0, right > 0)")

    if quantity == "trace":
        plt.ylabel("Std. Dev of QGT Trace over BZ")
        plt.title("Fluctuation of QGT Trace vs signed 1/omega (joined left | right-reversed)")
    elif quantity == "berry":
        plt.ylabel("Std. Dev of Berry Curvature over BZ")
        plt.title("Fluctuation of Berry Curvature vs signed 1/omega (joined left | right-reversed)")
    elif quantity == "trace_minus_berry":
        plt.ylabel(r"$\int_{\mathrm{BZ}} [\,\mathrm{Tr}(g) - \Omega\,]\, d^2k$")
        plt.title(r"Integrated Tr(g) - Omega vs signed 1/omega (joined left | right-reversed)")

    plt.grid(True, axis='both', alpha=0.35)
    plt.tight_layout()
    plt.show()


def plot_1d_field_std(root_dir, quantity="trace"):
    """
    Plot the standard deviation of a requested field over the BZ for a 1D parameter sweep.
    """
    data, names, axes, shape, kx, ky, meta = _load_nd_bundle(root_dir)
    
    if len(names) != 1:
        raise ValueError(f"plot_1d_field_std expects exactly 1 parameter, but found {len(names)}: {names}")
        
    param_name = names[0]
    param_axis = axes[0]
    field_grid = _pick_field_grid(data, quantity)
    
    # Calculate std over the BZ (the last two dimensions Ny, Nx)
    std_values = np.nanstd(field_grid, axis=(-2, -1))
    
    title = _label_from_quantity(quantity)
    
    plt.figure(figsize=(8, 5))
    plt.plot(param_axis, std_values, marker='o', linestyle='-')
    
    if "omega" in param_name.lower():
        plt.xscale('log')
        
    plt.xlabel(param_name)
    plt.ylabel(f"Std. Dev of {title} over BZ")
    plt.title(f"Fluctuation of {title} vs {param_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_qgt_metric_vs_signed_invomega_joined(
        "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1",
        "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1",
        quantity="trace",
        omega_min_left=33,
        omega_min_right=50,
    )

    plot_qgt_metric_vs_signed_invomega_joined(
        "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1",
        "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1",
        quantity="berry",
        omega_min_left=33,
        omega_min_right=50,
    )

    plot_qgt_metric_vs_signed_invomega_joined(
        "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1",
        "ChiralHamiltonian/omega6.283185307179586_A00.1_polarizationright_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_kx-0.82_0.82_ky-0.82_0.82_mesh150_omega5.00e_00_5.00e_03_spacing_log_points64_1",
        quantity="trace_minus_berry",
        omega_min_left=33,
        omega_min_right=50,
    )
