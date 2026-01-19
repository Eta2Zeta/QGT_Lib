import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import sys
import Library.Hamiltonian.Hamiltonian_v2
# Patch for backward compatibility (unpickling old data)
sys.modules["Library.Hamiltonian_v2"] = Library.Hamiltonian.Hamiltonian_v2
from Library.Hamiltonian.Chiral_Hamiltonian_Projected import ChiralHamiltonianProjected
Library.Hamiltonian.Hamiltonian_v2.RhombohedralGrapheneHamiltonian = ChiralHamiltonianProjected

def _bz_hexagon_from_bvecs(b1, b2):
    """
    Return (xv, yv) arrays of the hexagon vertices (closed) for the 1st BZ
    built from reciprocal vectors b1, b2 (shape (2,) arrays).
    """
    import numpy as np

    # 6 nearest reciprocal vectors
    Gs = np.array([
        +b1,
        +b2,
        +(b1 + b2),
        -b1,
        -b2,
        -(b1 + b2)
    ], dtype=float)

    # order by polar angle so adjacent normals give consecutive edges
    ang = np.arctan2(Gs[:,1], Gs[:,0])
    order = np.argsort(ang)
    Gs = Gs[order]

    # for each pair, intersect n_i·k = |n_i|^2/2 with n_{i+1}·k = |n_{i+1}|^2/2
    verts = []
    for i in range(6):
        g1 = Gs[i]
        g2 = Gs[(i+1) % 6]
        A = np.stack([g1, g2], axis=0)                    # 2x2
        c = 0.5 * np.array([np.dot(g1, g1), np.dot(g2, g2)])
        # solve A k = c
        kx_ky = np.linalg.solve(A, c)
        verts.append(kx_ky)

    verts = np.array(verts)
    # close the polygon for plotting
    xv = np.append(verts[:,0], verts[0,0])
    yv = np.append(verts[:,1], verts[0,1])
    return xv, yv

def _get_bvecs_from_meta(meta):
    """
    Try to extract b1, b2 from the saved Hamiltonian template in meta.
    Fallback to graphene defaults if needed.
    """
    import numpy as np
    b1 = b2 = None
    if meta is not None:
        Htemp = meta.get("Hamiltonian_Template", None)
        if Htemp is not None and hasattr(Htemp, "b1") and hasattr(Htemp, "b2"):
            b1 = np.asarray(Htemp.b1, dtype=float)
            b2 = np.asarray(Htemp.b2, dtype=float)

        # fallback: use 'a' if present
        if (b1 is None or b2 is None):
            a = getattr(Htemp, "a", 1.0) if Htemp is not None else 1.0
            b1 = (2*np.pi/(3*a)) * np.array([1.0,  np.sqrt(3.0)], dtype=float)
            b2 = (2*np.pi/(3*a)) * np.array([1.0, -np.sqrt(3.0)], dtype=float)

    else:
        # last-resort fallback
        a = 1.0
        b1 = (2*np.pi/(3*a)) * np.array([1.0,  np.sqrt(3.0)], dtype=float)
        b2 = (2*np.pi/(3*a)) * np.array([1.0, -np.sqrt(3.0)], dtype=float)

    return b1, b2

def _load_nd_bundle(root_dir):
    """
    root_dir: directory that contains 'qgt_nd_bundle.npz' (and meta.pkl)
    returns:
      data   : np.load(...) object (use data.files to see available arrays)
      names  : list[str] of parameter names (order matches axes/shape)
      axes   : list[np.ndarray] parameter value arrays (one per name)
      shape  : tuple[int] parameter grid shape
      kx, ky : 2D grids (Ny, Nx)
      meta   : dict loaded from meta.pkl (or None if missing)

    Notes:
      - This matches the layout produced by compute_qgt_nd_parallel(...).
      - Axis arrays are stored with keys: f"axis_{i}_{names[i]}".
    """
    bundle_path = os.path.join(root_dir, "qgt_nd_bundle.npz")
    meta_path   = os.path.join(root_dir, "meta.pkl")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Cannot find bundle at: {bundle_path}")

    # Load npz
    data  = np.load(bundle_path, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    shape = tuple(int(x) for x in data["shape"])
    axes  = [np.asarray(data[f"axis_{i}_{names[i]}"]) for i in range(len(names))]
    kx    = np.asarray(data["kx"])
    ky    = np.asarray(data["ky"])

    # Load meta (if present)
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

    return data, names, axes, shape, kx, ky, meta


def _pick_field_grid(data, quantity="trace", *, convert_berry_from_imQ=True):
    """
    Returns a (param_shape + Ny + Nx) array for the requested quantity.
    quantity: "trace" | "berry" | "imqxy" | "trace_minus_berry"
    """
    q = quantity.lower()

    if q == "trace":
        return np.asarray(data["trace_grid"])

    if q in ("berry", "berry_curvature", "omega"):
        if "berry_grid" in data.files:
            return np.asarray(data["berry_grid"])
        gxyi = np.asarray(data["g_xy_imag_grid"])
        return (-2.0 * gxyi) if convert_berry_from_imQ else gxyi

    if q in ("imqxy", "im(q_xy)", "im_qxy"):
        return np.asarray(data["g_xy_imag_grid"])

    if q in ("trace_minus_berry", "trace_minus_omega"):
        trace = np.asarray(data["trace_grid"])
        if "berry_grid" in data.files:
            berry = np.asarray(data["berry_grid"])
        else:
            gxyi = np.asarray(data["g_xy_imag_grid"])
            berry = (-2.0 * gxyi) if convert_berry_from_imQ else gxyi
        return trace - berry

    raise ValueError(f"Unknown quantity '{quantity}'.")



def dynamic_nd_field_with_bands(
    root_dir,
    *,
    quantity="trace",            # "trace" | "berry" | "imqxy"
    bands_to_plot=None,          # None -> all; int or iterable[int] -> selection
    ky_slice="mid",              # "mid" or int index to slice eigenvalues vs kx
    convert_berry_from_imQ=True,
    cmap="inferno",
    symmetric_cbar=None,         # None -> True for non-trace; False for trace
    title=None,
    show_integral=True
):
    """
    Shows a slim 1D eigenvalue cut (top) and a 2D field heatmap (bottom).
    Sliders for parameters remain along the bottom.
    """
    data, names, axes, shape, kx, ky, meta = _load_nd_bundle(root_dir)
    field_grid = _pick_field_grid(data, quantity, convert_berry_from_imQ=convert_berry_from_imQ)
    
    # grab dkx,dky from bundle if present (saved by your compute_qgt_nd_parallel)
    dkx = float(data["dkx"]) if "dkx" in data.files else (kx[0,1]-kx[0,0])
    dky = float(data["dky"]) if "dky" in data.files else (ky[1,0]-ky[0,0])
    area_element = float(dkx * dky)

    # --- bands present? ---
    if "eigenvalues_grid" not in data.files:
        raise KeyError("Bundle missing 'eigenvalues_grid'. Please save eigenvalues into the npz.")

    evals_grid = np.asarray(data["eigenvalues_grid"])   # (*param_shape, Ny, Nx, Nb)
    Ny, Nx = kx.shape
    Nb = evals_grid.shape[-1]

    # --- band selection ---
    if bands_to_plot is None:
        bands = list(range(Nb))
    elif isinstance(bands_to_plot, int):
        if not (0 <= bands_to_plot < Nb):
            raise IndexError(f"Band {bands_to_plot} out of range [0,{Nb-1}]")
        bands = [bands_to_plot]
    else:
        bands = list(bands_to_plot)
        bad = [b for b in bands if not (0 <= b < Nb)]
        if bad:
            raise IndexError(f"bands_to_plot has out-of-range indices {bad}; valid is [0,{Nb-1}]")

    # --- ky slice index for the 1D cut ---
    if ky_slice == "mid":
        ky_idx = Ny // 2
    elif isinstance(ky_slice, int):
        if not (0 <= ky_slice < Ny):
            raise IndexError(f"ky_slice {ky_slice} out of range [0,{Ny-1}]")
        ky_idx = ky_slice
    else:
        raise ValueError("ky_slice must be 'mid' or an integer ky index.")

    # --- color scale for the 2D field (global over all parameter points) ---
    if symmetric_cbar is None:
        symmetric_cbar = (quantity.lower() != "trace")
    if symmetric_cbar:
        vmax_abs = max(abs(np.nanmin(field_grid)), abs(np.nanmax(field_grid)))
        vmin, vmax = -vmax_abs, vmax_abs
    else:
        vmin, vmax = np.nanmin(field_grid), np.nanmax(field_grid)

    # --- initial parameter indices: middle of each axis ---
    init_idx = [ax.size // 2 for ax in axes]
    idx_tuple = tuple(init_idx)

    # --- initial 2D slice ---
    Z0 = field_grid[(*idx_tuple, slice(None), slice(None))]  # (Ny,Nx)

    # --- layout: top band plot (short), bottom heatmap; sliders at bottom ---
    n_params = len(names)
    slider_h = 0.03
    slider_gap = 0.0
    bottom_margin = 0.09 + n_params * (slider_h + slider_gap)
    bottom_margin = min(0.45, bottom_margin)

    fig = plt.figure(figsize=(12, 13))
    # axes positions: leave room for sliders; make a slim top axis
    top_rect    = [0.12, bottom_margin + 0.66, 0.73, 0.14]  # [left, bottom, width, height]
    bottom_rect = [0.12, bottom_margin, 0.92, 0.6]

    ax_top = fig.add_axes(top_rect)
    ax2d   = fig.add_axes(bottom_rect)

    # --- overlay Brillouin-zone hexagon (black outline) ---
    b1, b2 = _get_bvecs_from_meta(meta)
    bx, by = _bz_hexagon_from_bvecs(b1, b2)
    ax2d.plot(bx, by, 'w-', lw=1.8, alpha=0.9)   # black hexagon outline

    # --- draw initial bands (top) ---
    # grab the eigenvalues line at ky_idx for current parameter index
    ev_line = evals_grid[(*idx_tuple, ky_idx, slice(None), slice(None))]  # (Nx, Nb)
    # sort by kx so the line is monotonic in x
    kx_line = kx[ky_idx, :] if kx.ndim == 2 else kx
    order   = np.argsort(kx_line)
    kxs     = kx_line[order]
    ev_line_sorted = ev_line[order, :]  # (Nx, Nb)

    band_lines = []
    for b in bands:
        line, = ax_top.plot(kxs, ev_line_sorted[:, b], lw=1.2, label=f"band {b}")
        band_lines.append(line)

    ax_top.set_ylabel("Eigenvalue")
    ax_top.set_xticklabels([])     # no x labels on top strip
    ax_top.legend(loc="upper right", ncol=min(3, len(bands)), fontsize=8, frameon=True)

    # --- draw initial 2D field (bottom) ---
    im = ax2d.imshow(
        Z0, origin="lower",
        extent=[kx.min(), kx.max(), ky.min(), ky.max()],
        cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"
    )

    def _label_from_quantity(q):
        q = q.lower()
        return {
            "trace": "QGT Trace",
            "berry": "Berry Curvature Ω",
            "imqxy": "Im(Q_xy)",
            "trace_minus_berry": "Tr(g) − Ω",
        }.get(q, "Field")

    if title is None:
        title = _label_from_quantity(quantity)

    def _title_for(idx_t, Z):
        parts = [f"{names[i]}={axes[i][idx_t[i]]:.6g}" for i in range(len(names))]
        extra = []
        # std readout (handy for all)
        extra.append(f"std={float(np.nanstd(Z)):.3e}")
        # optional BZ integral for trace−berry
        if show_integral and quantity.lower() == "trace_minus_berry":
            integral = float(np.nansum(Z) * area_element)
            extra.append(f"∫(Tr−Ω) d^2k = {integral:.6g}")
        return f"{title} — " + ", ".join(parts) + "  |  " + "  |  ".join(extra)

    ax2d.set_title(_title_for(idx_tuple, Z0))
    ax2d.set_xlabel("$k_x$")
    ax2d.set_ylabel("$k_y$")
    cbar = plt.colorbar(im, ax=ax2d)
    cbar.set_label(title)

    # --- sliders ---
    sliders = []
    left, width = 0.12, 0.738
    y0 = 0.06
    for i, name in enumerate(names):
        y = y0 + i * (slider_h + slider_gap)
        ax_sl = fig.add_axes([left, y, width, slider_h], facecolor='lightgoldenrodyellow')
        s = Slider(ax_sl, f"{name}", 0, axes[i].size - 1, valinit=init_idx[i], valstep=1)
        sliders.append(s)

    # --- update handler ---
    def _update(_):
        idx = tuple(int(s.val) for s in sliders)

        # Update 2D field
        Z = field_grid[(*idx, slice(None), slice(None))]
        im.set_data(Z)
        ax2d.set_title(_title_for(idx, Z))

        # Update band cut
        ev_line = evals_grid[(*idx, ky_idx, slice(None), slice(None))]  # (Nx, Nb)
        ev_sorted = ev_line[order, :]
        for j, b in enumerate(bands):
            band_lines[j].set_data(kxs, ev_sorted[:, b])

        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(_update)

    plt.show()


# root_right = "results/QGT_ND/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10-RANGES_V_10.000_50.000-omega_50.000_5000.000_-SPACING_V_16_linear-omega_16_log_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1"
# root_left = "results/QGT_ND/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_left-t1_355.16-vF_542.10-RANGES_V_10.000_50.000-omega_50.000_5000.000_-SPACING_V_16_linear-omega_16_log_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1"

# New
# root_right = "results/QGT_ND/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10-RANGES_V_10.000_50.000-omega_50.000_5000.000_-SPACING_V_32_linear-omega_32_linear_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1"
# root_left = "results/QGT_ND/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_left-t1_355.16-vF_542.10-RANGES_V_10.000_50.000-omega_50.000_5000.000_-SPACING_V_32_linear-omega_32_linear_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1"

# V all the way to -10
root_right = "results/QGT_ND/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10-RANGES_V_-10.000_50.000-omega_50.000_5000.000_-SPACING_V_48_linear-omega_32_linear_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1"
root_left = "results/QGT_ND/ChiralHamiltonian/A0_0.10-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-omega_6.28-polarization_left-t1_355.16-vF_542.10-RANGES_V_-10.000_50.000-omega_50.000_5000.000_-SPACING_V_48_linear-omega_32_linear_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1"


# The chiral basis projected Hamiltonian
# root_right = "results/QGT_ND/RhombohedralGrapheneHamiltonian/A0_0.10-V_30-analytic_magnus_False-magnus_order_1-n_5-omega_6.28-polarization_right-t1_355.16-vF_542.10-RANGES_V_-10.000_50.000-omega_50.000_5000.000_-SPACING_V_1_linear-omega_1_linear_-kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set2"

# View the trace (default)
dynamic_nd_field_with_bands(root_right, quantity="trace", bands_to_plot=[0,1,2,3,4,5,6,7,8,9])
dynamic_nd_field_with_bands(root_left, quantity="trace", bands_to_plot=[0,1,2,3,4,5,6,7,8,9])


