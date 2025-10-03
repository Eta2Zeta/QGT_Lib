import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def _load_nd_bundle(root_dir):
    """
    root_dir: directory that contains 'qgt_nd_bundle.npz' (and meta.pkl)
    returns:
      data   : np.load(...) object
      names  : list[str] of parameter names, order matches axes/shape
      axes   : list[np.ndarray] parameter value arrays (one per name)
      shape  : tuple[int] parameter grid shape
      kx, ky : 2D grids
    """
    bundle_path = os.path.join(root_dir, "qgt_nd_bundle.npz")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Cannot find bundle at: {bundle_path}")

    data  = np.load(bundle_path, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    shape = tuple(int(x) for x in data["shape"])
    axes  = [np.asarray(data[f"axis_{i}_{names[i]}"]) for i in range(len(names))]
    kx    = np.asarray(data["kx"])
    ky    = np.asarray(data["ky"])
    return data, names, axes, shape, kx, ky


def _pick_field_grid(data, quantity="trace", *, convert_berry_from_imQ=True):
    """
    Returns a  (param_shape + Ny + Nx) array for the requested quantity.
    quantity: "trace" | "berry" | "imqxy"
    """
    q = quantity.lower()
    if q == "trace":
        return np.asarray(data["trace_grid"])
    if q in ("berry", "berry_curvature", "omega"):
        # prefer explicit berry if ever stored; otherwise derive from Im(Q_xy)
        if "berry_grid" in data.files:
            return np.asarray(data["berry_grid"])
        gxyi = np.asarray(data["g_xy_imag_grid"])
        return (-2.0 * gxyi) if convert_berry_from_imQ else gxyi
    if q in ("imqxy", "im(q_xy)", "im_qxy"):
        return np.asarray(data["g_xy_imag_grid"])
    raise ValueError(f"Unknown quantity '{quantity}'.")


# def dynamic_nd_field_from_bundle(
#     root_dir,
#     *,
#     quantity="trace",           # "trace" | "berry" | "imqxy"
#     convert_berry_from_imQ=True,
#     cmap="inferno",
#     symmetric_cbar=None,        # None -> True for non-trace; False for trace
#     title=None,
# ):
#     """
#     Interactive viewer for N-D bundle with one slider per parameter.
#     Shows a 2D heatmap of the chosen quantity at the selected parameter indices.
#     """
#     data, names, axes, shape, kx, ky = _load_nd_bundle(root_dir)
#     field_grid = _pick_field_grid(data, quantity, convert_berry_from_imQ=convert_berry_from_imQ)

#     # Figure out color scale (global, across the whole bundle)
#     if symmetric_cbar is None:
#         symmetric_cbar = (quantity.lower() != "trace")
#     if symmetric_cbar:
#         vmax_abs = 0.0
#         # robust global extrema without loading the whole array twice
#         vmax_abs = max(abs(np.nanmin(field_grid)), abs(np.nanmax(field_grid)))
#         vmin, vmax = -vmax_abs, vmax_abs
#     else:
#         vmin, vmax = np.nanmin(field_grid), np.nanmax(field_grid)

#     # Initial parameter indices: middle of each axis
#     init_idx = [ax.size // 2 for ax in axes]
#     idx_tuple = tuple(init_idx)

#     # Initial 2D slice
#     # Z0 = field_grid[idx_tuple, :, :]
#     Z0 = field_grid[(*idx_tuple, slice(None), slice(None))]  # -> (128, 128)


#     # Layout: leave room at bottom for N sliders
#     n_params = len(names)
#     slider_height = 0.035
#     slider_gap    = 0.010
#     bottom_margin = 0.10 + n_params * (slider_height + slider_gap)  # dynamic
#     bottom_margin = min(0.40, bottom_margin)                         # cap so it doesn't get silly

#     fig, ax = plt.subplots(figsize=(8, 6))
#     fig.subplots_adjust(bottom=bottom_margin)

#     im = ax.imshow(
#         Z0,
#         origin="lower",
#         extent=[kx.min(), kx.max(), ky.min(), ky.max()],
#         cmap=cmap,
#         vmin=vmin, vmax=vmax,
#         aspect="auto",
#     )

#     # Title/labels
#     if title is None:
#         label = {"trace":"QGT Trace", "berry":"Berry Curvature Ω", "imqxy":"Im(Q_xy)"} \
#                 .get(quantity.lower(), "Field")
#         title = label
#     def _title_for(idx_tuple):
#         parts = [f"{names[i]}={axes[i][idx_tuple[i]]:.6g}" for i in range(n_params)]
#         return f"{title} — " + ", ".join(parts)

#     ax.set_title(_title_for(idx_tuple))
#     ax.set_xlabel("$k_x$")
#     ax.set_ylabel("$k_y$")
#     cbar = plt.colorbar(im, ax=ax)
#     cbar.set_label(title)

#     # Build sliders
#     sliders = []
#     left, width = 0.12, 0.76
#     # y coordinate for the bottom slider row:
#     y0 = 0.06
#     for i, name in enumerate(names):
#         y = y0 + i * (slider_height + slider_gap)
#         ax_sl = plt.axes([left, y, width, slider_height], facecolor='lightgoldenrodyellow')
#         # slider spans discrete indices [0..Ni-1]
#         s = Slider(
#             ax_sl,
#             f"{name}",
#             0, axes[i].size - 1,
#             valinit=init_idx[i],
#             valstep=1,
#         )
#         sliders.append(s)

#     # Update handler
#     def _update(_):
#         idx = tuple(int(s.val) for s in sliders)
#         Z   = field_grid[(*idx, slice(None), slice(None))]

#         im.set_data(Z)
#         ax.set_title(_title_for(idx))
#         fig.canvas.draw_idle()

#     for s in sliders:
#         s.on_changed(_update)

#     plt.show()

def dynamic_nd_field_from_bundle(
    root_dir,
    *,
    quantity="trace",           # "trace" | "berry" | "imqxy"
    convert_berry_from_imQ=True,
    cmap="inferno",
    symmetric_cbar=None,        # None -> True for non-trace; False for trace
    title=None,
):
    """
    Interactive viewer for N-D bundle with one slider per parameter.
    Shows a 2D heatmap of the chosen quantity at the selected parameter indices,
    and displays the std[quantity] over kx,ky for that slice.
    """
    data, names, axes, shape, kx, ky = _load_nd_bundle(root_dir)
    field_grid = _pick_field_grid(data, quantity, convert_berry_from_imQ=convert_berry_from_imQ)

    # Figure out color scale (global, across the whole bundle)
    if symmetric_cbar is None:
        symmetric_cbar = (quantity.lower() != "trace")
    if symmetric_cbar:
        vmax_abs = max(abs(np.nanmin(field_grid)), abs(np.nanmax(field_grid)))
        vmin, vmax = -vmax_abs, vmax_abs
    else:
        vmin, vmax = np.nanmin(field_grid), np.nanmax(field_grid)

    # Initial parameter indices: middle of each axis
    init_idx = [ax.size // 2 for ax in axes]
    idx_tuple = tuple(init_idx)

    # Initial slice
    Z0 = field_grid[(*idx_tuple, slice(None), slice(None))]  # (Ny,Nx)

    # Layout
    n_params = len(names)
    slider_height = 0.035
    slider_gap    = 0.010
    bottom_margin = 0.10 + n_params * (slider_height + slider_gap)
    bottom_margin = min(0.40, bottom_margin)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(bottom=bottom_margin)

    im = ax.imshow(
        Z0,
        origin="lower",
        extent=[kx.min(), kx.max(), ky.min(), ky.max()],
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        aspect="auto",
    )

    # Title/labels
    if title is None:
        label = {"trace":"QGT Trace", "berry":"Berry Curvature Ω", "imqxy":"Im(Q_xy)"} \
                .get(quantity.lower(), "Field")
        title = label

    def _title_for(idx_tuple, Z):
        parts = [f"{names[i]}={axes[i][idx_tuple[i]]:.6g}" for i in range(n_params)]
        std_val = float(np.nanstd(Z))
        return f"{title} — " + ", ".join(parts) + f"  |  std={std_val:.3e}"

    ax.set_title(_title_for(idx_tuple, Z0))
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(title)

    # Build sliders
    sliders = []
    left, width = 0.12, 0.76
    y0 = 0.06
    for i, name in enumerate(names):
        y = y0 + i * (slider_height + slider_gap)
        ax_sl = plt.axes([left, y, width, slider_height], facecolor='lightgoldenrodyellow')
        s = Slider(ax_sl, f"{name}", 0, axes[i].size - 1,
                   valinit=init_idx[i], valstep=1)
        sliders.append(s)

    # Update handler
    def _update(_):
        idx = tuple(int(s.val) for s in sliders)
        Z   = field_grid[(*idx, slice(None), slice(None))]
        im.set_data(Z)
        ax.set_title(_title_for(idx, Z))
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(_update)

    plt.show()


# Suppose your bundle directory is something like:
root = "results/QGT_ND/ChiralHamiltonian/RANGES_V_10.000_50.000-omega_50.000_5000.000___SPACING_V_16_linear-omega_16_log___kx-0.90_0.90__ky-0.90_0.90__mesh100_data_set1"

# View the trace (default)
dynamic_nd_field_from_bundle(root)

# # Or view Berry curvature with symmetric colorbar:
# dynamic_nd_field_from_bundle(root, quantity="trace", symmetric_cbar=True, cmap="coolwarm")
