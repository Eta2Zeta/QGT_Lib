import os
import numpy as np
import matplotlib.pyplot as plt

def plot_chern_phase_diagram(
    sweep_root,
    x_param=None,
    y_param=None,
    slice_indices=None,   # dict like {"psi": 3, "A0": 0} for the non-plotted params
    reduce_other="mean",  # "mean" | "median" | "max" | "min" if not slicing
    cmap="coolwarm",
    save_path=None
):
    """
    Plot a 2D phase diagram of Chern number from a saved sweep folder.

    sweep_root:   path containing chern_grid.npy and param_axes.npz
    x_param/y_param: names of parameters to place on X/Y. If None, uses first two.
    slice_indices: dict of {param_name: index} to select along non-plotted params.
                   If an other-param is not in slice_indices, it will be reduced
                   across that axis using `reduce_other`.
    reduce_other: aggregation for non-plotted, non-sliced params: "mean"|"median"|"max"|"min"
    cmap:         matplotlib colormap
    save_path:    optional path to save the figure (PNG)
    """
    chern_path = os.path.join(sweep_root, "chern_grid.npy")
    axes_path  = os.path.join(sweep_root, "param_axes.npz")
    if not (os.path.exists(chern_path) and os.path.exists(axes_path)):
        raise FileNotFoundError("Missing chern_grid.npy or param_axes.npz in sweep_root.")

    chern_grid = np.load(chern_path)
    axes_npz   = np.load(axes_path, allow_pickle=True)
    names      = list(axes_npz["names"])
    axes       = [axes_npz[f"axis_{i}_{names[i]}"] for i in range(len(names))]

    # Choose x/y params
    if x_param is None or y_param is None:
        if len(names) < 2:
            raise ValueError("Need at least 2 parameters to plot a 2D phase diagram.")
        x_param = names[0] if x_param is None else x_param
        y_param = names[1] if y_param is None else y_param

    if x_param not in names or y_param not in names:
        raise ValueError("x_param or y_param not found in parameter names.")

    xi = names.index(x_param)
    yi = names.index(y_param)

    # Build a slicer for all dims
    slicer = [slice(None)] * chern_grid.ndim
    slice_indices = slice_indices or {}

    # For non-plotted params, apply either slicing (if given) or reduction later
    reduce_axes = []
    for i, pname in enumerate(names):
        if i in (xi, yi):
            continue
        if pname in slice_indices:
            slicer[i] = int(slice_indices[pname])
        else:
            reduce_axes.append(i)

    # Slice then reduce
    sub = chern_grid[tuple(slicer)]
    if reduce_axes:
        # after slicing, axis indices may shift; compute map of old->new
        # Find remaining axes order
        keep_idxs = [ax for ax in range(sub.ndim)]
        # Determine positions of x and y in sub
        # They were at xi, yi in the original; after slicing, build a mapping
        orig_to_new = []
        k = 0
        for i in range(chern_grid.ndim):
            if isinstance(slicer[i], slice):
                orig_to_new.append(k)
                k += 1
            else:
                orig_to_new.append(None)

        reduce_axes_new = [orig_to_new[i] for i in reduce_axes if orig_to_new[i] is not None]
        if reduce_other == "mean":
            sub = np.nanmean(sub, axis=tuple(reduce_axes_new), keepdims=False)
        elif reduce_other == "median":
            sub = np.nanmedian(sub, axis=tuple(reduce_axes_new), keepdims=False)
        elif reduce_other == "max":
            sub = np.nanmax(sub, axis=tuple(reduce_axes_new))
        elif reduce_other == "min":
            sub = np.nanmin(sub, axis=tuple(reduce_axes_new))
        else:
            raise ValueError("reduce_other must be one of: mean, median, max, min")

        xi_new = orig_to_new[xi]
        yi_new = orig_to_new[yi]
    else:
        # nothing reduced
        xi_new = [j for j,i in enumerate(range(chern_grid.ndim)) if isinstance(slicer[i], slice)][
            [i for i in range(chern_grid.ndim) if isinstance(slicer[i], slice)].index(xi)
        ]
        yi_new = [j for j,i in enumerate(range(chern_grid.ndim)) if isinstance(slicer[i], slice)][
            [i for i in range(chern_grid.ndim) if isinstance(slicer[i], slice)].index(yi)
        ]

    # Ensure sub is (y, x) for imshow
    if yi_new == 0 and xi_new == 1:
        img = sub
    elif yi_new == 1 and xi_new == 0:
        img = sub.T
    else:
        # bring axes so that (yi_new, xi_new) -> (0,1)
        img = np.moveaxis(sub, (yi_new, xi_new), (0, 1))

    x_axis = axes[xi]
    y_axis = axes[yi]

    plt.figure(figsize=(6,5))
    plt.imshow(
        img,
        origin="lower",
        extent=[x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()],
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",   # <-- no smoothing between pixels
        resample=False             # <-- prevents resampling when resizing
    )
    plt.colorbar(label="Chern number")
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title("Chern number phase diagram")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
    plt.show()


def plot_chern_vs_param(
    sweep_root: str,
    x_param: str | None = None,     # name of the parameter to plot on X; if None, uses the only/first one
    slice_indices: dict | None = None,  # e.g. {"psi": 3} to pick fixed indices for other params
    reduce_other: str = "mean",     # "mean" | "median" | "max" | "min" (applied to non-sliced, non-x axes)
    marker: str = "o",
    save_path: str | None = None
):
    """
    Plot Chern number vs a single sweep parameter from a saved sweep folder.

    Expects in `sweep_root`:
      - chern_grid.npy            (N-d array of Chern values)
      - param_axes.npz            (contains 'names' and axis arrays as 'axis_{i}_{name}')

    Behavior:
      - If only one parameter was swept, it just plots that.
      - If multiple parameters exist, choose `x_param` (or the first), then:
          * for any other parameter in `slice_indices`, pick that single index
          * for the rest, reduce along that axis with `reduce_other`.
    """
    chern_path = os.path.join(sweep_root, "chern_grid.npy")
    axes_path  = os.path.join(sweep_root, "param_axes.npz")
    if not (os.path.exists(chern_path) and os.path.exists(axes_path)):
        raise FileNotFoundError("Missing chern_grid.npy or param_axes.npz in sweep_root.")

    chern_grid = np.load(chern_path)
    axes_npz   = np.load(axes_path, allow_pickle=True)
    names      = list(axes_npz["names"])
    axes       = [np.array(axes_npz[f"axis_{i}_{names[i]}"], dtype=float) for i in range(len(names))]
    slice_indices = slice_indices or {}

    # Choose x_param (the only or the first by default)
    if x_param is None:
        x_param = names[0]
    if x_param not in names:
        raise ValueError(f"x_param '{x_param}' not found. Available: {names}")
    xi = names.index(x_param)

    # Build slicer: leave x as slice(None); others either fixed index or marked for reduction
    slicer = [slice(None)] * chern_grid.ndim
    reduce_axes = []
    for i, pname in enumerate(names):
        if i == xi:
            continue
        if pname in slice_indices:
            idx = int(slice_indices[pname])
            if idx < 0 or idx >= axes[i].size:
                raise IndexError(f"slice index {idx} out of bounds for parameter '{pname}' (size={axes[i].size}).")
            slicer[i] = idx
        else:
            reduce_axes.append(i)

    sub = chern_grid[tuple(slicer)]

    # After slicing, x lives at the position corresponding to where slicer kept a slice
    # Compute map: original axis -> new axis index (or None if sliced to scalar)
    orig_to_new = []
    k = 0
    for i in range(chern_grid.ndim):
        if isinstance(slicer[i], slice):
            orig_to_new.append(k); k += 1
        else:
            orig_to_new.append(None)

    # Reduce remaining non-x axes
    if reduce_axes:
        reduce_axes_new = tuple(a for a in (orig_to_new[i] for i in reduce_axes) if a is not None)
        if len(reduce_axes_new) > 0:
            if reduce_other == "mean":
                sub = np.nanmean(sub, axis=reduce_axes_new)
            elif reduce_other == "median":
                sub = np.nanmedian(sub, axis=reduce_axes_new)
            elif reduce_other == "max":
                sub = np.nanmax(sub, axis=reduce_axes_new)
            elif reduce_other == "min":
                sub = np.nanmin(sub, axis=reduce_axes_new)
            else:
                raise ValueError("reduce_other must be one of: mean, median, max, min")

    # Now sub should be 1D along x
    if sub.ndim == 0:
        # Everything reduced to a scalar — plot a single point at the (only) x value
        x_vals = np.array([axes[xi][0]])
        y_vals = np.array([float(sub)])
    elif sub.ndim == 1:
        x_vals = axes[xi]
        y_vals = np.asarray(sub, dtype=float)
        if x_vals.size != y_vals.size:
            raise ValueError(f"Size mismatch: x has {x_vals.size} points but y has {y_vals.size}.")
    else:
        # Defensive fallback: collapse all but x by mean
        x_axis_pos = orig_to_new[xi] if orig_to_new[xi] is not None else 0
        collapse = tuple(ax for ax in range(sub.ndim) if ax != x_axis_pos)
        y_vals = np.nanmean(sub, axis=collapse)
        x_vals = axes[xi]

    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals, marker=marker)
    plt.xlabel(x_param)
    plt.ylabel("Chern number")
    plt.title(f"Chern vs {x_param}")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
    plt.show()

# simplest: use the first two parameters as axes, average over the rest
# plot_chern_phase_diagram("./results/phase_diagram/HaldaneHamiltonian/M_-2.09_2.09-psi_-3.14_3.14_data_set1")

# plot_chern_vs_param("./results/phase_diagram/HaldaneHamiltonian/M_-2.09_2.09-psi_-3.14_3.14_data_set1", x_param="M", slice_indices={"psi": np.pi/2})

# plot_chern_vs_param("./results/phase_diagram/HaldaneHamiltonian/M_-2.09_2.09_N100_data_set1", x_param="M")

# plot_chern_vs_param("./results/phase_diagram/ChiralHamiltonian/V_20.00_40.00_data_set1", x_param="V")

# plot_chern_vs_param("./results/phase_diagram/ChiralHamiltonian/V_-10.00_10.00_N100_data_set1", x_param="V")