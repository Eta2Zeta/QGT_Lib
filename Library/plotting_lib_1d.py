import numpy as np
import matplotlib.pyplot as plt

def plot_eigenvalues_line(
    k_line,
    eigenvalues,
    dim=None,
    bands_to_plot=None,
    plot_style="line",   # "line", "points", or "both"
    marker="o",
    markersize=3,
    linewidth=1.5,
    alpha=1.0,
):
    """
    Plot eigenvalues for selected bands along a 1D k-path.

    Parameters:
    - k_line: 1D array of k-values along the line.
    - eigenvalues: 2D array of eigenvalues (shape: [num_points, num_bands]).
    - dim: Number of bands (optional). If None, inferred from eigenvalues.
    - bands_to_plot: band indices to plot. int or iterable. None => all.
    - plot_style: "line" (default), "points", or "both".
    - marker, markersize: used for points/both.
    - linewidth: used for line/both.
    - alpha: transparency.
    """
    # Infer the number of bands from eigenvalues if not provided
    if dim is None:
        dim = eigenvalues.shape[1]

    # Default to plotting all bands if no specific selection is given
    if bands_to_plot is None:
        bands_to_plot = range(dim)

    # Ensure bands_to_plot is iterable
    if isinstance(bands_to_plot, int):
        bands_to_plot = (bands_to_plot,)

    # Collect eigenvalues for selected bands (for y-lims)
    selected_eigenvalues = np.array(
        [eigenvalues[:, band] for band in bands_to_plot if 0 <= band < dim]
    )

    if selected_eigenvalues.size == 0:
        print("Warning: No valid bands selected for plotting.")
        return

    # Determine y-axis limits
    ymin, ymax = np.min(selected_eigenvalues), np.max(selected_eigenvalues)
    y_padding = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    ymin, ymax = ymin - y_padding, ymax + y_padding

    plot_style = plot_style.lower()
    if plot_style not in {"line", "points", "both"}:
        raise ValueError("plot_style must be one of: 'line', 'points', 'both'")

    # Set up the plot
    plt.figure(figsize=(10, 6))

    for band in bands_to_plot:
        if not (0 <= band < dim):
            print(f"Warning: Band {band} is out of range and will not be plotted.")
            continue

        y = eigenvalues[:, band]

        if plot_style in {"line", "both"}:
            plt.plot(
                k_line, y,
                label=f"Band {band}" if plot_style != "both" else None,
                linewidth=linewidth,
                alpha=alpha
            )

        if plot_style in {"points", "both"}:
            # For "both", give the legend to the scatter (cleaner) or vice versa
            plt.scatter(
                k_line, y,
                label=f"Band {band}" if plot_style == "points" else f"Band {band}",
                marker=marker,
                s=markersize**2,
                alpha=alpha
            )

    plt.title("Eigenvalues Along the Line in k-Space")
    plt.xlabel("k (along the line)")
    plt.ylabel("Eigenvalue")
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.grid(True)
    plt.show()
