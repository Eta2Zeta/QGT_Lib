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
    title=None,          # Custom title
    ax=None,             # Optional axes to plot on
    show=True            # Whether to call plt.show()
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
    - title: Custom title for the plot.
    - ax: Optional matplotlib axes to plot on.
    - show: Whether to call plt.show() at the end.
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
        return None, None

    # Determine y-axis limits
    ymin, ymax = np.min(selected_eigenvalues), np.max(selected_eigenvalues)
    y_padding = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    ymin, ymax = ymin - y_padding, ymax + y_padding

    plot_style = plot_style.lower()
    if plot_style not in {"line", "points", "both"}:
        raise ValueError("plot_style must be one of: 'line', 'points', 'both'")

    # Set up the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    for band in bands_to_plot:
        if not (0 <= band < dim):
            print(f"Warning: Band {band} is out of range and will not be plotted.")
            continue

        y = eigenvalues[:, band]

        if plot_style in {"line", "both"}:
            ax.plot(
                k_line, y,
                label=f"Band {band}" if plot_style != "both" else None,
                linewidth=linewidth,
                alpha=alpha
            )

        if plot_style in {"points", "both"}:
            # For "both", give the legend to the scatter (cleaner) or vice versa
            ax.scatter(
                k_line, y,
                label=f"Band {band}" if plot_style == "points" else f"Band {band}",
                marker=marker,
                s=markersize**2,
                alpha=alpha
            )

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Eigenvalues Along the Line in k-Space")
        
    ax.set_xlabel("k (along the line)")
    ax.set_ylabel("Eigenvalue")
    ax.set_ylim(ymin, ymax)
    ax.legend()
    ax.grid(True)
    
    if show:
        plt.show()
        
    return fig, ax

def plot_band_structure_sym(
    k_dist,
    eigenvalues,
    k_node_indices, 
    k_node_labels,
    bands_to_plot=None,
    title="Band Structure",
    ylabel="Energy (eV)",
    figsize=(10, 6),
    linewidth=1.5,
    ylim=None
):
    """
    Plots the band structure along a path in k-space with high-symmetry points labeled.

    Parameters:
    - k_dist: 1D array of cumulative distance along the path.
    - eigenvalues: 2D array of eigenvalues [num_points, num_bands].
    - k_node_indices: List of indices in k_dist corresponding to high-symmetry points.
    - k_node_labels: List of labels for the high-symmetry points.
    - bands_to_plot: List of band indices to plot. If None, plots all.
    """
    
    num_bands = eigenvalues.shape[1]
    if bands_to_plot is None:
        bands_to_plot = range(num_bands)
    
    plt.figure(figsize=figsize)
    
    # Plot bands - use list comprehension for efficiency if num_bands is large, but loop is fine here
    k_vals = np.array(k_dist)
    
    # Plot each band. Using 'k-' for all might make them hard to distinguish, 
    # but standard band structure plots focus on the dispersion, not distinguishing bands by color usually.
    # Let's use a cycle or just black.
    # Define styles: Solid Blue and Dashed Red
    styles = [
        {'color': 'blue', 'linestyle': '-'},
        {'color': 'red', 'linestyle': '--'}
    ]

    for i, band_idx in enumerate(bands_to_plot):
        style = styles[i % 2]
        plt.plot(
            k_vals, 
            eigenvalues[:, band_idx], 
            linestyle=style['linestyle'], 
            color=style['color'], 
            linewidth=linewidth, 
            alpha=0.8
        )
        
    # Vertical lines at high-symmetry points
    for idx in k_node_indices:
        # idx might be larger than len(k_dist) if user index is 1-based or something, assuming 0-based index into k_dist 
        if 0 <= idx < len(k_dist):
            plt.axvline(x=k_dist[idx], color='gray', linestyle='--', linewidth=0.8)
        
    # Set x-ticks and labels
    # Filter indices to be within range
    valid_indices = [idx for idx in k_node_indices if 0 <= idx < len(k_dist)]
    valid_labels = [label for i, label in enumerate(k_node_labels) if 0 <= k_node_indices[i] < len(k_dist)]
    
    plt.xticks([k_dist[i] for i in valid_indices], valid_labels)
    plt.xlim(k_dist[0], k_dist[-1])
    
    if ylim:
        plt.ylim(ylim)
        
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
