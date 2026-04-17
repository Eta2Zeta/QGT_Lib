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
    ylim=None,
    results_dir=None,
    save_fig=False,
    use_analytical=False
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
        bands_to_plot = list(range(num_bands))
    
    plt.figure(figsize=figsize)
    
    # Plot bands - use list comprehension for efficiency if num_bands is large, but loop is fine here
    k_vals = np.array(k_dist)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    if use_analytical:
        for i, band_idx in enumerate(bands_to_plot):
            if band_idx == 0:
                combo = r"$\alpha=-1, \beta=+1$"
            elif band_idx == 1:
                combo = r"$\alpha=-1, \beta=-1$"
            elif band_idx == 2:
                combo = r"$\alpha=+1, \beta=-1$"
            elif band_idx == 3:
                combo = r"$\alpha=+1, \beta=+1$"
            else:
                combo = ""
            label_text = f'Band {band_idx + 1} ({combo})' if combo else f'Band {band_idx + 1}'
            
            plt.plot(
                k_vals, 
                eigenvalues[:, band_idx], 
                linestyle='-', 
                color=colors[i % len(colors)], 
                linewidth=linewidth, 
                alpha=0.8,
                label=label_text
            )
        plt.legend(loc='best')
    else:
        # Sort bands from top to bottom
        bands_to_plot_sorted = sorted(bands_to_plot, reverse=True)
        for i, band_idx in enumerate(bands_to_plot_sorted):
            rank = i + 1
            plt.plot(
                k_vals, 
                eigenvalues[:, band_idx], 
                linestyle='-', 
                color=colors[i % len(colors)], 
                linewidth=linewidth, 
                alpha=0.8,
                label=f'Rank {rank}'
            )
        plt.legend(loc='best')
        
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
    
    if save_fig and results_dir:
        import os
        filename = "band_structure_sym.png"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved band structure to: {filepath}")
    else:
        plt.show()
        
    plt.close()

def plot_band_structure_angles_slider(
    k_vals,
    eigenvalues,
    angles,
    bands_to_plot=None,
    title="1D Band Structure vs Angle",
    results_dir=None,
    save_fig=False,
    show=True
):
    """
    Creates an interactive Plotly plot with a slider for the angle.
    
    Parameters
    ----------
    k_vals : ndarray, shape (num_points_per_line,)
        The 1D k-coordinates (from -k_max to k_max).
    eigenvalues : ndarray, shape (num_angles, num_points_per_line, num_bands)
        The eigenvalues for each angle.
    angles : ndarray, shape (num_angles,)
        The angles in radians.
    bands_to_plot : list or tuple, optional
        Indices of bands to plot. Defaults to all bands.
    title : str
        Plot title.
    results_dir : str, optional
        Path to save the HTML interact file.
    save_fig : bool
        If True, saves to 'band_structure_angles_slider.html' in results_dir.
    show : bool
        If True, opens the Plotly figure.
    """
    import plotly.graph_objects as go
    import os
    
    num_angles, num_points, num_bands = eigenvalues.shape
    
    if bands_to_plot is None:
        bands_to_plot = range(num_bands)
        
    fig = go.Figure()
    
    # Add traces for the first angle (index 0)
    for b in bands_to_plot:
        fig.add_trace(go.Scatter(
            x=k_vals,
            y=eigenvalues[0, :, b],
            name=f'Band {b}',
            mode='lines'
        ))
        
    # Create frames across all angles
    frames = []
    for i in range(num_angles):
        data_frame = [
            go.Scatter(
                x=k_vals,
                y=eigenvalues[i, :, b]
            ) for b in bands_to_plot
        ]
        frames.append(go.Frame(data=data_frame, name=str(i)))
        
    fig.frames = frames
    
    # Create slider steps
    steps = []
    for i, angle in enumerate(angles):
        step = dict(
            method='animate',
            args=[
                [str(i)],
                dict(
                    mode='immediate',
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )
            ],
            label=f"{angle:.3f} rad"
        )
        steps.append(step)
        
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Angle: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        title=title,
        xaxis_title="k",
        yaxis_title="Energy",
        sliders=sliders,
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, dict(frame=dict(duration=100, redraw=True), 
                                 fromcurrent=True, transition=dict(duration=0, easing="linear"))]
            ), dict(
                label="Pause",
                method="animate",
                args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]
            )],
            direction="left",
            pad={"r": 10, "t": 87},
            showactive=False,
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top"
        )]
    )
    
    # Save/Show logic
    if save_fig and results_dir:
        filepath = os.path.join(results_dir, "band_structure_angles_slider.html")
        fig.write_html(filepath, include_plotlyjs='cdn')
        print(f"Saved interactive angles plot to: {filepath}")
        
    if show:
        fig.show()
