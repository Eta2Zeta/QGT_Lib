import html
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from .plotting_lib_1d import plot_eigenvalues_line
from .plotting_utils import overlay_hamiltonian_symmetry_path
from .utilities import replace_zeros_with_nan


__all__ = [
    "plot_eigenvalues_line_cut",
    "plot_eigenvalue_line_slider",
    "plot_eigenvalue_surfaces",
    "plot_individual_eigenvalue_surfaces",
    "plot_eigenfunction_component_scatter",
    "plot_degeneracy_heatmap",
]


def _format_html_meta_value(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_html_meta_value(item) for item in value) + "]"
    if value is None:
        return "None"
    return str(value)


def _load_html_metadata_text(results_dir):
    if not results_dir:
        return None

    meta_path = os.path.join(results_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"Warning: Could not display metadata from {meta_path}: {exc}")
        return None

    lines = ["<b>Run Metadata</b>"]
    hamiltonian_params = metadata.get("hamiltonian_params", {})
    for key, value in metadata.items():
        if key == "hamiltonian_params":
            continue
        lines.append(
            f"{html.escape(str(key))} = "
            f"{html.escape(_format_html_meta_value(value))}"
        )

    if hamiltonian_params:
        lines.extend(["", "<b>Hamiltonian Parameters</b>"])
        for key, value in hamiltonian_params.items():
            lines.append(
                f"{html.escape(str(key))} = "
                f"{html.escape(_format_html_meta_value(value))}"
            )
    return "<br>".join(lines)


def plot_eigenvalues_line_cut(kx_grid, ky_grid, eigenvalues, start_k, end_k, num_points=100, bands_to_plot=None, results_dir=None, save_fig=False):
    """
    Extracts eigenvalues along a linear path in 2D k-space and plots them.

    Parameters:
    - kx_grid, ky_grid: 2D arrays (meshgrid) representing the k-space.
    - eigenvalues: 3D array of eigenvalues [nkx, nky, nbands].
    - start_k: Tuple (kx_start, ky_start).
    - end_k: Tuple (kx_end, ky_end).
    - num_points: Number of points along the interpolation line.
    - bands_to_plot: Tuple of band indices to plot.
    - results_dir: Directory where the plots will be saved.
    - save_fig: Boolean determining whether to save the plot or show it.
    """
    # Generate points along the line
    k_path_x = np.linspace(start_k[0], end_k[0], num_points)
    k_path_y = np.linspace(start_k[1], end_k[1], num_points)

    # Calculate path distance for x-axis of the plot
    dist = np.sqrt((k_path_x - start_k[0])**2 + (k_path_y - start_k[1])**2)

    nbands = eigenvalues.shape[2]
    result_array = np.zeros((num_points, nbands))

    # Flatten the grid coordinates
    flat_kx = kx_grid.flatten()
    flat_ky = ky_grid.flatten()

    print(f"Extracting cut with {num_points} points...")

    for i in range(num_points):
        tx, ty = k_path_x[i], k_path_y[i]

        # Distance squared
        d2 = (flat_kx - tx)**2 + (flat_ky - ty)**2
        nearest_idx = np.argmin(d2)

        # Convert flat index back to 2D index
        idx_y, idx_x = divmod(nearest_idx, kx_grid.shape[1])

        result_array[i, :] = eigenvalues[idx_y, idx_x, :]

    print("Extraction complete. Plotting...")

    # Construct descriptive title
    title = f"Eigenvalues along line: ({start_k[0]:.2f}, {start_k[1]:.2f}) -> ({end_k[0]:.2f}, {end_k[1]:.2f})"

    # Create figure and axes manually to add inset
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the eigenvalues
    plot_eigenvalues_line(dist, result_array, dim=nbands, bands_to_plot=bands_to_plot,
                          plot_style="points", title=title, ax=ax, show=False)

    # Add miniplot (inset)
    # Location: Lower left corner, small
    # Alternatively, using relative coordinates: [left, bottom, width, height]
    ax_inset = ax.inset_axes([0.05, 0.05, 0.2, 0.2])

    # Plot the bounding box of the grid (assuming rectangular grid aligned with axes)
    kx_min, kx_max = kx_grid.min(), kx_grid.max()
    ky_min, ky_max = ky_grid.min(), ky_grid.max()

    # Draw a box for the domain
    ax_inset.plot([kx_min, kx_max, kx_max, kx_min, kx_min],
                  [ky_min, ky_min, ky_max, ky_max, ky_min], 'k-', linewidth=0.5)

    # Plot the path line
    ax_inset.plot([start_k[0], end_k[0]], [start_k[1], end_k[1]], 'r-', linewidth=2)

    # Plot start and end points
    ax_inset.plot(start_k[0], start_k[1], 'go', markersize=3) # Start Green
    ax_inset.plot(end_k[0], end_k[1], 'rs', markersize=3)   # End Red

    ax_inset.set_aspect('equal')
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_title("Path in 2D", fontsize=8)

    if save_fig and results_dir:
        # Clean up coordinates for filename
        s_k = f"{start_k[0]:.2f}_{start_k[1]:.2f}".replace('.', 'p')
        e_k = f"{end_k[0]:.2f}_{end_k[1]:.2f}".replace('.', 'p')
        filename = f"eigenvalues_along_cut_{s_k}_to_{e_k}.png"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved Eigenvalue cut plot to: {filepath}")
    else:
        plt.show()

    plt.close()


def plot_eigenvalue_line_slider(
    ki,
    kj,
    eigenvalues,
    *,
    axis_order="ij",
    first_axis_label="ki",
    second_axis_label="kj",
    bands_to_plot=None,
    title=None,
    results_dir=None,
    save_fig=False,
    filename=None,
    show=True,
):
    """Plot 1D band slices of a 2D eigenvalue grid with a slice slider.

    ``axis_order='ij'`` plots the first sampled coordinate on the horizontal
    axis and uses the second coordinate as the slider. ``axis_order='ji'``
    reverses those roles. Band traces remain individually toggleable through
    the Plotly legend.
    """
    import plotly.graph_objects as go

    ki = np.asarray(ki)
    kj = np.asarray(kj)
    eigenvalues = np.asarray(eigenvalues)
    axis_order = str(axis_order).lower()

    if axis_order not in {"ij", "ji"}:
        raise ValueError("axis_order must be either 'ij' or 'ji'")
    if ki.ndim != 2 or kj.ndim != 2 or ki.shape != kj.shape:
        raise ValueError("ki and kj must be 2D meshgrid arrays with equal shapes")
    if eigenvalues.ndim != 3 or eigenvalues.shape[:2] != ki.shape:
        raise ValueError(
            "eigenvalues must have shape ki.shape + (num_bands,); "
            f"received {eigenvalues.shape} for grid shape {ki.shape}"
        )

    first_values = ki[0, :]
    second_values = kj[:, 0]
    if not np.allclose(ki, first_values[np.newaxis, :], equal_nan=True):
        raise ValueError("ki must be a separable meshgrid for an ij/ji line slider")
    if not np.allclose(kj, second_values[:, np.newaxis], equal_nan=True):
        raise ValueError("kj must be a separable meshgrid for an ij/ji line slider")

    num_bands = eigenvalues.shape[2]
    if bands_to_plot is None:
        bands = list(range(num_bands))
    elif isinstance(bands_to_plot, int):
        bands = [bands_to_plot]
    else:
        bands = list(bands_to_plot)
    invalid_bands = [band for band in bands if not 0 <= band < num_bands]
    if invalid_bands:
        raise IndexError(
            f"Invalid bands {invalid_bands}; valid band indices are 0 to {num_bands - 1}."
        )
    if not bands:
        raise ValueError("bands_to_plot must contain at least one band")

    if axis_order == "ij":
        x_values = first_values
        slider_values = second_values
        x_label = first_axis_label
        slider_label = second_axis_label

        def eigenvalue_slice(slider_index, band):
            return eigenvalues[slider_index, :, band]
    else:
        x_values = second_values
        slider_values = first_values
        x_label = second_axis_label
        slider_label = first_axis_label

        def eigenvalue_slice(slider_index, band):
            return eigenvalues[:, slider_index, band]

    fig = go.Figure()
    for band in bands:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=eigenvalue_slice(0, band),
                mode="lines",
                name=f"Band {band}",
                legendgroup=f"band_{band}",
                showlegend=True,
                hovertemplate=(
                    f"{x_label}: %{{x:.6g}}<br>"
                    "Energy: %{y:.6g}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    frames = []
    for slider_index in range(len(slider_values)):
        frame_data = [
            go.Scatter(
                x=x_values,
                y=eigenvalue_slice(slider_index, band),
            )
            for band in bands
        ]
        frames.append(
            go.Frame(
                data=frame_data,
                traces=list(range(len(bands))),
                name=str(slider_index),
            )
        )
    fig.frames = frames

    slider_steps = [
        {
            "method": "animate",
            "args": [
                [str(slider_index)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
            "label": f"{value:.6g}",
        }
        for slider_index, value in enumerate(slider_values)
    ]

    meta_text = _load_html_metadata_text(results_dir)
    if title is None:
        title = f"Eigenvalue line slices ({axis_order})"
    fig.update_layout(
        title=title,
        xaxis={"title": x_label},
        yaxis={"title": "Energy", "autorange": True},
        hovermode="x unified",
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": f"Fixed {slider_label}: "},
                "pad": {"t": 55},
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "showactive": False,
                "x": 0.0,
                "y": -0.18,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 100, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        legend={
            "title": {"text": "Bands"},
            "x": 0.01,
            "y": 0.99,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255, 255, 255, 0.85)",
            "bordercolor": "black",
            "borderwidth": 1,
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        showlegend=True,
        margin={"r": 340 if meta_text else 40, "b": 120, "l": 70, "t": 60},
    )

    if meta_text:
        fig.add_annotation(
            text=meta_text,
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bordercolor="black",
            borderwidth=1,
            borderpad=10,
            bgcolor="white",
            font={"size": 11},
        )

    if filename is None:
        filename = f"eigenvalue_line_slider_{axis_order}.html"
    if save_fig and results_dir:
        filepath = os.path.join(results_dir, filename)
        fig.write_html(filepath, include_plotlyjs="cdn")
        print(f"Eigenvalue line-slider plot saved to {filepath}")
    elif save_fig:
        fig.write_html(filename, include_plotlyjs="cdn")
        print(f"Eigenvalue line-slider plot saved to {filename}")

    if show and not save_fig:
        fig.show()
    return fig


def plot_eigenvalue_surfaces(
    kx,
    ky,
    eigenvalues,
    dim=None,                 # if None, infer from eigenvalues.shape[2]
    z_limit=300,
    norm=True,                # if True, all bands share Normalize(-z_limit, z_limit); if False, auto per-band
    stride_size=3,
    color_maps='default',
    bands_to_plot=None,       # NEW: int | iterable[int] | None (None => all bands)
    results_dir=None,
    save_fig=False,
    filename="eigenvalue_surfaces.html",
    show=True,
    x_label="kx",
    y_label="ky",
):
    """
    Plot selected eigenvalue bands as 3D interactive HTML surfaces with toggleable bands.

    Parameters:
    - kx, ky            : 2D arrays (meshgrid) for k-space
    - eigenvalues       : 3D array, shape (Nk, Nk, Nb)
    - dim               : number of bands; if None, inferred as eigenvalues.shape[2]
    - z_limit           : used when norm=True (shared limits)
    - norm              : True -> shared scaling; False -> per-band autoscale
    - stride_size       : surface stride for downsampling
    - color_maps        : 'default' | str | list[str]
    - bands_to_plot     : which band indices to plot (e.g. 0, or (0,2,5)); None -> all
    - results_dir       : Output directory to save the file.
    - save_fig          : Whether to save the HTML file locally in the `results_dir`.
    - filename          : Output filename.
    - show              : Whether to show the plot if not saving.
    - x_label, y_label  : Labels for the two sampled coordinates.

    If ``results_dir/meta.json`` exists, its run metadata and Hamiltonian
    parameters are displayed in a box beside the interactive plot.
    """
    import plotly.graph_objects as go
    meta_text = _load_html_metadata_text(results_dir)

    # Infer number of bands if not provided
    if dim is None:
        if eigenvalues.ndim != 3:
            raise ValueError(f"`eigenvalues` must be 3D (Nk, Nk, Nb); got shape {eigenvalues.shape}")
        dim = eigenvalues.shape[2]

    # Resolve which bands to plot
    if bands_to_plot is None:
        bands = list(range(dim))
    elif isinstance(bands_to_plot, int):
        bands = [bands_to_plot]
    else:
        bands = list(bands_to_plot)

    # Validate band indices
    bad = [b for b in bands if not (0 <= b < dim)]
    if bad:
        print("The bands you are asking to plot exceed the dimension of the Hamiltonian")
        raise IndexError(f"bands_to_plot contains out-of-range indices {bad}; valid range is [0, {dim-1}]")

    # Prepare colormaps for Plotly
    if color_maps == 'default':
        color_maps = ['Viridis', 'RdBu', 'Plasma', 'Earth', 'Inferno', 'Jet', 'Cividis', 'Hot', 'Spectral']
    elif isinstance(color_maps, str):
        color_maps = [color_maps] * max(1, len(bands))

    fig = go.Figure()

    # Downsample if stride given
    if stride_size > 1:
        kx_plot = kx[::stride_size, ::stride_size]
        ky_plot = ky[::stride_size, ::stride_size]
    else:
        kx_plot = kx
        ky_plot = ky

    for j, band in enumerate(bands):
        Z = replace_zeros_with_nan(eigenvalues[:, :, band])
        if stride_size > 1:
            Z = Z[::stride_size, ::stride_size]

        cmap = color_maps[j % len(color_maps)]

        if norm and z_limit is not None:
            cmin = -z_limit
            cmax = z_limit
        else:
            cmin = np.nanmin(Z)
            cmax = np.nanmax(Z)

        surface = go.Surface(
            x=kx_plot, y=ky_plot, z=Z,
            colorscale=cmap,
            cmin=cmin,
            cmax=cmax,
            opacity=0.8,
            showscale=False, # Plotly shows colorbars for each trace, hiding them reduces clutter
            name=f"Band {band}",
            showlegend=True
        )
        fig.add_trace(surface)

    fig.update_layout(
        title='Eigenvalue Surfaces',
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title='Eigenvalue',
            zaxis_range=[-z_limit, z_limit] if (norm and z_limit is not None) else None
        ),
        legend=dict(
            title=dict(text='Bands'),
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.85)',
            bordercolor='black',
            borderwidth=1,
            itemclick='toggle',
            itemdoubleclick='toggleothers',
        ),
        margin=dict(r=340 if meta_text else 20, b=10, l=10, t=40)
    )

    if meta_text:
        fig.add_annotation(
            text=meta_text,
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=1.02,
            y=1.0,
            xanchor='left',
            yanchor='top',
            bordercolor='black',
            borderwidth=1,
            borderpad=10,
            bgcolor='white',
            font=dict(size=11),
        )

    if save_fig and results_dir:
        filepath = os.path.join(results_dir, filename)
        fig.write_html(filepath, include_plotlyjs='cdn')
        print(f"Surface plot saved to {filepath}")
    elif filename and save_fig:
        fig.write_html(filename, include_plotlyjs='cdn')
        print(f"Surface plot saved to {filename}")

    if show and not (save_fig and results_dir):
        fig.show()


def plot_individual_eigenvalue_surfaces(kx, ky, eigenvalues, dim=6, z_limit=300, stride_size=3, color_maps='default'):
    """
    Plot individual eigenvalues for each band as separate 3D surface plots in a grid layout.

    Parameters:
    - kx, ky: 2D arrays for the k-space grid.
    - eigenvalues: 3D array of eigenvalues for each (kx, ky) grid point and band.
    - dim: Number of eigenvalue bands.
    - z_limit: Z-axis limit for plotting.
    - stride_size: Controls the density of plotted surfaces.
    - color_maps: List of color maps for each band, or a single color map for all bands.
    """
    # Determine grid layout based on the number of bands
    cols = math.ceil(math.sqrt(dim))
    rows = math.ceil(dim / cols)

    # Default color maps if none is provided
    if color_maps == 'default':
        color_maps = ['viridis', 'magma', 'coolwarm', 'plasma', 'inferno', 'cividis']
    elif isinstance(color_maps, str):
        color_maps = [color_maps] * dim  # Use the specified color map for all bands

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), subplot_kw={'projection': '3d'})
    fig.suptitle('Eigenvalues for Different Bands', fontsize=16)

    # Flatten axes array for easy indexing if grid is larger than 1x1
    axes = axes.ravel() if rows * cols > 1 else [axes]

    for band in range(dim):
        ax = axes[band]

        # Get the Z data for the eigenvalues and replace zeros with NaN
        Z_eigenvalue = replace_zeros_with_nan(eigenvalues[:, :, band])

        # Plot the surface for each eigenvalue band
        ax.plot_surface(kx, ky, Z_eigenvalue, cmap=color_maps[band % len(color_maps)],
                        rstride=stride_size, cstride=stride_size, alpha=0.6)

        ax.set_title(f'Eigenvalue {band + 1}')
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_zlabel('Eigenvalue')
        if z_limit is None:
            # Dynamically determine z-axis limits based on data in this band
            zmin = np.nanmin(Z_eigenvalue)
            zmax = np.nanmax(Z_eigenvalue)
            margin = 0.05 * (zmax - zmin)  # 5% margin
            ax.set_zlim(zmin - margin, zmax + margin)
        else:
            ax.set_zlim(-z_limit, z_limit)

    # Hide any unused subplots if rows * cols > dim
    for idx in range(dim, rows * cols):
        fig.delaxes(axes[idx])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_eigenfunction_component_scatter(kx, ky, eigenfunctions, band_index=None, components_to_plot=None, stride_size=3):
    """
    Plot specified eigenfunction components for a specific band or all bands as separate 3D scatter plots.

    Parameters:
    - kx, ky: 2D arrays for the k-space grid.
    - eigenfunctions: 4D array of eigenfunctions for each (kx, ky) grid point, band, and component.
    - band_index: Index of the band to plot. If None, plots all bands.
    - components_to_plot: List of component indices to plot. If None, plots all components.
    - stride_size: Controls the density of points plotted. Larger values skip more points.
    """
    # Determine which bands to plot
    if band_index is None:
        bands_to_plot = range(eigenfunctions.shape[2])  # Plot all bands
    else:
        bands_to_plot = [band_index]  # Plot only the specified band

    for band in bands_to_plot:
        # Determine the components to plot
        if components_to_plot is None:
            components_to_plot = range(eigenfunctions.shape[-1])  # Plot all components

        num_components = len(components_to_plot)
        cols = math.ceil(math.sqrt(num_components))
        rows = math.ceil(num_components / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), subplot_kw={'projection': '3d'})
        fig.suptitle(f'Band {band + 1} Eigenfunction Components', fontsize=16)

        # Flatten axes array for easy indexing if grid is larger than 1x1
        axes = axes.ravel() if rows * cols > 1 else [axes]

        for idx, component in enumerate(components_to_plot):
            ax = axes[idx]

            # Apply stride size to reduce the number of points
            stride = slice(None, None, stride_size)
            kx_strided = kx[stride, stride]
            ky_strided = ky[stride, stride]
            Z_eigenfunction = eigenfunctions[stride, stride, band, component].flatten()
            Z_eigenfunction = replace_zeros_with_nan(Z_eigenfunction)  # Replace zeros with NaN
            Z_magnitude = np.real(Z_eigenfunction)  # Use the real part for magnitude

            # Plot a 3D scatter plot
            sc = ax.scatter(kx_strided.flatten(), ky_strided.flatten(), Z_magnitude, c=Z_magnitude, cmap='viridis', s=1)

            ax.set_title(f'Component {component + 1}')
            ax.set_xlabel('kx')
            ax.set_ylabel('ky')
            ax.set_zlabel('Magnitude')

            # Add a color bar to each component plot
            fig.colorbar(sc, ax=ax, shrink=0.6, aspect=5)

        # Hide any unused subplots if rows * cols > num_components
        for idx in range(len(components_to_plot), rows * cols):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()
        plt.close()


def plot_degeneracy_heatmap(
    kx,
    ky,
    eigenvalues,
    threshold=0.01,
    title="Degeneracy Heatmap",
    sym_points=True,
    hamiltonian=None,
    kk=0.0,
    sym_kz_threshold=0.02,
    results_dir=None,
    save_fig=False,
    x_label="kx",
    y_label="ky",
):
    """
    Plot a 2D map where colors indicate the number of band degeneracies at each k-point.

    Parameters:
    - kx, ky: 2D arrays (meshgrid) for k-space.
    - eigenvalues: 3D array of eigenvalues [nkx, nky, nbands].
    - threshold: Relative threshold (fraction of max gap) to consider bands degenerate.
    - title: Plot title.
    - sym_points: If True, overlays Hamiltonian.get_sym_path() when available.
                  If False, no symmetry overlay is drawn.
    - hamiltonian: Optional Hamiltonian object used to read get_sym_path().
    - kk: Fixed kz value of the plotted 2D slice.
    - sym_kz_threshold: Fraction in [0, 1] of max(kx range, ky range). For 3D
                        symmetry points, only points with |kz - kk| within
                        this tolerance are projected onto the 2D map.
    - x_label, y_label: Labels for the two sampled coordinates.
    """
    nkx, nky, nbands = eigenvalues.shape
    degeneracy_map = np.zeros((nkx, nky), dtype=int)

    print("Calculating degeneracy heatmap...")

    # Iterate over grid
    for i in range(nkx):
        for j in range(nky):
            evals = np.sort(eigenvalues[i, j, :])
            diffs = np.diff(evals)

            if len(diffs) > 0:
                max_gap = np.max(diffs)
                # Avoid zero division or extremely small gaps
                current_thresh = max(threshold * max_gap, 1e-10)

                degeneracy_map[i, j] = np.sum(diffs <= current_thresh)
            else:
                degeneracy_map[i, j] = 0

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("0.35")

    # Define colors for 0, 1, 2, 3+
    colors = ['blue', 'green', 'red', 'black']
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 10.5] # Bins centered at 0, 1, 2, 3+
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.pcolormesh(kx, ky, degeneracy_map, cmap=cmap, norm=norm, shading='auto')

    if sym_points:
        overlay_hamiltonian_symmetry_path(
            ax,
            kx,
            ky,
            hamiltonian,
            kk=kk,
            sym_kz_threshold=sym_kz_threshold,
            line_width=2,
            point_size=50,
            label_fontsize=12,
            show_legend=True,
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Colorbar with discrete ticks
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['0 (Non-deg)', '1 (2-fold)', '2 (3-fold/2x2)', '3+ (>3-fold)'])
    cbar.set_label('Degeneracy Count (small gaps)')

    plt.tight_layout()
    if save_fig and results_dir:
        filename = "degeneracy_heatmap.png"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved degeneracy heatmap to: {filepath}")
    else:
        plt.show()
    plt.close()
