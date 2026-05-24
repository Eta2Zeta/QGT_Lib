import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from .utilities import replace_zeros_with_nan
from .plotting_lib_1d import plot_eigenvalues_line
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, BoundaryNorm


def extract_and_plot_eigenvalues_along_line(kx_grid, ky_grid, eigenvalues, start_k, end_k, num_points=100, bands_to_plot=None, results_dir=None, save_fig=False):
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
        import os
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







def plot_eigenvalues_surface_colorbar(
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
    filename="eigenvalues_surface_2d.html",
    show=True
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
    """
    import plotly.graph_objects as go
    
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
        title='Eigenvalues (2D Surface)',
        scene=dict(
            xaxis_title='kx',
            yaxis_title='ky',
            zaxis_title='Eigenvalue',
            zaxis_range=[-z_limit, z_limit] if (norm and z_limit is not None) else None
        ),
        margin=dict(r=20, b=10, l=10, t=40)
    )

    if save_fig and results_dir:
        import os
        filepath = os.path.join(results_dir, filename)
        fig.write_html(filepath, include_plotlyjs='cdn')
        print(f"Surface plot saved to {filepath}")
    elif filename and save_fig:
        fig.write_html(filename, include_plotlyjs='cdn')
        print(f"Surface plot saved to {filename}")
        
    if show and not (save_fig and results_dir):
        fig.show()


def plot_individual_eigenvalues(kx, ky, eigenvalues, dim=6, z_limit=300, stride_size=3, color_maps='default'):
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




def plot_eigenfunction_components(kx, ky, eigenfunctions, band_index=None, components_to_plot=None, stride_size=3):
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





def plot_QGT_components_3d(
    kx, ky, g_xx_array, g_xy_array, g_xy_array_imag, g_yy_array,
    stride_size=3, results_dir=None, save_fig=False, filename="QGT_components_3d.html", show=False
):
    """
    Plot g_xx, g_xy, g_yx, and g_yy arrays as 3D surface plots in a single figure (Plotly).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import os

    if stride_size > 1:
        kx = kx[::stride_size, ::stride_size]
        ky = ky[::stride_size, ::stride_size]
        g_xx_array = g_xx_array[::stride_size, ::stride_size]
        g_xy_array = g_xy_array[::stride_size, ::stride_size]
        g_xy_array_imag = g_xy_array_imag[::stride_size, ::stride_size]
        g_yy_array = g_yy_array[::stride_size, ::stride_size]

    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=[
            'Numerical g_xx (real part)',
            'Numerical g_xy (real part)',
            'Numerical g_xy (imaginary part)',
            'Numerical g_yy (real part)'
        ]
    )

    g_xy_real_min = np.nanmin(g_xy_array)
    g_xy_real_max = np.nanmax(g_xy_array)
    g_xy_imag_min = np.nanmin(g_xy_array_imag)
    g_xy_imag_max = np.nanmax(g_xy_array_imag)

    # Helper to add surface
    def add_surf(Z, col, cmin=None, cmax=None):
        if cmin is None: cmin = np.nanmin(Z)
        if cmax is None: cmax = np.nanmax(Z)
        fig.add_trace(go.Surface(
            x=kx, y=ky, z=Z, colorscale='Plasma', cmin=cmin, cmax=cmax,
            showscale=False
        ), row=1, col=col)
        
        # Update axis titles for this specific subplot
        fig.update_scenes(
            xaxis_title='kx', yaxis_title='ky',
            row=1, col=col
        )

    add_surf(g_xx_array, 1)
    add_surf(g_xy_array, 2, g_xy_real_min, g_xy_real_max)
    add_surf(g_xy_array_imag, 3, g_xy_imag_min, g_xy_imag_max)
    add_surf(g_yy_array, 4)
    
    fig.update_scenes(zaxis_title='g_xx', row=1, col=1)
    fig.update_scenes(zaxis_title='g_xy (real)', zaxis_range=[g_xy_real_min, g_xy_real_max], row=1, col=2)
    fig.update_scenes(zaxis_title='g_xy (imag)', zaxis_range=[g_xy_imag_min, g_xy_imag_max], row=1, col=3)
    fig.update_scenes(zaxis_title='g_yy', row=1, col=4)

    fig.update_layout(title_text='QGT Components (3D)', height=500, width=1600, margin=dict(r=10, b=10, l=10, t=60))

    if save_fig and results_dir:
        filepath = os.path.join(results_dir, filename)
        fig.write_html(filepath, include_plotlyjs='cdn')
        print(f"Saved QGT components (HTML) to: {filepath}")
    
    if show:
        fig.show()

# If looking for plot_QMT_wtrace_3d, use plot_trace_w_eigenvalue instead.
def plot_g_components_2d(g_xx_array, g_yy_array, trace_array, k_max=10):
    """
    Plot g_xx, g_yy, and trace arrays as 2D heatmaps in a single figure.

    Parameters:
    - kx, ky: 2D arrays for the k-space grid.
    - g_xx_array, g_yy_array, trace_array: 2D arrays to be plotted as heatmaps.
    - k_max: Maximum k-value for the extent of the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot g_xx_array
    ax1 = axes[0]
    c1 = ax1.imshow(g_xx_array, extent=(-k_max, k_max, -k_max, k_max), origin='lower', cmap='viridis')
    ax1.set_title('$g_{xx}$ (Numerical)')
    ax1.set_xlabel('kx')
    ax1.set_ylabel('ky')
    fig.colorbar(c1, ax=ax1)

    # Plot g_yy_array
    ax2 = axes[1]
    c2 = ax2.imshow(g_yy_array, extent=(-k_max, k_max, -k_max, k_max), origin='lower', cmap='plasma')
    ax2.set_title('$g_{yy}$ (Numerical)')
    ax2.set_xlabel('kx')
    ax2.set_ylabel('ky')
    fig.colorbar(c2, ax=ax2)

    # Plot trace_array
    ax3 = axes[2]
    c3 = ax3.imshow(trace_array, extent=(-k_max, k_max, -k_max, k_max), origin='lower', cmap='plasma')
    ax3.set_title('Trace (Numerical)')
    ax3.set_xlabel('kx')
    ax3.set_ylabel('ky')
    fig.colorbar(c3, ax=ax3)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_qmt_eig_berry_trace_3d(
    kx, ky,
    eigenvalues,              # shape: (Nk, Nk, Nb)
    g_xy_imag,                # shape: (Nk, Nk); Im(Q_xy)
    trace_array,              # shape: (Nk, Nk)
    eigenvalue_band=0,
    stride_size=2,
    convert_berry_from_imQ=True,  # If True, Ω = -2 * Im(Q_xy) by the standard convention Q_xy = g_xy - i Ω/2
    zlim_berry=None,
    zlim_trace=None,
    title="QGT: Eigenvalue, Berry Curvature, and Trace (3D)",
    results_dir=None,
    save_fig=False,
    filename="qmt_eig_berry_trace_3d.html",
    show=False
):
    """
    Make a 1×3 row of 3D surfaces for Eigenvalue, Berry curvature, and Trace using Plotly.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import os

    if stride_size > 1:
        kx = kx[::stride_size, ::stride_size]
        ky = ky[::stride_size, ::stride_size]
        if eigenvalues is not None:
            eigenvalues = eigenvalues[::stride_size, ::stride_size, :]
        g_xy_imag = g_xy_imag[::stride_size, ::stride_size]
        trace_array = trace_array[::stride_size, ::stride_size]

    if eigenvalues is not None:
        Z_eig = replace_zeros_with_nan(eigenvalues[:, :, eigenvalue_band])
    else:
        Z_eig = None

    if convert_berry_from_imQ:
        Z_berry = replace_zeros_with_nan(-2.0 * g_xy_imag) 
        berry_title = "Berry Curvature Ω"
    else:
        Z_berry = replace_zeros_with_nan(g_xy_imag)
        berry_title = "Im(Q_xy)"
        
    Z_trace = replace_zeros_with_nan(trace_array)

    zlim_eig   = get_limits_asym(Z_eig, None, zlim_percentile=None)
    zlim_berry_tuple = get_plot_limits(Z_berry, zlim_berry)
    zlim_trace_tuple = get_limits_asym(Z_trace, zlim_trace)

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=[
            f"Eigenvalue Band {eigenvalue_band+1}" if Z_eig is not None else "Eigenvalue (No Data)",
            berry_title,
            "Trace Tr[g]"
        ]
    )

    panels = [
        dict(Z=Z_eig,   cmap='Viridis', zlim=zlim_eig, title="Eigenvalue"),
        dict(Z=Z_berry, cmap='RdBu',    zlim=zlim_berry_tuple, title=berry_title),
        dict(Z=Z_trace, cmap='Plasma',  zlim=zlim_trace_tuple, title="Tr[g]"),
    ]

    for col, cfg in enumerate(panels, start=1):
        Z = cfg['Z']
        cmap = cfg['cmap']
        zlim_vals = cfg['zlim']

        if Z is not None:
            cmin = zlim_vals[0] if zlim_vals is not None else np.nanmin(Z)
            cmax = zlim_vals[1] if zlim_vals is not None else np.nanmax(Z)
            
            fig.add_trace(go.Surface(
                x=kx, y=ky, z=Z, colorscale=cmap, cmin=cmin, cmax=cmax,
                showscale=False
            ), row=1, col=col)
            
            fig.update_scenes(
                xaxis_title='kx', yaxis_title='ky', zaxis_title=cfg['title'],
                zaxis_range=zlim_vals,
                row=1, col=col
            )

    fig.update_layout(title_text=title, height=500, width=1500, margin=dict(r=10, b=10, l=10, t=60))

    if save_fig and results_dir:
        filepath = os.path.join(results_dir, filename)
        fig.write_html(filepath, include_plotlyjs='cdn')
        print(f"Saved QMT 3D components (HTML) to: {filepath}")
        
    if show:
        fig.show()

def plot_g_components_line(k_line, g_xx, g_yy, trace, angle_deg):
    # Plot QGT components
    plt.figure(figsize=(12, 6))
    plt.plot(k_line, g_xx, label='$g_{xx}$', color='blue')
    plt.plot(k_line, g_yy, label='$g_{yy}$', color='green')
    plt.plot(k_line, trace, label='Trace', color='red')

    plt.title(f'QGT Components Along Line at {angle_deg}°')
    plt.xlabel('k (along line)')
    plt.ylabel('$g$-metric components')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def get_plot_limits(Z, limit=None, zlim_percentile=None):
    """
    Calculate symmetric z-limits for plotting, optionally constrained by
    an absolute limit or a percentile of the data. No margin is added.
    """
    if Z is None:
        return None

    # Filter out NaNs and infs
    valid_Z = Z[np.isfinite(Z)]
    if len(valid_Z) == 0:
        return None

    zmin = np.min(valid_Z)
    zmax = np.max(valid_Z)

    # Handle constant field
    if zmax == zmin:
        delta = 1.0 if zmax == 0 else 0.05 * abs(zmax)
        return (zmin - delta, zmax + delta)

    # Symmetric about 0
    abs_max = max(abs(zmin), abs(zmax))
    
    abs_use = abs_max
    if zlim_percentile is not None:
        abs_use = np.percentile(np.abs(valid_Z), zlim_percentile)

    # If an absolute limit is provided and is tighter than the data/percentile, clamp to it
    if limit is not None:
        lim = float(limit)
        if lim < abs_use:
            abs_use = lim

    # Avoid degenerate range
    if abs_use == 0.0:
        abs_use = 1.0

    return (-abs_use, abs_use)


def get_limits_asym(Z, limit=None, zlim_percentile=99):
    """
    Calculate asymmetric z-limits for plotting metric or eigenvalue data.
    The maximum limit is determined by the data's percentile (e.g. 99th),
    and the minimum is determined by the true minimum of the valid data.
    """
    if Z is None:
        return None

    # Filter out NaNs and infs
    valid_Z = Z[np.isfinite(Z)]
    if len(valid_Z) == 0:
        return None

    zmin = np.min(valid_Z)
    zmax = np.max(valid_Z)

    # Handle constant field
    if zmax == zmin:
        delta = 1.0 if zmax == 0 else 0.05 * abs(zmax)
        return (zmin - delta, zmax + delta)

    # The minimum is just the actual minimum of the data
    actual_min = zmin
    
    # Calculate upper limit based on percentile if requested
    actual_max = zmax
    if zlim_percentile is not None:
        actual_max = np.percentile(valid_Z, zlim_percentile)

    # If an absolute limit is provided and tighter than the data/percentile, clamp it
    if limit is not None:
        lim = float(limit)
        if actual_max > lim:
            actual_max = lim

    # Ensure min isn't somehow greater than clamped max
    if actual_min >= actual_max:
        actual_min = actual_max - 1.0

    return (actual_min, actual_max)


def plot_qmt_eig_berry_trace_2d(
    kx, ky,
    eigenvalues,              # shape: (Nk, Nk, Nb)
    g_xy_imag,                # shape: (Nk, Nk); Im(Q_xy)
    trace_array,              # shape: (Nk, Nk)
    eigenvalue_band=0,
    convert_berry_from_imQ=True,  # If True, Ω = -2 * Im(Q_xy)
    cmaps=('viridis', 'coolwarm', 'plasma'),
    zlim_berry=None,
    zlim_trace=None,
    title="QGT: Eigenvalue, Berry Curvature, and Trace (2D Heatmaps)",
    components="xy",
    results_dir=None,
    save_fig=False,
    space_group=None
):
    """
    Make a 1×3 row of 2D heatmaps for:
      - Eigenvalue band 'eigenvalue_band'
      - Berry curvature Ω (from Im(Q_xy) if convert_berry_from_imQ=True)
      - Trace of the QGT

    Args:
      kx, ky            : 2D grids (meshgrid)
      eigenvalues       : 3D array (Nk, Nk, Nb)
      g_xy_imag         : 2D array Im(Q_xy)
      trace_array       : 2D array Tr[g]
      eigenvalue_band   : which band to plot from eigenvalues
      convert_berry_from_imQ : if True, uses Ω = -2 * Im(Q_xy)
      cmaps             : (cmap_eig, cmap_berry, cmap_trace)
      zlim_berry        : max absolute limit for berry panel; None -> auto limit with 5% margin
      zlim_trace        : max absolute limit for trace panel; None -> auto limit with 5% margin
      title             : figure title
    """
    # Extract data
    if eigenvalues is not None:
        Z_eig = replace_zeros_with_nan(eigenvalues[:, :, eigenvalue_band])
    else:
        Z_eig = None

    if convert_berry_from_imQ:
        Z_berry = replace_zeros_with_nan(-2.0 * g_xy_imag)  # Ω = -2 Im(Q_xy)
        berry_label = f"Berry Curvature Ω ({components})"
    else:
        Z_berry = replace_zeros_with_nan(g_xy_imag)
        berry_label = f"Im(Q_xy) ({components})"
    Z_trace = replace_zeros_with_nan(trace_array)

    zlim_eig   = get_limits_asym(Z_eig, None, zlim_percentile=None)
    zlim_berry_tuple = get_plot_limits(Z_berry, zlim_berry)
    zlim_trace_tuple = get_limits_asym(Z_trace, zlim_trace)

    # Figure & axes (1 row, 3 cols)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if title:
        fig.suptitle(title, fontsize=14)

    panels = [
        dict(Z=Z_eig,   cmap=cmaps[0], title=f"Eigenvalue Band {eigenvalue_band+1}" if Z_eig is not None else "Eigenvalue (No Data)", zlim=zlim_eig),
        dict(Z=Z_berry, cmap=cmaps[1], title=berry_label, zlim=zlim_berry_tuple),
        dict(Z=Z_trace, cmap=cmaps[2], title="Trace Tr[g]", zlim=zlim_trace_tuple),
    ]

    for ax, cfg in zip(axes, panels):
        Z = cfg["Z"]
        cmap = cfg["cmap"]
        zlim = cfg["zlim"]
        vmin, vmax = zlim if zlim is not None else (None, None)

        if Z is not None:
            norm = None

            # Make 0 map to the *center color* (white-ish in many diverging maps)
            if cfg["title"] in [berry_label, "Trace Tr[g]"]:
                if vmin is not None and vmax is not None and vmin < 0 < vmax:
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

            im = ax.pcolormesh(kx, ky, Z, cmap=cmap, shading="auto",
                            norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Use native matplotlib format_coord to show the exact z-value on hover
            def make_format_coord(ax_curr, Z_arr, kx_arr, ky_arr):
                def format_coord(x, y):
                    x_idx = np.abs(kx_arr[0, :] - x).argmin()
                    y_idx = np.abs(ky_arr[:, 0] - y).argmin()
                    
                    if 0 <= y_idx < Z_arr.shape[0] and 0 <= x_idx < Z_arr.shape[1]:
                        z_val = Z_arr[y_idx, x_idx]
                        return f"kx={x:.4f}, ky={y:.4f}  |  Value = {z_val:.6g}"
                    return f"kx={x:.4f}, ky={y:.4f}"
                return format_coord

            ax.format_coord = make_format_coord(ax, Z, kx, ky)

        ax.set_title(cfg['title'])
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.axis('scaled') # square aspect ratio
        
        # Add symmetry triangle
        if space_group == 194:
            kM_y = 2 * np.pi / np.sqrt(3)
            kK_x = 2 * np.pi / 3
            kE_x = np.pi
            kE_y = np.pi / np.sqrt(3)
            pts = {"G": (0, 0), "M": (0, kM_y), "K": (kK_x, kM_y), "E": (kE_x, kE_y)}
            path = ["G", "M", "K", "G", "E", "K"]
            x_vals = [pts[p][0] for p in path]
            y_vals = [pts[p][1] for p in path]
            
            # Use white so it's visible on most dark colormaps
            ax.plot(x_vals, y_vals, color='white', linewidth=1.5, linestyle='--')
            
            for p_name, (px, py) in pts.items():
                ax.scatter(px, py, color='white', s=30, zorder=5)
                # Add label
                if p_name == "G":
                    offset = (-10, -10)
                elif p_name == "M":
                    offset = (-10, -15)
                elif p_name == "E":
                    offset = (10, -5)
                else:  # K
                    offset = (10, -15)
                disp_name = r'$\Gamma$' if p_name == "G" else p_name
                ax.annotate(disp_name, (px, py), textcoords="offset points", xytext=offset, ha='center', color='white', fontsize=12, fontweight='bold')
        
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    
    if save_fig and results_dir:
        import os
        filename = f"qgt_2d_band_{eigenvalue_band}_{components}.png"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved 2D QGT plot to: {filepath}")
    else:
        plt.show()
        
    plt.close()


def plot_eigen_and_all_berry_2d(
    kx, ky,
    eigenvalues,              # shape: (Nk, Nk, Nb)
    g_xy_imag,                # shape: (Nk, Nk); Im(Q_xy)
    g_xz_imag,                # shape: (Nk, Nk); Im(Q_xz)
    g_yz_imag,                # shape: (Nk, Nk); Im(Q_yz)
    eigenvalue_band=0,
    convert_berry_from_imQ=True,  # If True, Ω = -2 * Im(Q_ij)
    cmaps=('viridis', 'coolwarm', 'coolwarm', 'coolwarm'),
    zlim_berry=None,
    zlim_percentile=None,
    title="Eigenvalue and Berry Curvature Components (2D Heatmaps)",
    results_dir=None,
    save_fig=False,
    space_group=194
):
    """
    Make a 1x4 row of 2D heatmaps for:
      - Eigenvalue band 'eigenvalue_band'
      - Berry curvature Ω_xy
      - Berry curvature Ω_xz
      - Berry curvature Ω_yz

    Args:
      kx, ky            : 2D grids (meshgrid)
      eigenvalues       : 3D array (Nk, Nk, Nb)
      g_xy_imag         : 2D array Im(Q_xy)
      g_xz_imag         : 2D array Im(Q_xz)
      g_yz_imag         : 2D array Im(Q_yz)
      eigenvalue_band   : which band to plot from eigenvalues
      convert_berry_from_imQ : if True, uses Ω = -2 * Im(Q_ij)
      cmaps             : tuple of 4 colormaps
      zlim_berry        : max absolute limit for berry panels; None -> auto limit
      zlim_percentile   : limit automatically calculated as a percentile of the abs data
      title             : figure title
    """
    # Extract data
    if eigenvalues is not None:
        Z_eig = replace_zeros_with_nan(eigenvalues[:, :, eigenvalue_band])
    else:
        Z_eig = None

    if convert_berry_from_imQ:
        Z_berry_xy = replace_zeros_with_nan(-2.0 * g_xy_imag)
        Z_berry_xz = replace_zeros_with_nan(-2.0 * g_xz_imag)
        Z_berry_yz = replace_zeros_with_nan(-2.0 * g_yz_imag)
        berry_label_xy = "Berry Curvature Ω (xy)"
        berry_label_xz = "Berry Curvature Ω (xz)"
        berry_label_yz = "Berry Curvature Ω (yz)"
    else:
        Z_berry_xy = replace_zeros_with_nan(g_xy_imag)
        Z_berry_xz = replace_zeros_with_nan(g_xz_imag)
        Z_berry_yz = replace_zeros_with_nan(g_yz_imag)
        berry_label_xy = "Im(Q_xy)"
        berry_label_xz = "Im(Q_xz)"
        berry_label_yz = "Im(Q_yz)"

    zlim_eig   = get_limits_asym(Z_eig, None, zlim_percentile=None)
    zlim_berry_xy_tuple = get_plot_limits(Z_berry_xy, zlim_berry, zlim_percentile)
    zlim_berry_xz_tuple = get_plot_limits(Z_berry_xz, zlim_berry, zlim_percentile)
    zlim_berry_yz_tuple = get_plot_limits(Z_berry_yz, zlim_berry, zlim_percentile)

    # Figure & axes (1 row, 4 cols)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    if title:
        fig.suptitle(title, fontsize=14)

    panels = [
        dict(Z=Z_eig,      cmap=cmaps[0], title=f"Eigenvalue Band {eigenvalue_band+1}" if Z_eig is not None else "Eigenvalue (No Data)", zlim=zlim_eig),
        dict(Z=Z_berry_xy, cmap=cmaps[1], title=berry_label_xy, zlim=zlim_berry_xy_tuple),
        dict(Z=Z_berry_xz, cmap=cmaps[2], title=berry_label_xz, zlim=zlim_berry_xz_tuple),
        dict(Z=Z_berry_yz, cmap=cmaps[3], title=berry_label_yz, zlim=zlim_berry_yz_tuple),
    ]

    for ax, cfg in zip(axes, panels):
        Z = cfg["Z"]
        cmap = cfg["cmap"]
        zlim = cfg["zlim"]
        vmin, vmax = zlim if zlim is not None else (None, None)

        if Z is not None:
            norm = None

            # Make 0 map to the *center color* (white-ish in many diverging maps)
            if "Berry" in cfg["title"] or "Im(Q_" in cfg["title"]:
                if vmin is not None and vmax is not None and vmin < 0 < vmax:
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

            im = ax.pcolormesh(kx, ky, Z, cmap=cmap, shading="auto",
                            norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Use native matplotlib format_coord to show the exact z-value on hover
            def make_format_coord(ax_curr, Z_arr, kx_arr, ky_arr):
                def format_coord(x, y):
                    x_idx = np.abs(kx_arr[0, :] - x).argmin()
                    y_idx = np.abs(ky_arr[:, 0] - y).argmin()
                    
                    if 0 <= y_idx < Z_arr.shape[0] and 0 <= x_idx < Z_arr.shape[1]:
                        z_val = Z_arr[y_idx, x_idx]
                        return f"kx={x:.4f}, ky={y:.4f}  |  Value = {z_val:.6g}"
                    return f"kx={x:.4f}, ky={y:.4f}"
                return format_coord

            ax.format_coord = make_format_coord(ax, Z, kx, ky)

        ax.set_title(cfg['title'])
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.axis('scaled') # square aspect ratio
        
        # Add symmetry triangle
        if space_group == 194:
            kM_y = 2 * np.pi / np.sqrt(3)
            kK_x = 2 * np.pi / 3
            kE_x = np.pi
            kE_y = np.pi / np.sqrt(3)
            pts = {"G": (0, 0), "M": (0, kM_y), "K": (kK_x, kM_y), "E": (kE_x, kE_y)}
            path = ["G", "M", "K", "G", "E", "K"]
            x_vals = [pts[p][0] for p in path]
            y_vals = [pts[p][1] for p in path]
            
            # Use white so it's visible on most dark colormaps
            ax.plot(x_vals, y_vals, color='white', linewidth=1.5, linestyle='--')
            
            for p_name, (px, py) in pts.items():
                ax.scatter(px, py, color='white', s=30, zorder=5)
                # Add label
                if p_name == "G":
                    offset = (-10, -10)
                elif p_name == "M":
                    offset = (-10, -15)
                elif p_name == "E":
                    offset = (10, -5)
                else:  # K
                    offset = (10, -15)
                disp_name = r'$\Gamma$' if p_name == "G" else p_name
                ax.annotate(disp_name, (px, py), textcoords="offset points", xytext=offset, ha='center', color='white', fontsize=12, fontweight='bold')
        
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    
    if save_fig and results_dir:
        import os
        filename = f"qgt_2d_all_berry_band_{eigenvalue_band}.png"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved 2D all-Berry plot to: {filepath}")
    else:
        plt.show()
        
    plt.close()


def plot_degeneracy_2d(kx, ky, eigenvalues, threshold=0.01, title="Degeneracy Map (2D)", sym_points=True, results_dir=None, save_fig=False):
    """
    Plot a 2D map where colors indicate the number of band degeneracies at each k-point.
    
    Parameters:
    - kx, ky: 2D arrays (meshgrid) for k-space.
    - eigenvalues: 3D array of eigenvalues [nkx, nky, nbands].
    - threshold: Relative threshold (fraction of max gap) to consider bands degenerate.
    - title: Plot title.
    - sym_points: If True, overlays default high-symmetry points ((0,0), (pi,0), (pi,pi), (0,pi)).
                  If array-like, uses the provided points.
    """
    nkx, nky, nbands = eigenvalues.shape
    degeneracy_map = np.zeros((nkx, nky), dtype=int)
    
    print("Calculating 2D degeneracy map...")
    
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

    # Define colors for 0, 1, 2, 3+
    colors = ['blue', 'green', 'red', 'black']
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 10.5] # Bins centered at 0, 1, 2, 3+
    norm = BoundaryNorm(bounds, cmap.N)
    
    im = ax.pcolormesh(kx, ky, degeneracy_map, cmap=cmap, norm=norm, shading='auto')
    
    # Overlay Symmetry Points
    if sym_points is not False:
        if sym_points is True:
            # Connect all points to form a perimeter and diagonals
            # Gamma -> X -> M -> Y -> Gamma -> M
            path = np.array([
                [0, 0],
                [np.pi, 0],
                [np.pi, np.pi],
                [0, np.pi],
                [0, 0],
                [np.pi, np.pi], # Cross-diagonal from (0,0) to (pi,pi)
                [np.pi, 0],     # Jump back to X (without drawing line to Y)
                [0, np.pi]      # Cross-diagonal from (pi,0) to (0,pi)
            ])
            
            # Since the path jumps and crosses, the simplest robust pyplot approach to draw connecting lines between all is to plot the perimeter box and diagonals manually.
            pass  # we will handle drawing differently below to prevent weird connecting segments
            
        else:
            path = np.array(sym_points)
            labels = [f"P{idx}" for idx in range(len(path))]
            ax.plot(path[:, 0], path[:, 1], color='white', linewidth=2, linestyle='--', label='Symmetry Path')
            ax.scatter(path[:, 0], path[:, 1], color='white', s=50, zorder=5)
            
            for (x, y), label in zip(path, labels):
                if label: 
                    ax.annotate(label, (x, y), textcoords="offset points", xytext=(5,5), ha='left', color='white', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')

        if sym_points is True:
            # Custom drawing to ensure all 4 points connect to every other point
            pts = {
                "Gamma": (0, 0),
                "X": (np.pi, 0),
                "M": (np.pi, np.pi),
                "Y": (0, np.pi)
            }
            labels_disp = {"Gamma": "(0,0)", "X": "($\pi$,0)", "M": "($\pi$,$\pi$)", "Y": "(0,$\pi$)"}
            
            # Draw all pairs
            lines_to_draw = [
                ("Gamma", "X"), ("X", "M"), ("M", "Y"), ("Y", "Gamma"), # Perimeter
                ("Gamma", "M"), ("X", "Y")                              # Diagonals
            ]
            
            for p1, p2 in lines_to_draw:
                x_vals = [pts[p1][0], pts[p2][0]]
                y_vals = [pts[p1][1], pts[p2][1]]
                # Only add the label to the legend once
                lbl = 'Symmetry Lines' if p1 == "Gamma" and p2 == "X" else "_nolegend_"
                ax.plot(x_vals, y_vals, color='white', linewidth=2, linestyle='--', label=lbl)
                
            # Draw point markers and labels
            for k, (x, y) in pts.items():
                ax.scatter(x, y, color='white', s=50, zorder=5)
                ax.annotate(labels_disp[k], (x, y), textcoords="offset points", xytext=(5,5), ha='left', color='white', fontsize=12, fontweight='bold')
                
            ax.legend(loc='upper right')
    
    ax.set_title(title)
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    
    # Colorbar with discrete ticks
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['0 (Non-deg)', '1 (2-fold)', '2 (3-fold/2x2)', '3+ (>3-fold)'])
    cbar.set_label('Degeneracy Count (small gaps)')
    
    plt.tight_layout()
    if save_fig and results_dir:
        import os
        filename = f"degeneracy_2d.png"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved Degeneracy map to: {filepath}")
    else:
        plt.show()
    plt.close()
