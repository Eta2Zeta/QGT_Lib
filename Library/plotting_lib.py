import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from .utilities import replace_zeros_with_nan

def plot_k_line_on_grid(line_kx, line_ky, k_max):
    """
    Plot the k-path line (line_kx, line_ky) on a 2D grid representing the Brillouin zone.

    Parameters:
    - line_kx: 1D array of kx values along the line.
    - line_ky: 1D array of ky values along the line.
    - k_max: Maximum k-value defining the grid boundary.
    """
    # Generate the grid for visualization
    kx_grid, ky_grid = np.meshgrid(np.linspace(-k_max, k_max, 100), 
                                   np.linspace(-k_max, k_max, 100))

    # Plot the grid and the line
    plt.figure(figsize=(8, 8))
    plt.plot(line_kx, line_ky, color='red', label='k-path line', linewidth=2)  # Plot the line
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Optional x-axis
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # Optional y-axis
    plt.xlim(-k_max, k_max)
    plt.ylim(-k_max, k_max)
    plt.title("k-Path Line on 2D k-Space Grid")
    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.grid(True)
    plt.legend()
    plt.show()



def plot_eigenvalues_line(k_line, eigenvalues, dim=None, bands_to_plot=None):
    """
    Plot eigenvalues for selected bands along a 1D k-path.

    Parameters:
    - k_line: 1D array of k-values along the line.
    - eigenvalues: 2D array of eigenvalues (shape: [num_points, num_bands]).
    - dim: Number of bands (optional). If None, it will be inferred from the eigenvalues array.
    - bands_to_plot: Tuple of band indices to plot (e.g., (0,1,3)). Default is None, which plots all bands.
    """
    # Infer the number of bands from eigenvalues if not provided
    if dim is None:
        dim = eigenvalues.shape[1]

    # Default to plotting all bands if no specific selection is given
    if bands_to_plot is None:
        bands_to_plot = range(dim)

    # Ensure bands_to_plot is a tuple
    if isinstance(bands_to_plot, int):
        bands_to_plot = (bands_to_plot,)

    # Collect eigenvalues for selected bands
    selected_eigenvalues = np.array([eigenvalues[:, band] for band in bands_to_plot if 0 <= band < dim])

    if selected_eigenvalues.size == 0:
        print("Warning: No valid bands selected for plotting.")
        return

    # Determine y-axis limits
    ymin, ymax = np.min(selected_eigenvalues), np.max(selected_eigenvalues)
    y_padding = 0.05 * (ymax - ymin)  # Add 5% padding
    ymin, ymax = ymin - y_padding, ymax + y_padding

    # Set up the plot
    plt.figure(figsize=(10, 6))
    for band in bands_to_plot:
        if 0 <= band < dim:  # Ensure the band index is within valid range
            plt.plot(k_line, eigenvalues[:, band], label=f'Band {band}')
        else:
            print(f"Warning: Band {band} is out of range and will not be plotted.")

    # Add plot details
    plt.title('Eigenvalues Along the Line in k-Space')
    plt.xlabel('k (along the line)')
    plt.ylabel('Eigenvalue')
    plt.ylim(ymin, ymax)  # Dynamically adjust the y-axis range
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_eigenvalues_surface(kx, ky, eigenvalues, dim=6, z_limit=300, stride_size=3, color_maps='default'):
    """
    Plot eigenvalues as 3D surface plots, with an option to specify color maps.

    Parameters:
    - kx, ky: 2D arrays for the k-space grid.
    - eigenvalues: 3D array of eigenvalues for each (kx, ky) grid point and band.
    - dim: Number of eigenvalue bands.
    - z_limit: Z-axis limit for plotting.
    - stride_size: Controls the density of plotted surfaces.
    - color_maps: List of color maps for each band, or a single color map for all bands.
    """
    # Default color maps if none is provided
    if color_maps == 'default':
        color_maps = ['viridis', 'magma', 'coolwarm', 'plasma', 'inferno', 'cividis']
    elif isinstance(color_maps, str):
        color_maps = [color_maps] * dim  # Use the specified color map for all bands

    fig = plt.figure(figsize=(24, 8))
    ax = fig.add_subplot(111, projection='3d')

    for band in range(dim):
        Z = replace_zeros_with_nan(eigenvalues[:, :, band])
        ax.plot_surface(kx, ky, Z, cmap=color_maps[band % len(color_maps)], 
                        rstride=stride_size, cstride=stride_size, alpha=0.8)

    ax.set_title('Eigenvalues for All Bands with Touching Points')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('Eigenvalue')
    ax.set_zlim(-z_limit, z_limit)

    plt.show()
    plt.close()


def plot_eigenvalues_surface_colorbar(kx, ky, eigenvalues, dim=6, z_limit=300, norm = True, stride_size=3, color_maps='default'):
    """
    Plot eigenvalues as 3D surface plots, with an option to specify color maps.

    Parameters:
    - kx, ky: 2D arrays for the k-space grid.
    - eigenvalues: 3D array of eigenvalues for each (kx, ky) grid point and band.
    - dim: Number of eigenvalue bands.
    - z_limit: Z-axis limit for plotting.
    - stride_size: Controls the density of plotted surfaces.
    - color_maps: List of color maps for each band, or a single color map for all bands.
    """
    # Default color maps if none is provided
    if color_maps == 'default':
        color_maps = ['viridis', 'magma', 'coolwarm', 'plasma', 'inferno', 'cividis']
    elif isinstance(color_maps, str):
        color_maps = [color_maps] * dim  # Use the specified color map for all bands

    fig = plt.figure(figsize=(24, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each band and add a color bar
    for band in range(dim):
        Z = replace_zeros_with_nan(eigenvalues[:, :, band])
        cmap = plt.get_cmap(color_maps[band % len(color_maps)])
        if norm is not None: 
            norm = plt.Normalize(vmin=-z_limit, vmax=z_limit)
        else: 
            norm = None
        
        # Plot the surface
        surf = ax.plot_surface(kx, ky, Z, cmap=cmap, norm=norm,
                               rstride=stride_size, cstride=stride_size, alpha=0.8)

        # Add a color bar
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(Z)
        # cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=30, pad=0.01)
        # cbar.set_label(f'Band {band + 1} Eigenvalues', fontsize=10)



    ax.set_title('Eigenvalues for All Bands with Touching Points')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('Eigenvalue')
    ax.set_zlim(-z_limit, z_limit)

    plt.show()
    plt.close()


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



def plot_phases(kx, ky, phasefactors, dim=6, z_limit=(-2, 2), color_maps='default'):
    """
    Plot the phases for different bands as 3D scatter plots in a grid layout.

    Parameters:
    - kx, ky: 2D arrays for the k-space grid.
    - phasefactors: 3D array of phase factors for each (kx, ky) grid point and band.
    - dim: Number of bands.
    - z_limit: Tuple for Z-axis limits.
    - color_maps: List of color maps for each band, or a single color map for all bands.
    """
    cols = math.ceil(math.sqrt(dim))
    rows = math.ceil(dim / cols)

    # Default color maps if none is provided
    if color_maps == 'default':
        color_maps = ['viridis', 'magma', 'coolwarm', 'plasma', 'inferno', 'cividis']
    elif isinstance(color_maps, str):
        color_maps = [color_maps] * dim

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), subplot_kw={'projection': '3d'})
    fig.suptitle('Phases', fontsize=16)

    axes = axes.ravel() if rows * cols > 1 else [axes]

    for band in range(dim):
        ax = axes[band]
        Z_phasefactor = replace_zeros_with_nan(phasefactors[:, :, band].flatten())

        # Create scatter plot
        sc = ax.scatter(kx.flatten(), ky.flatten(), Z_phasefactor, c=Z_phasefactor, cmap=color_maps[band % len(color_maps)], s=3)

        ax.set_title(f'Phase Factor {band + 1}')
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_zlabel('Phase Factor')
        ax.set_zlim(*z_limit)

        fig.colorbar(sc, ax=ax, shrink=0.6, aspect=5)

    for idx in range(dim, rows * cols):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_neighbor_phases(kx, ky, overall_neighbor_phase_array, dim=6, z_limit=(-2, 2), color_maps='default'):
    """
    Plot the overall neighbor phase array for different bands as 3D scatter plots in a grid layout.

    Parameters:
    - kx, ky: 2D arrays for the k-space grid.
    - overall_neighbor_phase_array: 3D array of neighbor phases for each (kx, ky) grid point and band.
    - dim: Number of bands.
    - z_limit: Tuple for Z-axis limits.
    - color_maps: List of color maps for each band, or a single color map for all bands.
    """
    cols = math.ceil(math.sqrt(dim))
    rows = math.ceil(dim / cols)

    # Default color maps if none is provided
    if color_maps == 'default':
        color_maps = ['viridis', 'magma', 'coolwarm', 'plasma', 'inferno', 'cividis']
    elif isinstance(color_maps, str):
        color_maps = [color_maps] * dim

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), subplot_kw={'projection': '3d'})
    fig.suptitle('Overall Neighbor Phase Factors', fontsize=16)

    axes = axes.ravel() if rows * cols > 1 else [axes]

    for band in range(dim):
        ax = axes[band]
        Z_neighbor_phase = replace_zeros_with_nan(overall_neighbor_phase_array[:, :, band].flatten())

        # Create scatter plot
        sc = ax.scatter(kx, ky, Z_neighbor_phase, c=Z_neighbor_phase, cmap=color_maps[band % len(color_maps)], s=3)

        ax.set_title(f'Neighbor Phase Factor {band + 1}')
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_zlabel('Neighbor Phase Factor')
        ax.set_zlim(*z_limit)

        fig.colorbar(sc, ax=ax, shrink=0.6, aspect=5)

    for idx in range(dim, rows * cols):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()
    plt.close()

def plot_QGT_components_3d(kx, ky, g_xx_array, g_xy_array, g_xy_array_imag, g_yy_array, stride_size=3):
    """
    Plot g_xx, g_xy, g_yx, and g_yy arrays as 3D surface plots in a single figure.

    Parameters:
    - kx, ky: 2D arrays for the k-space grid.
    - g_xx_array, g_xy_array, g_xy_array_imag, g_yy_array: 2D arrays to be plotted as surfaces.
    - stride_size: Controls the density of points in the surface plot.
    """
    fig = plt.figure(figsize=(24, 6))

    # Determine common z-limits for g_xy_array (real) and g_xy_array_imag (imaginary)
    # Use nanmin/nanmax to ignore NaNs when computing limits
    # Separate z-limits
    g_xy_real_min = np.nanmin(g_xy_array)
    g_xy_real_max = np.nanmax(g_xy_array)

    g_xy_imag_min = np.nanmin(g_xy_array_imag)
    g_xy_imag_max = np.nanmax(g_xy_array_imag)

    # Plot g_xx_array
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.plot_surface(kx, ky, g_xx_array, cmap='plasma', rstride=stride_size, cstride=stride_size)
    ax1.set_title('Numerical $g_{xx}$ (real part)')
    ax1.set_xlabel('kx')
    ax1.set_ylabel('ky')
    ax1.set_zlabel('$g_{xx}$')

    # Plot g_xy_array (real part)
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.plot_surface(kx, ky, g_xy_array, cmap='plasma', rstride=stride_size, cstride=stride_size)
    ax2.set_title('Numerical $g_{xy}$ (real part)')
    ax2.set_xlabel('kx')
    ax2.set_ylabel('ky')
    ax2.set_zlabel('$g_{xy}$ (real)')
    ax2.set_zlim(g_xy_real_min, g_xy_real_max)

    # Plot g_xy_array_imag (imaginary part)
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.plot_surface(kx, ky, g_xy_array_imag, cmap='plasma', rstride=stride_size, cstride=stride_size)
    ax3.set_title('Numerical $g_{xy}$ (imaginary part)')
    ax3.set_xlabel('kx')
    ax3.set_ylabel('ky')
    ax3.set_zlabel('$g_{xy}$ (imag)')
    ax3.set_zlim(g_xy_imag_min, g_xy_imag_max)


    # Plot g_yy_array
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.plot_surface(kx, ky, g_yy_array, cmap='plasma', rstride=stride_size, cstride=stride_size)
    ax4.set_title('Numerical $g_{yy}$ (real part)')
    ax4.set_xlabel('kx')
    ax4.set_ylabel('ky')
    ax4.set_zlabel('$g_{yy}$')

    plt.tight_layout()
    plt.show()
    plt.close()

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


def plot_trace_w_eigenvalue(kx, ky, g_xx_array, g_yy_array, eigenvalues, trace_array, eigenvalue_band=0, stride_size=4):
    """
    Plot a 2x2 grid with:
    - Top-left: 3D plot of g_xx
    - Top-right: 3D plot of g_yy
    - Bottom-left: 3D plot of a single eigenvalue band
    - Bottom-right: 2D heatmap of the trace.

    Parameters:
    - kx, ky: 2D arrays for the k-space grid.
    - g_xx_array, g_yy_array, trace_array: Arrays for g_xx, g_yy, and trace data.
    - eigenvalues: 3D array of eigenvalues for each (kx, ky) grid point and band.
    - eigenvalue_band: Index of the eigenvalue band to plot in the bottom-left plot.
    - stride_size: Controls the density of points in the 3D surface plots.
    """
    fig = plt.figure(figsize=(9, 9))

    # Top-left: 3D plot of g_xx
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(kx, ky, g_xx_array, cmap='plasma', rstride=stride_size, cstride=stride_size)
    ax1.set_title('$g_{xx}$ (real part)')
    ax1.set_xlabel('kx')
    ax1.set_ylabel('ky')
    ax1.set_zlabel('$g_{xx}$')

    # Top-right: 3D plot of g_yy
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot_surface(kx, ky, g_yy_array, cmap='plasma', rstride=stride_size, cstride=stride_size)
    ax2.set_title('$g_{yy}$ (real part)')
    ax2.set_xlabel('kx')
    ax2.set_ylabel('ky')
    ax2.set_zlabel('$g_{yy}$')

    # Bottom-left: 3D plot of the specified eigenvalue band
    ax3 = fig.add_subplot(223, projection='3d')
    eigenvalue_band_data = eigenvalues[:, :, eigenvalue_band]  # Extract the specified band
    ax3.plot_surface(kx, ky, eigenvalue_band_data, cmap='viridis', rstride=stride_size, cstride=stride_size)
    ax3.set_title(f'Eigenvalue Band {eigenvalue_band + 1}')
    ax3.set_xlabel('kx')
    ax3.set_ylabel('ky')
    ax3.set_zlabel('Eigenvalue')

    # Bottom-right: 2D heatmap of the trace
    ax4 = fig.add_subplot(224, projection = '3d')
    ax4.plot_surface(kx, ky, trace_array, cmap='plasma', rstride=stride_size, cstride=stride_size)
    ax4.set_title('Trace (Numerical)')
    ax4.set_xlabel('kx')
    ax4.set_ylabel('ky')
    ax4.set_zlabel('Trace')

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_qmt_eig_berry_trace_3d(
    kx, ky,
    eigenvalues,              # shape: (Nk, Nk, Nb)
    g_xy_imag,                # shape: (Nk, Nk); Im(Q_xy)
    trace_array,              # shape: (Nk, Nk)
    eigenvalue_band=0,
    stride_size=3,
    convert_berry_from_imQ=True,  # If True, Ω = -2 * Im(Q_xy) by the standard convention Q_xy = g_xy - i Ω/2
    cmaps=('viridis', 'coolwarm', 'plasma'),
    zlims=(None, None, None),     # (zlim_eig, zlim_berry, zlim_trace); each entry None -> auto
    cbar_shrink=0.7,
    cbar_aspect=30,
    title="QGT: Eigenvalue, Berry Curvature, and Trace (3D)"
):
    """
    Make a 1×3 row of 3D surfaces for:
      - Eigenvalue band 'eigenvalue_band'
      - Berry curvature Ω (from Im(Q_xy) if convert_berry_from_imQ=True)
      - Trace of the QGT

    Args:
      kx, ky            : 2D grids
      eigenvalues       : 3D array (Nk, Nk, Nb)
      g_xy_imag         : 2D array Im(Q_xy)
      trace_array       : 2D array Tr[g]
      eigenvalue_band   : which band to plot from eigenvalues
      stride_size       : surface stride
      convert_berry_from_imQ : if True, uses Ω = -2 * Im(Q_xy) (sign per usual QGT convention)
      cmaps             : (cmap_eig, cmap_berry, cmap_trace)
      zlims             : tuple of z-limits for each panel; any None -> auto limit with 5% margin
      cbar_shrink       : colorbar shrink factor
      cbar_aspect       : colorbar aspect (larger -> thinner)
      title             : figure title
    """
    # Extract data
    Z_eig = replace_zeros_with_nan(eigenvalues[:, :, eigenvalue_band])
    if convert_berry_from_imQ:
        Z_berry = replace_zeros_with_nan(-2.0 * g_xy_imag)  # Ω = -2 Im(Q_xy)
    else:
        Z_berry = replace_zeros_with_nan(g_xy_imag)         # show Im(Q_xy) directly
    Z_trace = replace_zeros_with_nan(trace_array)

    # Auto z-limits with 5% margin if not provided
    def auto_limits(Z):
        zmin = np.nanmin(Z)
        zmax = np.nanmax(Z)
        if not np.isfinite(zmin) or not np.isfinite(zmax):
            return None
        if zmax == zmin:
            delta = 1.0
            return (zmin - delta, zmax + delta)
        margin = 0.05 * (zmax - zmin)
        return (zmin - margin, zmax + margin)

    zlim_eig   = zlims[0] if zlims[0] is not None else auto_limits(Z_eig)
    zlim_berry = zlims[1] if zlims[1] is not None else auto_limits(Z_berry)
    zlim_trace = zlims[2] if zlims[2] is not None else auto_limits(Z_trace)

    # Figure & axes (1 row, 3 cols)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
    if title:
        fig.suptitle(title, fontsize=14)

    # Panels config
    panels = [
        dict(Z=Z_eig,   cmap=cmaps[0], title=f"Eigenvalue Band {eigenvalue_band+1}",
             zlabel="Eigenvalue", zlim=zlim_eig),
        dict(Z=Z_berry, cmap=cmaps[1], title="Berry Curvature Ω" if convert_berry_from_imQ else "Im(Q_xy)",
             zlabel="Ω" if convert_berry_from_imQ else "Im(Q_xy)", zlim=zlim_berry),
        dict(Z=Z_trace, cmap=cmaps[2], title="Trace Tr[g]", zlabel="Tr[g]", zlim=zlim_trace),
    ]

    for ax, cfg in zip(axes, panels):
        Z = cfg['Z']
        cmap = cfg['cmap']
        norm = None  # you can put Normalize(...) if you want matched color scaling
        surf = ax.plot_surface(
            kx, ky, Z, cmap=cmap, norm=norm,
            rstride=stride_size, cstride=stride_size, alpha=0.9
        )
        ax.set_title(cfg['title'])
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_zlabel(cfg['zlabel'])
        if cfg['zlim'] is not None:
            ax.set_zlim(*cfg['zlim'])

        # Colorbar per panel (thin)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(Z)
        cbar = fig.colorbar(mappable, ax=ax, shrink=cbar_shrink, aspect=cbar_aspect, pad=0.03)
        cbar.ax.tick_params(labelsize=9)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    plt.close()

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
