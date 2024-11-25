# plot_eigenvalues.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from utilities import replace_zeros_with_nan


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

def plot_QGT_components_3d(kx, ky, g_xx_array, g_xy_array, g_yx_array, g_yy_array, stride_size=3):
    """
    Plot g_xx, g_xy, g_yx, and g_yy arrays as 3D surface plots in a single figure.

    Parameters:
    - kx, ky: 2D arrays for the k-space grid.
    - g_xx_array, g_xy_array, g_yx_array, g_yy_array: 2D arrays to be plotted as surfaces.
    - stride_size: Controls the density of points in the surface plot.
    """
    fig = plt.figure(figsize=(24, 6))

    # Plot g_xx_array
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.plot_surface(kx, ky, g_xx_array, cmap='plasma', rstride=stride_size, cstride=stride_size)
    ax1.set_title('Numerical $g_{xx}$ (real part)')
    ax1.set_xlabel('kx')
    ax1.set_ylabel('ky')
    ax1.set_zlabel('$g_{xx}$')

    # Plot g_xy_array
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.plot_surface(kx, ky, g_xy_array, cmap='plasma', rstride=stride_size, cstride=stride_size)
    ax2.set_title('Numerical $g_{xy}$ (real part)')
    ax2.set_xlabel('kx')
    ax2.set_ylabel('ky')
    ax2.set_zlabel('$g_{xy}$')

    # Plot g_yx_array
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.plot_surface(kx, ky, g_yx_array, cmap='plasma', rstride=stride_size, cstride=stride_size)
    ax3.set_title('Numerical $g_{yx}$ (real part)')
    ax3.set_xlabel('kx')
    ax3.set_ylabel('ky')
    ax3.set_zlabel('$g_{yx}$')

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


def plot_QMT_wtrace_3d(kx, ky, g_xx_array, g_yy_array, trace_array, stride_size=3):
    """
    Plot g_xx, g_yy, and trace arrays as 3D surface plots in a single figure.

    Parameters:
    - kx, ky: 2D arrays for the k-space grid.
    - g_xx_array, g_yy_array, trace_array: 2D arrays to be plotted as surfaces.
    - stride_size: Controls the density of points in the surface plot.
    """
    fig = plt.figure(figsize=(18, 6))

    # Plot g_xx_array
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(kx, ky, g_xx_array, cmap='plasma', rstride=stride_size, cstride=stride_size)
    ax1.set_title('Numerical $g_{xx}$ (real part)')
    ax1.set_xlabel('kx')
    ax1.set_ylabel('ky')
    ax1.set_zlabel('$g_{xx}$')

    # Plot g_yy_array
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(kx, ky, g_yy_array, cmap='plasma', rstride=stride_size, cstride=stride_size)
    ax2.set_title('Numerical $g_{yy}$ (real part)')
    ax2.set_xlabel('kx')
    ax2.set_ylabel('ky')
    ax2.set_zlabel('$g_{yy}$')

    # Plot trace_array
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(kx, ky, trace_array, cmap='plasma', rstride=stride_size, cstride=stride_size)
    ax3.set_title('Numerical Trace (real part)')
    ax3.set_xlabel('kx')
    ax3.set_ylabel('ky')
    ax3.set_zlabel('Trace')

    plt.tight_layout()
    plt.show()
    plt.close()


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
