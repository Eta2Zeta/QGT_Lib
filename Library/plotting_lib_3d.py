import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go
from .utilities import replace_zeros_with_nan

def plot_3d_stacked_slices(kx, ky, kz_values, eigenvalues_slices, band_index=0, 
                           z_limit=None, cmap='viridis', title=None):
    """
    Plot stacked 2D slices of eigenvalues in a 3D plot.
    
    Parameters:
    - kx, ky: 2D meshgrid arrays for the XY plane (shaped [nkx, nky]).
    - kz_values: 1D array of Z values where slices are taken.
    - eigenvalues_slices: 3D array of shape [n_slices, nkx, nky]. 
                          Contains eigenvalue data for the selected band at each slice.
    - band_index: Integer, index of the band being plotted (for labeling).
    - z_limit: Tuple (min, max) for colorbar limits. If None, auto-scaled.
    - cmap: Colormap name.
    - title: Plot title.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    n_slices = len(kz_values)
    
    # Determine color limits
    if z_limit is None:
        vmin = np.nanmin(eigenvalues_slices)
        vmax = np.nanmax(eigenvalues_slices)
    else:
        vmin, vmax = z_limit
        
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])
    
    for i, kz in enumerate(kz_values):
        # Data for this slice
        Z_data = eigenvalues_slices[i]
        
        # We plot a surface at constant z = kz
        # Create a Z coordinate array filled with kz
        Z_coord = np.full_like(kx, kz)
        
        # Plot surface
        # We map eigenvalue to color
        # facecolors expect mapped RGBA values
        fcolors = m.to_rgba(Z_data)
        
        surf = ax.plot_surface(kx, ky, Z_coord, rstride=5, cstride=5,
                               facecolors=fcolors, shade=False, alpha=0.8)
        
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('kz')
    
    if title is None:
        title = f'Eigenvalues for Band {band_index} (Stacked Slices)'
    ax.set_title(title)
    
    # Add colorbar
    cbar = fig.colorbar(m, ax=ax, shrink=0.6, aspect=20, pad=0.05)
    cbar.set_label('Eigenvalue')
    
    plt.show()

def plot_3d_stacked_slices_from_volume(eigenvalues_3d, kx_vals, ky_vals, kz_vals, band_index=0, num_slices=10, title=None):
    """
    Helper function to plot stacked slices directly from the full 3D volume.
    
    Parameters:
    - eigenvalues_3d: 4D array [nkx, nky, nkz, dim] (or consistent with compute_eigenvalues_3d output)
    - kx_vals, ky_vals, kz_vals: 1D arrays
    """
    mesh_size_z = len(kz_vals)
    slice_indices = np.linspace(0, mesh_size_z-1, num_slices, dtype=int)
    kz_plot_values = kz_vals[slice_indices]
    
    # eigenvalues_3d is [x, y, z, dim]
    # We want [n_slices, x, y]
    
    eig_band = eigenvalues_3d[:, :, :, band_index] # [x, y, z]
    eig_slices = np.zeros((num_slices, len(kx_vals), len(ky_vals)))
    
    for i, idx in enumerate(slice_indices):
        eig_slices[i] = eig_band[:, :, idx]
        
    kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals)
    
    if title is None:
        title = f"Band {band_index} Eigenvalues ({num_slices} Z-Slices)"
        
    plot_3d_stacked_slices(kx_grid, ky_grid, kz_plot_values, eig_slices, 
                           band_index=band_index, title=title)

def plot_3d_volumetric_scatter(kx_grid, ky_grid, kz_grid, eigenvalues, band_index=0, 
                               sparsity=5, cmap='viridis', title=None):
    """
    Plot a 3D scatter plot where points are colored by eigenvalue.
    Useful as an alternative "cube" visualization.
    
    Parameters:
    - kx_grid, ky_grid, kz_grid: 3D meshgrid arrays.
    - eigenvalues: 3D array of eigenvalues [nkx, nky, nkz].
    - sparsity: Integer, step size to reduce point density (e.g., plot every 5th point).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample data
    s = slice(None, None, sparsity)
    kx_s = kx_grid[s, s, s].flatten()
    ky_s = ky_grid[s, s, s].flatten()
    kz_s = kz_grid[s, s, s].flatten()
    vals_s = eigenvalues[s, s, s].flatten()
    
    sc = ax.scatter(kx_s, ky_s, kz_s, c=vals_s, cmap=cmap, s=5, alpha=0.6)
    
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('kz')
    
    if title is None:
        title = f'Eigenvalues for Band {band_index} (Volumetric Scatter)'
    ax.set_title(title)
    
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Eigenvalue')
    
    plt.show()

def plot_isosurface(eigenvalues, isovalue, kx_vals, ky_vals, kz_vals, 
                    band_index=0, alpha=0.5, color='royalblue', title=None, step_size=1):
    """
    Plot an isosurface for a specific eigenvalue using marching cubes.
    
    Parameters:
    - eigenvalues: 3D array of eigenvalues [nkx, nky, nkz].
    - isovalue: The threshold value to draw the surface at.
    - kx_vals, ky_vals, kz_vals: 1D arrays defining the grid coordinates.
    - band_index: Band index for labeling.
    - alpha: Transparency of the surface.
    - color: Color of the surface.
    """
    # Use marching cubes to obtain the surface mesh
    # Note: eigenvalues axis order in calculation_3d was [x, y, z]
    # marching_cubes expects volume.
    
    try:
        # marching_cubes returns verts, faces, normals, values
        # We downsample the array by slicing with step_size
        verts, faces, normals, values = measure.marching_cubes(eigenvalues[::step_size, ::step_size, ::step_size], isovalue)
    except (RuntimeError, ValueError):
        print(f"Could not find isosurface for level {isovalue}")
        return

    # Transform indices to real coordinates
    # verts contains indices (x_idx, y_idx, z_idx)
    # We need to map these to (kx, ky, kz)
    
    dx = kx_vals[1] - kx_vals[0]
    dy = ky_vals[1] - ky_vals[0]
    dz = kz_vals[1] - kz_vals[0]
    
    x0, y0, z0 = kx_vals[0], ky_vals[0], kz_vals[0]
    
    real_verts = np.zeros_like(verts)
    real_verts[:, 0] = x0 + verts[:, 0] * dx * step_size
    real_verts[:, 1] = y0 + verts[:, 1] * dy * step_size
    real_verts[:, 2] = z0 + verts[:, 2] * dz * step_size
    
    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    mesh = Poly3DCollection(real_verts[faces], alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor('k') # Optional: add edges for better visibility
    mesh.set_linewidth(0.05)
    
    ax.add_collection3d(mesh)
    
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('kz')
    
    ax.set_xlim(kx_vals.min(), kx_vals.max())
    ax.set_ylim(ky_vals.min(), ky_vals.max())
    ax.set_zlim(kz_vals.min(), kz_vals.max())
    
    if title is None:
        title = f'Isosurface (Band {band_index}, E={isovalue:.3f})'
    ax.set_title(title)
    
    plt.show()

def plot_arbitrary_slice(eigenvalues, orientation, shift, kx_vals, ky_vals, kz_vals, 
                         title=None, cmap='viridis', density=100, stride=3, alpha=0.3):
    """
    Extract and plot a slice of the 3D eigenvalue data along a specified orientation.
    Plots the eigenvalues as a height map (surface) over the 2D plane coordinates.
    All bands are plotted in the same figure.
    
    Parameters:
    - eigenvalues: 4D array [nkx, nky, nkz, nbands].
    - orientation: One of 'x', 'y', 'z', 'xy', 'yz', 'xz', 'xyz'.
    - shift: value determining the position of the plane.
    - kx_vals, ky_vals, kz_vals: 1D arrays of grid coordinates.
    - stride: Downsampling stride for the surface plot (default 3).
    - alpha: Transparency of the surface plot (default 0.3).
    """
    
    # 1. Setup Interpolator
    # We create a list of interpolators, one for each band, or one vector interpolator if supported.
    # To be safe and compatible, let's just create one interp per band or loop during interpolation?
    # Actually, RegularGridInterpolator supports trailing dimensions.
    # We pass the full 4D array.
    
    interp = RegularGridInterpolator((kx_vals, ky_vals, kz_vals), eigenvalues, bounds_error=False, fill_value=np.nan)
    
    # 2. Define Plane Coordinates (u, v) and 3D Coordinates (X, Y, Z)
    k_min = kx_vals.min()
    k_max = kx_vals.max()
    
    # Range for u, v depends on orientation
    # We'll use a generic range and then mask/label appropriately
    
    u_label = "u"
    v_label = "v"
    
    u = np.linspace(k_min, k_max, density)
    v = np.linspace(k_min, k_max, density)
    
    X, Y, Z = None, None, None
    U, V = None, None # For plotting axes
    
    if orientation == 'x': # x = shift
        # Plane: y-z
        u_label, v_label = "ky", "kz"
        U, V = np.meshgrid(u, v)
        X = np.full_like(U, shift)
        Y = U
        Z = V
    
    elif orientation == 'y': # y = shift
        # Plane: x-z
        u_label, v_label = "kx", "kz"
        U, V = np.meshgrid(u, v)
        X = U
        Y = np.full_like(U, shift)
        Z = V
        
    elif orientation == 'z': # z = shift
        # Plane: x-y
        u_label, v_label = "kx", "ky"
        U, V = np.meshgrid(u, v)
        X = U
        Y = V
        Z = np.full_like(U, shift)
        
    elif orientation == 'xy': # x - y = shift
        # Plane defined by u=(x+y), v=z
        u_label, v_label = "k_{x+y}", "kz"
        
        # Range for x+y can be larger, approx [-2pi, 2pi]
        alpha_range = np.linspace(k_min * 2, k_max * 2, density)
        U, V = np.meshgrid(alpha_range, v) 
        
        # x = (u + shift)/2
        # y = (u - shift)/2
        X = (U + shift) / 2
        Y = (U - shift) / 2
        Z = V
        
    elif orientation == 'yz': # y - z = shift
        u_label, v_label = "kx", "k_{y+z}"
        alpha_range = np.linspace(k_min * 2, k_max * 2, density)
        U, V = np.meshgrid(u, alpha_range) # u=x, v=y+z
        
        X = U
        Y = (V + shift) / 2
        Z = (V - shift) / 2
        
    elif orientation == 'xz': # x - z = shift
        u_label, v_label = "k_{x+z}", "ky"
        alpha_range = np.linspace(k_min * 2, k_max * 2, density)
        U, V = np.meshgrid(alpha_range, v) # u=x+z, v=y
        
        X = (U + shift) / 2
        Y = V
        Z = (U - shift) / 2
        
    elif orientation == 'xyz': # x + y + z = shift
        # Plane normal (1,1,1).
        # u along (1,-1,0), v along (1,1,-2)
        u_label, v_label = "u (perp to v, n)", "v (proj of z)"
        
        c = shift / 3.0
        # Need coverage
        span = (k_max - k_min) * 1.5
        ur = np.linspace(-span, span, density)
        vr = np.linspace(-span, span, density)
        U, V = np.meshgrid(ur, vr)
        
        X = c + U/np.sqrt(2) + V/np.sqrt(6)
        Y = c - U/np.sqrt(2) + V/np.sqrt(6)
        Z = c - 2*V/np.sqrt(6)
        
    else:
        print(f"Unknown orientation: {orientation}")
        return

    # 3. Interpolate
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1) # (N_pts, 3)
    vals = interp(pts) # (N_pts, nbands)
    
    nbands = vals.shape[-1]
    vals_grid = vals.reshape(X.shape + (nbands,)) # (density, density, nbands)
    
    # 4. Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Replace nans?
    # vals_grid = replace_zeros_with_nan(vals_grid) # Only if zeros are issue using util
    
    # Colormaps? cycling or same?
    # Let's use different cmaps or cycle
    cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    
    for b in range(nbands):
        E = vals_grid[:, :, b]
        
        # To avoid clutter, we might want to plot wireframe or semi-transparent surface
        # Surface with alpha
        surf = ax.plot_surface(U, V, E, cmap=cmaps[b % len(cmaps)], 
                               rstride=stride, cstride=stride, alpha=alpha, vmin=np.nanmin(E), vmax=np.nanmax(E))
        
        # Add a text label? Or legend?
        # Legend for surface is tricky.
    
    ax.set_xlabel(u_label)
    ax.set_ylabel(v_label)
    ax.set_zlabel('Eigenvalue E')
    
    if title is None:
        title = f"Slice {orientation}, shift={shift:.2f}"
    ax.set_title(title)
    
    # Add a colorbar for reference (maybe just one generic or none to avoid overlapping)
    # plt.colorbar(surf, ax=ax, label="Energy (last band)")
    
    plt.tight_layout()
    plt.show()

def plot_volumetric_cloud(eigenvalues_3d, kx_vals, ky_vals, kz_vals, band_index=0, 
                          opacity=0.1, surface_count=20, title=None, filename=None):
    """
    Create a volumetric rendering (cloud plot) of the 3D eigenvalue data using Plotly.
    
    Parameters:
    - eigenvalues_3d: 4D array [nkx, nky, nkz, dim]
    - kx_vals, ky_vals, kz_vals: 1D arrays defining the grid.
    - band_index: Index of the band to visualize.
    - opacity: Opacity of the volume (default 0.1).
    - surface_count: Number of iso-surfaces to render in the volume (default 20).
    - title: Plot title.
    - filename: Output filename (e.g., 'volume_plot.html'). If None, shows in browser/notebook.
    """
    
    # Extract band data
    # eigenvalues_3d is [x, y, z, dim]
    values = eigenvalues_3d[:, :, :, band_index]
    
    # Grid coordinates
    X, Y, Z = np.meshgrid(kx_vals, ky_vals, kz_vals, indexing='ij')
    
    if title is None:
        title = f"Volumetric Cloud: Band {band_index}"
        
    vmin, vmax = np.min(values), np.max(values)
        
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=vmin,
        isomax=vmax,
        opacity=opacity, # needs to be small to see through!
        surface_count=surface_count, # number of isosurfaces, 20 is good default
        caps=dict(x_show=False, y_show=False, z_show=False), # Hide the box caps
        colorscale='Viridis',
        colorbar=dict(title='Energy'),
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='kx',
            yaxis_title='ky',
            zaxis_title='kz'
        ),
        margin=dict(r=20, b=10, l=10, t=40)
    )
    
    if filename:
        fig.write_html(filename)
        print(f"Volumetric plot saved to {filename}")
    else:
        fig.show()
