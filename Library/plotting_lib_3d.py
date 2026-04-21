import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go
from .utilities import replace_zeros_with_nan
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from itertools import product
from matplotlib.lines import Line2D


def plot_degeneracy_3d(kx_vals, ky_vals, kz_vals, eigenvalues, threshold=0.01, title="Degeneracy Map (3D)",
                       opacity=0.3, results_dir=None, save_fig=False, filename=None):
    """
    Plot a 3D scatter map where points indicate k-points with band degeneracies.
    
    Parameters:
    - kx_vals, ky_vals, kz_vals: 1D arrays defining the grid.
    - eigenvalues: 4D array of eigenvalues [nkx, nky, nkz, nbands].
    - threshold: Relative threshold (fraction of energy range) or absolute threshold to consider bands degenerate.
    - title: Plot title.
    - results_dir: Output directory to save the file.
    - save_fig: Whether to save the HTML file locally in the `results_dir`.
    - filename: Output filename, defaults to `degeneracy_3d.html`.
    """
    import plotly.graph_objects as go
    
    nkx, nky, nkz, nbands = eigenvalues.shape
    
    print("Calculating 3D degeneracy map...")
    
    # Calculate degeneracy counts
    # A vectorized approach:
    # 1. Sort eigenvalues along the band axis
    sorted_evals = np.sort(eigenvalues, axis=-1)
    # 2. Compute differences between adjacent bands
    diffs = np.diff(sorted_evals, axis=-1)
    
    # Using an absolute threshold for simplicity, matching the 2D version logic roughly
    # Alternatively, you can calculate the max gap to make it relative as in the 2D script:
    # max_gap = np.max(diffs)
    # actual_threshold = threshold * max_gap if max_gap > 0 else threshold
    actual_threshold = threshold 
    
    # Count how many gaps are smaller than the threshold at each k-point
    degeneracy_map = np.sum(diffs < actual_threshold, axis=-1)
    
    # Find indices where degeneracy > 0
    deg_indices = np.where(degeneracy_map > 0)
    
    if len(deg_indices[0]) == 0:
        print("No degeneracies found at the given threshold.")
        return
        
    X, Y, Z = np.meshgrid(kx_vals, ky_vals, kz_vals, indexing='ij')
    
    # Extract coordinates and counts for the degenerate points
    x_deg = X[deg_indices]
    y_deg = Y[deg_indices]
    z_deg = Z[deg_indices]
    counts = degeneracy_map[deg_indices]
    
    # Plot using Plotly 3D scatter
    fig = go.Figure()
    
    unique_counts = np.unique(counts)
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    base_cmap = plt.get_cmap('tab10')
    
    for i, uc in enumerate(unique_counts):
        mask = (counts == uc)
        color = mcolors.to_hex(base_cmap(i % 10))
        
        fig.add_trace(go.Scatter3d(
            x=x_deg[mask],
            y=y_deg[mask],
            z=z_deg[mask],
            mode='markers',
            marker=dict(
                size=4,
                color=color,
                opacity=opacity,
            ),
            name=f"Degeneracy: {uc}",
            text=[f"Degeneracy: {uc}"] * np.sum(mask),
            hoverinfo='text+x+y+z'
        ))
    
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='kx',
            yaxis_title='ky',
            zaxis_title='kz'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    if save_fig and results_dir:
        import os
        if not filename:
            filename = 'degeneracy_3d.html'
        filepath = os.path.join(results_dir, filename)
        fig.write_html(filepath, include_plotlyjs='cdn')
        print(f"Saved 3D Degeneracy map to: {filepath}")
    elif filename:
        fig.write_html(filename, include_plotlyjs='cdn')
        print(f"Saved 3D Degeneracy map to: {filename}")
    else:
        fig.show()


def plot_grid_slice(
    eigenvalues, orientation, shift,
    kx_vals, ky_vals, kz_vals,
    bands=None,
    kind="pcolormesh",   # "pcolormesh" or "imshow"
    cmap="viridis",
    per_slice_colorbar=True,
    title=None,
):
    """
    Plot an axis-aligned slice of eigenvalues as-is (NO interpolation).

    eigenvalues: array [nkx, nky, nkz, nbands]
    orientation:
      'x' -> fix kx ~ shift, plot over (ky,kz)
      'y' -> fix ky ~ shift, plot over (kx,kz)
      'z' -> fix kz ~ shift, plot over (kx,ky)
    shift: the coordinate value where you want the slice (snaps to nearest grid plane)

    bands: list of band indices to plot (default: all)
    kind: 'pcolormesh' (uses coords) or 'imshow' (fast, robust)
    per_slice_colorbar: each subplot gets its own colorbar (auto-normalized)
    """
    orientation = orientation.lower()
    kx_vals = np.asarray(kx_vals)
    ky_vals = np.asarray(ky_vals)
    kz_vals = np.asarray(kz_vals)

    if eigenvalues.ndim != 4:
        raise ValueError(f"eigenvalues must be 4D [nkx,nky,nkz,nbands], got shape {eigenvalues.shape}")

    nkx, nky, nkz, nbands = eigenvalues.shape
    if bands is None:
        bands = list(range(nbands))
    else:
        bands = list(bands)

    # --- choose nearest plane + slice ---
    if orientation == "x":
        ix = int(np.argmin(np.abs(kx_vals - shift)))
        coord_fixed = kx_vals[ix]
        slice_3d = eigenvalues[ix, :, :, :]      # [nky, nkz, nbands]
        X, Y = ky_vals, kz_vals
        xlabel, ylabel = "$k_y$", "$k_z$"
        fixed_label = f"$k_x$ = {coord_fixed:.3g} (i={ix})"

        # for plotting convenience we want [len(Y), len(X)]
        # slice_2d for one band is [nky, nkz] -> transpose to [nkz, nky]
        get_E = lambda b: slice_3d[:, :, b].T

    elif orientation == "y":
        iy = int(np.argmin(np.abs(ky_vals - shift)))
        coord_fixed = ky_vals[iy]
        slice_3d = eigenvalues[:, iy, :, :]      # [nkx, nkz, nbands]
        X, Y = kx_vals, kz_vals
        xlabel, ylabel = "$k_x$", "$k_z$"
        fixed_label = f"$k_y$ = {coord_fixed:.3g} (j={iy})"
        get_E = lambda b: slice_3d[:, :, b].T    # [nkz, nkx]

    elif orientation == "z":
        iz = int(np.argmin(np.abs(kz_vals - shift)))
        coord_fixed = kz_vals[iz]
        slice_3d = eigenvalues[:, :, iz, :]      # [nkx, nky, nbands]
        X, Y = kx_vals, ky_vals
        xlabel, ylabel = "$k_x$", "$k_y$"
        fixed_label = f"$k_z$ = {coord_fixed:.3g} (k={iz})"
        get_E = lambda b: slice_3d[:, :, b].T    # [nky, nkx]

    else:
        raise ValueError("orientation must be 'x', 'y', or 'z' for grid slicing (no interpolation).")

    # --- layout ---
    n = len(bands)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.2*nrows), squeeze=False)
    axes = axes.ravel()

    # --- plot each band ---
    for ax, b in zip(axes, bands):
        E = get_E(b)

        if kind == "imshow":
            im = ax.imshow(
                E, origin="lower",
                extent=[X[0], X[-1], Y[0], Y[-1]],
                cmap=cmap, aspect="auto"
            )
        else:
            # pcolormesh with coords-as-centers is usually OK with shading="auto"
            im = ax.pcolormesh(X, Y, E, shading="auto", cmap=cmap)

        ax.set_title(f"band {b}\n{fixed_label}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if per_slice_colorbar:
            fig.colorbar(im, ax=ax, shrink=0.85)

    # turn off unused
    for ax in axes[len(bands):]:
        ax.axis("off")

    if title is None:
        title = f"Grid slice ({orientation})"
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_arbitrary_slice_no_interp(
    eigenvalues, orientation, shift,
    kx_vals, ky_vals, kz_vals,
    title=None, stride=1, alpha=0.9,
    cmaps=None, filename=None, show=True,
    results_dir=None, save_fig=False
):
    """
    Plot axis-aligned slices as 3D surfaces for each band, using ONLY native grid data, using Plotly.
    NO interpolation. eigenvalues shape: [nkx, nky, nkz, nbands].

    orientation:
      'x' -> fix kx ~ shift, surface over (ky,kz)
      'y' -> fix ky ~ shift, surface over (kx,kz)
      'z' -> fix kz ~ shift, surface over (kx,ky)
    """
    import plotly.graph_objects as go

    orientation = orientation.lower()
    kx_vals = np.asarray(kx_vals)
    ky_vals = np.asarray(ky_vals)
    kz_vals = np.asarray(kz_vals)

    if eigenvalues.ndim != 4:
        raise ValueError(f"eigenvalues must be 4D [nkx,nky,nkz,nbands], got {eigenvalues.shape}")

    nkx, nky, nkz, nbands = eigenvalues.shape
    if cmaps is None:
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

    # --- choose nearest grid plane and build U,V,E for plotting ---
    if orientation == "x":
        ix = int(np.argmin(np.abs(kx_vals - shift)))
        x0 = float(kx_vals[ix])

        slice_3d = eigenvalues[ix, :, :, :]
        U, V = np.meshgrid(ky_vals, kz_vals, indexing="xy")
        get_E = lambda b: slice_3d[:, :, b].T

        u_label, v_label = "ky", "kz"
        fixed_label = f"kx = {x0:.4g} (ix={ix})"

    elif orientation == "y":
        iy = int(np.argmin(np.abs(ky_vals - shift)))
        y0 = float(ky_vals[iy])

        slice_3d = eigenvalues[:, iy, :, :]
        U, V = np.meshgrid(kx_vals, kz_vals, indexing="xy")
        get_E = lambda b: slice_3d[:, :, b].T

        u_label, v_label = "kx", "kz"
        fixed_label = f"ky = {y0:.4g} (iy={iy})"

    elif orientation == "z":
        iz = int(np.argmin(np.abs(kz_vals - shift)))
        z0 = float(kz_vals[iz])

        slice_3d = eigenvalues[:, :, iz, :]
        U, V = np.meshgrid(kx_vals, ky_vals, indexing="xy")
        get_E = lambda b: slice_3d[:, :, b].T

        u_label, v_label = "kx", "ky"
        fixed_label = f"kz = {z0:.4g} (iz={iz})"

    else:
        raise ValueError("No-interp version supports only orientation 'x', 'y', or 'z'.")

    # Downsample if stride given
    if stride > 1:
        U = U[::stride, ::stride]
        V = V[::stride, ::stride]

    fig = go.Figure()

    for b in range(nbands):
        E = get_E(b)
        if stride > 1:
            E = E[::stride, ::stride]

        surface = go.Surface(
            x=U, y=V, z=E,
            colorscale=cmaps[b % len(cmaps)],
            opacity=alpha,
            showscale=False,
            name=f"Band {b}",
            showlegend=True
        )
        fig.add_trace(surface)

    if title is None:
        title = f"Grid slice ({orientation}) | {fixed_label}"
        
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=u_label,
            yaxis_title=v_label,
            zaxis_title="Eigenvalue E"
        ),
        margin=dict(r=20, b=10, l=10, t=40)
    )

    if save_fig and results_dir:
        import os
        if not filename:
            shift_str = f"{shift:.1f}".replace('.', 'p')
            filename = f"slice_{orientation}_shift_{shift_str}.html"
        filepath = os.path.join(results_dir, filename)
        fig.write_html(filepath, include_plotlyjs='cdn')
        print(f"Slice plot saved to {filepath}")
    elif filename:
        fig.write_html(filename, include_plotlyjs='cdn')
        print(f"Slice plot saved to {filename}")
        
    if show and not (save_fig and results_dir):
        fig.show()

def plot_arbitrary_slice_interpolated(eigenvalues, orientation, shift, kx_vals, ky_vals, kz_vals, 
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

def plot_volumetric_cloud(values, kx_vals, ky_vals, kz_vals,
                          opacity=0.1, surface_count=20, levels=None, color_sequence=None, 
                          opacity_sequence=None, title=None, filename=None, stride=1,
                          results_dir=None, save_fig=False):
    """
    Create a volumetric rendering (cloud plot) of the 3D eigenvalue data using Plotly.
    
    Parameters:
    - values: 3D array of values to plot [nkx, nky, nkz]
    - kx_vals, ky_vals, kz_vals: 1D arrays defining the grid.
    - opacity: Opacity of the volume (default 0.1).
    - surface_count: Number of iso-surfaces to render in the volume (default 20), used if levels is None.
    - levels: (Optional) List of specific scalar real values at which to plot isosurfaces.
    - color_sequence: (Optional) List of colors (e.g. hex or string) corresponding to each level in `levels`.
    - opacity_sequence: (Optional) List of opacities corresponding to each level in `levels`.
    - title: Plot title.
    - filename: Output filename (e.g., 'volume_plot.html'). If None, shows in browser/notebook.
    - stride: Integer stride to downsample the array data for faster rendering (default 1).
    - results_dir: Output directory to save the file.
    - save_fig: Whether to save the HTML file locally in the `results_dir`.
    """
    
    # Extract band data
    # eigenvalues_3d is [x, y, z, dim]
    
    # Grid coordinates
    X, Y, Z = np.meshgrid(kx_vals, ky_vals, kz_vals, indexing='ij')
    
    # Downsample for faster loading and smaller sizes
    if stride > 1:
        X = X[::stride, ::stride, ::stride]
        Y = Y[::stride, ::stride, ::stride]
        Z = Z[::stride, ::stride, ::stride]
        values = values[::stride, ::stride, ::stride]
    
    if title is None:
        title = f"Volumetric Cloud"
        
    vmin, vmax = np.min(values), np.max(values)
    
    go_kwargs = dict(
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
        colorbar=dict(title='Value'),
    )
    
    if levels is not None:
        if color_sequence is None:
            # Fall back to a default colorscale mapping if no sequence provided
            base = plt.get_cmap("RdBu_r")
            L = len(levels)
            base_colors = base(np.linspace(0.0, 1.0, L))
            color_sequence = [mcolors.to_hex(c) for c in base_colors]
            
        data = []
        for i, level in enumerate(levels):
            col = color_sequence[i % len(color_sequence)]
            # Custom colorscale that maps strictly to this color
            cscale = [[0, col], [1, col]]
            
            # Determine opacity for this level
            if opacity_sequence is not None:
                current_opacity = opacity_sequence[i % len(opacity_sequence)]
            else:
                current_opacity = opacity
            
            iso = go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=values.flatten(),
                isomin=level,
                isomax=level,
                surface_count=1,
                opacity=current_opacity,
                caps=dict(x_show=False, y_show=False, z_show=False),
                colorscale=cscale,
                showscale=False,
                name=f'Value: {level:.3g}',
                showlegend=True
            )
            data.append(iso)
        fig = go.Figure(data=data)
        _fig_created = True
    
    if 'fig' not in locals() or not _fig_created:
        fig = go.Figure(data=go.Volume(**go_kwargs))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='kx',
            yaxis_title='ky',
            zaxis_title='kz'
        ),
        margin=dict(r=20, b=10, l=10, t=40)
    )
    
    if save_fig and results_dir:
        import os
        if not filename:
            filename = 'volume_plot.html'
        filepath = os.path.join(results_dir, filename)
        fig.write_html(filepath, include_plotlyjs='cdn')
        print(f"Volumetric plot saved to {filepath}")
    elif filename:
        fig.write_html(filename, include_plotlyjs='cdn')
        print(f"Volumetric plot saved to {filename}")
    else:
        fig.show()


def plot_slice_stack(
    data_3d, kx_vals, ky_vals, kz_vals,
    plane="xy", n_slices=10, include_endpoints=True,
    cmap="bwr", title="Slice stack"
):
    """
    plane:
      'xy' -> slices at various kz (perp to z)
      'xz' -> slices at various ky (perp to y)
      'yz' -> slices at various kx (perp to x)

    Behavior:
      - each slice uses its OWN normalization (auto from that slice)
      - each slice has its OWN colorbar
    """
    plane = plane.lower()

    if plane == "xy":
        coord = np.asarray(kz_vals, dtype=float)
        xvals = np.asarray(kx_vals, dtype=float)
        yvals = np.asarray(ky_vals, dtype=float)
        get_slice = lambda i: data_3d[:, :, i].T
        label = r"$k_z$"
        xlabel, ylabel = "$k_x$", "$k_y$"

    elif plane == "xz":
        coord = np.asarray(ky_vals, dtype=float)
        xvals = np.asarray(kx_vals, dtype=float)
        yvals = np.asarray(kz_vals, dtype=float)
        get_slice = lambda i: data_3d[:, i, :].T
        label = r"$k_y$"
        xlabel, ylabel = "$k_x$", "$k_z$"

    elif plane == "yz":
        coord = np.asarray(kx_vals, dtype=float)
        xvals = np.asarray(ky_vals, dtype=float)
        yvals = np.asarray(kz_vals, dtype=float)
        get_slice = lambda i: data_3d[i, :, :].T
        label = r"$k_x$"
        xlabel, ylabel = "$k_y$", "$k_z$"

    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    # --- pick targets / indices ---
    if coord.size == 0:
        raise ValueError("Empty coord array; check k arrays.")

    c_min = float(coord.min())
    c_max = float(coord.max())

    if n_slices <= 1:
        if include_endpoints:
            t0, t1 = c_min, c_max
        else:
            t0, t1 = (float(coord[1]), float(coord[-2])) if coord.size >= 3 else (c_min, c_max)
        targets = np.array([(t0 + t1) / 2.0], dtype=float)
    else:
        if include_endpoints:
            t0, t1 = c_min, c_max
        else:
            t0, t1 = (float(coord[1]), float(coord[-2])) if coord.size >= 3 else (c_min, c_max)
            if not (t0 < t1):
                t0, t1 = c_min, c_max
        targets = np.linspace(t0, t1, n_slices)

    idxs = [int(np.argmin(np.abs(coord - t))) for t in targets]
    idxs = sorted(set(idxs))
    n = len(idxs)
    if n == 0:
        raise RuntimeError("No slice indices selected (idxs empty).")

    # --- layout ---
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.2*nrows), squeeze=False)
    axes = axes.ravel()

    # --- plot each slice with its OWN normalization + OWN colorbar ---
    for ax, i in zip(axes, idxs):
        sl = get_slice(i)

        # auto-normalize per slice: do NOT pass vmin/vmax
        im = ax.pcolormesh(xvals, yvals, sl, shading="auto", cmap=cmap)

        ax.set_title(f"{label} = {coord[i]:.3g}  (i={i})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # individual colorbar for this subplot
        fig.colorbar(im, ax=ax, shrink=0.85)

    # turn off unused axes
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"{title} ({plane.upper()} stack)")
    plt.tight_layout()
    plt.show()

def plot_degeneracy_on_path_3d(k_points, eigenvalues, threshold=0.01, title="Degeneracy along Path", results_dir=None, save_fig=False):
    """
    Plot the k-path colored by the degree of band degeneracy.

    Automatically handles both 2D k-paths (N, 2) and 3D k-paths (N, 3):
      - 2D paths → matplotlib 2D scatter saved as PNG.
      - 3D paths → interactive Plotly 3D scatter saved as HTML.

    Parameters
    ----------
    k_points : ndarray, shape (N, 2) or (N, 3)
        k-coordinates along the path.
    eigenvalues : ndarray, shape (N, num_bands)
        Eigenvalues at each k-point.
    threshold : float
        Relative energy-gap threshold to consider bands degenerate.
    title : str
        Plot title.
    results_dir : str, optional
        Directory to save the plot.
    save_fig : bool
        If True, saves the figure to results_dir.
    """
    import os

    k_points = np.asarray(k_points)
    k_dim = k_points.shape[1]   # 2 or 3

    num_points = k_points.shape[0]
    degeneracies = np.zeros(num_points, dtype=int)

    # Calculate max degeneracy for each k-point
    for i in range(num_points):
        evals = np.sort(eigenvalues[i])
        diffs = np.diff(evals)
        if len(diffs) > 0:
            max_gap = np.max(diffs)
            current_threshold = max(threshold * max_gap, 1e-10)
            degeneracies[i] = int(np.sum(diffs <= current_threshold))

    color_map = {
        0: 'blue',
        1: 'green',
        2: 'red',
        3: 'black',
    }

    # ---------------------------------------------------------------
    # 2D path → matplotlib
    # ---------------------------------------------------------------
    if k_dim == 2:
        fig, ax = plt.subplots(figsize=(8, 6))

        for deg in sorted(set(degeneracies)):
            idx = np.where(degeneracies == deg)[0]
            color = color_map.get(deg, 'black')
            label = 'Non-degenerate' if deg == 0 else f'{deg + 1}-fold degenerate'
            ax.scatter(k_points[idx, 0], k_points[idx, 1],
                       c=color, s=20, label=label, zorder=3)

        # Draw connecting lines coloured by degeneracy
        for i in range(num_points - 1):
            deg = degeneracies[i]
            if degeneracies[i + 1] == deg:
                color = color_map.get(deg, 'black')
                ax.plot([k_points[i, 0], k_points[i + 1, 0]],
                        [k_points[i, 1], k_points[i + 1, 1]],
                        color=color, linewidth=1.5, zorder=2)

        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.set_aspect('equal', adjustable='datalim')
        plt.tight_layout()

        if save_fig and results_dir:
            filepath = os.path.join(results_dir, "degeneracy_on_path.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved degeneracy path plot to: {filepath}")
        else:
            plt.show()

        plt.close(fig)

    # ---------------------------------------------------------------
    # 3D path → Plotly
    # ---------------------------------------------------------------
    else:
        import plotly.graph_objects as go

        fig = go.Figure()

        # Faint BZ reference box
        box_min = np.array([0, 0, 0])
        box_max = np.array([np.pi, np.pi, np.pi])

        for y in [box_min[1], box_max[1]]:
            for z in [box_min[2], box_max[2]]:
                fig.add_trace(go.Scatter3d(
                    x=[box_min[0], box_max[0]], y=[y, y], z=[z, z],
                    mode='lines', line=dict(color='grey', width=2), opacity=0.3, showlegend=False
                ))
        for x in [box_min[0], box_max[0]]:
            for z in [box_min[2], box_max[2]]:
                fig.add_trace(go.Scatter3d(
                    x=[x, x], y=[box_min[1], box_max[1]], z=[z, z],
                    mode='lines', line=dict(color='grey', width=2), opacity=0.3, showlegend=False
                ))
        for x in [box_min[0], box_max[0]]:
            for y in [box_min[1], box_max[1]]:
                fig.add_trace(go.Scatter3d(
                    x=[x, x], y=[y, y], z=[box_min[2], box_max[2]],
                    mode='lines', line=dict(color='grey', width=2), opacity=0.3, showlegend=False
                ))

        # Points grouped by degeneracy
        for deg in sorted(set(degeneracies)):
            idx = np.where(degeneracies == deg)[0]
            color = color_map.get(deg, 'black')
            label = 'Non-degenerate (1)' if deg == 0 else f'{deg}-fold'
            fig.add_trace(go.Scatter3d(
                x=k_points[idx, 0], y=k_points[idx, 1], z=k_points[idx, 2],
                mode='markers',
                marker=dict(color=color, size=4),
                name=label
            ))

        # Connecting lines
        deg_to_lines = {deg: {'x': [], 'y': [], 'z': []} for deg in set(degeneracies)}
        for i in range(num_points - 1):
            deg = degeneracies[i]
            if degeneracies[i + 1] == deg:
                deg_to_lines[deg]['x'].extend([k_points[i, 0], k_points[i + 1, 0], None])
                deg_to_lines[deg]['y'].extend([k_points[i, 1], k_points[i + 1, 1], None])
                deg_to_lines[deg]['z'].extend([k_points[i, 2], k_points[i + 1, 2], None])

        for deg, lines in deg_to_lines.items():
            if lines['x']:
                fig.add_trace(go.Scatter3d(
                    x=lines['x'], y=lines['y'], z=lines['z'],
                    mode='lines',
                    line=dict(color=color_map.get(deg, 'black'), width=4),
                    showlegend=False
                ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='kx',
                yaxis_title='ky',
                zaxis_title='kz',
                aspectmode='data'
            )
        )

        if save_fig and results_dir:
            filepath = os.path.join(results_dir, "degeneracy_on_path_3d.html")
            fig.write_html(filepath)
            print(f"Saved interactive degeneracy 3D plot to: {filepath}")
        else:
            fig.show()
