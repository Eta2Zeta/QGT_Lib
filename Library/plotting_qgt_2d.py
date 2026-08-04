import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from .dimension_lib import coordinate_order_info
from .plotting_utils import overlay_hamiltonian_symmetry_path
from .utilities import replace_zeros_with_nan


DEFAULT_BERRY_HEATMAP_PERCENTILE = 99.0


def get_coordinate_axis_labels(order, *, backend):
    labels = coordinate_order_info(order)["coordinate_labels"][:2]

    if backend == "matplotlib":
        formatted = {
            "r": r"$r$",
            "phi": r"$\phi$ (rad)",
            "kx": r"$k_x$",
            "ky": r"$k_y$",
            "kz": r"$k_z$",
        }
    elif backend == "plotly":
        formatted = {
            "r": "r",
            "phi": "phi (rad)",
            "kx": "k<sub>x</sub>",
            "ky": "k<sub>y</sub>",
            "kz": "k<sub>z</sub>",
        }
    else:
        raise ValueError("backend must be 'matplotlib' or 'plotly'")

    return tuple(formatted[label] for label in labels)


__all__ = [
    "get_symmetric_plot_limits",
    "get_asymmetric_plot_limits",
    "get_coordinate_axis_labels",
    "plot_qgt_components_line",
    "plot_qgt_component_surfaces",
    "plot_qgt_eigenvalue_berry_trace_surfaces",
    "plot_qgt_component_heatmaps",
    "plot_qgt_eigenvalue_berry_trace_heatmaps",
    "plot_qgt_eigenvalue_berry_component_heatmaps",
    "plot_berry_irrep_projection_heatmaps",
]


# Other/shared functions


def get_symmetric_plot_limits(Z, limit=None, zlim_percentile=None):
    """
    Calculate symmetric z-limits for plotting, optionally constrained by
    an absolute limit or a percentile of the data. No margin is added.
    If clipping collapses the range, use the full finite-data extrema.
    """
    if Z is None:
        return None

    # Filter out NaNs and infs
    valid_Z = Z[np.isfinite(Z)]
    if len(valid_Z) == 0:
        return None

    zmin = np.min(valid_Z)
    zmax = np.max(valid_Z)

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

    # A percentile or cap can collapse the displayed range even when the full
    # field is nonzero. Fall back to the actual data rather than inventing a
    # constant plotting range.
    if not np.isfinite(abs_use) or abs_use <= 0.0:
        abs_use = abs_max

    return (-abs_use, abs_use)


def get_asymmetric_plot_limits(Z, limit=None, zlim_percentile=99):
    """
    Calculate asymmetric z-limits for plotting metric or eigenvalue data.
    The maximum limit is determined by the data's percentile (e.g. 99th),
    and the minimum is determined by the true minimum of the valid data.
    If clipping collapses or reverses the range, use the full data extrema.
    """
    if Z is None:
        return None

    # Filter out NaNs and infs
    valid_Z = Z[np.isfinite(Z)]
    if len(valid_Z) == 0:
        return None

    zmin = np.min(valid_Z)
    zmax = np.max(valid_Z)

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

    # A percentile or cap can collapse/reverse the range. Fall back to the
    # actual extrema rather than manufacturing a constant-width interval.
    if (
        not np.isfinite(actual_min)
        or not np.isfinite(actual_max)
        or actual_min >= actual_max
    ):
        return (zmin, zmax)

    return (actual_min, actual_max)


def plot_qgt_components_line(k_line, g_xx, g_yy, trace, angle_deg):
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


# Surface plots


def plot_qgt_component_surfaces(
    kx, ky, g_xx_array, g_xy_array, g_xy_array_imag, g_yy_array,
    stride_size=3, results_dir=None, save_fig=False,
    filename="qgt_component_surfaces.html", show=False, order="xyz",
):
    """
    Plot g_xx, g_xy, g_yx, and g_yy arrays as 3D surface plots in a single figure (Plotly).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import os

    x_label, y_label = get_coordinate_axis_labels(order, backend="plotly")

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
            'Numerical g<sub>xx</sub> (real part)',
            'Numerical g<sub>xy</sub> (real part)',
            'Numerical g<sub>xy</sub> (imaginary part)',
            'Numerical g<sub>yy</sub> (real part)'
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
            xaxis_title=x_label, yaxis_title=y_label,
            row=1, col=col
        )

    add_surf(g_xx_array, 1)
    add_surf(g_xy_array, 2, g_xy_real_min, g_xy_real_max)
    add_surf(g_xy_array_imag, 3, g_xy_imag_min, g_xy_imag_max)
    add_surf(g_yy_array, 4)
    
    fig.update_scenes(zaxis_title='g<sub>xx</sub>', row=1, col=1)
    fig.update_scenes(zaxis_title='g<sub>xy</sub> (real)', zaxis_range=[g_xy_real_min, g_xy_real_max], row=1, col=2)
    fig.update_scenes(zaxis_title='g<sub>xy</sub> (imag)', zaxis_range=[g_xy_imag_min, g_xy_imag_max], row=1, col=3)
    fig.update_scenes(zaxis_title='g<sub>yy</sub>', row=1, col=4)

    fig.update_layout(title_text='QGT Component Surfaces', height=500, width=1600, margin=dict(r=10, b=10, l=10, t=60))

    if save_fig and results_dir:
        filepath = os.path.join(results_dir, filename)
        fig.write_html(filepath, include_plotlyjs='cdn')
        print(f"Saved QGT components (HTML) to: {filepath}")
    
    if show:
        fig.show()


def plot_qgt_eigenvalue_berry_trace_surfaces(
    kx, ky,
    eigenvalues,              # shape: (Nk, Nk, Nb)
    g_xy_imag,                # shape: (Nk, Nk); Im(Q_xy)
    trace_array,              # shape: (Nk, Nk)
    eigenvalue_band=0,
    stride_size=2,
    convert_berry_from_imQ=True,  # If True, Ω = -2 * Im(Q_xy) by the standard convention Q_xy = g_xy - i Ω/2
    zlim_berry=None,
    zlim_trace=None,
    title="QGT: Eigenvalue, Berry Curvature, and Trace Surfaces",
    results_dir=None,
    save_fig=False,
    filename="qgt_eigenvalue_berry_trace_surfaces.html",
    show=False,
    order="xyz",
):
    """
    Make a 1×3 row of 3D surfaces for Eigenvalue, Berry curvature, and Trace using Plotly.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import os

    x_label, y_label = get_coordinate_axis_labels(order, backend="plotly")

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
        berry_title = "Berry Curvature Ω<sub>xy</sub>"
    else:
        Z_berry = replace_zeros_with_nan(g_xy_imag)
        berry_title = "Im(Q<sub>xy</sub>)"
        
    Z_trace = replace_zeros_with_nan(trace_array)

    zlim_eig = get_asymmetric_plot_limits(Z_eig, None, zlim_percentile=None)
    zlim_berry_tuple = get_symmetric_plot_limits(Z_berry, zlim_berry)
    zlim_trace_tuple = get_asymmetric_plot_limits(Z_trace, zlim_trace)

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=[
            f"Eigenvalue Band {eigenvalue_band+1}" if Z_eig is not None else "Eigenvalue (No Data)",
            berry_title,
            "Trace Tr(g)"
        ]
    )

    panels = [
        dict(Z=Z_eig,   cmap='Viridis', zlim=zlim_eig, title="Eigenvalue"),
        dict(Z=Z_berry, cmap='RdBu',    zlim=zlim_berry_tuple, title=berry_title),
        dict(Z=Z_trace, cmap='Plasma',  zlim=zlim_trace_tuple, title="Tr(g)"),
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
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=cfg['title'],
                zaxis_range=zlim_vals,
                row=1, col=col
            )

    fig.update_layout(title_text=title, height=500, width=1500, margin=dict(r=10, b=10, l=10, t=60))

    if save_fig and results_dir:
        filepath = os.path.join(results_dir, filename)
        fig.write_html(filepath, include_plotlyjs='cdn')
        print(f"Saved QGT surface plots (HTML) to: {filepath}")
        
    if show:
        fig.show()


# Heatmap plots


def plot_qgt_component_heatmaps(
    g_xx_array,
    g_yy_array,
    trace_array,
    k_max=10,
    order="xyz",
):
    """
    Plot g_xx, g_yy, and trace arrays as 2D heatmaps in a single figure.

    Parameters:
    - kx, ky: 2D arrays for the k-space grid.
    - g_xx_array, g_yy_array, trace_array: 2D arrays to be plotted as heatmaps.
    - k_max: Maximum k-value for the extent of the plot.
    """
    coordinate_info = coordinate_order_info(order)
    x_label, y_label = get_coordinate_axis_labels(order, backend="matplotlib")
    if coordinate_info["coordinate_system"] == "cylindrical":
        extent = (0.0, k_max, 0.0, 2.0 * np.pi)
        aspect = "auto"
    else:
        extent = (-k_max, k_max, -k_max, k_max)
        aspect = "equal"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot g_xx_array
    ax1 = axes[0]
    c1 = ax1.imshow(g_xx_array, extent=extent, origin='lower', cmap='viridis', aspect=aspect)
    ax1.set_title('$g_{xx}$ (Numerical)')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    fig.colorbar(c1, ax=ax1)

    # Plot g_yy_array
    ax2 = axes[1]
    c2 = ax2.imshow(g_yy_array, extent=extent, origin='lower', cmap='plasma', aspect=aspect)
    ax2.set_title('$g_{yy}$ (Numerical)')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    fig.colorbar(c2, ax=ax2)

    # Plot trace_array
    ax3 = axes[2]
    c3 = ax3.imshow(trace_array, extent=extent, origin='lower', cmap='plasma', aspect=aspect)
    ax3.set_title('Trace (Numerical)')
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    fig.colorbar(c3, ax=ax3)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_qgt_eigenvalue_berry_trace_heatmaps(
    kx, ky,
    eigenvalues,              # shape: (Nk, Nk, Nb)
    g_xy_imag,                # shape: (Nk, Nk); Im(Q_xy)
    trace_array,              # shape: (Nk, Nk)
    eigenvalue_band=0,
    convert_berry_from_imQ=True,  # If True, Ω = -2 * Im(Q_xy)
    cmaps=('viridis', 'coolwarm', 'plasma'),
    zlim_berry=None,
    zlim_percentile=DEFAULT_BERRY_HEATMAP_PERCENTILE,
    zlim_trace=None,
    title="QGT: Eigenvalue, Berry Curvature, and Trace Heatmaps",
    components="xy",
    results_dir=None,
    save_fig=False,
    hamiltonian=None,
    kk=0.0,
    sym_kz_threshold=0.02,
    order="xyz",
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
      zlim_berry        : optional absolute cap for the Berry color scale
      zlim_percentile   : percentile of |Omega| used for the symmetric Berry limits
      zlim_trace        : optional upper cap for the trace panel
      title             : figure title
      hamiltonian       : Hamiltonian providing get_sym_path(); None disables the overlay
      kk                : fixed kz value of the plotted kx-ky slice
      sym_kz_threshold  : fraction of the in-plane span used to select 3D path nodes
    """
    coordinate_info = coordinate_order_info(order)
    x_label, y_label = get_coordinate_axis_labels(order, backend="matplotlib")
    equal_aspect = coordinate_info["coordinate_system"] == "cartesian"

    # Extract data
    if eigenvalues is not None:
        Z_eig = replace_zeros_with_nan(eigenvalues[:, :, eigenvalue_band])
    else:
        Z_eig = None

    component_subscript = str(components).replace("_", "")
    if convert_berry_from_imQ:
        Z_berry = replace_zeros_with_nan(-2.0 * g_xy_imag)  # Ω = -2 Im(Q_xy)
        berry_label = rf"Berry Curvature $\Omega_{{{component_subscript}}}$"
    else:
        Z_berry = replace_zeros_with_nan(g_xy_imag)
        berry_label = rf"$\operatorname{{Im}}(Q_{{{component_subscript}}})$"
    Z_trace = replace_zeros_with_nan(trace_array)

    zlim_eig = get_asymmetric_plot_limits(Z_eig, None, zlim_percentile=None)
    zlim_berry_tuple = get_symmetric_plot_limits(
        Z_berry,
        zlim_berry,
        zlim_percentile,
    )
    zlim_trace_tuple = get_asymmetric_plot_limits(Z_trace, zlim_trace)

    # Figure & axes (1 row, 3 cols)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if title:
        fig.suptitle(title, fontsize=14)

    panels = [
        dict(Z=Z_eig, cmap=cmaps[0], title=f"Eigenvalue Band {eigenvalue_band+1}" if Z_eig is not None else "Eigenvalue (No Data)", zlim=zlim_eig, center_zero=False),
        dict(Z=Z_berry, cmap=cmaps[1], title=berry_label, zlim=zlim_berry_tuple, center_zero=True),
        dict(Z=Z_trace, cmap=cmaps[2], title=r"Trace $\operatorname{Tr}(g)$", zlim=zlim_trace_tuple, center_zero=True),
    ]

    for ax, cfg in zip(axes, panels):
        Z = cfg["Z"]
        cmap = cfg["cmap"]
        zlim = cfg["zlim"]
        vmin, vmax = zlim if zlim is not None else (None, None)

        if Z is not None:
            norm = None

            # Make 0 map to the *center color* (white-ish in many diverging maps)
            if cfg["center_zero"]:
                if vmin is not None and vmax is not None and vmin < 0 < vmax:
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

            im = ax.pcolormesh(kx, ky, Z, cmap=cmap, shading="auto",
                            norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            im.set_mouseover(True)

        ax.set_title(cfg['title'])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect("equal" if equal_aspect else "auto", adjustable="box")

    overlay_hamiltonian_symmetry_path(
        axes,
        kx,
        ky,
        hamiltonian,
        kk=kk,
        sym_kz_threshold=sym_kz_threshold,
    )
        
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    
    if save_fig and results_dir:
        import os
        filename = f"qgt_eigenvalue_berry_trace_heatmaps_band_{eigenvalue_band}_{components}.png"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved QGT heatmaps to: {filepath}")
    else:
        plt.show()
        
    plt.close()


def plot_qgt_eigenvalue_berry_component_heatmaps(
    kx, ky,
    eigenvalues,              # shape: (Nk, Nk, Nb)
    g_xy_imag,                # shape: (Nk, Nk); Im(Q_xy)
    g_xz_imag,                # shape: (Nk, Nk); Im(Q_xz)
    g_yz_imag,                # shape: (Nk, Nk); Im(Q_yz)
    eigenvalue_band=0,
    convert_berry_from_imQ=True,  # If True, Ω = -2 * Im(Q_ij)
    cmaps=('viridis', 'coolwarm', 'coolwarm', 'coolwarm'),
    zlim_berry=None,
    zlim_percentile=DEFAULT_BERRY_HEATMAP_PERCENTILE,
    title="QGT Eigenvalue and Berry Curvature Component Heatmaps",
    results_dir=None,
    save_fig=False,
    hamiltonian=None,
    kk=0.0,
    sym_kz_threshold=0.02,
    order="xyz",
):
    """
    Make a 1x4 row of 2D heatmaps for:
      - Eigenvalue band 'eigenvalue_band'
      - Berry curvature Ω_x = Ω_yz
      - Berry curvature Ω_y = Ω_zx
      - Berry curvature Ω_z = Ω_xy

    Args:
      kx, ky            : 2D grids (meshgrid)
      eigenvalues       : 3D array (Nk, Nk, Nb)
      g_xy_imag         : 2D array Im(Q_xy)
      g_xz_imag         : 2D array Im(Q_xz)
      g_yz_imag         : 2D array Im(Q_yz)
      eigenvalue_band   : which band to plot from eigenvalues
      convert_berry_from_imQ : if True, uses Ω_ij = -2 * Im(Q_ij)
      cmaps             : tuple of 4 colormaps
      zlim_berry        : max absolute limit for berry panels; None -> auto limit
      zlim_percentile   : limit automatically calculated as a percentile of the abs data
      title             : figure title
      hamiltonian       : Hamiltonian providing get_sym_path(); None disables the overlay
      kk                : fixed kz value of the plotted kx-ky slice
      sym_kz_threshold  : fraction of the in-plane span used to select 3D path nodes
    """
    coordinate_info = coordinate_order_info(order)
    x_label, y_label = get_coordinate_axis_labels(order, backend="matplotlib")
    equal_aspect = coordinate_info["coordinate_system"] == "cartesian"

    # Extract data
    if eigenvalues is not None:
        Z_eig = replace_zeros_with_nan(eigenvalues[:, :, eigenvalue_band])
    else:
        Z_eig = None

    if convert_berry_from_imQ:
        Z_berry_x = replace_zeros_with_nan(-2.0 * g_yz_imag)
        Z_berry_y = replace_zeros_with_nan(2.0 * g_xz_imag)
        Z_berry_z = replace_zeros_with_nan(-2.0 * g_xy_imag)
        berry_label_x = r"Berry Curvature $\Omega_x=\Omega_{yz}$"
        berry_label_y = r"Berry Curvature $\Omega_y=\Omega_{zx}$"
        berry_label_z = r"Berry Curvature $\Omega_z=\Omega_{xy}$"
    else:
        Z_berry_x = replace_zeros_with_nan(g_yz_imag)
        Z_berry_y = replace_zeros_with_nan(-g_xz_imag)
        Z_berry_z = replace_zeros_with_nan(g_xy_imag)
        berry_label_x = r"$\operatorname{Im}(Q_{yz})$"
        berry_label_y = r"$\operatorname{Im}(Q_{zx})=-\operatorname{Im}(Q_{xz})$"
        berry_label_z = r"$\operatorname{Im}(Q_{xy})$"

    zlim_eig = get_asymmetric_plot_limits(Z_eig, None, zlim_percentile=None)
    zlim_berry_x_tuple = get_symmetric_plot_limits(Z_berry_x, zlim_berry, zlim_percentile)
    zlim_berry_y_tuple = get_symmetric_plot_limits(Z_berry_y, zlim_berry, zlim_percentile)
    zlim_berry_z_tuple = get_symmetric_plot_limits(Z_berry_z, zlim_berry, zlim_percentile)

    # Figure & axes (1 row, 4 cols)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    if title:
        fig.suptitle(title, fontsize=14)

    panels = [
        dict(Z=Z_eig, cmap=cmaps[0], title=f"Eigenvalue Band {eigenvalue_band+1}" if Z_eig is not None else "Eigenvalue (No Data)", zlim=zlim_eig, center_zero=False),
        dict(Z=Z_berry_x, cmap=cmaps[1], title=berry_label_x, zlim=zlim_berry_x_tuple, center_zero=True),
        dict(Z=Z_berry_y, cmap=cmaps[2], title=berry_label_y, zlim=zlim_berry_y_tuple, center_zero=True),
        dict(Z=Z_berry_z, cmap=cmaps[3], title=berry_label_z, zlim=zlim_berry_z_tuple, center_zero=True),
    ]

    for ax, cfg in zip(axes, panels):
        Z = cfg["Z"]
        cmap = cfg["cmap"]
        zlim = cfg["zlim"]
        vmin, vmax = zlim if zlim is not None else (None, None)

        if Z is not None:
            norm = None

            # Make 0 map to the *center color* (white-ish in many diverging maps)
            if cfg["center_zero"]:
                if vmin is not None and vmax is not None and vmin < 0 < vmax:
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

            im = ax.pcolormesh(kx, ky, Z, cmap=cmap, shading="auto",
                            norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            im.set_mouseover(True)

        ax.set_title(cfg['title'])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect("equal" if equal_aspect else "auto", adjustable="box")

    overlay_hamiltonian_symmetry_path(
        axes,
        kx,
        ky,
        hamiltonian,
        kk=kk,
        sym_kz_threshold=sym_kz_threshold,
    )
        
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    
    if save_fig and results_dir:
        import os
        filename = f"qgt_eigenvalue_berry_component_heatmaps_band_{eigenvalue_band}.png"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved QGT Berry-component heatmaps to: {filepath}")
    else:
        plt.show()
        
    plt.close()


def plot_berry_irrep_projection_heatmaps(
    kx,
    ky,
    omega_x_original,
    omega_y_original,
    omega_x_projected,
    omega_y_projected,
    omega_x_residual,
    omega_y_residual,
    *,
    irrep="Eg",
    band_index=None,
    cmap="coolwarm",
    zlim_berry=None,
    zlim_residual=None,
    zlim_percentile=DEFAULT_BERRY_HEATMAP_PERCENTILE,
    residual_zlim_percentile=DEFAULT_BERRY_HEATMAP_PERCENTILE,
    title=None,
    hamiltonian=None,
    kk=0.0,
    sym_kz_threshold=0.02,
    results_dir=None,
    save_fig=False,
    filename=None,
):
    """Plot original, irrep-projected, and residual Berry x/y grids.

    Rows contain the original field, its irrep projection, and the residual
    ``Omega - P^irrep Omega``. Columns contain ``Omega_x`` and ``Omega_y``.
    Original and projected fields share a color scale within each column;
    residuals use independent scales so small projection errors remain visible.
    """
    kx = np.asarray(kx)
    ky = np.asarray(ky)
    if kx.shape != ky.shape or kx.ndim != 2:
        raise ValueError("kx and ky must be two-dimensional grids with matching shapes.")

    field_inputs = {
        "omega_x_original": omega_x_original,
        "omega_y_original": omega_y_original,
        "omega_x_projected": omega_x_projected,
        "omega_y_projected": omega_y_projected,
        "omega_x_residual": omega_x_residual,
        "omega_y_residual": omega_y_residual,
    }
    fields = {}
    for field_name, field in field_inputs.items():
        field = np.real_if_close(np.asarray(field))
        if np.iscomplexobj(field):
            raise ValueError(f"{field_name} must be real-valued.")
        if field.shape != kx.shape:
            raise ValueError(
                f"{field_name} has shape {field.shape}; expected {kx.shape}."
            )
        fields[field_name] = field

    x_main_limits = get_symmetric_plot_limits(
        np.stack(
            (
                fields["omega_x_original"],
                fields["omega_x_projected"],
            )
        ),
        zlim_berry,
        zlim_percentile,
    )
    y_main_limits = get_symmetric_plot_limits(
        np.stack(
            (
                fields["omega_y_original"],
                fields["omega_y_projected"],
            )
        ),
        zlim_berry,
        zlim_percentile,
    )
    x_residual_limits = get_symmetric_plot_limits(
        fields["omega_x_residual"],
        zlim_residual,
        residual_zlim_percentile,
    )
    y_residual_limits = get_symmetric_plot_limits(
        fields["omega_y_residual"],
        zlim_residual,
        residual_zlim_percentile,
    )

    def centered_norm(limits):
        if limits is None:
            return None
        abs_limit = max(abs(float(limits[0])), abs(float(limits[1])))
        if not np.isfinite(abs_limit) or abs_limit <= 0.0:
            return None
        return TwoSlopeNorm(vmin=-abs_limit, vcenter=0.0, vmax=abs_limit)

    irrep_math = r"E_g" if irrep == "Eg" else rf"\mathrm{{{irrep}}}"
    panels = (
        (
            fields["omega_x_original"],
            fields["omega_y_original"],
        ),
        (
            fields["omega_x_projected"],
            fields["omega_y_projected"],
        ),
        (
            fields["omega_x_residual"],
            fields["omega_y_residual"],
        ),
    )
    panel_titles = (
        (r"Original $\Omega_x$", r"Original $\Omega_y$"),
        (
            rf"$P^{{{irrep_math}}}\Omega_x$",
            rf"$P^{{{irrep_math}}}\Omega_y$",
        ),
        (
            rf"Residual $\Omega_x-P^{{{irrep_math}}}\Omega_x$",
            rf"Residual $\Omega_y-P^{{{irrep_math}}}\Omega_y$",
        ),
    )
    panel_limits = (
        (x_main_limits, y_main_limits),
        (x_main_limits, y_main_limits),
        (x_residual_limits, y_residual_limits),
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 14),
        sharex=True,
        sharey=True,
    )
    if title is None:
        band_text = "" if band_index is None else f", band {band_index}"
        title = rf"Berry-curvature projection onto ${irrep_math}${band_text}"
    if title:
        fig.suptitle(title, fontsize=16)

    for row in range(3):
        for column in range(2):
            ax = axes[row, column]
            field = np.ma.masked_invalid(panels[row][column])
            norm = centered_norm(panel_limits[row][column])
            mesh = ax.pcolormesh(
                kx,
                ky,
                field,
                cmap=cmap,
                shading="auto",
                norm=norm,
            )
            fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            mesh.set_mouseover(True)

            ax.set_title(panel_titles[row][column])
            ax.set_xlabel(r"$k_x$")
            ax.set_ylabel(r"$k_y$")
            ax.set_aspect("equal", adjustable="box")

    overlay_hamiltonian_symmetry_path(
        axes,
        kx,
        ky,
        hamiltonian,
        kk=kk,
        sym_kz_threshold=sym_kz_threshold,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = None
    if save_fig and results_dir:
        import os

        if filename is None:
            band_label = "unspecified" if band_index is None else str(band_index)
            filename = (
                f"berry_{irrep}_projection_heatmaps_band_{band_label}.png"
            )
        output_path = os.path.join(results_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved Berry-projection heatmaps to: {output_path}")
    else:
        plt.show()

    plt.close()
    return output_path
