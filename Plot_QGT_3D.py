import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
import numpy as np
from Library.plotting_lib_3d import plot_slice_stack
import sys
import trimesh
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from skimage.measure import marching_cubes
from Library.plotting_lib_3d import plot_volumetric_cloud

def plot_3d_qgt_slices(results_dir, quantity, component="xy", slice_plane="xy",
                       n_slices=1, include_endpoints=True):
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    # --- load meta ---
    meta_file = os.path.join(results_dir, "meta_info.pkl")
    if not os.path.exists(meta_file):
        meta_file = os.path.join(results_dir, "qgt_meta_info.pkl")
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file not found in {results_dir}")
        return

    print(f"Loading data from {results_dir}...")
    with open(meta_file, "rb") as f:
        meta_info = pickle.load(f)

    kx_vals = meta_info["kx_vals"]
    ky_vals = meta_info["ky_vals"]
    kz_vals = meta_info["kz_vals"]

    hamiltonian = meta_info.get("Hamiltonian_Obj")
    hamiltonian_name = hamiltonian.name if hamiltonian else "Hamiltonian"

    # --- helpers ---
    def load_arr(name):
        path = os.path.join(results_dir, f"{name}.npy")
        return np.load(path) if os.path.exists(path) else None

    def load_component_real(comp):
        # try common naming conventions for REAL part
        for nm in (f"g_{comp}", f"g_{comp}_real", f"g_{comp}_re"):
            arr = load_arr(nm)
            if arr is not None:
                return arr
        return None

    def load_component_imag(comp):
        # your files appear to be named like g_xy_imag
        return load_arr(f"g_{comp}_imag")

    comp = component.lower()
    if len(comp) != 2 or comp[0] not in "xyz" or comp[1] not in "xyz" or comp[0] == comp[1]:
        print(f"Error: component must be one of 'xy','xz','yz' (got '{component}')")
        return

    data_3d = None
    title_base = ""

    if quantity == "berry":
        gij_im = load_component_imag(comp)
        if gij_im is None:
            print(f"Error: Could not load g_{comp}_imag.npy in {results_dir}")
            return

        # Ω_ij = -2 Im g_ij  (your prior code used a different "vector + magnitude" thing)
        data_3d = -2.0 * gij_im
        title_base = f"{hamiltonian_name} Berry Curvature Ω_{comp}"

    elif quantity == "metric":
        gij_re = load_component_real(comp)
        if gij_re is None:
            print(f"Error: Could not load real metric component for g_{comp}. "
                  f"Tried g_{comp}.npy, g_{comp}_real.npy, g_{comp}_re.npy")
            return

        data_3d = gij_re
        title_base = f"{hamiltonian_name} Metric g_{comp}"

    else:
        print(f"Error: Unknown quantity '{quantity}'. Use 'metric' or 'berry'.")
        return

    # handle bands
    n_bands = data_3d.shape[0]

    for b0 in range(n_bands):
        band_label = b0 + 1
        title_b = f"{title_base} (Band {band_label})"
        
        plot_slice_stack(
            data_3d[b0], kx_vals, ky_vals, kz_vals,
            plane=slice_plane,
            n_slices=n_slices,
            include_endpoints=include_endpoints,
            title=title_b
        )

def plot_3d_qgt_all_slices_dynamic(results_dir, quantity="berry", component="xy", slice_plane="xy",
                                   bands=None, cmap="RdBu_r", z_limit=None, debug=False, sigma_multiplier=1):
    """
    Plots all 2D slices along a specified slice_plane of a 3D dataset, wrapped in
    an interactive Plotly HTML file with a slider to navigate between slices.
    """
    import plotly.graph_objects as go
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    # --- load meta ---
    meta_file = os.path.join(results_dir, "meta_info.pkl")
    if not os.path.exists(meta_file):
        meta_file = os.path.join(results_dir, "qgt_meta_info.pkl")
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file not found in {results_dir}")
        return

    print(f"Loading data from {results_dir}...")
    with open(meta_file, "rb") as f:
        meta_info = pickle.load(f)

    kx_vals = meta_info["kx_vals"]
    ky_vals = meta_info["ky_vals"]
    kz_vals = meta_info["kz_vals"]

    hamiltonian = meta_info.get("Hamiltonian_Obj")
    hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")

    # --- helpers ---
    def load_arr(name):
        path = os.path.join(results_dir, f"{name}.npy")
        return np.load(path) if os.path.exists(path) else None

    def load_component_real(comp):
        for nm in (f"g_{comp}", f"g_{comp}_real", f"g_{comp}_re"):
            arr = load_arr(nm)
            if arr is not None:
                return arr
        return None

    def load_component_imag(comp):
        return load_arr(f"g_{comp}_imag")

    comp = component.lower()
    if len(comp) != 2 or comp[0] not in "xyz" or comp[1] not in "xyz" or comp[0] == comp[1]:
        print(f"Error: component must be one of 'xy','xz','yz' (got '{component}')")
        return

    data_3d = None
    title_base = ""

    if quantity == "berry":
        gij_im = load_component_imag(comp)
        if gij_im is None:
            print(f"Error: Could not load g_{comp}_imag.npy in {results_dir}")
            return
        data_3d = -2.0 * gij_im
        title_base = f"{hamiltonian_name} Berry Curvature Ω_{comp}"

    elif quantity == "metric":
        gij_re = load_component_real(comp)
        if gij_re is None:
            print(f"Error: Could not load real metric component for g_{comp}")
            return
        data_3d = gij_re
        title_base = f"{hamiltonian_name} Metric g_{comp}"

    else:
        print(f"Error: Unknown quantity '{quantity}'")
        return

    import math
    from plotly.subplots import make_subplots

    n_bands = data_3d.shape[0]
    if bands is None:
        bands_0 = list(range(n_bands))
    else:
        bands_0 = [int(b) - 1 for b in bands if 0 <= int(b) - 1 < n_bands]

    if len(bands_0) == 0:
        print("No bands to plot.")
        return

    # Slice configuration
    if slice_plane == "xy":
        x_ax, y_ax = kx_vals, ky_vals
        z_ax = kz_vals
        xlabel, ylabel, zlabel = "kx", "ky", "kz"
        get_slice = lambda b0, idx: data_3d[b0][:, :, idx].T
    elif slice_plane == "yz":
        x_ax, y_ax = ky_vals, kz_vals
        z_ax = kx_vals
        xlabel, ylabel, zlabel = "ky", "kz", "kx"
        get_slice = lambda b0, idx: data_3d[b0][idx, :, :].T
    elif slice_plane == "xz":
        x_ax, y_ax = kx_vals, kz_vals
        z_ax = ky_vals
        xlabel, ylabel, zlabel = "kx", "kz", "ky"
        get_slice = lambda b0, idx: data_3d[b0][:, idx, :].T
    else:
        print(f"Error: Unknown slice plane {slice_plane}")
        return

    num_plots = len(bands_0)
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    subplot_titles = [f"Band {b0 + 1}" for b0 in bands_0]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, horizontal_spacing=0.08, vertical_spacing=0.1)

    # Compute limits per band
    band_limits = {}
    
    if debug:
        import matplotlib.pyplot as plt

    for b0 in bands_0:
        band_label = b0 + 1
        vol_data = data_3d[b0]
        valid = vol_data[np.isfinite(vol_data)]
        
        if z_limit:
            vmin, vmax = z_limit
        else:
            if len(valid) == 0:
                vmin, vmax = -1, 1
            else:
                p_low, p_high = np.percentile(valid, [5, 95])
                bulk_valid = valid[(valid >= p_low) & (valid <= p_high)]
                if len(bulk_valid) > 0:
                    avg = np.mean(bulk_valid)
                    std = np.std(bulk_valid)
                else:
                    avg = np.mean(valid)
                    std = np.std(valid)
                if std == 0:
                    std = 1.0
                
                filtered = valid[(valid >= avg - sigma_multiplier * std) & (valid <= avg + sigma_multiplier * std)]
                if len(filtered) == 0:
                    vmin, vmax = -1.0, 1.0
                else:
                    vmin = np.min(filtered)
                    vmax = np.max(filtered)
                    abs_max = max(abs(vmin), abs(vmax))
                    if abs_max == 0:
                        abs_max = 1.0
                    vmin, vmax = -abs_max, abs_max
                    
        band_limits[b0] = (vmin, vmax, avg, std)
        
        if debug:
            plt.figure(figsize=(10, 6))
            plot_valid = valid[(valid > avg - 5 * std) & (valid < avg + 5 * std)] if len(valid) > 0 else []
            if len(plot_valid) > 0:
                plt.hist(plot_valid, bins=100, alpha=0.7, color='blue', edgecolor='black')
                plt.axvline(avg, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg:.4g}')
                plt.axvline(vmin, color='green', linestyle='dashed', linewidth=2, label=f'-2 Sigma: {vmin:.4g}')
                plt.axvline(vmax, color='orange', linestyle='dashed', linewidth=2, label=f'+2 Sigma: {vmax:.4g}')
                plt.title(f'1D Data Distribution for Band {band_label} ({quantity}_{component})')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.legend()
                
                debug_fname = os.path.join(results_dir, f"debug_dist_{quantity}_{component}_band_{band_label}.png")
                plt.savefig(debug_fname, dpi=150)
                print(f"Saved debug distribution plot to: {debug_fname}")
            plt.close()

    steps = []
    
    for i, z_val in enumerate(z_ax):
        for b_idx, b0 in enumerate(bands_0):
            r = (b_idx // cols) + 1
            c = (b_idx % cols) + 1
            
            slice_2d = get_slice(b0, i)
            trace_name = f"Band {b0+1} | {zlabel}={z_val:.3g}"
            
            vmin, vmax, avg, std = band_limits[b0]
            
            # Position colorbars intelligently on the right side
            cb_x = 1.02 + 0.08 * (c - 1)
            cb_y = 1.0 - ((r - 1) + 0.5) / rows
            cb_len = min(0.9 / rows, 0.45)
            
            fig.add_trace(go.Heatmap(
                z=slice_2d, x=x_ax, y=y_ax,
                colorscale=cmap,
                zmin=vmin,
                zmax=vmax,
                visible=(i == 0),
                name=trace_name,
                colorbar=dict(
                    title=f"B{b0+1}",
                    thickness=15,
                    len=cb_len,
                    y=cb_y,
                    yanchor="middle",
                    x=cb_x
                ),
                hovertemplate=f"x: %{{x}}<br>y: %{{y}}<br>Value: %{{z}}<extra></extra>"
            ), row=r, col=c)

        step = dict(
            method="update",
            args=[{"visible": [False] * (len(z_ax) * num_plots)},
                  {"title": f"{title_base} Grid<br>Slice at {zlabel} = {z_val:.4f}"}],
            label=f"{z_val:.2f}"
        )
        
        start_idx = i * num_plots
        for j in range(num_plots):
            step["args"][0]["visible"][start_idx + j] = True
            
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": f"{zlabel}: "},
        pad={"t": 50},
        steps=steps
    )]

    # Load parameters from meta.json if it exists
    meta_json_path = os.path.join(results_dir, "meta.json")
    param_text = ""
    if os.path.exists(meta_json_path):
        import json
        try:
            with open(meta_json_path, "r") as f:
                meta_json = json.load(f)
            params = meta_json.get("hamiltonian_params", {})
            a0_val = params.get("A0", 0.0)
            ignore_keys = []
            if float(a0_val) == 0.0:
                ignore_keys = ["A0", "analytic_magnus", "magnus_order", "omega", "polarization"]
            
            lines = []
            for k, v in params.items():
                if k not in ignore_keys:
                    lines.append(f"{k} = {v}")
            if lines:
                param_text = "<b>Hamiltonian Params:</b><br>" + "<br>".join(lines)
        except Exception as e:
            print(f"Warning: Could not parse parameters from meta.json: {e}")

    # Build annotations
    annotations = []
    plot_width = 400 * cols + 150 + 50 * cols
    if param_text:
        cb_max_x = 1.02 + 0.08 * (cols - 1)
        annotations.append(dict(
            text=param_text,
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=cb_max_x + 0.15,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            bordercolor='black',
            borderwidth=1,
            borderpad=10,
            bgcolor='white',
            font=dict(size=12)
        ))
        plot_width += 250
        
    fig.update_layout(
        sliders=sliders,
        title=f"{title_base} Grid<br>Slice at {zlabel} = {z_ax[0]:.4f}",
        width=plot_width,
        height=400 * rows + 150,
        margin=dict(r=400) if param_text else None,
        annotations=annotations
    )
    
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    
    # Tie scaleanchor precisely to corresponding xaxis
    for i in range(1, num_plots + 1):
        r = ((i - 1) // cols) + 1
        c = ((i - 1) % cols) + 1
        x_anchor = f"x{i}" if i > 1 else "x"
        fig.update_yaxes(scaleanchor=x_anchor, scaleratio=1, row=r, col=c)

    fname = os.path.join(results_dir, f"dynamic_slices_{quantity}_{component}_plane_{slice_plane}_all_bands.html")
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=False)
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Dynamic Slices - All Bands</title>
        <style>
            body, html {{
                height: 100%;
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: white;
            }}
        </style>
    </head>
    <body>
        <div>
            {html_str}
        </div>
    </body>
    </html>
    """
    with open(fname, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Saved interactive grid to: {fname}")


def compute_equal_volume_levels(data_3d, count):
    """
    Computes `count` isosurface levels such that an approximately equal
    number of data points fall into the bins separated by these levels.
    """
    # Flatten the array and remove NaNs/Infs for percentile calculation
    valid_data = data_3d[np.isfinite(data_3d)]
    if len(valid_data) == 0:
        return np.array([])
        
    # We want `count` levels, which divides the data into `count + 1` bins.
    # Therefore, we need percentiles at fractions 1/(count+1), 2/(count+1), ... count/(count+1)
    percentiles = np.linspace(0, 100, count + 2)[1:-1]
    levels = np.percentile(valid_data, percentiles)
    return levels


def compute_dynamic_opacity(num_levels, min_op=0.3, max_op=0.8):
    """
    Computes a V-shaped or U-shaped opacity sequence where the extreme levels get max_op
    and the intermediate levels dip down towards min_op.
    """
    if num_levels == 0:
        return []
    if num_levels == 1:
        return [max_op]
    
    opacities = []
    for i in range(num_levels):
        x = i / (num_levels - 1)
        # abs(1.0 - 2*x) gives exactly 1 at x=0, 0 at x=0.5, 1 at x=1
        op = min_op + (max_op - min_op) * abs(1.0 - 2.0 * x)
        opacities.append(op)
    
    return opacities


def generate_volumetric_filename(results_dir, hamiltonian_name, quantity, plane, band_label, levels, kx_range=None, ky_range=None, kz_range=None):
    """
    Generates a descriptive filename for the volumetric cloud HTML plot.
    Format: {results_dir}/{hamiltonian_name}_{quantity}_{plane}_band{band_label}_levels_L1_L2.html
    where levels are formatted to 3 significant figures. Includes ranges if specified.
    """
    # Format levels to 3 significant figures
    formatted_levels = []
    for lvl in levels:
        if lvl == 0:
            formatted_levels.append("0.00")
        else:
            # Format to 3 sig figs
            formatted_levels.append(f"{lvl:.3g}")
            
    levels_str = "-".join(formatted_levels)
    fname = f"{hamiltonian_name}_{quantity}_{plane}_band{band_label}_levels_{levels_str}"
    
    if kx_range is not None:
        fname += f"_kx{kx_range[0]:.2f}-{kx_range[1]:.2f}"
    if ky_range is not None:
        fname += f"_ky{ky_range[0]:.2f}-{ky_range[1]:.2f}"
    if kz_range is not None:
        fname += f"_kz{kz_range[0]:.2f}-{kz_range[1]:.2f}"
        
    fname += ".html"
    
    # Sanitize filename
    fname = fname.replace(" ", "_").replace("/", "_")
    return os.path.join(results_dir, fname)


def plot_3d_qgt(
    results_dir,
    plane="all",
    quantity="berry",
    count=5,
    bands=None,
    levels=None,
    percentile_count=10,      # <-- NEW: segments for dividing percentiles to print
    kx_range=None,            # <-- NEW: tuple (min, max) for filtering kx
    ky_range=None,            # <-- NEW: tuple (min, max) for filtering ky
    kz_range=None,            # <-- NEW: tuple (min, max) for filtering kz
    *,
    export_dir=None,          # <-- NEW: where to save meshes; None => don't export
    export_fmt="ply",         # "ply", "stl", "obj", "glb" (depends on trimesh support)
    export_step_size=1,       # marching cubes step_size for export (1 = highest fidelity)
    plot_step_size=1,         # step_size for matplotlib plotting (bigger = faster)
    show=True,                # whether to show matplotlib figure
):
    """
    Plot 3D QGT isosurfaces for selected bands from STACKED arrays (n_bands, nx, ny, nz).

    If bands is None -> plot ALL bands (1..n_bands).
    Band indices are 1-based if you pass a list, e.g. [1,2,3].

    If levels is provided, it uses those exact levels.
    If levels is None and count is provided, it computes `count` levels such that
    an equal volume of data points falls between each isosurface (equal percentiles).
    """
    
    if plane is None or plane == "all":
        for p in ["xy", "yz", "xz"]:
            print(f"\n======================================================\n"
                  f"Plotting for plane {p}...\n"
                  f"======================================================")
            plot_3d_qgt(
                results_dir=results_dir,
                plane=p,
                quantity=quantity,
                count=count,
                bands=bands,
                levels=levels,
                percentile_count=percentile_count,
                kx_range=kx_range,
                ky_range=ky_range,
                kz_range=kz_range,
                export_dir=export_dir,
                export_fmt=export_fmt,
                export_step_size=export_step_size,
                plot_step_size=plot_step_size,
                show=show
            )
        return

    def save_isosurface_mesh(data_3d, level, kx_vals, ky_vals, kz_vals, out_path, step_size=1):
        # verts come back in index coordinates (i,j,k)
        verts_ijk, faces, normals, values = marching_cubes(
            data_3d, level=float(level), step_size=int(step_size)
        )

        # map index -> k coordinate (linear interpolation)
        i = verts_ijk[:, 0]
        j = verts_ijk[:, 1]
        k = verts_ijk[:, 2]
        kx = np.interp(i, np.arange(len(kx_vals)), kx_vals)
        ky = np.interp(j, np.arange(len(ky_vals)), ky_vals)
        kz = np.interp(k, np.arange(len(kz_vals)), kz_vals)
        verts_xyz = np.column_stack([kx, ky, kz])

        mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=False)
        mesh.export(out_path)

    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    # ---- meta ----
    meta_file = os.path.join(results_dir, "meta_info.pkl")

    print(f"Loading data from {results_dir}...")
    with open(meta_file, "rb") as f:
        meta_info = pickle.load(f)

    kx_vals_full = meta_info["kx_vals"]
    ky_vals_full = meta_info["ky_vals"]
    kz_vals_full = meta_info["kz_vals"]
    hamiltonian = meta_info.get("Hamiltonian_Obj", None)
    hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")

    # ---- choose components ----
    if plane == "xy":
        comp_11_name = "g_xx"
        comp_22_name = "g_yy"
        comp_12_imag_name = "g_xy_imag"
    elif plane == "yz":
        comp_11_name = "g_yy"
        comp_22_name = "g_zz"
        comp_12_imag_name = "g_yz_imag"
    elif plane == "xz":
        comp_11_name = "g_xx"
        comp_22_name = "g_zz"
        comp_12_imag_name = "g_xz_imag"
    else:
        print(f"Error: Unknown plane '{plane}'. Use 'xy', 'yz', or 'xz'.")
        return

    # ---- filtering indices based on ranges ----
    def get_indices(vals, r):
        if r is None:
            return np.arange(len(vals))
        return np.where((vals >= r[0]) & (vals <= r[1]))[0]

    ix_keep = get_indices(kx_vals_full, kx_range)
    iy_keep = get_indices(ky_vals_full, ky_range)
    iz_keep = get_indices(kz_vals_full, kz_range)

    if len(ix_keep) == 0 or len(iy_keep) == 0 or len(iz_keep) == 0:
        print("Error: Range filtering resulted in an empty grid along at least one dimension.")
        return

    # Sliced coordinate arrays
    kx_vals = kx_vals_full[ix_keep]
    ky_vals = ky_vals_full[iy_keep]
    kz_vals = kz_vals_full[iz_keep]

    # ---- load stacked arrays ----
    def load_arr(name):
        path = os.path.join(results_dir, f"{name}.npy")
        if not os.path.exists(path):
            print(f"Error: File {name}.npy not found in {results_dir}")
            return None
        return np.load(path)

    if quantity == "metric":
        val_11 = load_arr(comp_11_name)
        val_22 = load_arr(comp_22_name)
        if val_11 is None or val_22 is None:
            return
        if val_11.ndim != 4 or val_22.ndim != 4:
            raise ValueError(
                f"Expected stacked arrays (n_bands, nx, ny, nz). "
                f"Got {comp_11_name}.ndim={val_11.ndim}, {comp_22_name}.ndim={val_22.ndim}."
            )
        # Apply filter
        val_11 = val_11[:, ix_keep, :][:, :, iy_keep, :][:, :, :, iz_keep]
        val_22 = val_22[:, ix_keep, :][:, :, iy_keep, :][:, :, :, iz_keep]
        
        n_bands = val_11.shape[0]
        title_base = f"{hamiltonian_name} Metric Trace ({plane})"
    elif quantity == "berry":
        val_12_imag = load_arr(comp_12_imag_name)
        if val_12_imag is None:
            return
        if val_12_imag.ndim != 4:
            raise ValueError(
                f"Expected stacked array (n_bands, nx, ny, nz). "
                f"Got {comp_12_imag_name}.ndim={val_12_imag.ndim}."
            )
        # Apply filter
        val_12_imag = val_12_imag[:, ix_keep, :][:, :, iy_keep, :][:, :, :, iz_keep]

        n_bands = val_12_imag.shape[0]
        title_base = f"{hamiltonian_name} Berry Curvature ({plane})"
    else:
        print(f"Error: Unknown quantity '{quantity}'. Use 'metric' or 'berry'.")
        return

    # ---- bands: None => ALL (1..n_bands) ----
    if bands is None:
        bands_0 = list(range(n_bands))  # 0-based all bands
    else:
        bands_0 = []
        for b1 in list(bands):
            if not isinstance(b1, (int, np.integer)):
                raise TypeError(f"Band indices must be integers (1-based). Got {b1} ({type(b1)}).")
            b0 = int(b1) - 1
            if b0 < 0 or b0 >= n_bands:
                print(f"Skipping band {b1} (out of range: valid is 1..{n_bands})")
                continue
            bands_0.append(b0)

        if len(bands_0) == 0:
            print("No valid bands to plot after range checking.")
            return

    # ensure export directory exists if requested
    if export_dir is not None:
        os.makedirs(export_dir, exist_ok=True)

    # ---- plot each band in its own figure ----
    for b0 in bands_0:
        band_label = b0 + 1  # 1-based display

        if quantity == "metric":
            data_3d = val_11[b0] + val_22[b0]
        else:  # berry
            data_3d = -2.0 * val_12_imag[b0]

        dmin = float(np.min(data_3d))
        dmax = float(np.max(data_3d))

        # Calculate and print percentile segments
        valid_ds = data_3d[np.isfinite(data_3d)]
        if len(valid_ds) > 0:
            pct_levels = np.linspace(0, 100, percentile_count + 1)
            pct_vals = np.percentile(valid_ds, pct_levels)
            print(f"Data divided into {percentile_count} segments ({len(pct_vals)} boundary values):")
            for p_num, p_val in zip(pct_levels, pct_vals):
                print(f"  {p_num:5.1f}th percentile: {p_val:.6g}")
        
        # levels_use: do NOT mutate `levels` argument; make a per-band local array
        if levels is not None:
            levels_use = np.array(levels, dtype=float)
        else:
            # Use equal-volume percentiles if no exact levels are provided
            print(f"Computing {count} equal-volume isosurface levels for Band {band_label}...")
            levels_use = compute_equal_volume_levels(data_3d, count)

        # ensure sorted (so surfaces are sensible)
        levels_use = np.array(levels_use, dtype=float)
        levels_use.sort()
        L = len(levels_use)
        if L < 1:
            print("No levels to plot.")
            continue

        # 1) Pick L colors evenly from RdBu_r *by index*
        base = plt.get_cmap("RdBu_r")
        colors = base(np.linspace(0.0, 1.0, L))   # evenly spaced colors, independent of values

        # --- optionally make a matplotlib figure ---
        if show:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

        print(f"\nBand {band_label}: data range [{dmin:.6g}, {dmax:.6g}]")
        
        for i, level in enumerate(levels_use):
            if level < dmin or level > dmax:
                print(f"  - skipping level {level:.6g} (out of band data range)")
                continue

            # (A) EXPORT MESH (fast later)
            if export_dir is not None:
                fname = f"{hamiltonian_name}_{quantity}_{plane}_band{band_label:02d}_lvl{level:+.6g}.{export_fmt}"
                fname = fname.replace(" ", "_").replace("/", "_")
                out_path = os.path.join(export_dir, fname)
                print(f"  - exporting mesh at level {level:.6g} -> {out_path}")
                save_isosurface_mesh(
                    data_3d, level, kx_vals, ky_vals, kz_vals,
                    out_path, step_size=export_step_size
                )


        # Generate specific HTML filename 
        html_filename = generate_volumetric_filename(
            results_dir=results_dir, 
            hamiltonian_name=hamiltonian_name, 
            quantity=quantity, 
            plane=plane, 
            band_label=band_label, 
            levels=levels_use,
            kx_range=kx_range,
            ky_range=ky_range,
            kz_range=kz_range
        )

        # pass same levels and colors to the volume
        hex_colors = [mcolors.to_hex(c) for c in colors]
        opacity_seq = compute_dynamic_opacity(L, min_op=0.1, max_op=0.5)
        
        # Format levels for display and title
        formatted_levels = []
        for lvl in levels_use:
            if lvl == 0:
                formatted_levels.append("0.00")
            else:
                formatted_levels.append(f"{lvl:.3g}")
        levels_str = ", ".join(formatted_levels)
        
        print(f"  - Plotting levels: {levels_str}")
        print(f"  - Calculated dynamic opacities: {[round(op, 3) for op in opacity_seq]}")
        
        plot_title = f"{title_base} (Band {band_label})<br><sup>Levels: {levels_str}</sup>"
        
        plot_volumetric_cloud(data_3d, kx_vals, ky_vals, kz_vals, 
                                      opacity=0.1, levels=levels_use, color_sequence=hex_colors, 
                                      opacity_sequence=opacity_seq,
                                      title=plot_title, 
                                      filename=html_filename, stride=plot_step_size)

if __name__ == "__main__":
    # Example usage as requested
    
    # Using the latest result directory found
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # base_results_path = os.path.join(current_dir, "results/3D_QGT_results/RuO2Hamiltonian")

    # base_results_path = os.path.join(current_dir, "results/3D_QGT_results/AltermagnetHamiltonian")

    base_results_path = os.path.join(current_dir, "results/3D_QGT_results/gWaveAltermagnetHamiltonian")
    
    # Placeholder for the specific dataset, using one found in list_dir
    latest_dataset = "data_set_6"
    
    results_dir = os.path.join(base_results_path, latest_dataset)

    print(f"Running 3D QGT Slice Plot on: {results_dir}")
    # plot_3d_qgt_slices(results_dir=results_dir, quantity='berry', component='xy', slice_plane="xy", n_slices=3)
    
    # Run the new dynamic slicing visualization
    print(f"Generating dynamic all-slices HTML...")
    # plot_3d_qgt_all_slices_dynamic(results_dir=results_dir, quantity='berry', component='yz', slice_plane="xy", bands=[1, 2, 3, 4], debug=False)
    
    # plot_3d_qgt(results_dir=results_dir, quantity='berry', plane='xy', levels=[0.3, -0.3], kz_range = (0, 0.9*np.pi))

    # plot_3d_qgt(results_dir=results_dir, quantity='berry', plane='xy', count=4, bands=[1, 2, 3, 4], plot_step_size = 1, kz_range = (0, 1*np.pi))
    # plot_3d_qgt(results_dir=results_dir, quantity='berry', count=2, bands=[1, 2, 3, 4], plot_step_size = 1, kz_range = (-3.14, 3.14))
