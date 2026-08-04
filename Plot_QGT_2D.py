
import sys
import os
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt

from Library.dimension_lib import coordinate_order_info

# Import plotting functions
from Library.plotting_qgt_2d import (
    plot_qgt_component_surfaces,
    plot_qgt_eigenvalue_berry_trace_surfaces,
    plot_qgt_eigenvalue_berry_trace_heatmaps,
    plot_qgt_component_heatmaps,
    plot_qgt_eigenvalue_berry_component_heatmaps,
    get_coordinate_axis_labels,
    get_symmetric_plot_limits,
)

def plot_qgt_from_directory(
    target_dir,
    *,
    zlim_berry=1000,
    zlim_percentile=99,
):
    """
    Loads QGT results and metadata from the specified directory and generates plots.
    """
    print(f"Loading data from: {target_dir}")

    # Define file paths
    meta_info_file = os.path.join(target_dir, "meta_info.pkl") # try loading qgt_meta_info.pkl first if it exists? 

    if os.path.exists(os.path.join(target_dir, "qgt_meta_info.pkl")):
        meta_path = os.path.join(target_dir, "qgt_meta_info.pkl")
    elif os.path.exists(os.path.join(target_dir, "meta_info.pkl")):
        meta_path = os.path.join(target_dir, "meta_info.pkl")
    else:
        print(f"Error: No meta info file found in {target_dir}")
        sys.exit(1)

    # Load Metadata
    try:
        with open(meta_path, "rb") as f:
            meta_info = pickle.load(f)
        print("Loaded metadata.")
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        sys.exit(1)

    # Extract info from metadata
    try:
        kx = meta_info.get("kx", meta_info.get("ki", None))
        ky = meta_info.get("ky", meta_info.get("kj", None))
        if kx is None or ky is None:
            raise KeyError("Neither 'kx'/'ky' nor 'ki'/'kj' found in metadata.")
        Hamiltonian_Obj = meta_info.get("Hamiltonian_Obj", None)
        kk = meta_info.get("kk", 0.0)
        order = meta_info.get("order", "xyz")
        mesh_spacing = meta_info.get("mesh_spacing", "Unknown")
        print(f"Grid loaded. Mesh: {mesh_spacing}")
        if Hamiltonian_Obj:
            print(f"Hamiltonian: {Hamiltonian_Obj.name}")
    except KeyError as e:
        print(f"Error: Missing key in metadata: {e}")
        sys.exit(1)

    # Load QGT Data Arrays
    # Try loading from the standard names
    try:
        g_xx = np.load(os.path.join(target_dir, "g_xx.npy"))
        g_xy_real = np.load(os.path.join(target_dir, "g_xy_real.npy"))
        g_xy_imag = np.load(os.path.join(target_dir, "g_xy_imag.npy"))
        g_yy = np.load(os.path.join(target_dir, "g_yy.npy"))
        trace = np.load(os.path.join(target_dir, "trace.npy"))
        print("Loaded QGT arrays.")
    except FileNotFoundError as e:
        print(f"Error: Missing QGT data file: {e}")
        sys.exit(1)

    # Load Eigenvalues (Optional but needed for combined plots)
    eigenvalues = None
    eig_path = os.path.join(target_dir, "eigenvalues.npy")
    if os.path.exists(eig_path):
        eigenvalues = np.load(eig_path)
        print("Loaded eigenvalues.")
    else:
        print("Warning: 'eigenvalues.npy' not found. Some plots will be skipped.")

    # Plot Parameters

    # Check if we have stacked bands
    num_bands = 1
    if g_xx.ndim == 3:
        num_bands = g_xx.shape[0]

    for b in range(num_bands):
        # Extract band slice if 3D, else use arrays as is
        b_g_xx = g_xx[b] if g_xx.ndim == 3 else g_xx
        b_g_xy_real = g_xy_real[b] if g_xy_real.ndim == 3 else g_xy_real
        b_g_xy_imag = g_xy_imag[b] if g_xy_imag.ndim == 3 else g_xy_imag
        b_g_yy = g_yy[b] if g_yy.ndim == 3 else g_yy
        b_trace = trace[b] if trace.ndim == 3 else trace

        print(f"--- Generating Plots for Band {b} ---")

        # 1. QGT component surfaces
        print(f"Plotting QGT component surfaces for band {b}...")
        plot_qgt_component_surfaces(
            kx, ky, b_g_xx, b_g_xy_real, b_g_xy_imag, b_g_yy,
            stride_size=2,
            results_dir=target_dir,
            save_fig=True,
            filename=f"qgt_component_surfaces_band_{b}.html",
            show=False,
            order=order,
        )

        # 2. Combined Plots
        print(f"Plotting QGT eigenvalue/Berry/trace surfaces for band {b}...")
        plot_qgt_eigenvalue_berry_trace_surfaces(
            kx, ky, eigenvalues, b_g_xy_imag, b_trace,
            eigenvalue_band=b,

            title=f"Surface Results: {Hamiltonian_Obj.name if Hamiltonian_Obj else ''} (Band {b})",
            results_dir=target_dir,
            save_fig=True,
            filename=f"qgt_eigenvalue_berry_trace_surfaces_band_{b}.html",
            show=False,
            order=order,
        )

        print(f"Plotting QGT eigenvalue/Berry/trace heatmaps for band {b}...")
        plot_qgt_eigenvalue_berry_trace_heatmaps(
            kx, ky, eigenvalues, b_g_xy_imag, b_trace,
            eigenvalue_band=b,
            zlim_berry=zlim_berry,
            zlim_percentile=zlim_percentile,
            title=f"2D Results: {Hamiltonian_Obj.name if Hamiltonian_Obj else ''} (Band {b})",
            results_dir=target_dir,
            save_fig=True,
            hamiltonian=Hamiltonian_Obj if order == "xyz" else None,
            kk=kk,
            order=order,
        )
    print("Done.")

def plot_all_2d_berries_from_directory(target_dir):
    """
    Loads QGT results and optionally plotted the new 2D 1x4 horizontal plot 
    for Eigenvalues, and all three Berry curvature components.
    """
    print(f"Loading data for all-Berry plots from: {target_dir}")

    # Metadata
    meta_path = os.path.join(target_dir, "meta_info.pkl")
    if os.path.exists(os.path.join(target_dir, "qgt_meta_info.pkl")):
        meta_path = os.path.join(target_dir, "qgt_meta_info.pkl")

    if not os.path.exists(meta_path):
        print(f"Error: No meta info file found in {target_dir}")
        return

    try:
        with open(meta_path, "rb") as f:
            meta_info = pickle.load(f)
        kx = meta_info.get("kx", meta_info.get("ki", None))
        ky = meta_info.get("ky", meta_info.get("kj", None))
        if kx is None or ky is None:
            raise KeyError("Neither 'kx'/'ky' nor 'ki'/'kj' found in metadata.")
        Hamiltonian_Obj = meta_info.get("Hamiltonian_Obj", None)
        kk = meta_info.get("kk", 0.0)
        order = meta_info.get("order", "xyz")
    except Exception as e:
        print(f"Failed to load metadata/grid: {e}")
        return

    # Arrays
    try:
        g_xy_imag = np.load(os.path.join(target_dir, "g_xy_imag.npy"))
        g_xz_imag = np.load(os.path.join(target_dir, "g_xz_imag.npy"))
        g_yz_imag = np.load(os.path.join(target_dir, "g_yz_imag.npy"))
    except FileNotFoundError as e:
        print(f"Error: Missing QGT data file: {e}")
        return

    # Eigenvalues
    eigenvalues = None
    eig_path = os.path.join(target_dir, "eigenvalues.npy")
    if os.path.exists(eig_path):
        eigenvalues = np.load(eig_path)
    
    # Parameters matches Calc_QGT
    band = 1
    z_cutoff = 1000
    z_percentile = 99

    print("Plotting all-Berry 2D heatmaps...")

    if g_xy_imag.ndim > 2:
        num_bands = g_xy_imag.shape[0]
        for b in range(num_bands):
            print(f"Plotting all-Berry 2D for band {b}...")
            plot_qgt_eigenvalue_berry_component_heatmaps(
                kx, ky, eigenvalues, 
                g_xy_imag[b], g_xz_imag[b], g_yz_imag[b],
                eigenvalue_band=b,
                zlim_berry=z_cutoff,
                zlim_percentile=z_percentile,
                results_dir=target_dir,
                save_fig=True,
                hamiltonian=Hamiltonian_Obj if order == "xyz" else None,
                kk=kk,
                order=order,
            )
    else:
        print("Plotting all-Berry 2D for single band...")
        plot_qgt_eigenvalue_berry_component_heatmaps(
            kx, ky, eigenvalues, 
            g_xy_imag, g_xz_imag, g_yz_imag,
            eigenvalue_band=band,
            zlim_berry=z_cutoff,
            zlim_percentile=z_percentile,
            results_dir=target_dir,
            save_fig=True,
            hamiltonian=Hamiltonian_Obj if order == "xyz" else None,
            kk=kk,
            order=order,
        )

    print("Done all-Berry plots.")


def _load_2d_qgt_dataset(target_dir):
    """
    Load the standard 2D QGT result folder used by Calc_QGT_2D.py.
    """
    if os.path.exists(os.path.join(target_dir, "qgt_meta_info.pkl")):
        meta_path = os.path.join(target_dir, "qgt_meta_info.pkl")
    else:
        meta_path = os.path.join(target_dir, "meta_info.pkl")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No meta info file found in {target_dir}")

    with open(meta_path, "rb") as f:
        meta_info = pickle.load(f)

    kx = meta_info.get("kx", meta_info.get("ki", None))
    ky = meta_info.get("ky", meta_info.get("kj", None))
    if kx is None or ky is None:
        raise KeyError("Neither 'kx'/'ky' nor 'ki'/'kj' found in metadata.")

    data = {
        "target_dir": target_dir,
        "meta_info": meta_info,
        "kx": np.asarray(kx),
        "ky": np.asarray(ky),
        "g_xx": np.load(os.path.join(target_dir, "g_xx.npy")),
        "g_xy_real": np.load(os.path.join(target_dir, "g_xy_real.npy")),
        "g_xy_imag": np.load(os.path.join(target_dir, "g_xy_imag.npy")),
        "g_yy": np.load(os.path.join(target_dir, "g_yy.npy")),
        "trace": np.load(os.path.join(target_dir, "trace.npy")),
        "eigenvalues": None,
        "sym_line": None,
        "qgt_sym_paths": [],
    }

    eig_path = os.path.join(target_dir, "eigenvalues.npy")
    if os.path.exists(eig_path):
        data["eigenvalues"] = np.load(eig_path)

    sym_line_path = os.path.join(target_dir, "eigenvalues_sym_line.npz")
    if os.path.exists(sym_line_path):
        with np.load(sym_line_path, allow_pickle=True) as sym_data:
            data["sym_line"] = {
                "eigenvalues": np.asarray(sym_data["eigenvalues"]),
                "k_dist": np.asarray(sym_data["k_dist"]),
                "node_indices": np.asarray(sym_data["node_indices"], dtype=int),
                "path_labels": [str(label) for label in sym_data["path_labels"]],
                "k_path": np.asarray(sym_data["k_path"]) if "k_path" in sym_data else None,
                "path_points": np.asarray(sym_data["path_points"]) if "path_points" in sym_data else None,
            }

    for qgt_sym_path in sorted(glob.glob(os.path.join(target_dir, "qgt_sym_path_*.npz"))):
        with np.load(qgt_sym_path, allow_pickle=True) as sym_data:
            data["qgt_sym_paths"].append({
                "source_path": qgt_sym_path,
                "bands": np.asarray(sym_data["bands"], dtype=int),
                "k_path": np.asarray(sym_data["k_path"]),
                "k_dist": np.asarray(sym_data["k_dist"]),
                "node_indices": np.asarray(sym_data["node_indices"], dtype=int),
                "path_labels": [str(label) for label in sym_data["path_labels"]],
                "path_points": np.asarray(sym_data["path_points"]),
                "g_xx": np.asarray(sym_data["g_xx"]),
                "g_yy": np.asarray(sym_data["g_yy"]),
                "g_zz": np.asarray(sym_data["g_zz"]),
                "g_xy_real": np.asarray(sym_data["g_xy_real"]),
                "g_xy_imag": np.asarray(sym_data["g_xy_imag"]),
                "g_xz_real": np.asarray(sym_data["g_xz_real"]),
                "g_xz_imag": np.asarray(sym_data["g_xz_imag"]),
                "g_yz_real": np.asarray(sym_data["g_yz_real"]),
                "g_yz_imag": np.asarray(sym_data["g_yz_imag"]),
                "trace": np.asarray(sym_data["trace"]),
            })

    return data


def _qgt_quantity_spec(quantity):
    quantity_map = {
        "g_xx": ("g_xx", r"$g_{xx}$"),
        "g_yy": ("g_yy", r"$g_{yy}$"),
        "g_xy_real": ("g_xy_real", r"$\mathrm{Re}\ g_{xy}$"),
        "g_xy_imag": ("g_xy_imag", r"$\mathrm{Im}\ g_{xy}$"),
        "trace": ("trace", r"$\mathrm{Tr}\ g$"),
        "berry": ("g_xy_imag", r"$\Omega_{xy}$"),
        "berry_curvature": ("g_xy_imag", r"$\Omega_{xy}$"),
    }
    if quantity not in quantity_map:
        valid = ", ".join(quantity_map)
        raise ValueError(f"Unknown quantity '{quantity}'. Use one of: {valid}")
    return quantity_map[quantity]


def _qgt_quantity_array(dataset, quantity, band):
    key, label = _qgt_quantity_spec(quantity)
    arr = np.asarray(dataset[key])
    if arr.ndim == 3:
        if not (0 <= band < arr.shape[0]):
            raise IndexError(f"Band {band} is out of range for {key}; valid range is [0, {arr.shape[0] - 1}]")
        arr = arr[band]
    elif band != 0:
        raise IndexError(f"{key} is not band-stacked, so only band=0 is valid.")

    if quantity in ("berry", "berry_curvature"):
        arr = -2.0 * arr

    return np.asarray(arr), label


def _qgt_sym_quantity_array(dataset, quantity, band):
    key, label = _qgt_quantity_spec(quantity)
    for sym_qgt in dataset.get("qgt_sym_paths", []):
        bands = list(np.asarray(sym_qgt["bands"], dtype=int))
        if band not in bands:
            continue
        band_pos = bands.index(band)
        arr = np.asarray(sym_qgt[key])[band_pos]
        if quantity in ("berry", "berry_curvature"):
            arr = -2.0 * arr
        return np.asarray(arr), label, sym_qgt
    return None, label, None


def _plot_energy_panel(ax, dataset, *, bands_to_plot=None, title=""):
    sym_line = dataset.get("sym_line")
    if sym_line is None:
        ax.set_title(title)
        ax.set_ylabel("Energy")
        ax.text(0.5, 0.5, "eigenvalues_sym_line.npz not found", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    _plot_sym_line_energies(ax, sym_line, bands_to_plot=bands_to_plot, title=title)


def _plot_sym_line_energies(ax, sym_line, *, bands_to_plot=None, title=""):
    eigenvalues = np.real(np.asarray(sym_line["eigenvalues"]))
    k_dist = np.asarray(sym_line["k_dist"])
    node_indices = list(np.asarray(sym_line["node_indices"], dtype=int))
    path_labels = list(sym_line["path_labels"])

    ax.set_title(title)
    ax.set_ylabel("Energy")
    ax.set_xlabel("Symmetry path")

    if eigenvalues.ndim != 2:
        ax.text(0.5, 0.5, f"Unsupported symmetry eigenvalue shape {eigenvalues.shape}", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    nbands = eigenvalues.shape[-1]
    if bands_to_plot is None:
        bands = range(nbands)
    elif isinstance(bands_to_plot, int):
        bands = [bands_to_plot]
    else:
        bands = list(bands_to_plot)

    bad = [b for b in bands if not (0 <= b < nbands)]
    if bad:
        raise IndexError(f"Energy bands {bad} are out of range; valid range is [0, {nbands - 1}]")

    for band in bands:
        ax.plot(k_dist, eigenvalues[:, band], linewidth=1.2, label=f"band {band}")

    valid_nodes = [idx for idx in node_indices if 0 <= idx < len(k_dist)]
    valid_labels = [
        label
        for idx, label in zip(node_indices, path_labels)
        if 0 <= idx < len(k_dist)
    ]
    for idx in valid_nodes:
        ax.axvline(k_dist[idx], color="0.75", linewidth=0.8, linestyle="--")
    ax.set_xticks([k_dist[idx] for idx in valid_nodes])
    ax.set_xticklabels(valid_labels)
    ax.grid(alpha=0.25)


def _plot_qgt_sym_path_panel(ax, dataset, quantity, band, *, title="", ylim=None):
    qgt_line, qgt_label, sym_qgt = _qgt_sym_quantity_array(dataset, quantity, band)
    ax.set_title(title)
    ax.set_ylabel(qgt_label)
    ax.set_xlabel("Symmetry path")

    if qgt_line is None:
        ax.text(0.5, 0.5, "qgt_sym_path_*.npz not found for this band", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    k_dist = np.asarray(sym_qgt["k_dist"])
    node_indices = list(np.asarray(sym_qgt["node_indices"], dtype=int))
    path_labels = list(sym_qgt["path_labels"])
    ax.plot(k_dist, qgt_line, linewidth=1.4)

    valid_nodes = [idx for idx in node_indices if 0 <= idx < len(k_dist)]
    valid_labels = [
        label
        for idx, label in zip(node_indices, path_labels)
        if 0 <= idx < len(k_dist)
    ]
    for idx in valid_nodes:
        ax.axvline(k_dist[idx], color="0.75", linewidth=0.8, linestyle="--")
    ax.set_xticks([k_dist[idx] for idx in valid_nodes])
    ax.set_xticklabels(valid_labels)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)


def _shared_sym_qgt_ylim(left, right, quantity, left_band, right_band):
    left_line, _, _ = _qgt_sym_quantity_array(left, quantity, left_band)
    right_line, _, _ = _qgt_sym_quantity_array(right, quantity, right_band)
    finite_parts = []
    for line in (left_line, right_line):
        if line is not None:
            finite = np.asarray(line)[np.isfinite(line)]
            if finite.size:
                finite_parts.append(finite)
    if not finite_parts:
        return None

    finite_all = np.concatenate(finite_parts)
    ymin = float(np.nanmin(finite_all))
    ymax = float(np.nanmax(finite_all))
    if ymin == ymax:
        return ymin, ymax
    pad = 0.04 * (ymax - ymin)
    return ymin - pad, ymax + pad


def _sym_path_for_overlay(dataset):
    if dataset.get("qgt_sym_paths"):
        sym_qgt = dataset["qgt_sym_paths"][0]
        return sym_qgt.get("k_path"), sym_qgt.get("path_points"), sym_qgt.get("path_labels")
    sym_line = dataset.get("sym_line")
    if sym_line is not None:
        return sym_line.get("k_path"), sym_line.get("path_points"), sym_line.get("path_labels")
    return None, None, None


def _overlay_sym_path_on_qgt(ax, dataset):
    k_path, path_points, path_labels = _sym_path_for_overlay(dataset)
    if k_path is None:
        return

    k_path = np.asarray(k_path)
    if k_path.ndim != 2 or k_path.shape[1] < 2:
        return

    ax.plot(k_path[:, 0], k_path[:, 1], color="white", linewidth=1.5, linestyle="--", alpha=0.95)

    if path_points is not None:
        path_points = np.asarray(path_points)
        if path_points.ndim == 2 and path_points.shape[1] >= 2:
            ax.scatter(path_points[:, 0], path_points[:, 1], color="white", edgecolor="black", linewidth=0.4, s=22, zorder=5)
            if path_labels is not None:
                for point, label in zip(path_points, path_labels):
                    ax.annotate(
                        label,
                        (point[0], point[1]),
                        textcoords="offset points",
                        xytext=(4, 4),
                        color="white",
                        fontsize=8,
                        fontweight="bold",
                    )


def _image_extent(kx, ky):
    return [
        float(np.nanmin(kx)),
        float(np.nanmax(kx)),
        float(np.nanmin(ky)),
        float(np.nanmax(ky)),
    ]


def plot_qgt_2d_comparison_with_energies(
    left_dir,
    right_dir,
    *,
    quantity="trace",
    band=0,
    left_band=None,
    right_band=None,
    labels=("Dataset 1", "Dataset 2"),
    energy_bands_to_plot=None,
    share_sym_qgt_range=True,
    zlim_percentile=99.0,
    zlim_berry=None,
    cmap="inferno",
    background_color="0.35",
    output_path=None,
    figsize=(13, 10),
    dpi=220,
):
    """
    Plot two 2D QGT datasets side by side, with symmetry-path energy and QGT cuts above each QGT map.

    The top panels use eigenvalues_sym_line.npz from each dataset. If that file
    is missing, the corresponding top panel is left empty with a message. The
    Middle panels use qgt_sym_path_*.npz from each dataset. Bottom panels share
    one color scale for the requested QGT quantity and the figure is always saved.
    """
    left = _load_2d_qgt_dataset(left_dir)
    right = _load_2d_qgt_dataset(right_dir)
    left_band = band if left_band is None else left_band
    right_band = band if right_band is None else right_band

    left_qgt, qgt_label = _qgt_quantity_array(left, quantity, left_band)
    right_qgt, _ = _qgt_quantity_array(right, quantity, right_band)
    qgt_cmap = plt.get_cmap(cmap).copy()
    qgt_cmap.set_bad(color=background_color)
    sym_qgt_ylim = (
        _shared_sym_qgt_ylim(left, right, quantity, left_band, right_band)
        if share_sym_qgt_range
        else None
    )

    finite = np.concatenate([
        left_qgt[np.isfinite(left_qgt)].ravel(),
        right_qgt[np.isfinite(right_qgt)].ravel(),
    ])
    if finite.size == 0:
        raise ValueError(f"No finite values found for quantity '{quantity}'.")

    if quantity in ("berry", "berry_curvature"):
        qgt_vmin, qgt_vmax = get_symmetric_plot_limits(
            finite,
            zlim_berry,
            zlim_percentile,
        )
    else:
        qgt_vmin = float(np.nanmin(finite))
        if zlim_percentile is None:
            qgt_vmax = float(np.nanmax(finite))
        else:
            qgt_vmax = float(np.nanpercentile(finite, zlim_percentile))
        if qgt_vmax <= qgt_vmin:
            qgt_vmax = float(np.nanmax(finite))

    fig, axes = plt.subplots(
        3,
        2,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1.0, 1.0, 2.7]},
        constrained_layout=True,
    )

    _plot_energy_panel(
        axes[0, 0],
        left,
        bands_to_plot=energy_bands_to_plot,
        title=f"{labels[0]} energies",
    )
    _plot_energy_panel(
        axes[0, 1],
        right,
        bands_to_plot=energy_bands_to_plot,
        title=f"{labels[1]} energies",
    )

    _plot_qgt_sym_path_panel(
        axes[1, 0],
        left,
        quantity,
        left_band,
        title=f"{labels[0]} {qgt_label} along symmetry path",
        ylim=sym_qgt_ylim,
    )
    _plot_qgt_sym_path_panel(
        axes[1, 1],
        right,
        quantity,
        right_band,
        title=f"{labels[1]} {qgt_label} along symmetry path",
        ylim=sym_qgt_ylim,
    )

    images = []
    for ax, dataset, qgt, label, qgt_band in [
        (axes[2, 0], left, left_qgt, labels[0], left_band),
        (axes[2, 1], right, right_qgt, labels[1], right_band),
    ]:
        order = dataset["meta_info"].get("order", "xyz")
        coordinate_info = coordinate_order_info(order)
        x_label, y_label = get_coordinate_axis_labels(
            order,
            backend="matplotlib",
        )
        ax.set_facecolor(background_color)
        im = ax.imshow(
            np.ma.masked_invalid(qgt),
            origin="lower",
            extent=_image_extent(dataset["kx"], dataset["ky"]),
            aspect=(
                "equal"
                if coordinate_info["coordinate_system"] == "cartesian"
                else "auto"
            ),
            cmap=qgt_cmap,
            vmin=qgt_vmin,
            vmax=qgt_vmax,
        )
        images.append(im)
        if order == "xyz":
            _overlay_sym_path_on_qgt(ax, dataset)
        ax.set_title(f"{label}: {qgt_label}, band {qgt_band}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    cbar = fig.colorbar(images[-1], ax=axes[2, :], shrink=0.9, pad=0.02)
    cbar.set_label(qgt_label)

    if output_path is None:
        safe_quantity = quantity.replace(" ", "_")
        output_path = os.path.join(
            left_dir,
            f"qgt_2d_comparison_{safe_quantity}_left_band_{left_band}_right_band_{right_band}.png",
        )
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved side-by-side QGT comparison plot to: {output_path}")
    return output_path


if __name__ == "__main__":
    import os

    plot_qgt_2d_comparison_with_energies(
        "/Users/home/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/2D_QGT_results/ChiralHamiltonianChiralBasisProjected/dataset_1",
        "/Users/home/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/2D_QGT_results/ChiralHamiltonianChiralBasisProjected/dataset_2",
        quantity="berry",
        left_band=0,
        right_band=0,
        labels=("Chiral-basis Projected Naive Numerical", "Chiral-basis Projected Analytical"),
        output_path="/Users/home/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/2D_QGT_results/ChiralHamiltonian/chiral_basis_projected_naive_numerical_vs_chiral_basis_projected_analytical.png",
    )

    # base_dir = "/Users/home/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/2D_QGT_results/gWaveAltermagnetHamiltonian"
    # base_dir = "/Users/home/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/2D_QGT_results/THF_Hamiltonian"
    # base_dir = "/Users/home/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/2D_QGT_results/TwoOrbitalUnspinfulHamiltonian"
    #
    # for subdir in os.listdir(base_dir):
    #     target_dir = os.path.join(base_dir, subdir)
    #     if os.path.isdir(target_dir) and os.path.exists(os.path.join(target_dir, "meta_info.pkl")):
    #         print(f"\n--- Processing: {subdir} ---")
    #         # plot_all_2d_berries_from_directory(target_dir)
    #         # You can also uncomment the next line to run plot_qgt_from_directory
    #         plot_qgt_from_directory(target_dir)
