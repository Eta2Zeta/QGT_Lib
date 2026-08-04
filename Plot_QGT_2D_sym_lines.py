import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def _load_npz_dict(path):
    with np.load(path, allow_pickle=True) as data:
        loaded = {key: np.asarray(data[key]) for key in data.files}
    if "path_labels" in loaded:
        loaded["path_labels"] = [str(label) for label in loaded["path_labels"]]
    return loaded


def _load_sym_line_dataset(root_dir):
    eig_path = os.path.join(root_dir, "eigenvalues_sym_line.npz")
    if not os.path.exists(eig_path):
        raise FileNotFoundError(f"Cannot find symmetry-line eigenvalues: {eig_path}")

    qgt_paths = sorted(glob.glob(os.path.join(root_dir, "qgt_sym_path_*.npz")))
    if not qgt_paths:
        raise FileNotFoundError(f"Cannot find qgt_sym_path_*.npz in: {root_dir}")

    return {
        "root_dir": root_dir,
        "eigen": _load_npz_dict(eig_path),
        "qgt_paths": qgt_paths,
        "qgt": [_load_npz_dict(path) for path in qgt_paths],
    }


def _quantity_spec(quantity):
    quantity_map = {
        "g_xx": ("g_xx", r"$g_{xx}$"),
        "g_yy": ("g_yy", r"$g_{yy}$"),
        "g_zz": ("g_zz", r"$g_{zz}$"),
        "g_xy_real": ("g_xy_real", r"$\mathrm{Re}\ g_{xy}$"),
        "g_xy_imag": ("g_xy_imag", r"$\mathrm{Im}\ g_{xy}$"),
        "g_xz_real": ("g_xz_real", r"$\mathrm{Re}\ g_{xz}$"),
        "g_xz_imag": ("g_xz_imag", r"$\mathrm{Im}\ g_{xz}$"),
        "g_yz_real": ("g_yz_real", r"$\mathrm{Re}\ g_{yz}$"),
        "g_yz_imag": ("g_yz_imag", r"$\mathrm{Im}\ g_{yz}$"),
        "trace": ("trace", r"$\mathrm{Tr}\ g$"),
        "berry": ("g_xy_imag", r"$\Omega_{xy}$"),
        "berry_curvature": ("g_xy_imag", r"$\Omega_{xy}$"),
    }
    if quantity not in quantity_map:
        valid = ", ".join(quantity_map)
        raise ValueError(f"Unknown quantity '{quantity}'. Use one of: {valid}")
    return quantity_map[quantity]


def _select_qgt_line(dataset, quantity, band):
    key, label = _quantity_spec(quantity)
    for qgt in dataset["qgt"]:
        bands = list(np.asarray(qgt["bands"], dtype=int))
        if band not in bands:
            continue
        band_pos = bands.index(band)
        values = np.asarray(qgt[key])[band_pos]
        if quantity in ("berry", "berry_curvature"):
            values = -2.0 * values
        return np.asarray(values), label, qgt
    raise ValueError(f"No symmetry-path QGT data for band {band} in {dataset['root_dir']}")


def _as_band_list(bands, n_bands):
    if bands is None:
        return list(range(n_bands))
    if isinstance(bands, int):
        return [bands]
    return list(bands)


def _energy_bands_for_dataset(energy_bands_to_plot, qgt_band, n_bands):
    if energy_bands_to_plot is None:
        bands = [qgt_band]
    else:
        bands = _as_band_list(energy_bands_to_plot, n_bands)
    bad = [band for band in bands if not (0 <= band < n_bands)]
    if bad:
        raise IndexError(f"Energy bands {bad} are out of range; valid range is [0, {n_bands - 1}]")
    return bands


def _finite_range(arrays, pad_fraction=0.04):
    finite_parts = []
    for arr in arrays:
        arr = np.asarray(arr)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            finite_parts.append(finite)
    if not finite_parts:
        return None

    finite_all = np.concatenate(finite_parts)
    ymin = float(np.nanmin(finite_all))
    ymax = float(np.nanmax(finite_all))
    if ymin == ymax:
        return ymin, ymax
    pad = (ymax - ymin) * pad_fraction
    return ymin - pad, ymax + pad


def _format_sym_axis(ax, sym_data):
    k_dist = np.asarray(sym_data["k_dist"])
    node_indices = list(np.asarray(sym_data["node_indices"], dtype=int))
    path_labels = list(sym_data["path_labels"])

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


def plot_two_qgt_sym_lines(
    left_dir,
    right_dir,
    *,
    quantity="trace",
    left_band=0,
    right_band=0,
    energy_bands_to_plot=None,
    labels=("Dataset 1", "Dataset 2"),
    output_path=None,
    figsize=(9, 7),
    dpi=220,
):
    """
    Overlay symmetry-line energies and QGT quantity from two 2D QGT datasets.
    """
    left = _load_sym_line_dataset(left_dir)
    right = _load_sym_line_dataset(right_dir)

    left_qgt, qgt_label, left_qgt_meta = _select_qgt_line(left, quantity, left_band)
    right_qgt, _, right_qgt_meta = _select_qgt_line(right, quantity, right_band)

    left_eigs = np.real(np.asarray(left["eigen"]["eigenvalues"]))
    right_eigs = np.real(np.asarray(right["eigen"]["eigenvalues"]))
    left_k = np.asarray(left["eigen"]["k_dist"])
    right_k = np.asarray(right["eigen"]["k_dist"])

    left_energy_bands = _energy_bands_for_dataset(energy_bands_to_plot, left_band, left_eigs.shape[-1])
    right_energy_bands = _energy_bands_for_dataset(energy_bands_to_plot, right_band, right_eigs.shape[-1])

    energy_ylim = _finite_range(
        [left_eigs[:, band] for band in left_energy_bands]
        + [right_eigs[:, band] for band in right_energy_bands]
    )
    qgt_ylim = _finite_range([left_qgt, right_qgt])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=False,
        constrained_layout=True,
    )

    ax_e, ax_q = axes
    for band in left_energy_bands:
        ax_e.plot(left_k, left_eigs[:, band], linewidth=1.2, color="tab:blue", alpha=0.85, label=f"{labels[0]} band {band}")
    for band in right_energy_bands:
        ax_e.plot(right_k, right_eigs[:, band], linewidth=1.2, color="tab:orange", alpha=0.85, linestyle="--", label=f"{labels[1]} band {band}")
    _format_sym_axis(ax_e, left["eigen"])
    if energy_ylim is not None:
        ax_e.set_ylim(*energy_ylim)
    ax_e.set_ylabel("Energy")
    ax_e.set_title("Symmetry-line energies")
    ax_e.legend(loc="best", fontsize=8)

    ax_q.plot(left_qgt_meta["k_dist"], left_qgt, linewidth=1.6, color="tab:blue", label=f"{labels[0]} band {left_band}")
    ax_q.plot(right_qgt_meta["k_dist"], right_qgt, linewidth=1.6, color="tab:orange", linestyle="--", label=f"{labels[1]} band {right_band}")
    _format_sym_axis(ax_q, left_qgt_meta)
    if qgt_ylim is not None:
        ax_q.set_ylim(*qgt_ylim)
    ax_q.set_ylabel(qgt_label)
    ax_q.set_xlabel("Symmetry path")
    ax_q.set_title(f"{qgt_label} along symmetry path")
    ax_q.legend(loc="best", fontsize=8)

    if output_path is None:
        safe_quantity = quantity.replace(" ", "_")
        output_path = os.path.join(
            left_dir,
            f"qgt_sym_lines_overlay_{safe_quantity}_left_band_{left_band}_right_band_{right_band}.png",
        )
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved symmetry-line comparison plot to: {output_path}")
    return output_path


if __name__ == "__main__":
    plot_two_qgt_sym_lines(
        "results/2D_QGT_results/ChiralHamiltonian/dataset_1",
        "results/2D_QGT_results/ChiralHamiltonianChiralBasisProjected/dataset_2",
        quantity="berry",
        left_band=4,
        right_band=0,
        labels=("Full chiral Hamiltonian", "Chiral-basis projected"),
    )
