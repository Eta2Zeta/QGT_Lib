import sys
import os
import json
import numpy as np
from tqdm import tqdm
import copy
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial

from Library.Hamiltonian.Hamiltonian import *
from Library.Hamiltonian.ChiralHamiltonian import ChiralHamiltonian
from Library.Hamiltonian.gWaveAltermagnetHamiltonian import gWaveAltermagnetHamiltonian
from Library.utilities import generate_3d_sym_lines
from Library.Hamiltonian_helper import get_Hamiltonian
from Library.data_management_utils_nd import build_parameter_points
from Library.output_utils import print_calculation_complete


# ---------- per-point worker ----------
def _worker_sym_point(arg, h_template, k_path):
    """
    arg: (param_values_dict, idx_tuple) OR (param_values_dict, idx_tuple, progress_label)
    Computes eigenvalues along the symmetry k-path for a single parameter point.
    """
    if len(arg) == 3:
        param_values, idx, progress_label = arg
    else:
        param_values, idx = arg

    H = copy.deepcopy(h_template)
    for k, v in param_values.items():
        setattr(H, k, v)

    num_points = len(k_path)
    dim = int(H.dim)
    eigenvalues = np.zeros((num_points, dim))

    for i in range(num_points):
        k_curr = k_path[i]
        H_mat, _ = get_Hamiltonian(H, k_curr[0], k_curr[1], k_curr[2])
        evals = np.linalg.eigvalsh(H_mat)
        eigenvalues[i, :] = np.sort(evals)

    return (idx, eigenvalues)


# ---------- master driver ----------
def compute_sym_nd_parallel(
    hamiltonian_template,
    param_ranges,
    parameter_spacing,
    num_points_per_segment=100,
    space_group=194,
    processes=None,
    force_new_dir=False,
    float_dtype=np.float64,
):
    # TODO: This space group based sym path seems to be outdated, everything should be done in the Hamiltonian, maybe update it later
    """
    Computes eigenvalues along 3D high-symmetry lines for every point in an
    N-D parameter grid (in parallel).

    Saves:
      - sym_nd_bundle.npz  (k_path, k_dist, node_indices, path_labels,
                            per-axis arrays, eigenvalues_grid)
      - meta.json          (param_ranges, spacing, space_group)
    Returns: (root_dir, npz_path)
    """
    H_template = copy.deepcopy(hamiltonian_template)

    k_path, k_dist, node_indices, path_labels, path_points = \
        generate_3d_sym_lines(num_points_per_segment, space_group=space_group)
    num_k_points = len(k_path)
    dim = int(getattr(H_template, "dim", 0))

    points_with_idx, names, axes, shape = build_parameter_points(param_ranges, parameter_spacing)
    total_param_points = int(np.prod(shape))

    def _label_for_idx(idx_tuple):
        return f"{np.ravel_multi_index(idx_tuple, shape) + 1}/{total_param_points}"

    points_labeled = [(d, idx, _label_for_idx(idx)) for (d, idx) in points_with_idx]

    # Output array: (*param_shape, num_k_points, dim)
    out_shape = tuple(shape) + (num_k_points, dim)
    eigenvalues_grid = np.empty(out_shape, dtype=float_dtype)

    # Result directory
    base = os.path.join(os.getcwd(), "results", "Sym_Phase_Diagram")
    name = (
        f"{getattr(H_template, 'name', 'Model')}_Sym"
        + "".join([f"_{k}_{v[0]}to{v[1]}" for k, v in param_ranges.items()])
    )
    path = os.path.join(base, name)
    suffix, idx_dir = "", 1
    while not force_new_dir and os.path.exists(path + suffix):
        suffix = f"_{idx_dir}"
        idx_dir += 1
    root = path + suffix
    os.makedirs(root, exist_ok=True)

    bundle_path = os.path.join(root, "sym_nd_bundle.npz")

    # Worker
    worker = partial(_worker_sym_point, h_template=H_template, k_path=k_path)
    procs = processes or min(max(1, cpu_count() - 1), max(1, len(points_with_idx)))
    print(f"Launching 3D Sym N-D sweep on {procs} processes over {len(points_with_idx)} points ...")

    with Pool(processes=procs) as pool:
        with tqdm(total=len(points_labeled), desc="Sym points calc", unit="pt") as pbar:
            for result in pool.imap(worker, points_labeled):
                idx, eigenvalues = result
                eigenvalues_grid[idx] = eigenvalues
                pbar.update(1)

    # Save bundle
    np.savez_compressed(
        bundle_path,
        names=np.array(names, dtype=object),
        shape=np.array(shape, dtype=int),
        k_path=k_path,
        k_dist=k_dist,
        node_indices=node_indices,
        path_labels=path_labels,
        **{f"axis_{i}_{names[i]}": axes[i] for i in range(len(names))},
        eigenvalues_grid=eigenvalues_grid,
    )

    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(
            {"param_ranges": param_ranges, "spacing": parameter_spacing, "space_group": space_group},
            f, indent=2,
        )

    print_calculation_complete("N-D Symmetry-Path Eigenvalues", bundle_path, artifact="Bundle")
    return root, bundle_path


# =============================================================================
# Configuration — edit below to set up your sweep
# =============================================================================

H_template = gWaveAltermagnetHamiltonian(
    t1=0.3, t2=0.3, t3=0.3, t4=0.3, mu=0,
    Jx=0.2, Jy=0.0, Jz=0.0, lamb=0.1, lamb_z=0.1,
)

param_ranges = {
    "Jx":     (0.0, 1),
    "Jy":     (0.0, 1),
    "Jz":     (0.0, 1),
    "lamb":   (0.0, 0.3),
    "lamb_z": (0.0, 0.3),
    "t1":     (0.0, 0.3),
    "t2":     (0.0, 0.3),
    "t3":     (0.0, 0.3),
    "t4":     (0.0, 0.3),
}

_n = 3
parameter_spacing = {
    "Jx":     {"n": 2 * _n, "scale": "linear"},
    "Jy":     {"n": 2 * _n, "scale": "linear"},
    "Jz":     {"n": 2 * _n, "scale": "linear"},
    "lamb":   {"n": _n,     "scale": "linear"},
    "lamb_z": {"n": _n,     "scale": "linear"},
    "t1":     {"n": _n,     "scale": "linear"},
    "t2":     {"n": _n,     "scale": "linear"},
    "t3":     {"n": _n,     "scale": "linear"},
    "t4":     {"n": _n,     "scale": "linear"},
}


def main():
    root, bundle_path = compute_sym_nd_parallel(
        hamiltonian_template=H_template,
        param_ranges=param_ranges,
        parameter_spacing=parameter_spacing,
        num_points_per_segment=50,
        space_group=194,
        processes=None,
        force_new_dir=False,
        float_dtype=np.float32,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
