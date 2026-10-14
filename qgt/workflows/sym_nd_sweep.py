import os
import json
import numpy as np
from tqdm import tqdm
import copy
from multiprocessing import Pool, cpu_count
from functools import partial

from qgt.kspace.paths import generate_3d_sym_lines
from qgt.hamiltonians.evaluate import get_Hamiltonian
from qgt.io.setup_nd import build_parameter_points
from qgt.io.output import print_calculation_complete
from qgt.io.runs import new_run


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

    # Result directory (a new run every time)
    root, _ = new_run("sym_nd", getattr(H_template, 'name', 'Model'), force_new=True)

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
