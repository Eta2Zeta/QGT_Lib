"""Calculate eigenvalues along a Hamiltonian-defined high-symmetry path."""

import json
import pickle

import numpy as np

from Library.Hamiltonian import THF_Hamiltonian
from Library.data_management_utils_1d import setup_sym_points_results_directory
from Library.eigenvalue_calc_lib_1d import eigenvalues_along_path
from Library.output_utils import print_calculation_complete
from Library.plotting_lib_1d import plot_band_structure_sym
from Library.plotting_lib_3d import plot_degeneracy_on_path_3d


def calculation_sym_points(
    Hamiltonian_Obj,
    *,
    force_new=True,
    use_analytical=False,
    num_points_per_segment=100,
    bands_to_plot=None,
):
    """Calculate and plot bands on ``Hamiltonian_Obj.get_sym_path()``."""
    print(
        f"Performing symmetry-path eigenvalue calculation "
        f"({Hamiltonian_Obj.name})..."
    )

    sym_points, path_labels = Hamiltonian_Obj.get_sym_path()
    nodes = [np.asarray(sym_points[label], dtype=float) for label in path_labels]
    if not nodes:
        raise ValueError("The Hamiltonian returned an empty symmetry path.")

    k_dim = nodes[0].shape[0]
    if k_dim not in (2, 3):
        raise ValueError(f"Symmetry-path points must be 2D or 3D; received {k_dim}D.")
    if any(node.shape != nodes[0].shape for node in nodes):
        raise ValueError("All symmetry-path points must have the same dimensionality.")

    all_k_points = [nodes[0]]
    all_k_dist = [0.0]
    node_indices = [0]
    cumulative_distance = 0.0

    for start, end in zip(nodes[:-1], nodes[1:]):
        segment_distance = np.linalg.norm(end - start)
        segment_points = np.linspace(
            start,
            end,
            num_points_per_segment + 1,
        )[1:]
        segment_distances = np.linspace(
            0.0,
            segment_distance,
            num_points_per_segment + 1,
        )[1:]
        all_k_points.extend(segment_points)
        all_k_dist.extend(cumulative_distance + segment_distances)
        cumulative_distance += segment_distance
        node_indices.append(len(all_k_points) - 1)

    k_path = np.asarray(all_k_points)
    k_dist = np.asarray(all_k_dist)
    path_points = np.asarray(nodes)
    if k_dim == 2:
        k_path_for_calculation = np.column_stack(
            [k_path, np.zeros(len(k_path))]
        )
    else:
        k_path_for_calculation = k_path

    file_paths, use_existing, results_dir, meta_target = (
        setup_sym_points_results_directory(
            Hamiltonian_Obj,
            path_points,
            path_labels,
            num_points_per_segment,
            force_new=force_new,
        )
    )

    if use_existing:
        print("Loading existing symmetry-path results...")
        eigenvalues = np.load(file_paths["eigenvalues"])
    else:
        print("Calculating eigenvalues along the symmetry path...")
        eigenvalues, _ = eigenvalues_along_path(
            Hamiltonian_Obj,
            k_path_for_calculation,
            use_analytical=use_analytical,
        )
        np.save(file_paths["eigenvalues"], eigenvalues)
        with open(file_paths["meta_json"], "w") as meta_file:
            json.dump(meta_target, meta_file, indent=2)
        with open(file_paths["meta_pkl"], "wb") as meta_file:
            pickle.dump(meta_target, meta_file)

    plot_band_structure_sym(
        k_dist,
        eigenvalues,
        node_indices,
        path_labels,
        bands_to_plot=bands_to_plot,
        title=f"Band Structure along symmetry path ({Hamiltonian_Obj.name})",
        results_dir=results_dir,
        save_fig=True,
        use_analytical=use_analytical,
    )
    plot_degeneracy_on_path_3d(
        k_path,
        eigenvalues,
        threshold=0.02,
        title=f"Degeneracy along Path ({Hamiltonian_Obj.name})",
        results_dir=results_dir,
        save_fig=True,
    )

    print_calculation_complete(
        "Symmetry-Path Eigenvalues",
        results_dir,
        artifact="Results",
    )
    return eigenvalues, k_path, k_dist, results_dir


def main():
    Hamiltonian_Obj = THF_Hamiltonian(A0=0)
    calculation_sym_points(Hamiltonian_Obj, force_new=True)


if __name__ == "__main__":
    main()
