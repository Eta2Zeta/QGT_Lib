"""Calculate eigenvalues along multiple angled momentum-space lines."""

import json
import pickle

import numpy as np

from Library.Hamiltonian import THF_Hamiltonian
from Library.data_management_utils_1d import setup_1D_angles_results_directory
from Library.eigenvalue_calc_lib_1d import eigenvalues_along_path
from Library.output_utils import print_calculation_complete
from Library.plotting_lib_1d import plot_band_structure_angles_slider
from Library.utilities import generate_1d_lines_at_angles


def calculation_1d_at_angles(
    Hamiltonian_Obj,
    *,
    k_max=2.0 * np.pi,
    num_angles=10,
    num_points_per_line=1000,
    force_new=True,
    use_analytical=False,
    bands_to_plot=None,
    show=True,
):
    """Calculate bands on lines through the origin and plot an angle slider."""
    print(f"Performing 1D angled eigenvalue calculation ({Hamiltonian_Obj.name})...")

    k_path, k_vals, angles = generate_1d_lines_at_angles(
        k_max,
        num_angles,
        num_points_per_line,
    )
    k_path_3d = np.column_stack([k_path, np.zeros(len(k_path))])

    file_paths, use_existing, results_dir, meta_target = (
        setup_1D_angles_results_directory(
            Hamiltonian_Obj,
            k_max,
            num_angles,
            num_points_per_line,
            force_new=force_new,
        )
    )

    if use_existing:
        print("Loading existing 1D-angle results...")
        eigenvalues_flat = np.load(file_paths["eigenvalues"])
    else:
        print("Calculating eigenvalues along all angled paths...")
        eigenvalues_flat, _ = eigenvalues_along_path(
            Hamiltonian_Obj,
            k_path_3d,
            use_analytical=use_analytical,
        )
        np.save(file_paths["eigenvalues"], eigenvalues_flat)
        with open(file_paths["meta_json"], "w") as meta_file:
            json.dump(meta_target, meta_file, indent=2)
        with open(file_paths["meta_pkl"], "wb") as meta_file:
            pickle.dump(meta_target, meta_file)

    num_bands = eigenvalues_flat.shape[1]
    eigenvalues = eigenvalues_flat.reshape(
        (num_angles, num_points_per_line, num_bands)
    )

    plot_band_structure_angles_slider(
        k_vals=k_vals,
        eigenvalues=eigenvalues,
        angles=angles,
        bands_to_plot=bands_to_plot,
        title=f"1D Band Structure vs Angle ({Hamiltonian_Obj.name})",
        results_dir=results_dir,
        save_fig=True,
        show=show,
    )
    print_calculation_complete("1D Angled Eigenvalues", results_dir, artifact="Results")
    return eigenvalues, angles, results_dir


def main():
    Hamiltonian_Obj = THF_Hamiltonian(A0=0)
    k_max = getattr(Hamiltonian_Obj, "k_theta", 2.0 * np.pi)
    calculation_1d_at_angles(
        Hamiltonian_Obj,
        k_max=k_max,
        num_angles=200,
        bands_to_plot=(2, 3),
    )


if __name__ == "__main__":
    main()
