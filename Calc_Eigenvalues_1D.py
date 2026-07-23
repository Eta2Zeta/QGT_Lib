"""Calculate eigenvalues along a single straight momentum-space line."""

import pickle

import numpy as np

from Library.Hamiltonian import THF_Hamiltonian
from Library.data_management_utils_1d import setup_results_directory_1d
from Library.eigenvalue_calc_lib_1d import line_eigenvalues_eigenfunctions
from Library.output_utils import print_calculation_complete
from Library.plotting_lib_1d import plot_eigenvalues_line


def calculation_1d(
    Hamiltonian_Obj,
    *,
    angle_deg=30.0,
    kx_shift=0.0,
    ky_shift=0.0,
    num_points=100,
    k_max=np.sqrt(2.0) * np.pi,
    band_index=1,
    bands_to_plot=(0,),
    force_new=False,
):
    """Calculate and plot bands on a line through momentum space."""
    print(f"Performing 1D eigenvalue calculation ({Hamiltonian_Obj.name})...")

    k_angle = np.deg2rad(angle_deg)
    k_line = np.linspace(-k_max, k_max, num_points)
    line_kx = k_line * np.cos(k_angle) + kx_shift
    line_ky = k_line * np.sin(k_angle) + ky_shift

    file_paths, use_existing, results_dir = setup_results_directory_1d(
        Hamiltonian_Obj,
        angle_deg,
        kx_shift,
        ky_shift,
        num_points,
        k_max,
        force_new=force_new,
    )

    if use_existing:
        eigenvalues = np.load(file_paths["eigenvalues"])
        eigenfunctions = np.load(file_paths["eigenfunctions"])
        print("Loaded existing 1D eigenvalues and eigenfunctions.")
    else:
        eigenvalues, eigenfunctions, _, _, _ = line_eigenvalues_eigenfunctions(
            Hamiltonian_Obj,
            line_kx,
            line_ky,
            band_index,
        )
        np.save(file_paths["eigenvalues"], eigenvalues)
        np.save(file_paths["eigenfunctions"], eigenfunctions)

        meta_info = {
            "kx_line": line_kx,
            "ky_line": line_ky,
            "num_points": int(num_points),
            "Hamiltonian_Obj": Hamiltonian_Obj,
        }
        with open(file_paths["meta_info"], "wb") as meta_file:
            pickle.dump(meta_info, meta_file)

    plot_eigenvalues_line(
        k_line,
        eigenvalues,
        dim=None,
        bands_to_plot=bands_to_plot,
    )
    print_calculation_complete("1D Eigenvalues", results_dir, artifact="Results")
    return eigenvalues, eigenfunctions, results_dir


def main():
    Hamiltonian_Obj = THF_Hamiltonian(A0=0)
    calculation_1d(Hamiltonian_Obj)


if __name__ == "__main__":
    main()
