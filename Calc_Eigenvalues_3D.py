"""Calculate eigenvalues and eigenvectors on a three-dimensional momentum grid."""

import json
import os
import pickle
import shutil

import numpy as np

from Library.Hamiltonian import THF_Hamiltonian
from Library.data_management_utils_3d import setup_3D_Eigen_results_directory
from Library.eigenvalue_calc_lib import compute_eigenvalues_3d
from Library.output_utils import print_calculation_complete
from Library.plotting_lib_3d import (
    plot_arbitrary_slice_no_interp,
    plot_degeneracy_3d,
    plot_volumetric_cloud,
)
from Library.utilities import centered_kvals


def calculation_3d(
    Hamiltonian_Obj,
    *,
    force_new=True,
    include_end_points=True,
    k_range=np.pi,
    mesh=120,
    band_idx=2,
):
    """Calculate, save, and plot a 3D eigenvalue grid."""
    print(f"Performing 3D eigenvalue calculation ({Hamiltonian_Obj.name})...")

    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    if include_end_points:
        kx_vals = np.linspace(-k_range, k_range, mesh)
        ky_vals = np.linspace(-k_range, k_range, mesh)
        kz_vals = np.linspace(-k_range, k_range, mesh)
        kvals_mode = "endpoints"
    else:
        kx_vals = centered_kvals(k_range, mesh)
        ky_vals = centered_kvals(k_range, mesh)
        kz_vals = centered_kvals(k_range, mesh)
        kvals_mode = "centered"

    mesh_shape = (len(kx_vals), len(ky_vals), len(kz_vals))
    dkx = float(kx_vals[1] - kx_vals[0])
    dky = float(ky_vals[1] - ky_vals[0])
    dkz = float(kz_vals[1] - kz_vals[0])

    file_paths, use_existing, results_dir, meta_target = (
        setup_3D_Eigen_results_directory(
            Hamiltonian_Obj,
            [kx_vals[0], kx_vals[-1]],
            [ky_vals[0], ky_vals[-1]],
            [kz_vals[0], kz_vals[-1]],
            mesh_shape=mesh_shape,
            include_endpoints=include_end_points,
            force_new=force_new,
            kvals_mode=kvals_mode,
        )
    )

    if use_existing:
        print("Loading existing 3D results...")
        eigenvalues_3d = np.load(file_paths["eigenvalues"])
        eigenvectors_3d = np.load(file_paths["eigenfunctions"])
    else:
        eigenvalues_3d, eigenvectors_3d = compute_eigenvalues_3d(
            Hamiltonian_Obj,
            kx_vals,
            ky_vals,
            kz_vals,
        )
        np.save(file_paths["eigenvalues"], eigenvalues_3d)
        np.save(file_paths["eigenfunctions"], eigenvectors_3d)

        meta_info = meta_target.copy()
        meta_info["dk"] = [dkx, dky, dkz]
        with open(file_paths["meta_json"], "w") as meta_file:
            json.dump(meta_info, meta_file, indent=2, sort_keys=True)

        meta_pkl = {
            "kx_vals": kx_vals,
            "ky_vals": ky_vals,
            "kz_vals": kz_vals,
            "mesh_shape": mesh_shape,
            "dk": (dkx, dky, dkz),
            "include_endpoints": bool(include_end_points),
            "kvals_mode": kvals_mode,
            "Hamiltonian_Obj": Hamiltonian_Obj,
        }
        with open(file_paths["meta_pkl"], "wb") as meta_file:
            pickle.dump(meta_pkl, meta_file)

    print(f"Copying 3D results to temporary directory: {temp_dir}")
    shutil.copy(file_paths["meta_json"], os.path.join(temp_dir, "meta.json"))
    shutil.copy(file_paths["meta_pkl"], os.path.join(temp_dir, "meta_info.pkl"))
    shutil.copy(
        file_paths["eigenvalues"],
        os.path.join(temp_dir, "eigenvalues_3d.npy"),
    )
    shutil.copy(
        file_paths["eigenfunctions"],
        os.path.join(temp_dir, "eigenvectors_3d.npy"),
    )

    print("Generating 3D plots...")
    eig_band = eigenvalues_3d[:, :, :, band_idx]
    plot_volumetric_cloud(
        eig_band,
        kx_vals,
        ky_vals,
        kz_vals,
        opacity=0.2,
        levels=[0],
        results_dir=results_dir,
        save_fig=True,
    )
    plot_arbitrary_slice_no_interp(
        eigenvalues_3d,
        "z",
        0,
        kx_vals,
        ky_vals,
        kz_vals,
        title="Slice z (shift=0)",
        results_dir=results_dir,
        save_fig=True,
    )
    plot_degeneracy_3d(
        kx_vals,
        ky_vals,
        kz_vals,
        eigenvalues_3d,
        threshold=0.05,
        title=f"3D Band Degeneracy Map ({Hamiltonian_Obj.name})",
        results_dir=results_dir,
        save_fig=True,
    )

    print_calculation_complete(
        "3D Eigenvalues",
        results_dir,
        artifact="Results",
        copied_to=temp_dir,
    )
    return eigenvalues_3d, eigenvectors_3d, results_dir


def main():
    Hamiltonian_Obj = THF_Hamiltonian(A0=0)
    calculation_3d(
        Hamiltonian_Obj,
        force_new=False,
        include_end_points=True,
    )


if __name__ == "__main__":
    main()
