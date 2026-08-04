"""Calculate eigenvalues and eigenvectors on a two-dimensional momentum grid."""

import json
import os
import pickle
import shutil

import numpy as np

from Library.Hamiltonian import *
from Library.data_management_utils_2d import setup_2D_Eigen_results_directory
from Library.dimension_lib import (
    create_2d_coordinate_grid,
    is_cylindrical_order,
)
from Library.eigenvalue_calc_lib import (
    capping_eigenvalues,
    grid_eigenvalues_eigenfunctions,
)
from Library.output_utils import print_calculation_complete
from Library.plotting_eigenvalues_2d import (
    plot_degeneracy_heatmap,
    plot_eigenvalue_line_slider,
    plot_eigenvalue_surfaces,
)


def calculation_2d(
    Hamiltonian_Obj,
    *,
    force_new=True,
    include_end_points=True,
    kk=0.0,
    order="xyz",
    mesh_spacing=150,
    k_max=None,
    z_limit=1000,
    bands_to_plot=None,
    max_l=10,
):
    """Calculate, save, and plot a 2D eigenvalue grid.

    ``order`` accepts Cartesian permutations such as ``xyz`` and ``yzx``, or
    cylindrical orders ``xpz``, ``ypz``, ``xpy``, ``zpy``, ``ypx``, and
    ``zpx``. For a cylindrical order, the sampled coordinates are radius and
    phi while ``kk`` fixes the third physical axis.
    """
    print(f"Performing 2D eigenvalue calculation ({Hamiltonian_Obj.name})...")

    if isinstance(max_l, (bool, np.bool_)) or not isinstance(
        max_l,
        (int, np.integer),
    ):
        raise TypeError("max_l must be a positive integer")
    max_l = int(max_l)
    if max_l < 1:
        raise ValueError("max_l must be at least 1")

    if k_max is None:
        k_max = getattr(Hamiltonian_Obj, "k_theta", 2.0 * np.pi)

    ki, kj, grid_info = create_2d_coordinate_grid(
        k_max,
        mesh_spacing,
        order=order,
        include_endpoints=include_end_points,
    )
    order = grid_info["order"]
    cylindrical = is_cylindrical_order(order)
    dim = Hamiltonian_Obj.dim

    meta_params = {
        "hamiltonian_name": getattr(Hamiltonian_Obj, "name", "Hamiltonian_Obj"),
        "ki_range": grid_info["ki_range"],
        "kj_range": grid_info["kj_range"],
        "mesh_spacing": int(mesh_spacing),
        "dki": grid_info["dki"],
        "dkj": grid_info["dkj"],
        "kk": float(kk) if kk is not None else None,
        "kvals_mode": grid_info["sampling"],
        "order": order,
        "coordinate_system": grid_info["coordinate_system"],
        "coordinate_labels": grid_info["coordinate_labels"],
        "radial_reference_axis": grid_info["radial_reference_axis"],
        "rotation_axis": grid_info["rotation_axis"],
        "angular_orientation": grid_info["angular_orientation"],
        "phi_periodic": grid_info["phi_periodic"],
        "phi_domain": grid_info["phi_domain"],
        "phi_endpoint_included": grid_info["phi_endpoint_included"],
        "floquet_max_l": (
            max_l
            if float(getattr(Hamiltonian_Obj, "A0", 0.0)) != 0.0
            else None
        ),
        "floquet_ratio_band_basis": "zero_fourier_harmonic_energy_order",
        "floquet_ratio_index_order": ["coupled_band", "photon_index_l"],
        "floquet_ratio_includes_same_band": False,
        "hamiltonian_params": Hamiltonian_Obj.get_parameters_dict(parameter="2D"),
    }

    file_paths, use_existing, results_dir, meta_target = (
        setup_2D_Eigen_results_directory(
            meta_params=meta_params,
            force_new=force_new,
        )
    )

    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    if use_existing:
        eigenvalues = np.load(file_paths["eigenvalues"])
        eigenfunctions = np.load(file_paths["eigenfunctions"])
        floquet_max_ratio = np.load(file_paths["floquet_max_ratio"])
        floquet_max_ratio_indices = np.load(
            file_paths["floquet_max_ratio_indices"]
        )
        print(
            "Loaded existing 2D eigenvalues, eigenfunctions, Floquet ratios, "
            "and metadata."
        )

        for file_path in file_paths.values():
            shutil.copy(file_path, os.path.join(temp_dir, os.path.basename(file_path)))
        print(f"Copied existing results to temporary directory: {temp_dir}")
    else:
        (
            eigenvalues,
            eigenfunctions,
            _,
            _,
            floquet_max_ratio,
            floquet_max_ratio_indices,
        ) = grid_eigenvalues_eigenfunctions(
            Hamiltonian_Obj,
            ki,
            kj,
            mesh_spacing,
            dim=dim,
            kk=kk,
            order=order,
            max_l=max_l,
        )

        np.save(file_paths["eigenvalues"], eigenvalues)
        np.save(file_paths["eigenfunctions"], eigenfunctions)
        np.save(file_paths["floquet_max_ratio"], floquet_max_ratio)
        np.save(
            file_paths["floquet_max_ratio_indices"],
            floquet_max_ratio_indices,
        )
        np.save(os.path.join(temp_dir, "eigenvalues.npy"), eigenvalues)
        np.save(os.path.join(temp_dir, "eigenfunctions.npy"), eigenfunctions)
        np.save(
            os.path.join(temp_dir, "floquet_max_ratio.npy"),
            floquet_max_ratio,
        )
        np.save(
            os.path.join(temp_dir, "floquet_max_ratio_indices.npy"),
            floquet_max_ratio_indices,
        )

        with open(file_paths["meta_json"], "w") as meta_file:
            json.dump(meta_target, meta_file, indent=2, sort_keys=True)

        meta_info_pkl = meta_target.copy()
        meta_info_pkl.update(
            {
                "Hamiltonian_Obj": Hamiltonian_Obj,
                "ki": ki,
                "kj": kj,
            }
        )
        with open(file_paths["meta_pkl"], "wb") as meta_file:
            pickle.dump(meta_info_pkl, meta_file)
        with open(os.path.join(temp_dir, "meta_info.pkl"), "wb") as meta_file:
            pickle.dump(meta_info_pkl, meta_file)
        shutil.copy(file_paths["meta_json"], os.path.join(temp_dir, "meta.json"))

    eigenvalues_for_plot = capping_eigenvalues(
        eigenvalues=eigenvalues,
        z_limit=z_limit,
    )
    x_label = "r" if cylindrical else grid_info["coordinate_labels"][0]
    y_label = "phi (rad)" if cylindrical else grid_info["coordinate_labels"][1]
    plot_eigenvalue_surfaces(
        ki,
        kj,
        eigenvalues_for_plot,
        dim=dim,
        z_limit=z_limit,
        stride_size=2,
        color_maps="default",
        norm=None,
        bands_to_plot=bands_to_plot,
        results_dir=results_dir,
        save_fig=True,
        x_label=x_label,
        y_label=y_label,
    )

    print("Plotting 2D degeneracy map...")
    plot_degeneracy_heatmap(
        ki,
        kj,
        eigenvalues_for_plot,
        threshold=0.02,
        title=f"Band Degeneracy Map ({Hamiltonian_Obj.name})",
        sym_points=(order == "xyz"),
        hamiltonian=Hamiltonian_Obj if order == "xyz" else None,
        kk=kk,
        results_dir=results_dir,
        save_fig=True,
        x_label=x_label,
        y_label=y_label,
    )

    for axis_order in ("ij", "ji"):
        horizontal_label = x_label if axis_order == "ij" else y_label
        slider_label = y_label if axis_order == "ij" else x_label
        plot_eigenvalue_line_slider(
            ki,
            kj,
            eigenvalues_for_plot,
            axis_order=axis_order,
            first_axis_label=x_label,
            second_axis_label=y_label,
            bands_to_plot=bands_to_plot,
            title=(
                f"Eigenvalues vs {horizontal_label}; "
                f"slider: {slider_label} ({Hamiltonian_Obj.name})"
            ),
            results_dir=results_dir,
            save_fig=True,
            show=False,
        )

    print_calculation_complete(
        "2D Eigenvalues",
        results_dir,
        artifact="Results",
        copied_to=temp_dir,
    )
    return (
        eigenvalues,
        eigenfunctions,
        floquet_max_ratio,
        floquet_max_ratio_indices,
        results_dir,
    )


def main():
    # Hamiltonian_Obj = THF_Hamiltonian(A0=0, V=5)
    # coordinate_order = "xyz"
    # calculation_2d(
    #     Hamiltonian_Obj,
    #     force_new=False,
    #     include_end_points=False,
    #     kk=0.0,
    #     order=coordinate_order,
    # )

    Hamiltonian_Obj = gWaveAltermagnetHamiltonian(
        Jx=0.0,
        Jy=0.0,
        Jz=0.2,       # Jz turned on
        lamb=0.1,
        lamb_z=0.5,   # larger than the default 0.1
        A0=0.0,
        magnus_order=0,
        t1=0.3,
        t2=0.3,
        t3=0.3,
        t4=0.3
    )

    # Hamiltonian_Obj = MinimalHamSG127_2c2d(
    #     Jx = 0.0,
    #     Jy = 0.0,
    #     Jz = 0.2,
    #     lamb = 0.1,
    #     lamb_z = 0.1,
    #     A0 = 0.0,
    #     t1 = 0.3,
    #     t2 = 0.3,
    #     t3 = 0.3,
    #     t4 = 0.3
    # )

    # Hamiltonian_Obj = MinimalHamSG124_2b2d(
    #     Jx=0.0,
    #     Jy=0.0,
    #     Jz=0.2,
    #     lamb=0.1,
    #     lamb_z=0.1,
    #     A0=0.0,
    #     t1=0.3,
    #     t2=0.3,
    #     t3=0.3,
    #     t4=0.3,
    # )

    calculation_2d(
        Hamiltonian_Obj,
        force_new=True,
        kk=0.0,
        order="xpz",
        k_max=1.5 * np.pi,
    )


if __name__ == "__main__":
    main()
