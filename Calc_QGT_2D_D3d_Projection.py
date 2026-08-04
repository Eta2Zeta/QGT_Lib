"""Project g-wave Berry-curvature grids onto every irrep of D3d."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import tempfile

import numpy as np

from Library.GroupTheory import (
    D3dPointGroup,
    calculate_numerical_qgt_group_orbit,
)
from Library.Hamiltonian import gWaveAltermagnetHamiltonian
from Library.data_management_utils_common import (
    dump_metadata,
    pick_or_create_result_dir_simple,
)
from Library.dimension_lib import create_2d_coordinate_grid, map_k_by_order


BUNDLE_FILENAME = "qgt_d3d_all_irrep_projection_bundle.npz"


def _project_irrep_worker(
    irrep,
    characters,
    irrep_dimension,
    group_order,
    berry_path,
    output_path,
):
    """Project both stored Berry components for one irrep in one process."""
    berry_xy = np.load(berry_path, mmap_mode="r", allow_pickle=False)
    coefficients = (
        float(irrep_dimension)
        / float(group_order)
        * np.conjugate(np.asarray(characters))
    )
    projected = np.zeros(
        berry_xy.shape[1:],
        dtype=np.result_type(berry_xy.dtype, coefficients.dtype),
    )
    for element_index, coefficient in enumerate(coefficients):
        if coefficient != 0:
            projected += coefficient * berry_xy[element_index]

    projected = np.real_if_close(projected)
    np.save(output_path, projected)
    return irrep, output_path, os.getpid()


def _project_all_irreps_parallel(berry_xy_by_group, group, results_dir):
    """Run exactly one spawned worker for each irrep and preserve irrep order."""
    irreps = tuple(group.irreps)
    worker_pids = {}
    output_paths = {}

    with tempfile.TemporaryDirectory(
        prefix="d3d_irrep_projection_",
        dir=results_dir,
    ) as temp_dir:
        berry_path = os.path.join(temp_dir, "berry_xy_by_group.npy")
        np.save(berry_path, berry_xy_by_group)

        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(irreps),
            mp_context=spawn_context,
            max_tasks_per_child=1,
        ) as executor:
            futures = {}
            for irrep in irreps:
                output_path = os.path.join(temp_dir, f"projected_{irrep}.npy")
                future = executor.submit(
                    _project_irrep_worker,
                    irrep,
                    group.characters(irrep),
                    group.irrep_dimension(irrep),
                    group.order,
                    berry_path,
                    output_path,
                )
                futures[future] = irrep

            for future in as_completed(futures):
                irrep, output_path, worker_pid = future.result()
                output_paths[irrep] = output_path
                worker_pids[irrep] = worker_pid
                print(f"Finished {irrep} projection in process {worker_pid}.")

        projected_by_irrep = np.stack(
            [np.load(output_paths[irrep], allow_pickle=False) for irrep in irreps],
            axis=0,
        )

    if len(set(worker_pids.values())) != len(irreps):
        raise RuntimeError("Each irrep must be calculated by a separate process.")
    return projected_by_irrep, worker_pids


def _projection_diagnostics(original, projected):
    """Return squared projected-norm ratios and relative residuals."""
    grid_axes = (-2, -1)
    original_norm_squared = np.sum(np.abs(original) ** 2, axis=grid_axes)
    projected_norm_squared = np.sum(np.abs(projected) ** 2, axis=grid_axes)
    residual_norm_squared = np.sum(
        np.abs(original - projected) ** 2,
        axis=grid_axes,
    )

    weights = np.divide(
        projected_norm_squared,
        original_norm_squared,
        out=np.full_like(projected_norm_squared, np.nan, dtype=float),
        where=original_norm_squared > 0,
    )
    relative_residuals = np.sqrt(
        np.divide(
            residual_norm_squared,
            original_norm_squared,
            out=np.full_like(residual_norm_squared, np.nan, dtype=float),
            where=original_norm_squared > 0,
        )
    )
    return weights, relative_residuals


def calculate_d3d_all_irrep_berry_projections(
    Hamiltonian_Obj,
    *,
    mesh_spacing=150,
    k_max=1.5 * np.pi,
    kk=0.0,
    band_indices=None,
    delta_k=1e-5,
    orbit_workers=6,
    force_new=False,
    results_root=None,
    plot_results=True,
    plot_zlim_berry=None,
    plot_zlim_percentile=99.0,
    plot_zlim_residual=None,
    plot_residual_zlim_percentile=99.0,
):
    """Calculate the D3d projections of ``Omega_x`` and ``Omega_y``.

    The 12 group-element QGT jobs are evaluated with ``orbit_workers`` spawned
    processes. The six character projections are subsequently evaluated
    concurrently, with one process per irrep.
    """
    group = D3dPointGroup()
    irrep_names = tuple(group.irreps)
    if (
        isinstance(orbit_workers, bool)
        or not isinstance(orbit_workers, (int, np.integer))
        or orbit_workers <= 0
    ):
        raise ValueError("orbit_workers must be a positive integer.")
    if band_indices is None:
        band_indices = tuple(range(Hamiltonian_Obj.dim))
    else:
        band_indices = tuple(int(index) for index in band_indices)

    ki, kj, grid_info = create_2d_coordinate_grid(
        k_max,
        mesh_spacing,
        order="xyz",
        include_endpoints=True,
    )
    kx, ky, kz = map_k_by_order(ki, kj, kk, "xyz")
    kx, ky, kz = np.broadcast_arrays(kx, ky, kz)
    cartesian_k_grid = np.stack((kx, ky, kz), axis=-1)

    metadata = {
        "calculation": "D3d_all_irrep_character_projection",
        "hamiltonian_name": Hamiltonian_Obj.name,
        "hamiltonian_params": Hamiltonian_Obj.get_parameters_dict(parameter="2D"),
        "point_group": group.name,
        "group_element_order": list(group.element_names),
        "irreps": list(irrep_names),
        "irrep_dimensions": {
            irrep: group.irrep_dimension(irrep)
            for irrep in irrep_names
        },
        "projection_normalization": "d_irrep / group_order",
        "projected_components": ["Omega_x", "Omega_y"],
        "projection_action": "coordinate_only",
        "parallelization": {
            "group_orbit": {
                "axis": "group_element",
                "tasks": group.order,
                "requested_workers": int(orbit_workers),
                "effective_workers": min(int(orbit_workers), group.order),
                "task_scope": "one group element",
                "execution_mode": (
                    "serial"
                    if orbit_workers == 1
                    else "spawned_process_pool"
                ),
                "start_method": None if orbit_workers == 1 else "spawn",
            },
            "irrep_projection": {
                "axis": "irrep",
                "processes": len(irrep_names),
                "tasks_per_process": 1,
                "start_method": "spawn",
                "data_sharing": "read_only_npy_memmap",
            },
        },
        "qgt_method": "numerical_phase_corrected",
        "berry_convention": {
            "Omega_x": "-2 Im(Q_yz)",
            "Omega_y": "+2 Im(Q_xz)",
        },
        "band_indices": list(band_indices),
        "delta_k": float(delta_k),
        "mesh_spacing": int(mesh_spacing),
        "ki_range": grid_info["ki_range"],
        "kj_range": grid_info["kj_range"],
        "dki": grid_info["dki"],
        "dkj": grid_info["dkj"],
        "kk": float(kk),
        "kvals_mode": grid_info["sampling"],
        "order": "xyz",
    }

    if results_root is None:
        results_root = os.path.join(
            os.getcwd(),
            "results",
            "2D_QGT_Projection_results",
            Hamiltonian_Obj.name,
        )

    results_dir, use_existing = pick_or_create_result_dir_simple(
        base_root=results_root,
        base_name="dataset_",
        required_params=metadata,
        force_new=force_new,
        required_files=[BUNDLE_FILENAME, "meta.json"],
    )
    bundle_path = os.path.join(results_dir, BUNDLE_FILENAME)
    metadata_path = os.path.join(results_dir, "meta.json")

    if use_existing:
        print(f"Using existing D3d all-irrep projection: {bundle_path}")
        with np.load(bundle_path, allow_pickle=False) as bundle:
            saved_data = {key: bundle[key] for key in bundle.files}
    else:
        dump_metadata(metadata, metadata_path)

        orbit = calculate_numerical_qgt_group_orbit(
            Hamiltonian_Obj,
            cartesian_k_grid,
            group,
            band_indices=band_indices,
            delta_k=delta_k,
            max_workers=orbit_workers,
        )

        identity_position = group.element_names.index(group.identity.name)
        original_omega_x = orbit.berry_curvature[
            identity_position,
            ...,
            0,
        ]
        original_omega_y = orbit.berry_curvature[
            identity_position,
            ...,
            1,
        ]

        projected_xy_by_irrep, worker_pids = _project_all_irreps_parallel(
            orbit.berry_curvature[..., :2],
            group,
            results_dir,
        )
        projected_omega_x = projected_xy_by_irrep[..., 0]
        projected_omega_y = projected_xy_by_irrep[..., 1]
        residual_omega_x = (
            original_omega_x[np.newaxis, ...] - projected_omega_x
        )
        residual_omega_y = (
            original_omega_y[np.newaxis, ...] - projected_omega_y
        )
        omega_x_projected_norm_ratio, omega_x_relative_residual = (
            _projection_diagnostics(
                original_omega_x[np.newaxis, ...],
                projected_omega_x,
            )
        )
        omega_y_projected_norm_ratio, omega_y_relative_residual = (
            _projection_diagnostics(
                original_omega_y[np.newaxis, ...],
                projected_omega_y,
            )
        )

        saved_data = {
            "ki": ki,
            "kj": kj,
            "cartesian_k_grid": cartesian_k_grid,
            "group_element_names": np.asarray(group.element_names, dtype="U"),
            "irrep_names": np.asarray(irrep_names, dtype="U"),
            "irrep_dimensions": np.asarray(
                [group.irrep_dimension(irrep) for irrep in irrep_names],
                dtype=int,
            ),
            "irrep_characters_by_element": np.asarray(
                [group.characters(irrep) for irrep in irrep_names]
            ),
            "irrep_worker_pids": np.asarray(
                [worker_pids[irrep] for irrep in irrep_names],
                dtype=int,
            ),
            "band_indices": np.asarray(band_indices, dtype=int),
            "inverse_transformed_k_grids": orbit.inverse_transformed_k_grids,
            "eigenvalues_by_group": orbit.eigenvalues,
            "eigenfunctions_by_group": orbit.eigenfunctions,
            "berry_curvature_by_group": orbit.berry_curvature,
            "omega_x_original": original_omega_x,
            "omega_y_original": original_omega_y,
            "omega_x_projected_by_irrep": projected_omega_x,
            "omega_y_projected_by_irrep": projected_omega_y,
            "omega_x_residual_by_irrep": residual_omega_x,
            "omega_y_residual_by_irrep": residual_omega_y,
            "omega_x_projected_norm_ratio_by_irrep": (
                omega_x_projected_norm_ratio
            ),
            "omega_y_projected_norm_ratio_by_irrep": (
                omega_y_projected_norm_ratio
            ),
            "omega_x_relative_residual_by_irrep": omega_x_relative_residual,
            "omega_y_relative_residual_by_irrep": omega_y_relative_residual,
        }
        for component_name, component_data in orbit.qgt_components.items():
            saved_data[f"qgt_{component_name}_by_group"] = component_data

        temporary_bundle_path = f"{bundle_path}.tmp"
        with open(temporary_bundle_path, "wb") as temporary_bundle:
            np.savez_compressed(temporary_bundle, **saved_data)
        os.replace(temporary_bundle_path, bundle_path)

        print("D3d all-irrep Berry-curvature projections complete.")
        print(f"Saved projection bundle to: {bundle_path}")

    if plot_results:
        from Library.plotting_qgt_2d import (
            plot_berry_irrep_projection_heatmaps,
        )

    for irrep_position, irrep_value in enumerate(saved_data["irrep_names"]):
        irrep = str(irrep_value)
        for band_position, band_index in enumerate(saved_data["band_indices"]):
            print(
                f"{irrep}, band {band_index}: "
                f"Omega_x projected norm ratio="
                f"{saved_data['omega_x_projected_norm_ratio_by_irrep'][irrep_position, band_position]:.8g}, "
                f"relative residual="
                f"{saved_data['omega_x_relative_residual_by_irrep'][irrep_position, band_position]:.8g}; "
                f"Omega_y projected norm ratio="
                f"{saved_data['omega_y_projected_norm_ratio_by_irrep'][irrep_position, band_position]:.8g}, "
                f"relative residual="
                f"{saved_data['omega_y_relative_residual_by_irrep'][irrep_position, band_position]:.8g}"
            )

            if plot_results:
                plot_berry_irrep_projection_heatmaps(
                    saved_data["ki"],
                    saved_data["kj"],
                    saved_data["omega_x_original"][band_position],
                    saved_data["omega_y_original"][band_position],
                    saved_data["omega_x_projected_by_irrep"][
                        irrep_position,
                        band_position,
                    ],
                    saved_data["omega_y_projected_by_irrep"][
                        irrep_position,
                        band_position,
                    ],
                    saved_data["omega_x_residual_by_irrep"][
                        irrep_position,
                        band_position,
                    ],
                    saved_data["omega_y_residual_by_irrep"][
                        irrep_position,
                        band_position,
                    ],
                    irrep=irrep,
                    band_index=int(band_index),
                    zlim_berry=plot_zlim_berry,
                    zlim_residual=plot_zlim_residual,
                    zlim_percentile=plot_zlim_percentile,
                    residual_zlim_percentile=plot_residual_zlim_percentile,
                    hamiltonian=Hamiltonian_Obj,
                    kk=kk,
                    results_dir=results_dir,
                    save_fig=True,
                )

    return saved_data, results_dir


def main():
    Hamiltonian_Obj = gWaveAltermagnetHamiltonian(
        Jx=0.0,
        Jy=0.0,
        Jz=0.2,
        lamb=0.1,
        lamb_z=0.5,
        A0=0.0,
        magnus_order=0,
        t1=0.3,
        t2=0.3,
        t3=0.3,
        t4=0.3,
    )

    calculate_d3d_all_irrep_berry_projections(
        Hamiltonian_Obj,
        mesh_spacing=150,
        k_max=1.5 * np.pi,
        kk=0.0,
        band_indices=tuple(range(Hamiltonian_Obj.dim)),
        delta_k=1e-5,
        orbit_workers=6,
        force_new=False,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
