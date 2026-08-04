"""Numerical QGT calculations on complete point-group orbits of k-space grids."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from ..QGT_calc_functions_3comp import (
    quantum_geometric_tensor_3d_num_phase_corrected,
)
from ..QGT_lib import QGT_grid_num
from ..eigenvalue_calc_lib import grid_eigenvalues_eigenfunctions
from .point_groups import PointGroup


QGT_COMPONENT_NAMES = (
    "xx",
    "yy",
    "zz",
    "xy_real",
    "xy_imag",
    "xz_real",
    "xz_imag",
    "yz_real",
    "yz_imag",
)

BERRY_COMPONENT_INDICES = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def _calculate_numerical_qgt_group_element(
    hamiltonian,
    transformed_grid,
    element_index,
    element_name,
    band_indices,
    delta_k,
    *,
    show_progress,
):
    """Calculate the eigenpairs and QGT grids for one group element."""
    mesh_spacing = transformed_grid.shape[0]
    transformed_kx = transformed_grid[..., 0]
    transformed_ky = transformed_grid[..., 1]
    transformed_kz = transformed_grid[..., 2]
    fixed_kz = float(transformed_kz.flat[0])

    eigenvalues, eigenfunctions, *_ = grid_eigenvalues_eigenfunctions(
        hamiltonian,
        transformed_kx,
        transformed_ky,
        mesh_spacing,
        hamiltonian.dim,
        kk=fixed_kz,
        order="xyz",
        show_progress=show_progress,
    )

    qgt_for_element = {name: [] for name in QGT_COMPONENT_NAMES}
    for band_index in band_indices:
        qgt_result = QGT_grid_num(
            transformed_kx,
            transformed_ky,
            eigenvalues,
            eigenfunctions,
            quantum_geometric_tensor_3d_num_phase_corrected,
            hamiltonian,
            delta_k,
            band_index,
            progress_label=f"{element_name}, band {band_index}",
            kk=fixed_kz,
            order="xyz",
            show_progress=show_progress,
        )
        for component_name, component_grid in zip(
            QGT_COMPONENT_NAMES,
            qgt_result,
        ):
            qgt_for_element[component_name].append(component_grid)

    stacked_qgt = {
        component_name: np.stack(component_grids, axis=0)
        for component_name, component_grids in qgt_for_element.items()
    }
    return (
        element_index,
        element_name,
        eigenvalues,
        eigenfunctions,
        stacked_qgt,
    )


@dataclass
class NumericalQGTGroupOrbit:
    """Stored numerical data for every inverse-transformed group grid."""

    group_name: str
    element_names: tuple[str, ...]
    inverse_transformed_k_grids: np.ndarray
    eigenvalues: np.ndarray
    eigenfunctions: np.ndarray
    qgt_components: dict[str, np.ndarray]
    berry_curvature: np.ndarray
    band_indices: tuple[int, ...]
    delta_k: float

    def project_qgt_component(
        self,
        group,
        irrep,
        component,
        *,
        band_index=None,
    ):
        """Project one stored QGT component without recalculating the orbit."""
        self._validate_group(group)
        try:
            transformed_data = self.qgt_components[component]
        except KeyError as error:
            valid = ", ".join(self.qgt_components)
            raise KeyError(
                f"Unknown QGT component {component!r}. Valid: {valid}."
            ) from error
        transformed_data = self._select_band(transformed_data, band_index)
        return group.project_onto_irrep(transformed_data, irrep)

    def project_berry_component(
        self,
        group,
        irrep,
        component,
        *,
        band_index=None,
    ):
        """Project one Cartesian Berry-curvature component."""
        self._validate_group(group)
        try:
            component_index = BERRY_COMPONENT_INDICES[component]
        except KeyError as error:
            raise KeyError("Berry component must be 'x', 'y', or 'z'.") from error
        transformed_data = self.berry_curvature[..., component_index]
        transformed_data = self._select_band(transformed_data, band_index)
        return group.project_onto_irrep(transformed_data, irrep)

    def _select_band(self, transformed_data, band_index):
        if band_index is None:
            return transformed_data
        try:
            band_position = self.band_indices.index(int(band_index))
        except ValueError as error:
            valid = ", ".join(str(index) for index in self.band_indices)
            raise IndexError(
                f"Band {band_index} is not stored in this orbit. Valid: {valid}."
            ) from error
        return transformed_data[:, band_position]

    def _validate_group(self, group):
        if group.name != self.group_name:
            raise ValueError(
                f"Orbit was calculated for {self.group_name}, not {group.name}."
            )
        if tuple(group.element_names) != self.element_names:
            raise ValueError(
                "The group's element order does not match the stored orbit data."
            )


def calculate_numerical_qgt_group_orbit(
    hamiltonian,
    cartesian_k_grid,
    group: PointGroup,
    band_indices,
    *,
    delta_k=1e-5,
    max_workers=6,
    show_progress=True,
):
    """Calculate eigenpairs and QGT data on every ``R_g^{-1} k`` grid.

    The existing grid eigensolver and phase-corrected numerical QGT routine
    are used unchanged for each group element. ``cartesian_k_grid`` must have
    shape ``(N, N, 3)``. Its final axis contains ``(kx, ky, kz)``.

    The current two-dimensional grid routines require each transformed grid
    to lie in a plane of constant ``kz``. This condition is satisfied for
    horizontal planes acted on by ``D3d``.

    One task is submitted for each group element. At most ``max_workers``
    spawned processes evaluate those tasks concurrently; the default is six.
    Results are restored to ``group.elements`` order before they are returned.
    """
    cartesian_k_grid = np.asarray(cartesian_k_grid, dtype=float)
    if (
        cartesian_k_grid.ndim != 3
        or cartesian_k_grid.shape[-1] != 3
        or cartesian_k_grid.shape[0] != cartesian_k_grid.shape[1]
    ):
        raise ValueError("cartesian_k_grid must have shape (N, N, 3).")

    band_indices = tuple(int(index) for index in band_indices)
    if not band_indices:
        raise ValueError("band_indices must contain at least one band.")
    if len(set(band_indices)) != len(band_indices):
        raise ValueError("band_indices must not contain duplicates.")
    invalid_bands = [
        index for index in band_indices if not 0 <= index < hamiltonian.dim
    ]
    if invalid_bands:
        raise ValueError(
            f"Invalid band indices {invalid_bands}; valid indices are between "
            f"0 and {hamiltonian.dim - 1}."
        )

    delta_k = float(delta_k)
    if not np.isfinite(delta_k) or delta_k <= 0:
        raise ValueError("delta_k must be a finite positive number.")

    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, (int, np.integer))
        or max_workers <= 0
    ):
        raise ValueError("max_workers must be a positive integer.")
    worker_count = min(int(max_workers), group.order)

    transformed_grids = []
    for element in group.elements:
        transformed_grid = group.transform_k(
            element,
            cartesian_k_grid,
            inverse=True,
        )
        transformed_kz = transformed_grid[..., 2]
        fixed_kz = float(transformed_kz.flat[0])
        if not np.allclose(transformed_kz, fixed_kz, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Transformed grid for {element.name} does not have constant kz; "
                "the existing 2D grid routines cannot evaluate it."
            )
        transformed_grids.append(transformed_grid)

    transformed_grids = np.stack(transformed_grids, axis=0)
    received_elements = np.zeros(group.order, dtype=bool)
    eigenvalue_grids = None
    eigenfunction_grids = None
    qgt_grids = None

    def store_element_result(result):
        nonlocal eigenvalue_grids, eigenfunction_grids, qgt_grids

        element_index, element_name, eigenvalues, eigenfunctions, qgt = result
        expected_name = group.element_names[element_index]
        if element_name != expected_name:
            raise RuntimeError(
                "A QGT group-element worker returned mismatched output."
            )
        if received_elements[element_index]:
            raise RuntimeError(
                f"Received duplicate QGT output for {element_name}."
            )

        if eigenvalue_grids is None:
            eigenvalue_grids = np.empty(
                (group.order,) + eigenvalues.shape,
                dtype=eigenvalues.dtype,
            )
            eigenfunction_grids = np.empty(
                (group.order,) + eigenfunctions.shape,
                dtype=eigenfunctions.dtype,
            )
            qgt_grids = {
                component_name: np.empty(
                    (group.order,) + qgt[component_name].shape,
                    dtype=qgt[component_name].dtype,
                )
                for component_name in QGT_COMPONENT_NAMES
            }

        eigenvalue_grids[element_index] = eigenvalues
        eigenfunction_grids[element_index] = eigenfunctions
        for component_name in QGT_COMPONENT_NAMES:
            qgt_grids[component_name][element_index] = qgt[component_name]
        received_elements[element_index] = True

    if worker_count == 1:
        for element_index, (element, transformed_grid) in enumerate(
            zip(group.elements, transformed_grids)
        ):
            if show_progress:
                print(
                    f"Calculating QGT group orbit "
                    f"{element_index + 1}/{group.order}: {element.name}"
                )
            result = _calculate_numerical_qgt_group_element(
                hamiltonian,
                transformed_grid,
                element_index,
                element.name,
                band_indices,
                delta_k,
                show_progress=show_progress,
            )
            store_element_result(result)
            del result
    else:
        if show_progress:
            print(
                f"Calculating {group.order} QGT group-element jobs with "
                f"{worker_count} processes."
            )
        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=spawn_context,
        ) as executor:
            futures = {
                executor.submit(
                    _calculate_numerical_qgt_group_element,
                    hamiltonian,
                    transformed_grid,
                    element_index,
                    element.name,
                    band_indices,
                    delta_k,
                    show_progress=False,
                ): (element_index, element.name)
                for element_index, (element, transformed_grid) in enumerate(
                    zip(group.elements, transformed_grids)
                )
            }
            completed_futures = tqdm(
                as_completed(futures),
                total=group.order,
                desc="Calculating QGT group orbit",
                unit="element",
                disable=not show_progress,
            )
            for future in completed_futures:
                element_index, element_name = futures.pop(future)
                result = future.result()
                if result[0] != element_index or result[1] != element_name:
                    raise RuntimeError(
                        "A QGT group-element worker returned mismatched output."
                    )
                store_element_result(result)
                completed_futures.set_postfix_str(element_name)
                del result
                del future
            del completed_futures

    if not np.all(received_elements):
        missing_elements = np.asarray(group.element_names)[~received_elements]
        raise RuntimeError(
            f"Missing QGT results for group elements {missing_elements.tolist()}."
        )

    stacked_qgt = qgt_grids
    berry_curvature = np.stack(
        (
            -2.0 * stacked_qgt["yz_imag"],
            2.0 * stacked_qgt["xz_imag"],
            -2.0 * stacked_qgt["xy_imag"],
        ),
        axis=-1,
    )

    return NumericalQGTGroupOrbit(
        group_name=group.name,
        element_names=group.element_names,
        inverse_transformed_k_grids=transformed_grids,
        eigenvalues=eigenvalue_grids,
        eigenfunctions=eigenfunction_grids,
        qgt_components=stacked_qgt,
        berry_curvature=berry_curvature,
        band_indices=band_indices,
        delta_k=delta_k,
    )
