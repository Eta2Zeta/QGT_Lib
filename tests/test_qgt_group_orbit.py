import numpy as np
import pytest

from Library.GroupTheory import (
    D3dPointGroup,
    calculate_numerical_qgt_group_orbit,
)
from Library.GroupTheory import qgt_orbit as qgt_orbit_module


class FakeHamiltonian:
    dim = 2


def _cartesian_grid(kz=0.0):
    kx, ky = np.meshgrid(
        np.array([-0.5, 0.5]),
        np.array([-0.25, 0.25]),
    )
    return np.stack((kx, ky, np.full_like(kx, kz)), axis=-1)


def test_qgt_group_orbit_uses_existing_grid_calculations_for_every_element(
    monkeypatch,
):
    group = D3dPointGroup()
    hamiltonian = FakeHamiltonian()
    k_grid = _cartesian_grid(kz=0.4)
    eigen_calls = []
    qgt_calls = []

    def fake_grid_eigenvalues(
        hamiltonian_arg,
        kx,
        ky,
        mesh_spacing,
        dim,
        *,
        kk,
        order,
        show_progress,
    ):
        eigen_calls.append((kx.copy(), ky.copy(), kk, order))
        eigenvalues = np.zeros(kx.shape + (dim,))
        eigenfunctions = np.broadcast_to(
            np.eye(dim, dtype=complex),
            kx.shape + (dim, dim),
        ).copy()
        matrix_grid = np.zeros(kx.shape + (dim, dim), dtype=complex)
        return eigenvalues, eigenfunctions, matrix_grid, matrix_grid.copy()

    def fake_qgt_grid(
        kx,
        ky,
        eigenvalues,
        eigenfunctions,
        qgt_function,
        hamiltonian_arg,
        delta_k,
        band_index,
        *,
        progress_label,
        kk,
        order,
        show_progress,
    ):
        qgt_calls.append((kx.copy(), ky.copy(), kk, progress_label, order))
        return tuple(
            np.full(kx.shape, component_index + 1.0 + 10.0 * band_index)
            for component_index in range(9)
        )

    monkeypatch.setattr(
        qgt_orbit_module,
        "grid_eigenvalues_eigenfunctions",
        fake_grid_eigenvalues,
    )
    monkeypatch.setattr(qgt_orbit_module, "QGT_grid_num", fake_qgt_grid)

    orbit = calculate_numerical_qgt_group_orbit(
        hamiltonian,
        k_grid,
        group,
        band_indices=(0, 1),
        delta_k=2e-5,
        max_workers=1,
    )

    assert len(eigen_calls) == group.order
    assert len(qgt_calls) == group.order * 2
    assert orbit.element_names == group.element_names
    assert orbit.inverse_transformed_k_grids.shape == (12, 2, 2, 3)
    assert orbit.eigenvalues.shape == (12, 2, 2, 2)
    assert orbit.eigenfunctions.shape == (12, 2, 2, 2, 2)
    assert orbit.qgt_components["xz_imag"].shape == (12, 2, 2, 2)
    assert orbit.berry_curvature.shape == (12, 2, 2, 2, 3)

    for element_index, element in enumerate(group.elements):
        expected_grid = group.transform_k(element, k_grid, inverse=True)
        assert np.allclose(
            orbit.inverse_transformed_k_grids[element_index],
            expected_grid,
        )
        assert np.allclose(eigen_calls[element_index][0], expected_grid[..., 0])
        assert np.allclose(eigen_calls[element_index][1], expected_grid[..., 1])
        assert np.isclose(eigen_calls[element_index][2], expected_grid[0, 0, 2])
        assert qgt_calls[2 * element_index][3] == f"{element.name}, band 0"
        assert qgt_calls[2 * element_index + 1][3] == f"{element.name}, band 1"

    assert np.allclose(orbit.berry_curvature[:, 0, ..., 0], -18.0)
    assert np.allclose(orbit.berry_curvature[:, 0, ..., 1], 14.0)
    assert np.allclose(orbit.berry_curvature[:, 0, ..., 2], -10.0)
    assert np.allclose(orbit.berry_curvature[:, 1, ..., 0], -38.0)
    assert np.allclose(
        orbit.project_berry_component(
            group,
            "A1g",
            "x",
            band_index=0,
        ),
        -18.0,
    )
    assert np.allclose(
        orbit.project_berry_component(group, "Eg", "x"),
        0.0,
    )


def test_qgt_group_orbit_defaults_to_six_parallel_workers(monkeypatch):
    executor_calls = []
    submitted_element_indices = []

    class ImmediateFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class RecordingExecutor:
        def __init__(self, *, max_workers, mp_context):
            executor_calls.append((max_workers, mp_context.get_start_method()))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def submit(self, function, *args, **kwargs):
            submitted_element_indices.append(args[2])
            return ImmediateFuture(function(*args, **kwargs))

    def fake_element_calculation(
        hamiltonian,
        transformed_grid,
        element_index,
        element_name,
        band_indices,
        delta_k,
        *,
        show_progress,
    ):
        grid_shape = transformed_grid.shape[:2]
        eigenvalues = np.full(
            grid_shape + (hamiltonian.dim,),
            element_index,
            dtype=float,
        )
        eigenfunctions = np.full(
            grid_shape + (hamiltonian.dim, hamiltonian.dim),
            element_index,
            dtype=complex,
        )
        qgt = {
            component_name: np.full(
                (len(band_indices),) + grid_shape,
                element_index,
                dtype=float,
            )
            for component_name in qgt_orbit_module.QGT_COMPONENT_NAMES
        }
        return (
            element_index,
            element_name,
            eigenvalues,
            eigenfunctions,
            qgt,
        )

    monkeypatch.setattr(
        qgt_orbit_module,
        "ProcessPoolExecutor",
        RecordingExecutor,
    )
    monkeypatch.setattr(
        qgt_orbit_module,
        "as_completed",
        lambda futures: reversed(tuple(futures)),
    )
    monkeypatch.setattr(
        qgt_orbit_module,
        "_calculate_numerical_qgt_group_element",
        fake_element_calculation,
    )

    group = D3dPointGroup()
    orbit = calculate_numerical_qgt_group_orbit(
        FakeHamiltonian(),
        _cartesian_grid(),
        group,
        band_indices=(0,),
        show_progress=False,
    )

    assert executor_calls == [(6, "spawn")]
    assert submitted_element_indices == list(range(group.order))
    assert orbit.element_names == group.element_names
    assert np.allclose(
        orbit.eigenvalues[:, 0, 0, 0],
        np.arange(group.order),
    )
    assert np.allclose(
        orbit.qgt_components["xx"][:, 0, 0, 0],
        np.arange(group.order),
    )


def test_qgt_group_orbit_rejects_grid_without_constant_transformed_kz(
    monkeypatch,
):
    k_grid = _cartesian_grid()
    k_grid[..., 2] = np.array([[0.0, 0.1], [0.2, 0.3]])

    with pytest.raises(ValueError, match="does not have constant kz"):
        calculate_numerical_qgt_group_orbit(
            FakeHamiltonian(),
            k_grid,
            D3dPointGroup(),
            band_indices=(0,),
        )
