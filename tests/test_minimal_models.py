"""Contracts for the complete minimal-model catalog and existing QGT workflows."""

import pickle

import numpy as np
import pytest

import Library.Hamiltonian.MinimalModels as minimal_models
from Library.Hamiltonian.Hamiltonian import hamiltonian
from Library.Hamiltonian.MinimalModels import (
    MINIMAL_MODELS,
    MinimalAltermagnetHamiltonian,
    MinimalHamSG194_2a,
)
from Library.Hamiltonian_helper import get_Hamiltonian
from Library.QGT_calc_functions_2comp import quantum_geometric_tensor_num
from Library.diagnalization import eigenvalues_and_vectors_eigenvalue_ordering
from Library.eigenvalue_calc_lib_1d import eigenvalues_along_path


MODEL_CLASSES = tuple(MINIMAL_MODELS.values())
MOMENTA = np.array(
    [[0.37, -0.43, 0.81], [-0.71, 0.54, -0.93], [1.13, -0.62, 1.27]]
)


def test_catalog_exports_all_models_with_one_shared_base():
    assert len(MINIMAL_MODELS) == 27
    for name, model_class in MINIMAL_MODELS.items():
        assert name == model_class.__name__
        assert model_class.__bases__ == (MinimalAltermagnetHamiltonian,)
        assert issubclass(model_class, hamiltonian)
        assert getattr(minimal_models, name) is model_class


@pytest.mark.parametrize(
    "name, zone_corner, default_path, scalar_energy",
    [
        ("MinimalHamSG11_2a2b2c2d", "R", "GXSYGZURTZ", -2.6),
        ("MinimalHamSG51_2a2b2c2d", "R", "GXSYGZURTZ", -2.6),
        ("MinimalHamSG128_2a2b", "A", "GXMGZRAZ", -2.6),
        ("MinimalHamSG163_2b", "A", "GMKGALHA", 2.9),
        ("MinimalHamSG176_2b", "A", "GMKGALHA", 2.9),
        ("MinimalHamSG223_2a", "R", "GXMGRXMR", -2.1),
    ],
)
def test_shared_lattice_dispersion_and_symmetry_paths(
    name, zone_corner, default_path, scalar_energy,
):
    model = MINIMAL_MODELS[name](t1=1.1, t2=0.4, mu=0.0)
    points, labels = model.get_sym_path()
    assert labels == list(default_path)
    np.testing.assert_array_equal(points["G"], np.zeros(3))
    assert points[zone_corner][2] == np.pi
    assert model.get_sym_path(f"G{zone_corner}")[1] == ["G", zone_corner]
    # The nonidentity Pauli terms are traceless, so trace(H)/4 isolates eps0.
    energy = np.trace(model.compute_static(*points[zone_corner])) / 4
    np.testing.assert_allclose(energy, scalar_energy, atol=1e-13)
    with pytest.raises(ValueError, match=f"Unknown SG{model.space_group} symmetry labels"):
        model.get_sym_path("GQ")


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_scalar_and_vectorized_matrices_and_analytic_spectrum(model_class):
    model = model_class(Jx=0.17, Jy=-0.13, Jz=0.29)
    assert model.dim == 4
    assert model.name == model_class.__name__
    scalar = np.stack([model.compute_static(*k) for k in MOMENTA])
    vectorized = model.compute_static_vectorized(*MOMENTA.T)
    assert vectorized.shape == (3, 4, 4)
    assert np.isfinite(vectorized).all()
    np.testing.assert_allclose(vectorized, scalar, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(vectorized, vectorized.conj().swapaxes(-1, -2))
    np.testing.assert_allclose(
        model.get_analytical_eigenvalues(*MOMENTA.T),
        np.linalg.eigvalsh(vectorized),
        rtol=1e-11,
        atol=1e-12,
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_momentum_broadcasting_and_shape_contract(model_class):
    model = model_class(Jx=0.17, Jy=-0.13, Jz=0.29)
    kx = np.array([0.37, -0.71])[:, None]
    ky = np.array([-0.43, 0.54, -0.62])[None, :]
    kz = 0.81
    broadcast_x, broadcast_y = np.broadcast_arrays(kx, ky)
    expected = np.array([
        [model.compute_static(x, y, kz) for x, y in zip(row_x, row_y)]
        for row_x, row_y in zip(broadcast_x, broadcast_y)
    ])
    vectorized = model.compute_static_vectorized(kx, ky, kz)
    assert vectorized.shape == (2, 3, 4, 4)
    np.testing.assert_allclose(vectorized, expected)
    energies = model.get_analytical_eigenvalues(kx, ky, kz)
    assert energies.shape == (2, 3, 4)
    np.testing.assert_allclose(energies, np.linalg.eigvalsh(expected), atol=1e-12)

    # Scalar calls to the vectorized APIs retain the historical leading axis.
    scalar_matrix = model.compute_static_vectorized(*MOMENTA[0])
    scalar_energies = model.get_analytical_eigenvalues(*MOMENTA[0])
    assert scalar_matrix.shape == (1, 4, 4)
    assert scalar_energies.shape == (1, 4)
    np.testing.assert_allclose(scalar_matrix[0], expected[0, 0])
    np.testing.assert_allclose(scalar_energies[0], np.linalg.eigvalsh(expected[0, 0]))

    fixed_xy = model.compute_static_vectorized(0.37, -0.43, MOMENTA[:, 2])
    assert fixed_xy.shape == (3, 4, 4)
    np.testing.assert_allclose(
        fixed_xy,
        np.stack([model.compute_static(0.37, -0.43, z) for z in MOMENTA[:, 2]]),
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_parameter_mutation_metadata_and_pickle_round_trip(model_class):
    model = model_class(Jx=0.17, Jy=-0.13, Jz=0.29)
    before = model.compute_static(*MOMENTA[0])
    model.mu += 0.42
    np.testing.assert_allclose(
        model.compute_static(*MOMENTA[0]), before - 0.42 * np.eye(4), atol=1e-13
    )
    model.Jy += 0.23
    sigma_y = np.array([[0, -1j], [1j, 0]])
    expected = before - 0.42 * np.eye(4) + 0.23 * np.kron(np.diag([1, -1]), sigma_y)
    np.testing.assert_allclose(model.compute_static(*MOMENTA[0]), expected, atol=1e-13)
    parameters = model.get_parameters_dict()
    assert parameters["mu"] == model.mu
    assert parameters["Jy"] == model.Jy
    assert parameters["omega"] == model.omega
    assert "omega" not in model.get_parameters_dict(parameter="1D")
    restored = pickle.loads(pickle.dumps(model))
    assert type(restored) is model_class
    assert restored.get_parameters_dict() == parameters
    np.testing.assert_allclose(restored.compute_static(*MOMENTA[0]), expected)


def test_static_helper_and_path_eigensolvers_use_minimal_objects():
    model = MinimalHamSG194_2a(Jx=0.17, Jy=-0.13, Jz=0.29)
    effective, perturbation = get_Hamiltonian(model, *MOMENTA[0])
    np.testing.assert_allclose(effective, model.compute_static(*MOMENTA[0]))
    np.testing.assert_array_equal(perturbation, np.zeros((4, 4)))
    numerical, eigenvectors = eigenvalues_along_path(model, MOMENTA)
    analytical, no_eigenvectors = eigenvalues_along_path(
        model, MOMENTA, use_analytical=True
    )
    assert eigenvectors.shape == (3, 4, 4)
    assert no_eigenvectors is None
    np.testing.assert_allclose(numerical, analytical, atol=1e-12)


@pytest.mark.parametrize("polarization", ["left", "right", "linear_x", "linear_y"])
def test_driven_fft_and_magnus_integration(polarization):
    model = MinimalHamSG194_2a(
        Jx=0.17, Jy=-0.13, Jz=0.29,
        A0=0.14, omega=5.2, magnus_order=2, polarization=polarization,
    )
    kx, ky, kz = MOMENTA[0]
    harmonics = model.fourier_components_fft([-2, -1, 0, 1, 2], kx, ky, kz, M=128)
    for harmonic in (0, 1, 2):
        np.testing.assert_allclose(
            harmonics[-harmonic], harmonics[harmonic].conj().T, atol=1e-13
        )
    time = 0.19
    shift_x = 0 if polarization == "linear_y" else model.A0 * np.cos(model.omega * time)
    shift_y = {
        "left": model.A0 * np.sin(model.omega * time),
        "right": -model.A0 * np.sin(model.omega * time),
        "linear_x": 0,
        "linear_y": model.A0 * np.cos(model.omega * time),
    }[polarization]
    np.testing.assert_allclose(
        model.compute_driven(time, kx, ky, kz),
        model.compute_static(kx + shift_x, ky + shift_y, kz),
    )
    effective, perturbation = get_Hamiltonian(model, kx, ky, kz, harmonics=harmonics)
    np.testing.assert_allclose(effective, effective.conj().T, atol=1e-13)
    expected_perturbation = sum(
        (harmonics[n] @ harmonics[-n] - harmonics[-n] @ harmonics[n]) / (n * model.omega)
        for n in (1, 2)
    )
    np.testing.assert_allclose(perturbation, expected_perturbation, atol=1e-13)
    np.testing.assert_allclose(effective, model.compute_static(kx, ky, kz) + perturbation)


def test_numeric_qgt_matches_independent_matrix_derivative_formula():
    model = MinimalHamSG194_2a(Jx=0.17, Jy=-0.13, Jz=0.29)
    k = MOMENTA[0]
    energies, eigenstates = eigenvalues_and_vectors_eigenvalue_ordering(model, *k)
    assert np.min(np.diff(energies.real)) > 1e-3
    step = 1e-5
    actual = quantum_geometric_tensor_num(
        model, k[0], k[1], step, energies, eigenstates, band_index=0, kz=k[2]
    )
    # Spectral expression is independent of eigenvector phase-fixing and derivatives.
    derivatives = []
    for direction in np.eye(3)[:2]:
        derivatives.append(
            (model.compute_static(*(k + step * direction))
             - model.compute_static(*(k - step * direction))) / (2 * step)
        )
    qgt = np.zeros((2, 2), dtype=complex)
    for i, derivative_i in enumerate(derivatives):
        for j, derivative_j in enumerate(derivatives):
            for band in range(1, 4):
                qgt[i, j] += (
                    np.vdot(eigenstates[0], derivative_i @ eigenstates[band])
                    * np.vdot(eigenstates[band], derivative_j @ eigenstates[0])
                    / (energies[0] - energies[band]) ** 2
                )
    expected = (qgt[0, 0].real, qgt[0, 1].real, qgt[0, 1].imag, qgt[1, 1].real)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-7)
