"""The Kubo QGT and the link-variable Berry flux against closed forms and each other."""

import numpy as np
import pytest

from qgt import hamiltonians
from qgt.hamiltonians.base import hamiltonian
from qgt.hamiltonians.evaluate import hamiltonian_at
from qgt.hamiltonians.minimal_models import MINIMAL_MODELS
from qgt.geometry.kubo import berry_curvature, qgt_components, quantum_geometric_tensor
from qgt.geometry.qgt_2d import QGT_grid_kubo
from qgt.geometry.qgt_3d import QGT_3d_kubo
from qgt.topology import chern_number_from_links, monopole_charge

SIGMA = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)


class DVectorModel(hamiltonian):
    """H = d(k).sigma with the Qi-Wu-Zhang d = (sin kx, sin ky, m + cos kx + cos ky), copied ``copies`` times."""

    def __init__(self, m=1.0, copies=1):
        super().__init__(dim=2 * copies, omega=1.0)
        self.name = "DVector"
        self.m = m
        self.copies = copies

    def d(self, kx, ky):
        return np.array([np.sin(kx), np.sin(ky), self.m + np.cos(kx) + np.cos(ky)])

    def compute_static(self, kx, ky, kz=0):
        H = np.einsum("a,aij->ij", self.d(kx, ky), SIGMA)
        return np.kron(np.eye(self.copies), H)


class Weyl(hamiltonian):
    def __init__(self):
        super().__init__(dim=2, omega=1.0)
        self.name = "Weyl"

    def compute_static(self, kx, ky, kz=0):
        return np.einsum("a,aij->ij", np.array([kx, ky, kz]), SIGMA)


def closed_form_qgt(model, kx, ky, band, step=1e-6):
    """Two-band metric (1/4) dn_i.dn_j and curvature -/+ (1/2) n.(dn_x x dn_y), n = d/|d|."""
    n = lambda kx, ky: model.d(kx, ky) / np.linalg.norm(model.d(kx, ky))
    dn = [(n(kx + step, ky) - n(kx - step, ky)) / (2 * step),
          (n(kx, ky + step) - n(kx, ky - step)) / (2 * step)]
    metric = np.array([[dn[i] @ dn[j] / 4 for j in range(2)] for i in range(2)])
    curvature = (1 if band == 0 else -1) * 0.5 * n(kx, ky) @ np.cross(dn[0], dn[1])
    return metric, curvature


MOMENTA = np.random.default_rng(3).uniform(-np.pi, np.pi, size=(20, 3))


def test_vectorized_hamiltonians_match_pointwise_ones():
    models = [cls() for cls in MINIMAL_MODELS.values()] + [
        hamiltonians.AltermagnetHamiltonian(), hamiltonians.RuO2Hamiltonian(),
        hamiltonians.ChiralHamiltonianChiralBasisProjected(), hamiltonians.HaldaneHamiltonian(),
    ]
    for model in models:
        expected = np.stack([model.compute_static(*k) for k in MOMENTA])
        np.testing.assert_allclose(hamiltonian_at(model, MOMENTA), expected, atol=1e-12, err_msg=model.name)


def test_driven_models_use_the_effective_hamiltonian():
    model = hamiltonians.HaldaneHamiltonian(A0=0.2, omega=6.0)
    expected = np.stack([model.effective_hamiltonian(*k)[0] for k in MOMENTA[:3]])
    np.testing.assert_allclose(hamiltonian_at(model, MOMENTA[:3]), expected)
    assert not np.allclose(expected, np.stack([model.compute_static(*k) for k in MOMENTA[:3]]))


@pytest.mark.parametrize("band", [0, 1])
def test_two_band_qgt_matches_closed_form(band):
    model = DVectorModel(m=1.3)
    Q = quantum_geometric_tensor(model, MOMENTA, band)
    for k, Qk in zip(MOMENTA, Q):
        metric, curvature = closed_form_qgt(model, k[0], k[1], band)
        np.testing.assert_allclose(Qk[:2, :2].real, metric, rtol=1e-5, atol=1e-9)
        assert berry_curvature(Qk)[2] == pytest.approx(curvature, rel=1e-5, abs=1e-9)
        np.testing.assert_allclose(Qk[2], 0, atol=1e-12)  # H does not depend on kz


def test_every_band_at_once_matches_one_band_at_a_time():
    model = MINIMAL_MODELS["MinimalHamSG194_2a"]()
    every = quantum_geometric_tensor(model, MOMENTA, None)
    for band in range(model.dim):
        np.testing.assert_allclose(every[:, band], quantum_geometric_tensor(model, MOMENTA, band))


def test_stored_eigenvectors_set_the_band_numbering():
    model = DVectorModel(m=1.3)
    _, U = np.linalg.eigh(hamiltonian_at(model, MOMENTA))
    swapped = np.swapaxes(U, -1, -2)[..., ::-1, :] * np.exp(0.7j)  # rows, upper band first
    np.testing.assert_allclose(
        quantum_geometric_tensor(model, MOMENTA, 0, eigenvectors=swapped),
        quantum_geometric_tensor(model, MOMENTA, 1),
    )
    swapped[2] = np.nan
    assert np.isnan(quantum_geometric_tensor(model, MOMENTA, 0, eigenvectors=swapped)[2]).all()


def test_band_group_stays_finite_where_its_bands_are_degenerate():
    single, doubled = DVectorModel(m=1.3), DVectorModel(m=1.3, copies=2)
    group = quantum_geometric_tensor(doubled, MOMENTA, [0, 1])
    np.testing.assert_allclose(group, 2 * quantum_geometric_tensor(single, MOMENTA, 0), rtol=1e-6, atol=1e-10)
    # A single band of a degenerate pair has no QGT of its own.
    assert np.isnan(quantum_geometric_tensor(doubled, MOMENTA, 0)).all()


def test_grid_functions_return_the_nine_components():
    model = DVectorModel(m=1.3)
    ki, kj = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-2, 2, 3))
    grid = QGT_grid_kubo(ki, kj, model, 0, kk=0.4, order="xzy")
    k = np.stack([ki, np.full(ki.shape, 0.4), kj], axis=-1)  # 'xzy': ki -> kx, kj -> kz, kk -> ky
    for actual, expected in zip(grid, qgt_components(quantum_geometric_tensor(model, k, 0))):
        np.testing.assert_allclose(actual, expected)
    vals = np.linspace(-1, 1, 3)
    volume = QGT_3d_kubo(vals, vals, vals, model, 1)
    assert all(component.shape == (3, 3, 3) for component in volume)


@pytest.mark.parametrize("m, expected", [(1.0, 1), (-1.0, 1), (2.5, 0)])
def test_chern_number_from_links_matches_curvature_integral(m, expected):
    model = DVectorModel(m=m)
    b1, b2 = np.array([2 * np.pi, 0.0]), np.array([0.0, 2 * np.pi])
    chern = chern_number_from_links(model, 0, b1, b2, mesh=24)
    n = 200
    s = (np.arange(n) + 0.5) / n * 2 * np.pi
    kx, ky = np.meshgrid(s, s, indexing="ij")
    k = np.stack([kx, ky, np.zeros_like(kx)], axis=-1)
    integral = berry_curvature(quantum_geometric_tensor(model, k, 0))[..., 2].mean() * 2 * np.pi
    assert chern == pytest.approx(round(chern), abs=1e-9)
    assert abs(round(chern)) == expected
    assert chern == pytest.approx(integral, abs=1e-6)


def test_haldane_chern_number_is_minus_one_and_orientation_independent():
    model = hamiltonians.HaldaneHamiltonian()
    assert chern_number_from_links(model, 0, model.b1, model.b2, mesh=30) == pytest.approx(-1, abs=1e-9)
    assert chern_number_from_links(model, 0, model.b2, model.b1, mesh=30) == pytest.approx(-1, abs=1e-9)
    assert chern_number_from_links(model, [0, 1], model.b1, model.b2, mesh=30) == pytest.approx(0, abs=1e-9)


def test_weyl_monopole_charge_matches_curvature_flux():
    model = Weyl()
    lower, upper = monopole_charge(model, 0, (0, 0, 0)), monopole_charge(model, 1, (0, 0, 0))
    assert lower == pytest.approx(1, abs=1e-9) and upper == pytest.approx(-1, abs=1e-9)
    n = 120
    theta = (np.arange(n) + 0.5) * np.pi / n
    phi = (np.arange(2 * n) + 0.5) * np.pi / n
    T, P = np.meshgrid(theta, phi, indexing="ij")
    normal = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1)
    radius = 0.01
    curvature = berry_curvature(quantum_geometric_tensor(model, radius * normal, 0))
    flux = np.sum(np.sum(curvature * normal, axis=-1) * np.sin(T)) * radius**2 * (np.pi / n) ** 2
    assert flux / (2 * np.pi) == pytest.approx(lower, abs=1e-3)
