import numpy as np

from Library.Hamiltonian.Hamiltonian import hamiltonian
from Library.eigenvalue_calc_lib import grid_eigenvalues_eigenfunctions


class CircularTwoBandHamiltonian(hamiltonian):
    """Two-band model with exactly one circular-drive harmonic per direction."""

    def __init__(self, *, delta=1.5, coupling=2.5, A0=0.1, omega=2.0, magnus_order=0):
        super().__init__(
            dim=2,
            omega=omega,
            A0=A0,
            polarization="left",
            magnus_order=magnus_order,
            analytic_magnus=False,
        )
        self.delta = float(delta)
        self.coupling = float(coupling)
        self.fft_calls = 0

    def compute_static(self, kx, ky, kz=0):
        off_diagonal = self.coupling * (kx + 1j * ky)
        return np.array(
            [
                [0.0, off_diagonal],
                [np.conj(off_diagonal), self.delta],
            ],
            dtype=complex,
        )

    def fourier_components_fft(self, ns, kx, ky, kz=0, M=512):
        self.fft_calls += 1
        return super().fourier_components_fft(ns, kx, ky, kz=kz, M=M)


def _single_point_grid(
    hamiltonian_object,
    *,
    max_l=2,
    store_hamiltonians=True,
):
    ki = np.zeros((1, 1), dtype=float)
    kj = np.zeros((1, 1), dtype=float)
    return grid_eigenvalues_eigenfunctions(
        hamiltonian_object,
        ki,
        kj,
        mesh_spacing=1,
        dim=2,
        show_progress=False,
        max_l=max_l,
        store_hamiltonians=store_hamiltonians,
    )


def test_floquet_ratio_and_argmax_indices_are_correct():
    hamiltonian_object = CircularTwoBandHamiltonian()

    outputs = _single_point_grid(hamiltonian_object)
    max_ratios = outputs[4][0, 0]
    max_indices = outputs[5][0, 0]

    np.testing.assert_allclose(max_ratios, [0.5, 0.5], atol=1e-12)
    np.testing.assert_array_equal(max_indices, [[1, -1], [0, 1]])
    assert hamiltonian_object.fft_calls == 1


def test_exact_coupled_resonance_returns_infinite_ratio():
    hamiltonian_object = CircularTwoBandHamiltonian(delta=2.0, omega=2.0)

    max_ratios = _single_point_grid(hamiltonian_object)[4][0, 0]

    assert np.all(np.isinf(max_ratios))


def test_static_hamiltonian_skips_fft_and_returns_sentinels():
    hamiltonian_object = CircularTwoBandHamiltonian(A0=0.0)

    outputs = _single_point_grid(hamiltonian_object)

    np.testing.assert_array_equal(outputs[4], np.zeros((1, 1, 2)))
    np.testing.assert_array_equal(
        outputs[5],
        np.array([[[[-1, 0], [-1, 0]]]], dtype=np.int32),
    )
    assert hamiltonian_object.fft_calls == 0


def test_magnus_terms_reuse_the_ratio_fft():
    hamiltonian_object = CircularTwoBandHamiltonian(magnus_order=2)

    _single_point_grid(hamiltonian_object)

    assert hamiltonian_object.fft_calls == 1


def test_hamiltonian_grids_can_be_omitted_without_removing_eigenfunctions():
    outputs = _single_point_grid(
        CircularTwoBandHamiltonian(A0=0.0),
        store_hamiltonians=False,
    )

    assert outputs[1].shape == (1, 1, 2, 2)
    assert outputs[2] is None
    assert outputs[3] is None
