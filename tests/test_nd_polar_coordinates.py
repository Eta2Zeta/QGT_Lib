import numpy as np

from Calc_QGT_2D_nd_parameter_sweep import _worker_qgt_point
from Library.Hamiltonian.Hamiltonian import hamiltonian
from Library.dimension_lib import create_2d_coordinate_grid_from_ranges


class MassiveDiracHamiltonian(hamiltonian):
    def __init__(self, mass=0.7):
        super().__init__(
            dim=2,
            omega=1.0,
            A0=0.0,
            polarization="left",
            magnus_order=0,
            analytic_magnus=False,
        )
        self.mass = float(mass)

    def compute_static(self, kx, ky, kz=0):
        return np.array(
            [
                [self.mass, kx - 1j * ky],
                [kx + 1j * ky, -self.mass],
            ],
            dtype=complex,
        )


def test_nd_worker_evaluates_and_returns_polar_qgt_data():
    radius, phi, grid_info = create_2d_coordinate_grid_from_ranges(
        (0.0, 0.2),
        (0.0, 2.0 * np.pi),
        4,
        order="xpz",
    )
    hamiltonian_object = MassiveDiracHamiltonian()
    k_path = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    result = _worker_qgt_point(
        ({}, (0,), None),
        hamiltonian_object,
        radius,
        phi,
        4,
        0,
        k_path,
        1,
        0.0,
        "xpz",
        1e-5,
        grid_info["phi_periodic"],
    )

    for component in result[1:11]:
        assert np.asarray(component).shape == radius.shape
    assert result[12].shape == (radius.shape[1],)

    eigenvalues = result[13]
    expected_energy = np.sqrt(hamiltonian_object.mass**2 + 0.2**2)
    np.testing.assert_allclose(
        eigenvalues[1, -1],
        [-expected_energy, expected_energy],
        atol=1e-10,
    )
    assert len(result) == 17
