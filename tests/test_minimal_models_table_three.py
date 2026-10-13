"""Physical/source regressions for the additional Table III models."""

import importlib
import unittest

import numpy as np

from Library.QGT_calc_functions_3comp import quantum_geometric_tensor_3d_num_phase_corrected
from Library.diagnalization import eigenvalues_and_vectors_eigenvalue_ordering


class TestMinimalModelsTableThree(unittest.TestCase):
    # Coefficient order is (tx, tz, lambda_x, lambda_y, lambda_z).
    # Values were independently evaluated from Roig et al., Table III, at
    # k=(0.37, -0.61, 0.83); they deliberately use unequal SOC coefficients
    # and nonzero secondary bases to distinguish all comma-separated terms.
    reference_terms = {
        11: (
            0.6676928667974089, 0.07230298765863347, 0.1240001038338045,
            -0.08932690669150288, -0.1621539819365136,
        ),
        14: (
            0.472688213058261, 0.07230298765863347, -0.07793772244723007,
            -0.05641419097074962, 0.2294934628727403,
        ),
        51: (
            0.688055375441162, -0.2401626343719833, -0.01369900784934461,
            0.186757887619744, 0.07063897702966292,
        ),
        53: (
            0.6296507034702686, 0.380463183266607, 0.1169351306444785,
            -0.01832211749945131, -0.01260811630813421,
        ),
        55: (
            0.6562995230623957, 0.1864419426386156, 0.01683176388885383,
            -0.0413847550191234, -0.159387027029439,
        ),
        58: (
            0.6005904046872412, 0.1864419426386156, 0.009196516066807799,
            -0.02261174568319054, -0.1458576697097586,
        ),
        128: (
            0.6005904046872412, 0.02100815276289114, 0.009196516066807799,
            -0.01547119441481458, -0.1458576697097586,
        ),
        136: (
            0.6005904046872412, 0.1864419426386156, 0.009196516066807799,
            0.01547119441481458, -0.01643514417161665,
        ),
        176: (
            0.6405814243462422, 0.6830619653791289, -0.2433942807429158,
            0.008296318529910149, 0.07680661206303258,
        ),
        193: (
            0.6405814243462422, 0.6700125459444589, -0.1062707274694651,
            -0.05537683983540062, -0.00448921133932569,
        ),
    }

    @staticmethod
    def model_class(space_group):
        if space_group in (11, 14, 51, 53, 55, 58):
            suffix = "2a2b2c2d"
        elif space_group in (128, 136):
            suffix = "2a2b"
        else:
            suffix = "2b"
        name = f"MinimalHamSG{space_group}_{suffix}"
        module = importlib.import_module(f"Library.Hamiltonian.MinimalModels.{name}")
        return getattr(module, name)

    @staticmethod
    def extra_parameters(space_group):
        if space_group == 11:
            # Source regression excludes the extra hoppings derived in Note 3-3.
            return dict(
                t4_alt=0.27, lamb_y=0.19, lamb_y_alt=0.31,
                t3_add=0.0, lamb_z_add=0.0, lamb_z_xz_add=0.0,
            )
        if space_group == 14:
            return dict(
                t3_alt=0.43, t4_alt=0.27, lamb_y=0.19,
                lamb_x_alt=-0.11, lamb_y_alt=0.31, lamb_z_alt=0.29,
            )
        if space_group in (51, 53, 55, 58):
            return dict(lamb_y=0.19)
        if space_group == 176:
            return dict(t4_alt=0.27, lamb2=0.23, lamb_z_alt=0.29)
        return {}

    def test_table_three_numerical_reference(self):
        for space_group, expected in self.reference_terms.items():
            with self.subTest(space_group=space_group):
                model = self.model_class(space_group)(
                    t3=0.7, t4=-0.9, lamb=0.13, lamb_z=-0.17,
                    **self.extra_parameters(space_group),
                )
                actual = model._model_terms(0.37, -0.61, 0.83)[1:]
                np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)

    def test_fourfold_nodal_planes_without_soc_or_order(self):
        # These are the nonsymmorphic band degeneracies distinguishing
        # Table III from Table IV. Include secondary hopping bases as well.
        plane_axis = {
            11: 1, 14: 1, 51: 0, 53: 2, 55: 0,
            58: 0, 128: 0, 136: 0, 176: 2, 193: 2,
        }
        points = np.array([[0.21, -0.73, 0.49], [-0.87, 0.34, -1.12]])
        for space_group, axis in plane_axis.items():
            with self.subTest(space_group=space_group):
                extra = {
                    name: (0.0 if name.startswith("lamb") else value)
                    for name, value in self.extra_parameters(space_group).items()
                }
                model = self.model_class(space_group)(
                    lamb=0.0, lamb_z=0.0, Jx=0.0, Jy=0.0, Jz=0.0,
                    **extra,
                )
                plane_points = points.copy()
                plane_points[:, axis] = np.pi
                energies = np.linalg.eigvalsh(
                    model.compute_static_vectorized(*plane_points.T)
                )
                np.testing.assert_allclose(
                    np.ptp(energies, axis=-1), 0.0, atol=1e-13,
                )

    def test_sg11_added_hoppings_and_spectrum(self):
        model_class = self.model_class(11)
        paper = model_class(t3_add=0.0, lamb_z_add=0.0, lamb_z_xz_add=0.0)
        added = model_class()
        self.assertEqual(added.t3_add, 0.8)
        self.assertEqual(added.lamb_z_add, 0.3)
        self.assertEqual(added.lamb_z_xz_add, 0.3)
        self.assertEqual((added.Jx, added.Jy, added.omega, added.A0), (0, 0, 0, 0))
        parameters = added.get_parameters_dict()
        self.assertEqual(parameters["t3_add"], 0.8)
        self.assertEqual(parameters["lamb_z_add"], 0.3)
        self.assertEqual(parameters["lamb_z_xz_add"], 0.3)

        # Check the complete matrix increment, independently of _model_terms.
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.diag([1, -1])
        tau_x = np.kron(sigma_x, np.eye(2))
        tau_y_sigma_z = np.kron(sigma_y, sigma_z)
        momenta = np.array([[0.37, -0.61, 0.83], [-0.71, 0.54, -0.93]])
        kx, ky, kz = momenta.T
        expected_increment = (
            (0.8 * np.cos(kx) * np.cos(ky / 2))[:, None, None] * tau_x
            + (0.3 * (np.cos(kz) + np.sin(kx) * np.sin(kz))
               * np.cos(ky / 2))[:, None, None] * tau_y_sigma_z
        )
        matrices = added.compute_static_vectorized(kx, ky, kz)
        np.testing.assert_allclose(
            matrices - paper.compute_static_vectorized(kx, ky, kz),
            expected_increment, atol=1e-13,
        )
        np.testing.assert_allclose(
            matrices, np.stack([added.compute_static(*k) for k in momenta]),
            atol=1e-13,
        )
        np.testing.assert_allclose(
            added.get_analytical_eigenvalues(kx, ky, kz),
            np.linalg.eigvalsh(matrices), atol=1e-12,
        )

    def test_sg11_added_hoppings_preserve_space_group(self):
        model = self.model_class(11)()
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        # In the site-position Bloch convention, screw and mirror share this
        # sublattice/spin action up to an irrelevant scalar Bloch phase.
        exchange_and_spin_rotation = np.kron(sigma_x, -1j * sigma_y)
        operations = (
            (np.array([-1, 1, -1]), exchange_and_spin_rotation),  # C2y screw
            (np.array([1, -1, 1]), exchange_and_spin_rotation),  # my mirror
            (np.array([-1, -1, -1]), np.eye(4)),  # inversion
        )
        for k in (np.array([0.37, -0.61, 0.83]), np.array([-0.71, 0.54, -0.93])):
            original = model.compute_static(*k)
            for signs, representation in operations:
                np.testing.assert_allclose(
                    model.compute_static(*(signs * k)),
                    representation @ original @ representation.conj().T,
                    atol=1e-13,
                )

    def test_sg11_added_hoppings_lift_total_mirror_plane_node(self):
        # Note 3-3 predicts Omega_y on ky=0, including its sign and magnitude.
        # One added hopping alone leaves it zero; both independent terms lift it.
        k = np.array([0.6, 0.0, 2 * np.pi / 100])
        for t3_add, lamb_z_add in (
            (0, 0), (0.08, 0), (0, 0.06), (0.08, 0.06), (0.8, 0.3),
        ):
            with self.subTest(t3_add=t3_add, lamb_z_add=lamb_z_add):
                model = self.model_class(11)(
                    t3_add=t3_add, lamb_z_add=lamb_z_add, lamb_z_xz_add=0.0,
                )
                energies, states = eigenvalues_and_vectors_eigenvalue_ordering(model, *k)
                self.assertGreater(np.min(np.diff(energies.real)), 1e-3)
                qgt = quantum_geometric_tensor_3d_num_phase_corrected(
                    model, *k, 1e-5, energies, states, band_index=0,
                )
                berry = np.array([-2 * qgt[8], 2 * qgt[6], -2 * qgt[4]])
                mass = model.Jz + abs(model.lamb)
                energy = np.sqrt(
                    (model.t3 + t3_add * np.cos(k[0])) ** 2
                    + (model.lamb_z + lamb_z_add * np.cos(k[2])) ** 2
                    + mass ** 2
                )
                expected_y = (
                    -mass * t3_add * lamb_z_add * np.sin(k[0]) * np.sin(k[2])
                    / (2 * energy ** 3)
                )
                np.testing.assert_allclose(
                    berry, [0, expected_y, 0], rtol=3e-5, atol=2e-9,
                )
                if t3_add and lamb_z_add:
                    self.assertGreater(abs(berry[1]), 1e-4)

    def test_sg11_xz_hopping_lifts_kz_zero_line_for_all_bands(self):
        # The term itself vanishes on kz=0; the full 3D geometry detects its
        # nonzero z derivative. Check the block formula independently for
        # all four energy-ordered bands, on both computed slices.
        for kz in (0.0, 2 * np.pi / 100):
            k = np.array([1.8, 0.0, kz])
            model = self.model_class(11)()
            energies, states = eigenvalues_and_vectors_eigenvalue_ordering(model, *k)
            self.assertGreater(np.min(np.diff(energies.real)), 1e-3)
            a = model.t3 + model.t3_add * np.cos(k[0])
            b = (model.lamb_z + model.lamb_z_add * np.cos(kz)
                 + model.lamb_z_xz_add * np.sin(k[0]) * np.sin(kz))
            b_z = (-model.lamb_z_add * np.sin(kz)
                   + model.lamb_z_xz_add * np.sin(k[0]) * np.cos(kz))
            for band, (energy_sign, nu) in enumerate(((-1, 1), (-1, -1), (1, -1), (1, 1))):
                with self.subTest(kz=kz, band=band):
                    mass = model.Jz + nu * abs(model.lamb)
                    energy = np.sqrt(a * a + b * b + mass * mass)
                    expected_y = (
                        -energy_sign * mass * model.t3_add * np.sin(k[0]) * b_z
                        / (2 * energy ** 3)
                    )
                    qgt = quantum_geometric_tensor_3d_num_phase_corrected(
                        model, *k, 1e-5, energies, states, band_index=band,
                    )
                    berry = np.array([-2 * qgt[8], 2 * qgt[6], -2 * qgt[4]])
                    np.testing.assert_allclose(
                        berry, [0, expected_y, 0], rtol=3e-5, atol=2e-9,
                    )
                    self.assertGreater(abs(berry[1]), 0.1)

        # At kz=0 the added hopping leaves H and its spectrum unchanged,
        # yet it changes the curvature. This catches accidental 2D-only QGT.
        without_xz = self.model_class(11)(lamb_z_xz_add=0.0)
        k = np.array([1.8, 0.0, 0.0])
        np.testing.assert_allclose(model.compute_static(*k), without_xz.compute_static(*k))
        energies, states = eigenvalues_and_vectors_eigenvalue_ordering(without_xz, *k)
        qgt = quantum_geometric_tensor_3d_num_phase_corrected(
            without_xz, *k, 1e-5, energies, states, band_index=0,
        )
        self.assertLess(abs(2 * qgt[6]), 2e-9)

    def test_sg11_xz_full_qgt_matches_analytic_hamiltonian_derivatives(self):
        # Independent spectral formula Q_ij = sum_m <n|d_i H|m>
        # <m|d_j H|n> / (E_n-E_m)^2 checks all 9 tensor components,
        # including y derivatives, for every band on both requested slices.
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.diag([1, -1])
        tx = np.kron(sigma_x, np.eye(2))
        tz = np.kron(sigma_z, np.eye(2))
        yx = np.kron(sigma_y, sigma_x)
        yy = np.kron(sigma_y, sigma_y)
        yz = np.kron(sigma_y, sigma_z)
        model = self.model_class(11)()
        for kz in (0.0, 2 * np.pi / 100):
            for kx, ky in ((1.8, 0.0), (0.57, -0.34)):
                k = np.array([kx, ky, kz])
                cx, cy, cz = np.cos(k)
                sx, sy, sz = np.sin(k)
                cy2, sy2 = np.cos(ky / 2), np.sin(ky / 2)
                a = model.t3 + model.t3_add * cx
                b = model.lamb_z + model.lamb_z_add * cz + model.lamb_z_xz_add * sx * sz
                c = model.t4 * sx + model.t4_alt * sz
                d = model.lamb_y * sx + model.lamb_y_alt * sz
                derivatives = np.array([
                    -model.t1 * sx * np.eye(4) - model.t3_add * sx * cy2 * tx
                    + model.t4 * sy * cx * tz + model.lamb_y * sy2 * cx * yy
                    + model.lamb_z_xz_add * cy2 * cx * sz * yz,
                    -model.t1 * sy * np.eye(4) - 0.5 * sy2 * a * tx
                    + cy * c * tz - 0.5 * model.lamb * sy2 * yx
                    + 0.5 * cy2 * d * yy - 0.5 * sy2 * b * yz,
                    -model.t2 * sz * np.eye(4) + model.t4_alt * sy * cz * tz
                    + model.lamb_y_alt * sy2 * cz * yy
                    + cy2 * (-model.lamb_z_add * sz + model.lamb_z_xz_add * sx * cz) * yz,
                ])
                energies, eigenvectors = np.linalg.eigh(model.compute_static(*k))
                self.assertGreater(np.min(np.diff(energies)), 1e-3)
                transformed = np.einsum("an,iab,bm->inm", eigenvectors.conj(), derivatives, eigenvectors)
                routine_energies, states = eigenvalues_and_vectors_eigenvalue_ordering(model, *k)
                for band in range(4):
                    with self.subTest(k=k, band=band):
                        others = np.arange(4) != band
                        matrix_elements = transformed[:, band, others]
                        expected = np.einsum(
                            "im,jm,m->ij", matrix_elements, matrix_elements.conj(),
                            1 / (energies[band] - energies[others]) ** 2,
                        )
                        qgt = quantum_geometric_tensor_3d_num_phase_corrected(
                            model, *k, 1e-5, routine_energies, states, band_index=band,
                        )
                        xy, xz, yz_q = qgt[3] + 1j * qgt[4], qgt[5] + 1j * qgt[6], qgt[7] + 1j * qgt[8]
                        actual = np.array([
                            [qgt[0], xy, xz], [xy.conjugate(), qgt[1], yz_q],
                            [xz.conjugate(), yz_q.conjugate(), qgt[2]],
                        ])
                        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-9)


if __name__ == "__main__":
    unittest.main()
