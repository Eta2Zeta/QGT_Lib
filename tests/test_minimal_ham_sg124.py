import unittest

import numpy as np

from Library.Hamiltonian import MinimalHamSG124_2b2d


class TestMinimalHamSG124(unittest.TestCase):
    def test_table_iv_form_factors(self):
        model = MinimalHamSG124_2b2d(
            t1=1.1,
            t2=-0.4,
            t3=0.7,
            t4=-0.9,
            mu=0.2,
            lamb=0.13,
            lamb_z=-0.17,
        )
        kx, ky, kz = 0.37, -0.42, 0.81

        terms = model._model_terms(kx, ky, kz)
        expected = (
            1.1 * (np.cos(kx) + np.cos(ky)) - 0.4 * np.cos(kz) - 0.2,
            0.7 * np.cos(kz / 2.0),
            -0.9 * np.sin(kx) * np.sin(ky) * (np.cos(kx) - np.cos(ky)),
            0.13 * np.sin(kx) * np.sin(kz / 2.0),
            0.13 * np.sin(ky) * np.sin(kz / 2.0),
            -0.17 * np.cos(kz / 2.0),
        )

        self.assertTrue(np.allclose(terms, expected))

    def test_static_and_vectorized_hamiltonians_agree(self):
        model = MinimalHamSG124_2b2d(Jx=0.07, Jy=-0.11, Jz=0.23)
        k_points = np.array(
            [
                [0.17, -0.29, 0.41],
                [-0.83, 0.52, -0.37],
                [1.21, -0.68, 0.94],
            ]
        )

        vectorized = model.compute_static_vectorized(
            k_points[:, 0],
            k_points[:, 1],
            k_points[:, 2],
        )
        scalar = np.stack([model.compute_static(*k_point) for k_point in k_points])

        self.assertEqual(vectorized.shape, (3, 4, 4))
        self.assertTrue(np.allclose(vectorized, scalar))
        self.assertTrue(
            np.allclose(vectorized, vectorized.conj().transpose(0, 2, 1))
        )

    def test_analytical_eigenvalues_match_diagonalization(self):
        model = MinimalHamSG124_2b2d(
            Jx=0.09,
            Jy=-0.14,
            Jz=0.22,
            lamb=0.16,
            lamb_z=0.12,
        )
        kx = np.array([0.19, -0.71, 1.03, -1.37])
        ky = np.array([-0.31, 0.64, -0.88, 0.47])
        kz = np.array([0.43, -0.56, 0.91, -1.12])

        numerical = np.linalg.eigvalsh(model.compute_static_vectorized(kx, ky, kz))
        analytical = model.get_analytical_eigenvalues(kx, ky, kz)

        self.assertTrue(np.allclose(analytical, numerical))

    def test_a2g_form_factor_symmetry(self):
        model = MinimalHamSG124_2b2d(t4=1.0)
        kx, ky = 0.43, 0.78

        tz = model._model_terms(kx, ky, 0.0)[2]
        tz_c4 = model._model_terms(-ky, kx, 0.0)[2]
        tz_vertical_mirror = model._model_terms(-kx, ky, 0.0)[2]

        self.assertTrue(np.isclose(tz_c4, tz))
        self.assertTrue(np.isclose(tz_vertical_mirror, -tz))

    def test_symmetry_path(self):
        model = MinimalHamSG124_2b2d()

        points, labels = model.get_sym_path()

        self.assertEqual(labels, list("GXMGZRAZ"))
        self.assertTrue(np.allclose(points["G"], [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(points["A"], [np.pi, np.pi, np.pi]))

        with self.assertRaisesRegex(ValueError, "Unknown SG124 symmetry labels"):
            model.get_sym_path("GQ")


if __name__ == "__main__":
    unittest.main()
