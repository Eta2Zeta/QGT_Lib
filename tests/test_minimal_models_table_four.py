"""Independent coefficient regressions against PRB 110, 144412, Table IV."""

import importlib
import unittest

import numpy as np


def table_iv_bases(kx, ky, kz):
    """Return the five nonidentity coefficients for each printed basis.

    This reference uses direct NumPy expressions, independent of the
    Hamiltonian implementation and its shared trigonometric helper.
    """
    sx, sy, sz = np.sin(kx), np.sin(ky), np.sin(kz)
    cx, cy, cz = np.cos(kx), np.cos(ky), np.cos(kz)
    shx, shy, shz = np.sin(kx / 2), np.sin(ky / 2), np.sin(kz / 2)
    chx, chy, chz = np.cos(kx / 2), np.cos(ky / 2), np.cos(kz / 2)
    fx = np.sin(kx) + np.sin(kx / 2) * np.cos(np.sqrt(3) * ky / 2)
    fy = np.sqrt(3) * np.cos(kx / 2) * np.sin(np.sqrt(3) * ky / 2)
    d = cx - cy
    zero = np.zeros_like(kx)

    def row(tx=zero, tz=zero, lx=zero, ly=zero, lz=zero):
        return np.stack([tx, tz, lx, ly, lz])

    common_alternates = {
        "t4_alt": row(tz=sx * sy),
    }
    sixth_order = fx * fy * (fx**2 - 3 * fy**2) * (3 * fx**2 - fy**2)
    return {
        "MinimalHamSG13_2a2b2c2d": {
            "t3": row(tx=chz),
            "t3_alt": row(tx=sx * shz),
            "t4": row(tz=sy * sx),
            "t4_alt": row(tz=sy * sz),
            "lamb": row(lx=chz),
            "lamb_x_alt": row(lx=sx * shz),
            "lamb_y": row(ly=sy * shz),
            "lamb_y_alt": row(ly=sy * sx * chz),
            "lamb_z": row(lz=chz),
            "lamb_z_alt": row(lz=sx * shz),
        },
        "MinimalHamSG49_2a2b2c2d": {
            "t3": row(tx=chz), "t4": row(tz=sx * sy),
            "lamb": row(lx=sx * shz), "lamb_y": row(ly=sy * shz),
            "lamb_z": row(lz=chz),
        },
        "MinimalHamSG83_2e2f": {
            "t3": row(tx=chx * chy),
            "t3_alt": row(tx=shx * shy * d),
            "t4": row(tz=d), **common_alternates,
            "lamb": row(lx=shx * chy * sz, ly=-chx * shy * sz),
            "lamb2": row(lx=chx * shy * sz, ly=shx * chy * sz),
            "lamb_z": row(lz=shx * shy),
            "lamb_z_alt": row(lz=chx * chy * d),
        },
        "MinimalHamSG84_2a2b": {
            "t3": row(tx=chz), "t4": row(tz=d), **common_alternates,
            "lamb": row(lx=sx * shz, ly=-sy * shz),
            "lamb2": row(lx=sy * shz, ly=sx * shz),
            "lamb_z": row(lz=chz * d),
        },
        "MinimalHamSG84_2c2d": {
            "t3": row(tx=chx * chy * chz),
            "t3_alt": row(tx=shx * shy * chz),
            "t4": row(tz=d), **common_alternates,
            "lamb": row(lx=shx * chy * shz, ly=-chx * shy * shz),
            "lamb2": row(lx=chx * shy * shz, ly=shx * chy * shz),
            "lamb_z": row(lz=chx * chy * chz * d),
            # Both printed bases include (cos(kx) - cos(ky)).
            "lamb_z_alt": row(lz=shx * shy * chz * d),
        },
        "MinimalHamSG123_2e2f": {
            "t3": row(tx=chx * chy), "t4": row(tz=d),
            "lamb": row(lx=chx * shy * sz, ly=shx * chy * sz),
            "lamb_z": row(lz=shx * shy),
        },
        "MinimalHamSG131_2a2b": {
            "t3": row(tx=chz), "t4": row(tz=d),
            "lamb": row(lx=sy * shz, ly=sx * shz),
            "lamb_z": row(lz=sx * sy * chz),
        },
        "MinimalHamSG131_2c2d": {
            "t3": row(tx=chx * chy * chz), "t4": row(tz=d),
            "lamb": row(lx=chx * shy * shz, ly=shx * chy * shz),
            "lamb_z": row(lz=shx * shy * chz),
        },
        "MinimalHamSG132_2a2c": {
            "t3": row(tx=chz), "t4": row(tz=sx * sy),
            "lamb": row(lx=sx * shz, ly=-sy * shz),
            "lamb_z": row(lz=chz * d),
        },
        "MinimalHamSG163_2b": {
            "t3": row(tx=chz),
            "t3_alt": row(tx=fx * (3 * fy**2 - fx**2) * shz),
            "t4": row(tz=fy * (fy**2 - 3 * fx**2) * sz),
            "t4_alt": row(tz=sixth_order),
            "lamb": row(lx=fx * shz, ly=fy * shz),
            "lamb2": row(lx=(fx**2 - fy**2) * chz, ly=-2 * fx * fy * chz),
            "lamb_z": row(lz=chz),
            "lamb_z_alt": row(lz=fx * (3 * fy**2 - fx**2) * shz),
        },
        "MinimalHamSG165_2b": {
            "t3": row(tx=chz),
            "t3_alt": row(tx=fy * (3 * fx**2 - fy**2) * shz),
            "t4": row(tz=fx * (fx**2 - 3 * fy**2) * sz),
            "t4_alt": row(tz=sixth_order),
            "lamb": row(lx=fx * shz, ly=fy * shz),
            "lamb2": row(lx=2 * fx * fy * chz, ly=(fx**2 - fy**2) * chz),
            "lamb_z": row(lz=chz),
            "lamb_z_alt": row(lz=fy * (3 * fx**2 - fy**2) * shz),
        },
        "MinimalHamSG223_2a": {
            "t3": row(tx=chx * chy * chz),
            "t4": row(tz=(cx - cy) * (cy - cz) * (cz - cx)),
            "lamb": row(lx=chx * shy * shz, ly=shx * chy * shz, lz=shx * shy * chz),
        },
    }


def model_class(name):
    module = importlib.import_module(f"Library.Hamiltonian.MinimalModels.{name}")
    return getattr(module, name)


class TestTableIVCoefficients(unittest.TestCase):
    def test_every_printed_basis_has_an_independent_amplitude(self):
        kx = np.array([0.31, -0.84, 1.23, -1.47])
        ky = np.array([-0.62, 0.53, -1.18, 0.71])
        kz = np.array([0.83, -1.06, 0.47, -0.39])
        for name, bases in table_iv_bases(kx, ky, kz).items():
            cls = model_class(name)
            for amplitude, expected in bases.items():
                with self.subTest(model=name, amplitude=amplitude):
                    parameters = dict.fromkeys(bases, 0.0)
                    parameters[amplitude] = 1.0
                    model = cls(**parameters)
                    actual = np.stack(model._model_terms(kx, ky, kz)[1:])
                    np.testing.assert_allclose(actual, expected, atol=1e-14)
                    for index in range(kx.size):
                        scalar = model._model_terms(kx[index], ky[index], kz[index])[1:]
                        np.testing.assert_allclose(scalar, expected[:, index], atol=1e-14)

    def test_all_amplitudes_can_be_combined(self):
        points = np.array([[0.43, -0.67, 0.91], [-0.56, 1.17, -0.32]])
        for name, bases in table_iv_bases(*points.T).items():
            with self.subTest(model=name):
                parameters = {key: (-1)**i * (i + 1) / 13 for i, key in enumerate(bases)}
                expected = sum(parameters[key] * basis for key, basis in bases.items())
                model = model_class(name)(**parameters)
                actual = np.stack(model._model_terms(*points.T)[1:])
                np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_cubic_soc_uses_one_amplitude(self):
        cls = model_class("MinimalHamSG223_2a")
        with self.assertRaisesRegex(ValueError, "cubic symmetry"):
            cls(lamb=0.13, lamb_z=0.21)
        model = cls(lamb=0.13, lamb_z=0.13)
        model.lamb_z = -0.27
        self.assertEqual(model.lamb, -0.27)
        model.lamb = 0.39
        self.assertEqual(model.lamb_z, 0.39)
        k = np.array([0.43, -0.71, 0.83])
        before = model._model_terms(*k)[3:]
        after = model._model_terms(*np.roll(k, -1))[3:]
        np.testing.assert_allclose(after, np.roll(before, -1))


if __name__ == "__main__":
    unittest.main()
