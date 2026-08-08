import os

import numpy as np

import Calc_QGT_2D_nd_parameter_sweep as nd_sweep
from Library.Hamiltonian.Hamiltonian import hamiltonian


class TinyHamiltonian(hamiltonian):
    def __init__(self, mass=0.5):
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
        return np.array([[self.mass, 0.0], [0.0, -self.mass]])


def _fake_symmetry_path(*_args, **_kwargs):
    k_path = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    return (
        k_path,
        np.array([0.0, 0.1]),
        np.array([0, 1]),
        ["G", "X"],
        k_path.copy(),
    )


def _fake_worker(arg, *, ki, k_path, h_template, **_kwargs):
    _, idx, _ = arg
    value = float(idx[0] + 1)
    field = np.full(ki.shape, value)
    dim = h_template.dim
    eigenvalues = np.full(ki.shape + (dim,), value)
    ratios = np.full(ki.shape + (dim,), value)
    ratio_indices = np.full(ki.shape + (dim, 2), idx[0], dtype=np.int32)
    eigenvalues_sym = np.full((len(k_path), dim), value)

    return (
        idx,
        field,
        field,
        field,
        field,
        field,
        field,
        field,
        field,
        field,
        field,
        value,
        None,
        eigenvalues,
        ratios,
        ratio_indices,
        eigenvalues_sym,
    )


def test_nd_sweep_uses_temporary_memmaps_and_preserves_bundle_format(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        nd_sweep,
        "setup_qgt_nd_results_dir",
        lambda *_args, **_kwargs: (str(tmp_path), False),
    )
    monkeypatch.setattr(nd_sweep, "generate_2d_sym_lines", _fake_symmetry_path)
    monkeypatch.setattr(nd_sweep, "_worker_qgt_point", _fake_worker)

    opened_arrays = []
    real_open_memmap = nd_sweep._open_nd_output_memmap

    def tracking_open_memmap(*args, **kwargs):
        array = real_open_memmap(*args, **kwargs)
        opened_arrays.append(array)
        return array

    monkeypatch.setattr(
        nd_sweep,
        "_open_nd_output_memmap",
        tracking_open_memmap,
    )

    root, bundle_path = nd_sweep.compute_qgt_nd_parallel(
        hamiltonian_template=TinyHamiltonian(),
        param_ranges={"mass": (0.4, 0.6)},
        parameter_spacing={"mass": {"n": 2, "scale": "linear"}},
        ki_range=(-0.1, 0.1),
        kj_range=(-0.1, 0.1),
        mesh_spacing=2,
        band=0,
        num_points_per_segment=1,
        processes=1,
        force_new_dir=False,
        float_dtype=np.float32,
        max_l=1,
        order="xyz",
    )

    assert root == str(tmp_path)
    assert os.path.exists(bundle_path)
    assert opened_arrays
    assert all(isinstance(array, np.memmap) for array in opened_arrays)
    assert not (tmp_path / ".qgt_nd_memmap").exists()

    with np.load(bundle_path, allow_pickle=True) as bundle:
        assert bundle["g_xx_grid"].dtype == np.float32
        assert bundle["g_xx_grid"].shape == (2, 2, 2)
        np.testing.assert_allclose(bundle["g_xx_grid"][:, 0, 0], [1.0, 2.0])
        assert "eigenfunctions_grid" not in bundle
        assert "hamiltonian_grid" not in bundle
        assert "hamiltonian_prime_grid" not in bundle
