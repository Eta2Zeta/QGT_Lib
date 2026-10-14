import json
from pathlib import Path

from qgt.plotting.minimal_planes import _sg11_set, gallery_order, other_parameter_sets

SG11 = "MinimalHamSG11_2a2b2c2d"
SET_PARAMS = {1: {}, 2: {"t3_add": 0.08, "lamb_z_add": 0.06}, 3: {"t3_add": 0.8, "lamb_z_add": 0.3},
              4: {"t3_add": 0.8, "lamb_z_add": 0.3, "lamb_z_xz_add": 0.3}}


def test_sg11_sets_are_named_by_their_added_terms():
    for number, params in SET_PARAMS.items():
        assert _sg11_set({"t1": 0.8, **params}) == number
    assert _sg11_set({"t3_add": 0.5}) is None


def _dataset(folder, plane, kz, params, model=SG11):
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({"kk": kz, "hamiltonian_params": params}))
    (folder / "summary.json").write_text(json.dumps({"model": model, "plane": plane}))
    (folder / "plots.json").write_text("{}")
    return {"model": model, "plane": plane, "kz": kz, "dataset_dir": str(folder)}


def test_other_parameter_sets_and_their_order(tmp_path):
    run, legacy = tmp_path / "run", tmp_path / "legacy"
    results = [_dataset(run / SG11 / plane / "dataset_2", plane, kz, SET_PARAMS[4])
               for plane, kz in (("kz_0", 0.0), ("kz_0p01_BZ", 0.06))]
    _dataset(legacy / SG11 / "kz_0" / "dataset_1", "kz_0", 0.0, SET_PARAMS[1])
    _dataset(legacy / SG11 / "kz_0" / "dataset_3", "kz_0", 0.0, SET_PARAMS[4])  # same as the run: skipped
    _dataset(legacy / "MinimalHamSG999" / "kz_0" / "dataset_1", "kz_0", 0.0, {}, model="MinimalHamSG999")
    others = other_parameter_sets(results, run, root=legacy)
    assert [Path(o["dataset_dir"]).name for o in others] == ["dataset_1"]
    assert Path(others[0]["output_dir"]) == run / SG11 / "legacy_sets" / "kz_0" / "dataset_1"
    ordered = gallery_order(results + others)
    assert [(Path(i["dataset_dir"]).parent.name, Path(i["dataset_dir"]).name) for i in ordered] == [
        ("kz_0", "dataset_1"), ("kz_0", "dataset_2"), ("kz_0p01_BZ", "dataset_2")]


def test_old_pickles_without_new_terms_are_checked_against_saved_eigenvalues(tmp_path):
    import pickle

    import numpy as np
    import pytest

    from qgt.hamiltonians.minimal_models import MinimalHamSG11_2a2b2c2d
    from qgt.plotting.minimal_planes import _saved_hamiltonian

    k = np.linspace(-np.pi, np.pi, 9)
    kx, ky = np.meshgrid(k, k)
    mask = np.ones(kx.shape, bool)

    def spectrum(model):
        return np.array([[np.linalg.eigvalsh(model.compute_static(x, y, 0.5)) for x, y in zip(rx, ry)]
                         for rx, ry in zip(kx, ky)])

    old = MinimalHamSG11_2a2b2c2d(lamb_z_xz_add=0.0)
    del old.lamb_z_xz_add  # as pickled before the term existed
    (tmp_path / "meta_info.pkl").write_bytes(pickle.dumps({"Hamiltonian_Obj": old}))
    # kz = 0.5: the lamb_z_xz_add term carries sin kz, so it would not show at kz = 0
    loaded = _saved_hamiltonian(tmp_path, kx, ky, 0.5, mask, spectrum(MinimalHamSG11_2a2b2c2d(lamb_z_xz_add=0.0)))
    assert loaded.lamb_z_xz_add == 0
    with pytest.raises(ValueError, match="does not reproduce"):
        _saved_hamiltonian(tmp_path, kx, ky, 0.5, mask, spectrum(MinimalHamSG11_2a2b2c2d()))
