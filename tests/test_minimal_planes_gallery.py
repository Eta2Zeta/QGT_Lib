import json
from pathlib import Path

from qgt.plotting.minimal_planes import _sg11_variant, gallery_order

SG11 = "MinimalHamSG11_2a2b2c2d"
DEFAULT = {"t3_add": 0.8, "lamb_z_add": 0.3, "lamb_z_xz_add": 0.3}
ORIGINAL = {"t3_add": 0, "lamb_z_add": 0, "lamb_z_xz_add": 0}


def test_sg11_variants_are_named_by_their_added_terms():
    assert _sg11_variant({"t1": 0.8, **ORIGINAL})[1] == "original minimal model"
    assert _sg11_variant({"t1": 0.8})[1] == "original minimal model"
    assert _sg11_variant(DEFAULT)[1] == "added terms (default)"


def _dataset(folder, plane, kz, params):
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({"kk": kz, "hamiltonian_params": params}))
    return {"model": SG11, "plane": plane, "kz": kz, "dataset_dir": str(folder)}


def test_original_model_is_shown_before_the_default(tmp_path):
    results = [_dataset(tmp_path / plane / name, plane, kz, params)
               for name, params in (("dataset_2", DEFAULT), ("dataset_3", ORIGINAL))
               for plane, kz in (("kz_0", 0.0), ("kz_0p01_BZ", 0.06))]
    ordered = [(Path(i["dataset_dir"]).name, i["plane"]) for i in gallery_order(results)]
    assert ordered == [("dataset_3", "kz_0"), ("dataset_3", "kz_0p01_BZ"),
                       ("dataset_2", "kz_0"), ("dataset_2", "kz_0p01_BZ")]


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
