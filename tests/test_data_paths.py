import os

from qgt.io import paths as data_paths


def test_legacy_dir_follows_qgt_data(monkeypatch, tmp_path):
    monkeypatch.setenv("QGT_DATA", str(tmp_path))
    assert data_paths.legacy_dir("2D_QGT_ND", "ChiralHamiltonian") == os.path.join(
        str(tmp_path), "legacy", "2D_QGT_ND", "ChiralHamiltonian"
    )


def test_recorded_old_results_path_is_relocated(monkeypatch, tmp_path):
    monkeypatch.setenv("QGT_DATA", str(tmp_path))
    old = "/Users/someone/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/MinimalModels_QGT_planes/M/kz_0/dataset_1"
    assert data_paths.relocate_legacy_path(old) == os.path.join(
        str(tmp_path), "legacy", "MinimalModels_QGT_planes", "M", "kz_0", "dataset_1"
    )


def test_other_paths_are_unchanged():
    assert data_paths.relocate_legacy_path("/tmp/some/dataset_1") == "/tmp/some/dataset_1"


def test_default_data_root_is_next_to_the_repository(monkeypatch):
    monkeypatch.delenv("QGT_DATA", raising=False)
    assert (data_paths.REPO_ROOT / "pyproject.toml").is_file()
    assert data_paths.data_root() == str(data_paths.REPO_ROOT.parent / "Results")
