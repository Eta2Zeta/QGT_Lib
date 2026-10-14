import json
import os
import tomllib
import types
from pathlib import Path

import pytest

from qgt import config
from qgt.io import runs

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def data(monkeypatch, tmp_path):
    monkeypatch.setenv("QGT_DATA", str(tmp_path / "data"))
    return tmp_path / "data"


def _write(tmp_path, text):
    path = tmp_path / "setup.toml"
    path.write_text(text)
    return path


def test_every_shipped_config_parses_and_names_known_models():
    paths = sorted(CONFIGS.rglob("*.toml"))
    assert paths
    for path in paths:
        loaded = tomllib.loads(path.read_text())
        if "model" in loaded:
            assert isinstance(config.model_class(loaded["model"]["class"]), type), path


def test_references_resolve_to_data_folders(data):
    assert config.resolve_value("@legacy/2D_QGT_ND/x") == os.path.join(str(data), "legacy", "2D_QGT_ND", "x")
    assert config.resolve_value({"a": ["@runs/eigen_2d"]}) == {"a": [os.path.join(str(data), "runs", "eigen_2d")]}
    with pytest.raises(FileNotFoundError):
        config.resolve_value("@latest/eigen_2d")
    run_dir, _ = runs.new_run("eigen_2d", "M", None)
    assert config.resolve_value("@latest/eigen_2d/eigenvalues.npy") == os.path.join(run_dir, "eigenvalues.npy")
    assert config.resolve_value("plain") == "plain"


def test_build_model_with_params_and_attributes():
    model = config.build_model({"class": "SquareLatticeHamiltonian", "params": {"A0": 0.1, "omega": 1.0},
                                "attributes": {"polarization": "right"}})
    assert type(model).__name__ == "SquareLatticeHamiltonian"
    assert (model.A0, model.omega, model.polarization) == (0.1, 1.0, "right")
    assert type(config.build_model({"class": "MinimalHamSG194_2a"})).__name__ == "MinimalHamSG194_2a"
    with pytest.raises(ValueError, match="No Hamiltonian class"):
        config.model_class("NoSuchModel")


def test_run_passes_model_and_arguments_and_records_the_config(tmp_path, data):
    seen = []

    def calculation(model, *, mesh, label):
        run_dir, _ = runs.new_run("eigen_2d", type(model).__name__, None)
        seen.append((type(model).__name__, mesh, label, run_dir))
        return run_dir

    path = _write(tmp_path, '[model]\nclass = "MinimalHamSG194_2a"\n[model.params]\nJz = 0.3\n'
                            '[arguments]\nmesh = 7\nlabel = "@legacy/x"\n')
    run_dir = config.run(calculation, path)
    assert seen[0][:3] == ("MinimalHamSG194_2a", 7, os.path.join(str(data), "legacy", "x"))
    info = json.loads(Path(run_dir, runs.RUN_INFO).read_text())
    assert info["config_file"] == str(path) and info["config"]["arguments"]["mesh"] == 7
    # recording stops after the call
    later, _ = runs.new_run("eigen_2d", "M", None)
    assert "config" not in json.loads(Path(later, runs.RUN_INFO).read_text())


def test_run_with_keyword_model_calls_and_module_function(tmp_path, data):
    calls = []
    module = types.ModuleType("plots")
    module.first = lambda **kwargs: calls.append(("first", kwargs))
    module.second = lambda **kwargs: calls.append(("second", kwargs))
    path = _write(tmp_path, 'function = "second"\n[model]\nclass = "MinimalHamSG194_2a"\n'
                            'argument = "hamiltonian_template"\n[arguments]\nshared = 1\n'
                            '[[calls]]\nquantity = "trace"\n[[calls]]\nquantity = "berry"\n')
    results = config.run(module, path)
    assert len(results) == 2 and [c[0] for c in calls] == ["second", "second"]
    assert [c[1]["quantity"] for c in calls] == ["trace", "berry"]
    assert all(c[1]["shared"] == 1 and type(c[1]["hamiltonian_template"]).__name__ == "MinimalHamSG194_2a" for c in calls)


def test_prepare_can_derive_arguments_from_the_model(tmp_path, data):
    path = _write(tmp_path, '[model]\nclass = "MinimalHamSG194_2a"\n[arguments]\nband_indices = "all"\n')

    def prepare(model, arguments):
        arguments["band_indices"] = tuple(range(model.dim))
        return arguments

    assert config.run(lambda model, band_indices: band_indices, path, prepare=prepare) == (0, 1, 2, 3)
