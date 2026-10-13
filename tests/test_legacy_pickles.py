import importlib
import os
import pickle
from pathlib import Path

import pytest

from Library import legacy_pickles


def _pickled_global(module, name):
    """A protocol-0 pickle that references module.name, as an old result file would."""
    return f"c{module}\n{name}\n.".encode()


@pytest.mark.parametrize(
    "old_module, old_name, new_module, new_name",
    [
        ("Library.Hamiltonian_v2", "HaldaneHamiltonian", "Library.Hamiltonian.HaldaneHamiltonian", "HaldaneHamiltonian"),
        ("Library.Hamiltonian_v2", "RhombohedralGrapheneHamiltonian", "Library.Hamiltonian.ChiralHamiltonian", "ChiralHamiltonian"),
        ("Library.Hamiltonian.Altermagnet_Hamiltonian", "AltermagnetHamiltonian",
         "Library.Hamiltonian.AltermagnetHamiltonian", "AltermagnetHamiltonian"),
        ("Library.Hamiltonian.Chiral_Hamiltonian_Projected", "ChiralHamiltonianProjected",
         "Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected", "ChiralHamiltonianChiralBasisProjected"),
        ("Library.Hamiltonian.gWaveAltermagnetHamiltonian", "gWaveAltermagnetHamiltonian",
         "Library.Hamiltonian.MinimalModels.MinimalHamSG194_2a", "MinimalHamSG194_2a"),
        ("Library.Hamiltonian.MinimalHamSG1272a2b", "MinimalHamSG1272a2b",
         "Library.Hamiltonian.MinimalModels.MinimalHamSG127_2a2b", "MinimalHamSG127_2a2b"),
        ("Library.Hamiltonian.MinimalHamSG127_2c2d", "MinimalHamSG127_2c2d",
         "Library.Hamiltonian.MinimalModels.MinimalHamSG127_2c2d", "MinimalHamSG127_2c2d"),
    ],
)
def test_old_module_paths_unpickle_to_current_classes(old_module, old_name, new_module, new_name):
    cls = pickle.loads(_pickled_global(old_module, old_name))
    assert (cls.__module__, cls.__name__) == (new_module, new_name)


def test_every_explicit_entry_resolves():
    for old_module, table in legacy_pickles.MOVED_MODULES.items():
        for old_name in table:
            if old_name != "*":
                assert isinstance(legacy_pickles.resolve(old_module, old_name), type)


def test_unknown_name_in_moved_module_fails_clearly():
    module = importlib.import_module("Library.Hamiltonian.Altermagnet_Hamiltonian")
    with pytest.raises(AttributeError, match="no known successor"):
        module.NoSuchClass


def test_all_legacy_result_pickles_load():
    """Every metadata pickle under Results/legacy loads (skipped without the data)."""
    legacy = Path(os.environ.get("QGT_REAL_DATA", "")) / "legacy"
    if not legacy.is_dir():
        pytest.skip("Results/legacy is not available on this machine")
    failures = []
    paths = sorted(legacy.rglob("*.pkl"))
    for path in paths:
        try:
            with open(path, "rb") as handle:
                pickle.load(handle)
        except Exception as exc:
            failures.append(f"{path.relative_to(legacy)}: {type(exc).__name__}: {exc}")
    assert paths
    assert not failures, "\n".join(failures[:20])
