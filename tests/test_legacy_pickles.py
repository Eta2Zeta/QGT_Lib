import importlib
import os
import pickle
from pathlib import Path

import pytest

from qgt.io import legacy_pickles


def _pickled_global(module, name):
    """A protocol-0 pickle that references module.name, as an old result file would."""
    return f"c{module}\n{name}\n.".encode()


@pytest.mark.parametrize(
    "old_module, old_name, new_module, new_name",
    [
        ("Library.Hamiltonian_v2", "HaldaneHamiltonian", "qgt.hamiltonians.haldane", "HaldaneHamiltonian"),
        ("Library.Hamiltonian_v2", "RhombohedralGrapheneHamiltonian", "qgt.hamiltonians.chiral", "ChiralHamiltonian"),
        ("Library.Hamiltonian.Altermagnet_Hamiltonian", "AltermagnetHamiltonian",
         "qgt.hamiltonians.altermagnet", "AltermagnetHamiltonian"),
        ("Library.Hamiltonian.Chiral_Hamiltonian_Projected", "ChiralHamiltonianProjected",
         "qgt.hamiltonians.chiral_basis_projected", "ChiralHamiltonianChiralBasisProjected"),
        ("Library.Hamiltonian.gWaveAltermagnetHamiltonian", "gWaveAltermagnetHamiltonian",
         "qgt.hamiltonians.minimal_models.MinimalHamSG194_2a", "MinimalHamSG194_2a"),
        ("Library.Hamiltonian.MinimalHamSG1272a2b", "MinimalHamSG1272a2b",
         "qgt.hamiltonians.minimal_models.MinimalHamSG127_2a2b", "MinimalHamSG127_2a2b"),
        ("Library.Hamiltonian.MinimalHamSG127_2c2d", "MinimalHamSG127_2c2d",
         "qgt.hamiltonians.minimal_models.MinimalHamSG127_2c2d", "MinimalHamSG127_2c2d"),
        # Library -> qgt
        ("Library.Hamiltonian.ChiralHamiltonian", "ChiralHamiltonian", "qgt.hamiltonians.chiral", "ChiralHamiltonian"),
        ("Library.Hamiltonian.Hamiltonian", "GrapheneHamiltonian", "qgt.hamiltonians.graphene", "GrapheneHamiltonian"),
        ("Library.Hamiltonian.MinimalModels.MinimalHamSG194_2a", "MinimalHamSG194_2a",
         "qgt.hamiltonians.minimal_models.MinimalHamSG194_2a", "MinimalHamSG194_2a"),
        ("Library.Hamiltonian.THF_Hamiltonian", "THF_Hamiltonian", "qgt.hamiltonians.thf", "THF_Hamiltonian"),
    ],
)
def test_old_module_paths_unpickle_to_current_classes(old_module, old_name, new_module, new_name):
    cls = pickle.loads(_pickled_global(old_module, old_name))
    assert (cls.__module__, cls.__name__) == (new_module, new_name)


def test_every_table_entry_loads():
    for old_module, entry in legacy_pickles.MOVED_MODULES.items():
        module = importlib.import_module(old_module)
        for old_name in entry["renamed"]:
            assert isinstance(legacy_pickles.resolve(old_module, old_name), type)
            assert getattr(module, old_name) is legacy_pickles.resolve(old_module, old_name)


def test_old_import_statements_still_work():
    from Library.QGT_lib import QGT_grid_num
    from qgt.geometry.qgt_2d import QGT_grid_num as current

    assert QGT_grid_num is current
    namespace = {}
    exec("from Library.utilities import *", namespace)
    assert {"generate_2d_sym_lines", "replace_zeros_with_nan", "sign_check"} <= set(namespace)
    from Library.Hamiltonian import MinimalModels

    assert MinimalModels.MinimalHamSG194_2a.__module__ == "qgt.hamiltonians.minimal_models.MinimalHamSG194_2a"


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
