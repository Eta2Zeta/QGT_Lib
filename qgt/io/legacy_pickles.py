"""Serve old module paths so old results and scripts still load.

Result pickles (meta_info.pkl, meta.pkl, ...) store objects together with the
module path of their class, and old scripts import modules by their old names.
The library moved from ``Library`` to ``qgt``, and several modules were renamed
before that. ``install()`` adds an import hook that serves each old module path
as a module holding the names of its current replacement, so old pickles load
with plain ``pickle.load`` and ``np.load(allow_pickle=True)``, and
``from Library.X import ...`` keeps working. ``qgt`` and the ``Library`` stub
install it when imported.

To support another old path, add it to ``MOVED_MODULES``.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import types

_MINIMAL = "qgt.hamiltonians.minimal_models"


def _moved(*modules, package=False, **renamed):
    """Entry: the old module now provides the names of ``modules`` (first wins) and ``renamed``."""
    return {"from": list(modules), "package": package, "renamed": renamed}


MOVED_MODULES = {
    # The package itself (Library -> qgt, 2026-10-13).
    "Library.Eigenvector": _moved("qgt.spectra.eigenvectors"),
    "Library.diagnalization": _moved("qgt.spectra.diagonalization"),
    "Library.eigenvalue_calc_lib": _moved("qgt.spectra.eigenvalues"),
    "Library.eigenvalue_calc_lib_1d": _moved("qgt.spectra.eigenvalues_1d"),
    "Library.QGT_lib": _moved("qgt.geometry.qgt_2d"),
    "Library.QGT_Calc_3d_lib": _moved("qgt.geometry.qgt_3d"),
    "Library.QGT_calc_functions_2comp": _moved("qgt.geometry.pointwise_2d"),
    "Library.QGT_calc_functions_3comp": _moved("qgt.geometry.pointwise_3d"),
    "Library.topology": _moved("qgt.topology"),
    "Library.GroupTheory": _moved("qgt.symmetry", package=True),
    "Library.GroupTheory.point_groups": _moved("qgt.symmetry.point_groups"),
    "Library.GroupTheory.qgt_orbit": _moved("qgt.symmetry.qgt_orbit"),
    "Library.Geometry": _moved(package=True),
    "Library.Geometry.zones": _moved("qgt.symmetry.zones"),
    "Library.dimension_lib": _moved("qgt.kspace.grids"),
    "Library.indexing_lib": _moved("qgt.kspace.indexing"),
    "Library.utilities": _moved("qgt.kspace.paths", "qgt.utils"),
    "Library.data_paths": _moved("qgt.io.paths"),
    "Library.runs": _moved("qgt.io.runs"),
    "Library.legacy_pickles": _moved("qgt.io.legacy_pickles"),
    "Library.data_management_utils_common": _moved("qgt.io.metadata"),
    "Library.data_management_utils_1d": _moved("qgt.io.setup_1d"),
    "Library.data_management_utils_2d": _moved("qgt.io.setup_2d"),
    "Library.data_management_utils_3d": _moved("qgt.io.setup_3d"),
    "Library.data_management_utils_nd": _moved("qgt.io.setup_nd"),
    "Library.output_utils": _moved("qgt.io.output"),
    "Library.plotting_lib_1d": _moved("qgt.plotting.bands_1d"),
    "Library.plotting_eigenvalues_2d": _moved("qgt.plotting.bands_2d"),
    "Library.plotting_qgt_2d": _moved("qgt.plotting.qgt_2d"),
    "Library.plotting_qgt_polar": _moved("qgt.plotting.qgt_polar"),
    "Library.plotting_lib_3d": _moved("qgt.plotting.volume_3d"),
    "Library.plotting_utils": _moved("qgt.plotting.common"),
    "Library.basic_math": _moved("qgt.hamiltonians.base"),
    "Library.Hamiltonian_helper": _moved("qgt.hamiltonians.evaluate"),
    "Library.Hamiltonian": _moved("qgt.hamiltonians", package=True),
    "Library.Hamiltonian.Hamiltonian": _moved("qgt.hamiltonians.base", "qgt.hamiltonians.toy",
                                              "qgt.hamiltonians.graphene", "qgt.hamiltonians.rashba"),
    "Library.Hamiltonian.AltermagnetHamiltonian": _moved("qgt.hamiltonians.altermagnet"),
    "Library.Hamiltonian.ChiralHamiltonian": _moved("qgt.hamiltonians.chiral"),
    "Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected": _moved("qgt.hamiltonians.chiral_basis_projected"),
    "Library.Hamiltonian.ChiralHamiltonian_SW_Projected": _moved("qgt.hamiltonians.chiral_sw_projected"),
    "Library.Hamiltonian.HaldaneHamiltonian": _moved("qgt.hamiltonians.haldane"),
    "Library.Hamiltonian.RuO2Hamiltonian": _moved("qgt.hamiltonians.ruo2"),
    "Library.Hamiltonian.SquareLatticeHamiltonian": _moved("qgt.hamiltonians.square_lattice"),
    "Library.Hamiltonian.SquareLatticeHamiltonianMod": _moved("qgt.hamiltonians.square_lattice_mod"),
    "Library.Hamiltonian.THF_Hamiltonian": _moved("qgt.hamiltonians.thf"),
    "Library.Hamiltonian.TwoOrbitalSpinfulHamiltonian": _moved("qgt.hamiltonians.two_orbital_spinful"),
    "Library.Hamiltonian.TwoOrbitalUnspinfulHamiltonian": _moved("qgt.hamiltonians.two_orbital_unspinful"),
    "Library.Hamiltonian.MinimalModels": _moved(_MINIMAL, package=True),

    # Older renames, kept for results written before them.
    # Single-file Hamiltonian module, split up in v0.0.3, where
    # RhombohedralGrapheneHamiltonian was renamed ChiralHamiltonian.
    "Library.Hamiltonian_v2": _moved(
        "qgt.hamiltonians",
        RhombohedralGrapheneHamiltonian="qgt.hamiltonians.chiral:ChiralHamiltonian"),
    "Library.Hamiltonian.Altermagnet_Hamiltonian": _moved("qgt.hamiltonians.altermagnet"),
    # ChiralHamiltonianProjected was renamed ChiralHamiltonianChiralBasisProjected in v0.1.1.
    "Library.Hamiltonian.Chiral_Hamiltonian_Projected": _moved(
        "qgt.hamiltonians.chiral_basis_projected",
        ChiralHamiltonianProjected="qgt.hamiltonians.chiral_basis_projected:ChiralHamiltonianChiralBasisProjected"),
    # The g-wave altermagnet became the SG 194 minimal model.
    "Library.Hamiltonian.gWaveAltermagnetHamiltonian": _moved(
        f"{_MINIMAL}.MinimalHamSG194_2a",
        gWaveAltermagnetHamiltonian=f"{_MINIMAL}.MinimalHamSG194_2a:MinimalHamSG194_2a"),
    "Library.Hamiltonian.MinimalHamSG1272a2b": _moved(
        _MINIMAL, MinimalHamSG1272a2b=f"{_MINIMAL}.MinimalHamSG127_2a2b:MinimalHamSG127_2a2b"),
}
# Minimal models: Library/Hamiltonian/ -> Library/Hamiltonian/MinimalModels/ -> qgt/hamiltonians/minimal_models/.
for _name in ("MinimalAltermagnetHamiltonian", "MinimalHamSG11_2a2b2c2d", "MinimalHamSG123_2e2f",
              "MinimalHamSG124_2b2d", "MinimalHamSG127_2a2b", "MinimalHamSG127_2c2d", "MinimalHamSG128_2a2b",
              "MinimalHamSG131_2a2b", "MinimalHamSG131_2c2d", "MinimalHamSG132_2a2c", "MinimalHamSG136_2a2b",
              "MinimalHamSG13_2a2b2c2d", "MinimalHamSG14_2a2b2c2d", "MinimalHamSG163_2b", "MinimalHamSG165_2b",
              "MinimalHamSG176_2b", "MinimalHamSG192_2b", "MinimalHamSG193_2b", "MinimalHamSG194_2a",
              "MinimalHamSG223_2a", "MinimalHamSG49_2a2b2c2d", "MinimalHamSG51_2a2b2c2d",
              "MinimalHamSG53_2a2b2c2d", "MinimalHamSG55_2a2b2c2d", "MinimalHamSG58_2a2b2c2d",
              "MinimalHamSG83_2e2f", "MinimalHamSG84_2a2b", "MinimalHamSG84_2c2d"):
    MOVED_MODULES[f"Library.Hamiltonian.MinimalModels.{_name}"] = _moved(f"{_MINIMAL}.{_name}")
    MOVED_MODULES.setdefault(f"Library.Hamiltonian.{_name}", _moved(f"{_MINIMAL}.{_name}"))


def _load_target(spec):
    module_name, _, attribute = spec.partition(":")
    return getattr(importlib.import_module(module_name), attribute)


def resolve(old_module, name):
    """Return the current object for ``name`` as seen through ``old_module``."""
    entry = MOVED_MODULES[old_module]
    if name in entry["renamed"]:
        return _load_target(entry["renamed"][name])
    for module_name in entry["from"]:
        module = importlib.import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)


class _MovedModule(types.ModuleType):
    """Module object for an old path."""

    def __setattr__(self, name, value):
        # Importing an old submodule path would otherwise replace the class the
        # old package exported under the same name (e.g. MinimalModels.MinimalHamSG194_2a).
        current = self.__dict__.get(name)
        if isinstance(value, types.ModuleType) and current is not None and not isinstance(current, types.ModuleType):
            return
        super().__setattr__(name, value)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise AttributeError(
            f"{self.__name__}.{name} has no known successor; add it to "
            f"qgt.io.legacy_pickles.MOVED_MODULES"
        )


class _MovedModuleLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return _MovedModule(spec.name)

    def exec_module(self, module):
        entry = MOVED_MODULES[module.__name__]
        for module_name in entry["from"]:
            for key, value in vars(importlib.import_module(module_name)).items():
                if not key.startswith("__"):
                    module.__dict__.setdefault(key, value)
        for old_name, target in entry["renamed"].items():
            module.__dict__[old_name] = _load_target(target)


class _MovedModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        entry = MOVED_MODULES.get(fullname)
        if entry is None:
            return None
        return importlib.util.spec_from_loader(fullname, _MovedModuleLoader(), is_package=entry["package"])


def install():
    """Serve the old module paths. Safe to call more than once."""
    if not any(isinstance(finder, _MovedModuleFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _MovedModuleFinder())
