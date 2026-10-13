"""Load result pickles saved under module paths that have since moved.

Result metadata pickles (meta_info.pkl, meta.pkl, ...) store Hamiltonian
objects together with the module path of their class. Several modules have
been renamed or moved since those results were written. ``install()``
registers an import hook that serves each old path as a small module whose
attributes are the current classes, so old results load with plain
``pickle.load`` and ``np.load(allow_pickle=True)``. ``Library/__init__.py``
installs it, so any script that imports from ``Library`` can read old results.

To support another old path, add it to ``MOVED_MODULES``.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import types

# Old module path -> {old class name: (current module, current class name)}.
# The key "*" maps every other name to the class of the same name in the
# given package.
MOVED_MODULES = {
    # Single-file Hamiltonian module, split into Library/Hamiltonian/ (v0.0.3);
    # RhombohedralGrapheneHamiltonian was renamed ChiralHamiltonian there.
    "Library.Hamiltonian_v2": {
        "*": ("Library.Hamiltonian", None),
        "RhombohedralGrapheneHamiltonian": ("Library.Hamiltonian.ChiralHamiltonian", "ChiralHamiltonian"),
    },
    "Library.Hamiltonian.Altermagnet_Hamiltonian": {
        "AltermagnetHamiltonian": ("Library.Hamiltonian.AltermagnetHamiltonian", "AltermagnetHamiltonian"),
    },
    # ChiralHamiltonianProjected was renamed ChiralHamiltonianChiralBasisProjected (v0.1.1).
    "Library.Hamiltonian.Chiral_Hamiltonian_Projected": {
        "ChiralHamiltonianProjected": ("Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected",
                                       "ChiralHamiltonianChiralBasisProjected"),
    },
    # The g-wave altermagnet became the SG 194 minimal model.
    "Library.Hamiltonian.gWaveAltermagnetHamiltonian": {
        "gWaveAltermagnetHamiltonian": ("Library.Hamiltonian.MinimalModels.MinimalHamSG194_2a", "MinimalHamSG194_2a"),
        "MinimalHamSG194_2a": ("Library.Hamiltonian.MinimalModels.MinimalHamSG194_2a", "MinimalHamSG194_2a"),
    },
    "Library.Hamiltonian.MinimalHamSG1272a2b": {
        "*": ("Library.Hamiltonian.MinimalModels", None),
        "MinimalHamSG1272a2b": ("Library.Hamiltonian.MinimalModels.MinimalHamSG127_2a2b", "MinimalHamSG127_2a2b"),
    },
}
# Minimal models moved from Library/Hamiltonian/ into Library/Hamiltonian/MinimalModels/.
for _name in ("MinimalAltermagnetHamiltonian", "MinimalHamSG124_2b2d", "MinimalHamSG127_2a2b",
              "MinimalHamSG127_2c2d", "MinimalHamSG192_2b", "MinimalHamSG194_2a"):
    MOVED_MODULES[f"Library.Hamiltonian.{_name}"] = {"*": ("Library.Hamiltonian.MinimalModels", None)}


def resolve(old_module, name):
    """Return the current class for ``name`` pickled from ``old_module``."""
    table = MOVED_MODULES[old_module]
    module_name, class_name = table.get(name) or table["*"]
    return getattr(importlib.import_module(module_name), class_name or name)


class _MovedModuleLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = types.ModuleType(spec.name)

        def __getattr__(name):
            if name.startswith("__"):
                raise AttributeError(name)
            try:
                return resolve(spec.name, name)
            except (KeyError, AttributeError) as exc:
                raise AttributeError(
                    f"{spec.name}.{name} has no known successor; add it to "
                    f"Library.legacy_pickles.MOVED_MODULES"
                ) from exc

        module.__getattr__ = __getattr__
        return module

    def exec_module(self, module):
        pass


class _MovedModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in MOVED_MODULES:
            return importlib.util.spec_from_loader(fullname, _MovedModuleLoader())
        return None


def install():
    """Serve the old module paths; consulted only when a normal import fails."""
    if not any(isinstance(finder, _MovedModuleFinder) for finder in sys.meta_path):
        sys.meta_path.append(_MovedModuleFinder())
