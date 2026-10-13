"""Locations of calculation data, which live outside the repository.

The data folder defaults to ``Results/`` next to the repository
(``~/Documents/Quantum_Geometric_Tensor/Results``); set the ``QGT_DATA``
environment variable to use another location. Everything that used to live
in the repository's ``results/`` folder is under ``legacy/`` with its
original layout.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def data_root():
    """Return the data folder as a string path."""
    return os.path.expanduser(os.environ.get("QGT_DATA", str(REPO_ROOT.parent / "Results")))


def legacy_dir(*parts):
    """Return a path inside the former ``results/`` tree, e.g. ``legacy_dir("2D_QGT_ND", name)``."""
    return os.path.join(data_root(), "legacy", *parts)
