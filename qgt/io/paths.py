"""Locations of calculation data, which live outside the repository.

The data folder defaults to ``Results/`` next to the repository
(``~/Documents/Quantum_Geometric_Tensor/Results``); set the ``QGT_DATA``
environment variable to use another location. Everything that used to live
in the repository's ``results/`` folder is under ``legacy/`` with its
original layout.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root():
    """Return the data folder as a string path."""
    return os.path.expanduser(os.environ.get("QGT_DATA", str(REPO_ROOT.parent / "Results")))


def legacy_dir(*parts):
    """Return a path inside the former ``results/`` tree, e.g. ``legacy_dir("2D_QGT_ND", name)``."""
    return os.path.join(data_root(), "legacy", *parts)


_OLD_RESULTS_SEGMENT = os.sep + os.path.join("QGT_Lib", "results") + os.sep


def relocate_legacy_path(path):
    """Map a path recorded inside the former ``QGT_Lib/results/`` to its location under ``legacy/``.

    Older metadata (e.g. ``dataset_dir`` in MinimalModels manifests) stores absolute paths into
    the repository's old results folder. Other paths are returned unchanged.
    """
    path = os.fspath(path)
    if _OLD_RESULTS_SEGMENT in path:
        return legacy_dir(path.split(_OLD_RESULTS_SEGMENT, 1)[1])
    return path
