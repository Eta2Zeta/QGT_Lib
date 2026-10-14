"""Eigenvalues along 1D lines through Gamma at many angles.

Settings: configs/compute/eigen_1d_angles.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

import numpy as np  # noqa: E402
from qgt.config import config_path, run  # noqa: E402
from qgt.workflows.eigen_1d_angles import calculation_1d_at_angles  # noqa: E402


def prepare(model, arguments):
    arguments.setdefault("k_max", getattr(model, "k_theta", 2.0 * np.pi))
    return arguments


if __name__ == "__main__":
    run(calculation_1d_at_angles, config_path(sys.argv[1] if len(sys.argv) > 1 else "eigen_1d_angles.toml", "compute"), prepare=prepare)
