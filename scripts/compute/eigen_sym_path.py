"""Eigenvalues along the model's high-symmetry path.

Settings: configs/compute/eigen_sym_path.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.workflows.eigen_sym_path import calculation_sym_points  # noqa: E402


if __name__ == "__main__":
    run(calculation_sym_points, config_path(sys.argv[1] if len(sys.argv) > 1 else "eigen_sym_path.toml", "compute"))
