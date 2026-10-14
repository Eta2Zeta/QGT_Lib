"""Eigenvalues on a 3D k-grid; writes an eigen_3d run.

Settings: configs/compute/eigen_3d.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.workflows.eigen_3d import calculation_3d  # noqa: E402


if __name__ == "__main__":
    run(calculation_3d, config_path(sys.argv[1] if len(sys.argv) > 1 else "eigen_3d.toml", "compute"))
