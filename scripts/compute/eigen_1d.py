"""Eigenvalues along a 1D line through the Brillouin zone.

Settings: configs/compute/eigen_1d.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.workflows.eigen_1d import calculation_1d  # noqa: E402


if __name__ == "__main__":
    run(calculation_1d, config_path(sys.argv[1] if len(sys.argv) > 1 else "eigen_1d.toml", "compute"))
