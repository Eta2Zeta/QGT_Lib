"""QGT of all bands on the 2D grid of an eigen_2d run; writes a qgt_2d run.

Settings: configs/compute/qgt_2d.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.workflows.qgt_2d import calculate_2d_all_bands  # noqa: E402


if __name__ == "__main__":
    run(calculate_2d_all_bands, config_path(sys.argv[1] if len(sys.argv) > 1 else "qgt_2d.toml", "compute"))
