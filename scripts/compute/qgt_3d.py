"""QGT of all bands on the 3D grid of an eigen_3d run; writes a qgt_3d run.

Settings: configs/compute/qgt_3d.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.workflows.qgt_3d import calculate_3d_all_bands_parallel  # noqa: E402


if __name__ == "__main__":
    run(calculate_3d_all_bands_parallel, config_path(sys.argv[1] if len(sys.argv) > 1 else "qgt_3d.toml", "compute"))
