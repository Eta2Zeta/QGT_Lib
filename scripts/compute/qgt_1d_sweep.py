"""QGT along a 1D line while sweeping one model parameter.

Settings: configs/compute/qgt_1d_sweep.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.workflows.qgt_1d_sweep import sweep_single_param_1d  # noqa: E402


if __name__ == "__main__":
    run(sweep_single_param_1d, config_path(sys.argv[1] if len(sys.argv) > 1 else "qgt_1d_sweep.toml", "compute"))
