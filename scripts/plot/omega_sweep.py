"""QGT versus drive frequency for left and right drive.

Settings: configs/plot/omega_sweep.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.plotting import omega_sweep  # noqa: E402


if __name__ == "__main__":
    run(omega_sweep, config_path(sys.argv[1] if len(sys.argv) > 1 else "omega_sweep.toml", "plot"))
