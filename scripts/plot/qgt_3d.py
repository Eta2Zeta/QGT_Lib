"""Views of a 3D QGT run.

Settings: configs/plot/qgt_3d.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.plotting import qgt_3d  # noqa: E402


if __name__ == "__main__":
    run(qgt_3d, config_path(sys.argv[1] if len(sys.argv) > 1 else "qgt_3d.toml", "plot"))
