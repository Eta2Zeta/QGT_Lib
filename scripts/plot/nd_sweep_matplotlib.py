"""A qgt_nd sweep in a matplotlib window with sliders.

Settings: configs/plot/nd_sweep_matplotlib.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.plotting.nd_sweep_matplotlib import dynamic_nd_field_with_bands_matplotlib  # noqa: E402


if __name__ == "__main__":
    run(dynamic_nd_field_with_bands_matplotlib, config_path(sys.argv[1] if len(sys.argv) > 1 else "nd_sweep_matplotlib.toml", "plot"))
