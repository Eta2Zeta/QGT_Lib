"""A QGT metric versus signed 1/omega, left and right drive joined.

Settings: configs/plot/idealness_1d.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.plotting.idealness_1d import plot_qgt_metric_vs_signed_invomega_joined  # noqa: E402


if __name__ == "__main__":
    run(plot_qgt_metric_vs_signed_invomega_joined, config_path(sys.argv[1] if len(sys.argv) > 1 else "idealness_1d.toml", "plot"))
