"""Two QGT runs compared along the symmetry path.

Settings: configs/plot/qgt_sym_lines.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.plotting.qgt_sym_lines import plot_two_qgt_sym_lines  # noqa: E402


if __name__ == "__main__":
    run(plot_two_qgt_sym_lines, config_path(sys.argv[1] if len(sys.argv) > 1 else "qgt_sym_lines.toml", "plot"))
