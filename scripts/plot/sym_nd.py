"""Symmetry-path eigenvalues with sliders over a sym_nd parameter sweep.

Settings: configs/plot/sym_nd.toml, or pass another config file or a bundle .npz directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.plotting.sym_nd import dynamic_sym_nd  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].endswith(".npz"):
        dynamic_sym_nd(sys.argv[1])
    else:
        run(dynamic_sym_nd, config_path(sys.argv[1] if len(sys.argv) > 1 else "sym_nd.toml", "plot"))
