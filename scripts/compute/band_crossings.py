"""Exact band crossings of minimal models; writes one band_crossings run per model.

Settings: configs/compute/band_crossings.toml (one model), or pass another config
file, e.g. configs/compute/band_crossings_all.toml for all 27 models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.workflows import band_crossings  # noqa: E402


if __name__ == "__main__":
    run(band_crossings, config_path(sys.argv[1] if len(sys.argv) > 1 else "band_crossings.toml", "compute"))
