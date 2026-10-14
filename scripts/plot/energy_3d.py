"""Energy isosurfaces and slices of one band of a 3D eigenvalue run.

Settings: configs/plot/energy_3d.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.plotting.energy_3d import plot_3d_energy  # noqa: E402


if __name__ == "__main__":
    run(plot_3d_energy, config_path(sys.argv[1] if len(sys.argv) > 1 else "energy_3d.toml", "plot"))
