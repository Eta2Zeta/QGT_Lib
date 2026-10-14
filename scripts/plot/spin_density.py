"""Spin textures of the newest spin_density run for a model.

Settings: configs/plot/spin_density.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.plotting.spin_density import latest_spin_density_dataset, plot_all_spin_components  # noqa: E402


def plot_latest(model_name):
    dataset_dir = latest_spin_density_dataset(model_name)
    if dataset_dir is not None:
        plot_all_spin_components(dataset_dir)


if __name__ == "__main__":
    run(plot_latest, config_path(sys.argv[1] if len(sys.argv) > 1 else "spin_density.toml", "plot"))
