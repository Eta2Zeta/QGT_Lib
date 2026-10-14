"""Berry curvature projected onto every irrep of D3d.

Settings: configs/compute/qgt_2d_d3d_projection.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

import multiprocessing  # noqa: E402
from qgt.config import config_path, run  # noqa: E402
from qgt.workflows.qgt_2d_d3d_projection import calculate_d3d_all_irrep_berry_projections  # noqa: E402


def prepare(model, arguments):
    if arguments.get("band_indices") == "all":
        arguments["band_indices"] = tuple(range(model.dim))
    return arguments


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run(calculate_d3d_all_irrep_berry_projections, config_path(sys.argv[1] if len(sys.argv) > 1 else "qgt_2d_d3d_projection.toml", "compute"), prepare=prepare)
