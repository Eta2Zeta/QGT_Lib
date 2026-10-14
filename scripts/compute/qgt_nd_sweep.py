"""QGT on a 2D k-grid over an N-dimensional parameter grid; writes a qgt_nd run.

Settings: configs/compute/qgt_nd_sweep.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

import multiprocessing  # noqa: E402

import numpy as np  # noqa: E402
from qgt.config import config_path, run  # noqa: E402
from qgt.workflows.qgt_nd_sweep import compute_qgt_nd_parallel  # noqa: E402


def prepare(model, arguments):
    arguments["float_dtype"] = np.dtype(arguments.get("float_dtype", "float64")).type
    return arguments


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run(compute_qgt_nd_parallel, config_path(sys.argv[1] if len(sys.argv) > 1 else "qgt_nd_sweep.toml", "compute"), prepare=prepare)
