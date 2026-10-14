"""Fit the 1/omega and V slices exported by plot/idealness_2d.

Settings: configs/compute/fit_idealness_1d.toml, or pass another config file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.config import config_path, run  # noqa: E402
from qgt.workflows.fit_idealness_1d import process_slice_data  # noqa: E402

if __name__ == "__main__":
    try:
        run(process_slice_data, config_path(sys.argv[1] if len(sys.argv) > 1 else "fit_idealness_1d.toml", "compute"))
        print("Done.")
    except FileNotFoundError as error:
        print(error)
        print("Export the slices first: plot/idealness_2d with "
              "function = \"plot_slices_qgt_std_param2d_signed_invomega_joined\" and export_slice_path set.")
