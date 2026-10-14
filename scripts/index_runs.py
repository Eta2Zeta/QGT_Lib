"""Write Results/runs/index.csv: one row per calculation run, newest first."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from Library.runs import write_index  # noqa: E402

if __name__ == "__main__":
    print(f"Wrote {write_index()}")
