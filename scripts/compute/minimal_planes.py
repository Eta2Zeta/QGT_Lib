"""QGT on two kz planes for the 27 minimal altermagnet models (resumable batch).

Takes command-line options rather than a config file; see --help, e.g.
    python scripts/compute/minimal_planes.py --workers 6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.workflows.minimal_planes import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
