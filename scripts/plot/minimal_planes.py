"""Standard figures and the gallery for the MinimalModels plane datasets.

Takes command-line options rather than a config file; see --help, e.g.
    python scripts/plot/minimal_planes.py --workers 2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (not needed after `pip install -e .`)

from qgt.plotting.minimal_planes import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
