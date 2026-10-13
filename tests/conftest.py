"""Shared pytest setup.

- Draw matplotlib figures off-screen so tests never open windows.
- Point the data folder at a throwaway directory so no test can write into
  the real Results/ folder. QGT_REAL_DATA keeps the real location for tests
  that only read old results.
"""

import os
import tempfile

os.environ["MPLBACKEND"] = "Agg"

from Library.data_paths import data_root  # noqa: E402

os.environ["QGT_REAL_DATA"] = data_root()
os.environ["QGT_DATA"] = tempfile.mkdtemp(prefix="qgt_test_data_")
