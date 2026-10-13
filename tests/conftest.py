"""Shared pytest setup.

- Draw matplotlib figures off-screen so tests never open windows.
- Point the data folder at a throwaway directory so no test can write into
  the real Results/ folder.
"""

import os
import tempfile

os.environ["MPLBACKEND"] = "Agg"
os.environ["QGT_DATA"] = tempfile.mkdtemp(prefix="qgt_test_data_")
