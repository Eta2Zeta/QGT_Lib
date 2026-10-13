"""Shared pytest setup: draw matplotlib figures off-screen so tests never open windows."""

import os

os.environ["MPLBACKEND"] = "Agg"
