"""Quantum geometric tensor library."""

from . import legacy_pickles as _legacy_pickles

# Let old result pickles load although their classes have moved.
_legacy_pickles.install()
