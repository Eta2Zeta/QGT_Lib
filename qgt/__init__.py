"""Quantum geometric tensor library.

Subpackages: ``hamiltonians`` (models), ``spectra`` (eigenvalue solvers and
eigenvector alignment), ``geometry`` (QGT and wavefunction derivatives),
``symmetry`` (point groups, QGT group orbits, zones), ``kspace`` (grids,
indexing, symmetry paths), ``plotting``, and ``io`` (data folders, runs, old
results). ``topology`` holds Berry curvature, flux, Chern and winding numbers;
``utils`` small numerical helpers.
"""

from qgt.io import legacy_pickles as _legacy_pickles

# Let old results and scripts load although their modules have moved.
_legacy_pickles.install()
