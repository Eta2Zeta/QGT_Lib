"""Former name of the qgt package, kept so old results and scripts still load.

Every old module path (Library.Hamiltonian..., Library.QGT_lib, ...) is served
by qgt.io.legacy_pickles. New code imports from qgt.
"""

from qgt.io import legacy_pickles as _legacy_pickles

_legacy_pickles.install()
