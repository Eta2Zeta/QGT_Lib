"""All 27 explicit minimal altermagnet families in Roig et al., Tables III-IV.

Each model is a regular QGT ``hamiltonian`` object. ``MINIMAL_MODELS`` maps
public class names to classes for discovery or parameter sweeps.
"""

from qgt.hamiltonians.minimal_models.MinimalAltermagnetHamiltonian import (
    MinimalAltermagnetHamiltonian,
)
from qgt.hamiltonians.minimal_models.MinimalHamSG11_2a2b2c2d import MinimalHamSG11_2a2b2c2d
from qgt.hamiltonians.minimal_models.MinimalHamSG13_2a2b2c2d import MinimalHamSG13_2a2b2c2d
from qgt.hamiltonians.minimal_models.MinimalHamSG14_2a2b2c2d import MinimalHamSG14_2a2b2c2d
from qgt.hamiltonians.minimal_models.MinimalHamSG49_2a2b2c2d import MinimalHamSG49_2a2b2c2d
from qgt.hamiltonians.minimal_models.MinimalHamSG51_2a2b2c2d import MinimalHamSG51_2a2b2c2d
from qgt.hamiltonians.minimal_models.MinimalHamSG53_2a2b2c2d import MinimalHamSG53_2a2b2c2d
from qgt.hamiltonians.minimal_models.MinimalHamSG55_2a2b2c2d import MinimalHamSG55_2a2b2c2d
from qgt.hamiltonians.minimal_models.MinimalHamSG58_2a2b2c2d import MinimalHamSG58_2a2b2c2d
from qgt.hamiltonians.minimal_models.MinimalHamSG83_2e2f import MinimalHamSG83_2e2f
from qgt.hamiltonians.minimal_models.MinimalHamSG84_2a2b import MinimalHamSG84_2a2b
from qgt.hamiltonians.minimal_models.MinimalHamSG84_2c2d import MinimalHamSG84_2c2d
from qgt.hamiltonians.minimal_models.MinimalHamSG123_2e2f import MinimalHamSG123_2e2f
from qgt.hamiltonians.minimal_models.MinimalHamSG124_2b2d import MinimalHamSG124_2b2d
from qgt.hamiltonians.minimal_models.MinimalHamSG127_2a2b import MinimalHamSG127_2a2b
from qgt.hamiltonians.minimal_models.MinimalHamSG127_2c2d import MinimalHamSG127_2c2d
from qgt.hamiltonians.minimal_models.MinimalHamSG128_2a2b import MinimalHamSG128_2a2b
from qgt.hamiltonians.minimal_models.MinimalHamSG131_2a2b import MinimalHamSG131_2a2b
from qgt.hamiltonians.minimal_models.MinimalHamSG131_2c2d import MinimalHamSG131_2c2d
from qgt.hamiltonians.minimal_models.MinimalHamSG132_2a2c import MinimalHamSG132_2a2c
from qgt.hamiltonians.minimal_models.MinimalHamSG136_2a2b import MinimalHamSG136_2a2b
from qgt.hamiltonians.minimal_models.MinimalHamSG163_2b import MinimalHamSG163_2b
from qgt.hamiltonians.minimal_models.MinimalHamSG165_2b import MinimalHamSG165_2b
from qgt.hamiltonians.minimal_models.MinimalHamSG176_2b import MinimalHamSG176_2b
from qgt.hamiltonians.minimal_models.MinimalHamSG192_2b import MinimalHamSG192_2b
from qgt.hamiltonians.minimal_models.MinimalHamSG193_2b import MinimalHamSG193_2b
from qgt.hamiltonians.minimal_models.MinimalHamSG194_2a import MinimalHamSG194_2a
from qgt.hamiltonians.minimal_models.MinimalHamSG223_2a import MinimalHamSG223_2a

MINIMAL_MODELS = {
    "MinimalHamSG11_2a2b2c2d": MinimalHamSG11_2a2b2c2d,
    "MinimalHamSG13_2a2b2c2d": MinimalHamSG13_2a2b2c2d,
    "MinimalHamSG14_2a2b2c2d": MinimalHamSG14_2a2b2c2d,
    "MinimalHamSG49_2a2b2c2d": MinimalHamSG49_2a2b2c2d,
    "MinimalHamSG51_2a2b2c2d": MinimalHamSG51_2a2b2c2d,
    "MinimalHamSG53_2a2b2c2d": MinimalHamSG53_2a2b2c2d,
    "MinimalHamSG55_2a2b2c2d": MinimalHamSG55_2a2b2c2d,
    "MinimalHamSG58_2a2b2c2d": MinimalHamSG58_2a2b2c2d,
    "MinimalHamSG83_2e2f": MinimalHamSG83_2e2f,
    "MinimalHamSG84_2a2b": MinimalHamSG84_2a2b,
    "MinimalHamSG84_2c2d": MinimalHamSG84_2c2d,
    "MinimalHamSG123_2e2f": MinimalHamSG123_2e2f,
    "MinimalHamSG124_2b2d": MinimalHamSG124_2b2d,
    "MinimalHamSG127_2a2b": MinimalHamSG127_2a2b,
    "MinimalHamSG127_2c2d": MinimalHamSG127_2c2d,
    "MinimalHamSG128_2a2b": MinimalHamSG128_2a2b,
    "MinimalHamSG131_2a2b": MinimalHamSG131_2a2b,
    "MinimalHamSG131_2c2d": MinimalHamSG131_2c2d,
    "MinimalHamSG132_2a2c": MinimalHamSG132_2a2c,
    "MinimalHamSG136_2a2b": MinimalHamSG136_2a2b,
    "MinimalHamSG163_2b": MinimalHamSG163_2b,
    "MinimalHamSG165_2b": MinimalHamSG165_2b,
    "MinimalHamSG176_2b": MinimalHamSG176_2b,
    "MinimalHamSG192_2b": MinimalHamSG192_2b,
    "MinimalHamSG193_2b": MinimalHamSG193_2b,
    "MinimalHamSG194_2a": MinimalHamSG194_2a,
    "MinimalHamSG223_2a": MinimalHamSG223_2a,
}

__all__ = ["MinimalAltermagnetHamiltonian", "MINIMAL_MODELS", *MINIMAL_MODELS]
