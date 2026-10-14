"""Small numerical helpers shared across the package."""

import numpy as np


def replace_zeros_with_nan(Z):
    """Replace zero values in the array with NaN."""
    return np.where(Z == 0, np.nan, Z)


# Sign checker
def sign_check(vec1, vec2): 
    if np.dot(vec1, vec2) < 0: 
        return vec1, -vec2
    else: 
        return vec1, vec2


def in_range(w, omega_min, omega_max):
    if (omega_min is not None) and (w < omega_min): return False
    if (omega_max is not None) and (w > omega_max): return False
    return True
