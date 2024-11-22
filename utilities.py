import numpy as np


def replace_zeros_with_nan(Z):
    """Replace zero values in the array with NaN."""
    return np.where(Z == 0, np.nan, Z)