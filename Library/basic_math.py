import numpy as np
def commutator_static(H1, H2):
    """
    Static method to compute the commutator of two Hamiltonians.
    """
    return np.dot(H1, H2) - np.dot(H2, H1)