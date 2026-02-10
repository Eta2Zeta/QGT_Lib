import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from .indexing_lib import *
from .Hamiltonian_helper import get_Hamiltonian
from .Eigenvector import *


def get_eigenvalues_and_eigenvectors(Hamiltonian):
    """
    Hamiltonian is any Matrix

    This solves the Hamiltonian for its spectrum and Eigenstates
    """
    eigenvalues, eigenvectors = np.linalg.eig(Hamiltonian)
    eigenvectors = np.transpose(eigenvectors)
    return eigenvalues, eigenvectors

def eigenvalues_and_vectors_eigenvalue_ordering(
    Hamiltonian, kx, ky, kz=0, eigenvector: 'Eigenvectors' = None, band_index=None, calculate_perturbation=False
    ):
    """
    Calculate eigenvalues and eigenvectors, with optional reordering based on the zone number.
    Also calculates the maximum perturbation strength for a given band index.

    Parameters:
    - Hamiltonian: The Hamiltonian object.
    - kx: A number representing the kx value.
    - ky: A number representing the ky value.
    - kz: A number representing the kz value (default 0).
    - eigenvector: An optional Eigenvector object for phase correction.
    - band_index: Index of the band to compute perturbation strength.

    Returns:
    - eigenvalues: The sorted eigenvalues.
    - eigenvectors: The sorted and possibly reordered eigenvectors.
    - max_perturbation: Maximum perturbation strength for the given band (or None if band_index is None).
    """

    H_k, H_prime = get_Hamiltonian(Hamiltonian, kx, ky, kz=kz)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = get_eigenvalues_and_eigenvectors(H_k)

    # Sort the eigenvalues and eigenvectors in ascending order
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[idx, :]
    
    # Set the new eigenvectors with phase correction
    if eigenvector is not None:
        eigenvectors = eigenvector.set_eigenvectors_eigenvalue_preordered(eigenvectors, eigenvalues, kx, ky, ignore_small_phase_diff=False)

    # Initialize max perturbation strength
    max_perturbation = None

    # Compute the perturbation strength if band_index is specified
    if band_index is not None and calculate_perturbation:
        num_bands = len(eigenvalues)

        # Ensure the band index is within valid range
        if not (0 <= band_index < num_bands):
            raise ValueError(f"Invalid band_index {band_index}. Must be between 0 and {num_bands-1}.")

        perturbation_values = []

        # Check if the band is at the lower edge (band 0)
        if band_index == 0:
            # Compute only for (0,1)
            n, m = 0, 1
            delta_E = eigenvalues[n] - eigenvalues[m]
            if delta_E != 0:
                pert_strength = abs(eigenvectors[n].conj().T @ H_prime @ eigenvectors[m]) / delta_E
                perturbation_values.append(pert_strength)

        # Check if the band is at the upper edge (last band)
        elif band_index == num_bands - 1:
            # Compute only for (last-1, last)
            n, m = num_bands - 2, num_bands - 1
            delta_E = np.real(eigenvalues[n] - eigenvalues[m])
            if delta_E != 0:
                pert_strength = abs(eigenvectors[n].conj().T @ H_prime @ eigenvectors[m]) / delta_E
                perturbation_values.append(pert_strength)

        # Otherwise, compute for both (band-1, band) and (band, band+1)
        else:
            for m in [band_index - 1, band_index + 1]:
                n = band_index
                delta_E = eigenvalues[n] - eigenvalues[m]
                if delta_E != 0:
                    pert_strength = abs(eigenvectors[n].conj().T @ H_prime @ eigenvectors[m]) / delta_E
                    perturbation_values.append(pert_strength)

        # Find the maximum perturbation strength
        max_perturbation = max(perturbation_values) if perturbation_values else 0
    
    magnus_operator_norm = np.linalg.norm(H_prime, 2)
    if max_perturbation is not None:
        return eigenvalues, eigenvectors, max_perturbation, magnus_operator_norm
    else:   
        return eigenvalues, eigenvectors

