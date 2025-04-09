import numpy as np
from .Hamiltonian_v2 import hamiltonian

def get_Hamiltonian(Hamiltonian, kx, ky, get_first_magnus=False, get_second_magnus=False):
    """
    Get the Hamiltonian matrix for a given kx, ky. Optionally return the first 
    and/or second Magnus terms along with the effective Hamiltonian.

    Parameters:
    - Hamiltonian: The Hamiltonian object, function, or array.
    - kx, ky: The k-space coordinates.
    - get_first_magnus (bool): If True, return the first Magnus term.
    - get_second_magnus (bool): If True, return the second Magnus term.

    Returns:
    - If no additional terms are requested, returns the effective Hamiltonian (H_k).
    - If additional terms are requested, returns a tuple with the effective Hamiltonian and the requested Magnus terms.
    """
    if isinstance(Hamiltonian, hamiltonian):  # Check if it's a Hamiltonian class object
        H_k, H_prime = Hamiltonian.effective_hamiltonian(kx, ky)
        
        # Initialize a list for additional results
        results = [H_k]

        # Optionally compute the first Magnus term
        if get_first_magnus:
            first_magnus = Hamiltonian.magnus_first_term(kx, ky)
            results.append(first_magnus)

        # Optionally compute the second Magnus term
        if get_second_magnus:
            second_magnus = Hamiltonian.magnus_second_term(kx, ky)
            results.append(second_magnus)

        return tuple(results) if len(results) > 1 else (H_k, H_prime)

    elif callable(Hamiltonian):  # If it's a callable function
        H_k = Hamiltonian(kx, ky)
        return H_k

    elif isinstance(Hamiltonian, np.ndarray):  # If it's a static numpy array
        H_k = Hamiltonian  # Use it directly
        return H_k

    else:
        raise ValueError("Invalid Hamiltonian type. Must be a callable, a numpy array, or a Hamiltonian class object.")

