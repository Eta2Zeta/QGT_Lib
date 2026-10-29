import numpy as np

from qgt.hamiltonians.base import hamiltonian

def get_Hamiltonian(
    Hamiltonian,
    kx,
    ky,
    kz=0,
    get_first_magnus=False,
    get_second_magnus=False,
    harmonics=None,
):
    """
    Get the Hamiltonian matrix for a given kx, ky. Optionally return the first
    and/or second Magnus terms along with the effective Hamiltonian.

    Parameters:
    - Hamiltonian: A Hamiltonian class object.
    - kx, ky: The k-space coordinates.
    - kz: The z component of the momentum (default 0).
    - get_first_magnus (bool): If True, return the first Magnus term.
    - get_second_magnus (bool): If True, return the second Magnus term.
    - harmonics: Optional precomputed Fourier-component dictionary to reuse.

    Returns:
    - If no additional terms are requested, returns ``(H_k, H_prime)``.
    - If additional terms are requested, returns the effective Hamiltonian and
      the requested Magnus terms.
    """
    if not isinstance(Hamiltonian, hamiltonian):
        raise TypeError(
            "get_Hamiltonian expects an instance of qgt.hamiltonians.base.hamiltonian."
        )

    H_k, H_prime = Hamiltonian.effective_hamiltonian(
        kx,
        ky,
        kz,
        harmonics=harmonics,
    )

    results = [H_k]

    if get_first_magnus:
        first_magnus = Hamiltonian.magnus_first_term(
            kx,
            ky,
            kz,
            harmonics=harmonics,
        )
        results.append(first_magnus)

    if get_second_magnus:
        second_magnus = Hamiltonian.magnus_second_term(
            kx,
            ky,
            kz,
            harmonics=harmonics,
        )
        results.append(second_magnus)

    return tuple(results) if len(results) > 1 else (H_k, H_prime)


def hamiltonian_at(Hamiltonian, k):
    """
    H at momenta ``k`` of shape ``(..., 3)``, as an array of shape ``(..., dim, dim)``.

    A static model is evaluated with ``compute_static_vectorized``; a driven
    one (``A0 != 0``) point by point with its Floquet ``effective_hamiltonian``.
    """
    k = np.asarray(k, dtype=float)
    flat = k.reshape(-1, 3)
    if getattr(Hamiltonian, "A0", 0) == 0:
        H = Hamiltonian.compute_static_vectorized(flat[:, 0], flat[:, 1], flat[:, 2])
    else:
        H = np.stack([Hamiltonian.effective_hamiltonian(kx, ky, kz)[0] for kx, ky, kz in flat])
    return np.asarray(H, dtype=complex).reshape(k.shape[:-1] + (Hamiltonian.dim, Hamiltonian.dim))
