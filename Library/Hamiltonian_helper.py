from .Hamiltonian.Hamiltonian import hamiltonian

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
            "get_Hamiltonian expects an instance of Library.Hamiltonian.Hamiltonian.hamiltonian."
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
