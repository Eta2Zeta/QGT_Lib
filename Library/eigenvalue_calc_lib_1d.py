import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from itertools import permutations
from .indexing_lib import *
from .Hamiltonian.Hamiltonian import hamiltonian
from .Eigenvector import *
from .diagnalization import eigenvalues_and_vectors_eigenvalue_ordering, get_eigenvalues_and_eigenvectors
from .dimension_lib import map_k_by_order

# & Calculations in an angled line

def line_eigenvalues_eigenfunctions(Hamiltonian, line_kx, line_ky, band_index=None):
    """
    Calculate eigenvalues and eigenvectors along a line in the kx-ky plane.

    If Hamiltonian has a method called `valid_k_point(kx, ky)`, it will skip points where
    that method returns False by assigning np.nan to the output arrays.

    Parameters:
    - Hamiltonian: Object with a method compute_static(kx, ky) and possibly valid_k_point(kx, ky).
    - line_kx, line_ky: 1D arrays of the same length specifying the k-points.
    - band_index: If specified, filters eigenvalues/eigenvectors by band index.

    Returns:
    - eigenvalues: (num_points, dim) array
    - eigenfunctions: (num_points, dim, dim) array
    - phase_factors_array: (num_points, dim) array
    - perturbation_array: (num_points,) array [only if band_index is specified]
    """
    dim = Hamiltonian.dim
    num_points = len(line_kx)

    eigenvalues = np.full((num_points, dim), np.nan, dtype=float)
    eigenfunctions = np.full((num_points, dim, dim), np.nan, dtype=complex)
    phase_factors_array = np.full((num_points, dim), np.nan, dtype=float)
    perturbation_array = np.full(num_points, np.nan, dtype=float)
    magnus_operator_norm = np.full(num_points, np.nan, dtype=float)

    eigenvector = Eigenvectors(dim)

    has_valid_k = hasattr(Hamiltonian, "valid_k_point") and callable(getattr(Hamiltonian, "valid_k_point"))

    for i, (kx, ky) in enumerate(zip(line_kx, line_ky)):
        if has_valid_k and not Hamiltonian.valid_k_point(kx, ky):
            continue  # leave NaNs
        vals, vecs, pert, mon = eigenvalues_and_vectors_eigenvalue_ordering(
            Hamiltonian, kx, ky, eigenvector, band_index=band_index, calculate_perturbation=True
        )
        phase_factors = eigenvector.get_phase_factors()

        eigenvalues[i] = vals
        eigenfunctions[i] = vecs
        phase_factors_array[i] = phase_factors
        perturbation_array[i] = pert
        magnus_operator_norm[i] = mon

    if band_index is not None:
        return eigenvalues, eigenfunctions, phase_factors_array, perturbation_array, magnus_operator_norm
    else:
        return eigenvalues, eigenfunctions, phase_factors_array


def eigenvalues_along_path(Hamiltonian_Obj, k_path, use_analytical=False):
    """
    Compute eigenvalues and eigenvectors along an arbitrary k-path.

    Can optionally use an analytic eigenvalue expression if the Hamiltonian
    provides one. In that case, eigenvectors are returned as None.

    Parameters
    ----------
    Hamiltonian_Obj : hamiltonian
        A Hamiltonian object with a .dim attribute and compute_static method.
    k_path : ndarray, shape (N, 2) or (N, 3)
        Array of k-points along the path. 2-column arrays are treated as (kx, ky)
        with kz=0; 3-column arrays use (kx, ky, kz).
    use_analytical : bool
        If True and Hamiltonian_Obj has a get_analytical_eigenvalues method, use it
        instead of numerical diagonalization.
    Returns
    -------
    eigenvalues : ndarray, shape (N, dim)
        Eigenvalues sorted in ascending order at each k-point.
    eigenvectors : ndarray, shape (N, dim, dim) or None
        Eigenvectors ordered continuously along the path. None when
        use_analytical=True and an analytical eigenvalue expression is used.
    """
    k_path = np.asarray(k_path)
    num_points = len(k_path)
    dim = Hamiltonian_Obj.dim

    # --- Analytical shortcut ---
    if use_analytical and hasattr(Hamiltonian_Obj, 'get_analytical_eigenvalues'):
        print("Using analytical eigenvalue expression...")
        kx = k_path[:, 0]
        ky = k_path[:, 1]
        kz = k_path[:, 2] if k_path.shape[1] == 3 else np.zeros(num_points)
        return Hamiltonian_Obj.get_analytical_eigenvalues(kx, ky, kz), None

    if use_analytical:
        print("Analytical expression not found for this Hamiltonian. Falling back to numerical calculation...")

    # --- Numerical diagonalization ---
    eigenvalues = np.full((num_points, dim), np.nan, dtype=float)
    eigenvectors = np.full((num_points, dim, dim), np.nan, dtype=complex)
    eigenvector_tracker = Eigenvectors(dim)

    kz_col = k_path[:, 2] if k_path.shape[1] == 3 else np.zeros(num_points)

    for idx in tqdm(range(num_points), desc="Calculating path"):
        kx = k_path[idx, 0]
        ky = k_path[idx, 1]
        kz = kz_col[idx]

        evals, evecs = eigenvalues_and_vectors_eigenvalue_ordering(
            Hamiltonian_Obj,
            kx,
            ky,
            kz=kz,
            eigenvector=eigenvector_tracker,
            calculate_perturbation=False,
        )
        eigenvalues[idx] = np.real(evals)
        eigenvectors[idx] = evecs

    return eigenvalues, eigenvectors
