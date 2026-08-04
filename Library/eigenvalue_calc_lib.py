import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from itertools import permutations
from .indexing_lib import *
from .Hamiltonian.Hamiltonian import hamiltonian
from .Hamiltonian_helper import get_Hamiltonian
from .Eigenvector import *
from .diagnalization import eigenvalues_and_vectors_eigenvalue_ordering, get_eigenvalues_and_eigenvectors
from .dimension_lib import map_k_by_order


# * Checkers
def check_eigen_solution(Hamiltonian, kx, ky, eigenvalues, eigenvectors, tolerance=1e-6):
    """
    Checks if the calculated eigenvalues and eigenvectors satisfy the eigenvalue equation Hψ = λψ.
    
    Parameters:
    - Hamiltonian: function that generates the Hamiltonian matrix
    - kx, ky: the k-points for which to check the eigenvalue-eigenvector solution
    - eigenvalues: array of calculated eigenvalues
    - eigenvectors: array of calculated eigenvectors
    - tolerance: acceptable tolerance for checking if Hψ = λψ
    
    Returns:
    - valid: Boolean indicating if all eigenvalue-eigenvector pairs are valid
    """
    H_k = Hamiltonian(kx, ky)
    valid = True

    for i in range(len(eigenvalues)):
        # Compute Hψ
        H_psi = np.dot(H_k, eigenvectors[i])
        # Compute λψ
        lambda_psi = eigenvalues[i] * eigenvectors[i]
        # Calculate the norm of the difference
        diff = np.linalg.norm(H_psi - lambda_psi)
        
        if diff > tolerance:
            print(f"Eigenvalue {i + 1} at (kx, ky) = ({kx:.4f}, {ky:.4f}) does not satisfy the eigenvalue equation.")
            print(f"Difference norm: {diff}")
            valid = False

    return valid


# * Phase Calculations
def calculate_neighbor_phase_array(eigenfunctions, mesh_spacing, dim):
    # Initialize the neighbor_phase_array_after_calc with zeros
    neighbor_phase_array_after_calc = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)

    # Define the possible neighbor offsets
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Iterate over each point in the grid
    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            for d in range(dim):
                phase_sum = 0
                num_neighbors = 0
                
                # Iterate over each possible neighbor
                for offset in neighbor_offsets:
                    ni, nj = i + offset[0], j + offset[1]
                    
                    # Check if the neighbor is within bounds
                    if 0 <= ni < mesh_spacing and 0 <= nj < mesh_spacing:
                        # Calculate the phase difference between the current point and the neighbor
                        phase_diff = np.angle(np.vdot(eigenfunctions[i, j, d], eigenfunctions[ni, nj, d]))
                        phase_sum += np.abs(phase_diff)/np.pi
                        num_neighbors += 1
                
                # Normalize the sum by the number of neighbors
                if num_neighbors > 0:
                    neighbor_phase_array_after_calc[i, j, d] = phase_sum/num_neighbors

    return neighbor_phase_array_after_calc

def recursive_phase_correction(eigenfunctions, neighbor_phase_array, mesh_spacing, d, threshold=0.99):
    """
    Corrects the phase of the eigenvectors recursively for a specific dimension `d`
    until the maximum neighboring phase difference is below a given threshold.
    
    Parameters:
    - eigenfunctions: Array of eigenfunctions.
    - neighbor_phase_array: Array storing the neighboring phase differences.
    - mesh_spacing: Size of the grid.
    - d: Dimension (band) of interest to correct.
    - threshold: The threshold value for the phase difference. Recursion stops when all phases are below this value.
    """
    # Find the point with the largest neighboring phase difference
    max_phase = np.max(neighbor_phase_array[:, :, d])
    
    # Stop recursion if all neighboring phase differences are below the threshold
    if max_phase < threshold:
        return
    
    # Get the indices of the point with the largest neighboring phase
    max_indices = np.unravel_index(np.argmax(neighbor_phase_array[:, :, d]), (mesh_spacing, mesh_spacing))
    i, j = max_indices
    
    # Get the neighboring offsets (up, down, left, right)
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Adjust the eigenvector at the point with the largest neighboring phase
    phase_sum = 0
    num_neighbors = 0

    for offset in neighbor_offsets:
        ni, nj = i + offset[0], j + offset[1]
        if 0 <= ni < mesh_spacing and 0 <= nj < mesh_spacing:
            # Calculate the phase difference between the current point and the neighbor for the selected band `d`
            phase_diff = np.angle(np.vdot(eigenfunctions[i, j, d], eigenfunctions[ni, nj, d]))
            # Adjust the eigenvector to minimize the phase difference
            eigenfunctions[i, j, d] *= np.exp(-1j * phase_diff)
            phase_sum += np.abs(phase_diff) / np.pi  # Accumulate the phase difference
            num_neighbors += 1

    # Recalculate the neighboring phase array for the specific dimension `d`
    neighbor_phase_array = calculate_neighbor_phase_array(eigenfunctions, mesh_spacing, len(eigenfunctions[0, 0]))
    
    # Recursively correct the next point with the largest neighboring phase
    recursive_phase_correction(eigenfunctions, neighbor_phase_array, mesh_spacing, d, threshold)



# * Basic getting eigenvalues and eigenvectors 


# Function to calculate eigenvalues and eigenvectors
def eigenvalues_and_vectors_eigenvector_ordering(Hamiltonian, kx, ky, eigenvector = None):
    """
    Hamiltonian should be a function with Hamiltonian (kx, ky, args = args) as arguments

    This is ordered by the size of the eigenvalues, so there could be some discontinuities in the eivenvectors, 
    and this hurts the calculation of the quantum geometry.
    """
    H_k, _ = get_Hamiltonian(Hamiltonian, kx, ky)
    
    eigenvalues, eigenvectors = get_eigenvalues_and_eigenvectors(H_k)

    # Set the new eigenvectors with phase correction
    eigenvalues, eigenvectors = eigenvector.set_eigenvectors_eigenvector_ordered(eigenvectors, eigenvalues, kx, ky)
    
    return eigenvalues, eigenvectors


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

# & Analytical Calculations

def analytic_eigenvalues_2d(hamiltonian, kx, ky, mesh_spacing, dim):
    """
    Compute analytical eigenvalues on a 2D k-grid using the Hamiltonian's analytic expression.

    Parameters:
        hamiltonian: Hamiltonian object with `analytic_eigenvalues(kx, ky)` method.
        kx, ky: 2D meshgrid arrays of shape (mesh_spacing, mesh_spacing).
        mesh_spacing: Integer defining the number of points along each axis.
        dim: Number of bands (should be 2 for RhombohedralGrapheneHamiltonian).

    Returns:
        eigenvalues: Array of shape (mesh_spacing, mesh_spacing, dim) containing band energies.
    """
    eigenvalues = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)

    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            eigenvalues[i, j] = hamiltonian.analytic_eigenvalues(kx[i, j], ky[i, j])

    return eigenvalues


# & Calculations in a normal grid
_FLOQUET_FOURIER_SAMPLES = 512


def max_floquet_perturbative_ratios(harmonics, omega, max_l):
    """Find the strongest interband photon-sector coupling for every band.

    The Fourier convention is ``H(t) = sum_l H_l exp(i l omega t)``.  For a
    source band ``n`` of the zero Fourier harmonic ``H_0``, this routine
    evaluates

    ``|<u_m|H_l|u_n>| / |E_n - E_m - l*omega|``

    for every other band ``m != n`` and every nonzero integer
    ``-max_l <= l <= max_l``.  It returns the largest ratio for each source
    band and the corresponding ``(m, l)`` pair.

    A coupled state with machine-zero detuning and nonzero matrix element is
    assigned an infinite ratio.  If every allowed matrix element for a band
    vanishes, its ratio remains zero and its index remains ``(-1, 0)``.
    """
    if isinstance(max_l, (bool, np.bool_)) or not isinstance(
        max_l,
        (int, np.integer),
    ):
        raise TypeError("max_l must be a positive integer")
    max_l = int(max_l)
    if max_l < 1:
        raise ValueError("max_l must be at least 1")

    omega = float(omega)
    if omega == 0.0:
        raise ValueError(
            "omega must be nonzero when calculating Floquet perturbative ratios"
        )

    harmonic_orders = np.concatenate(
        (
            np.arange(-max_l, 0, dtype=int),
            np.arange(1, max_l + 1, dtype=int),
        )
    )
    missing = [
        int(l_value)
        for l_value in harmonic_orders
        if int(l_value) not in harmonics
    ]
    if 0 not in harmonics:
        missing.insert(0, 0)
    if missing:
        raise KeyError(
            "Missing Fourier harmonics required for ratio calculation: "
            f"{missing}"
        )

    H_0 = np.asarray(harmonics[0], dtype=complex)
    if H_0.ndim != 2 or H_0.shape[0] != H_0.shape[1]:
        raise ValueError("The zero Fourier harmonic must be a square matrix")
    if not np.all(np.isfinite(H_0)):
        raise ValueError("The zero Fourier harmonic contains non-finite values")

    hermiticity_scale = max(1.0, float(np.linalg.norm(H_0, ord=np.inf)))
    hermiticity_error = float(np.linalg.norm(H_0 - H_0.conj().T, ord=np.inf))
    if hermiticity_error > 1e-9 * hermiticity_scale:
        raise ValueError(
            "The zero Fourier harmonic is not Hermitian within numerical tolerance"
        )
    H_0 = 0.5 * (H_0 + H_0.conj().T)

    energies, eigenvectors = np.linalg.eigh(H_0)
    dim = energies.size
    max_ratios = np.zeros(dim, dtype=float)
    max_indices = np.empty((dim, 2), dtype=np.int32)
    max_indices[:, 0] = -1
    max_indices[:, 1] = 0
    if dim < 2:
        return max_ratios, max_indices

    harmonic_stack = np.stack(
        [
            np.asarray(harmonics[int(l_value)], dtype=complex)
            for l_value in harmonic_orders
        ],
        axis=0,
    )
    expected_shape = (harmonic_orders.size, dim, dim)
    if harmonic_stack.shape != expected_shape:
        raise ValueError(
            "Every Fourier harmonic must have the same shape as H_0; "
            f"expected {expected_shape}, received {harmonic_stack.shape}"
        )
    if not np.all(np.isfinite(harmonic_stack)):
        raise ValueError("A nonzero Fourier harmonic contains non-finite values")

    # Shape convention: (photon harmonic l, coupled band m, source band n).
    matrix_elements = np.einsum(
        "im,lij,jn->lmn",
        eigenvectors.conj(),
        harmonic_stack,
        eigenvectors,
        optimize=True,
    )
    couplings = np.abs(matrix_elements)
    detunings = np.abs(
        energies[None, None, :]
        - energies[None, :, None]
        - harmonic_orders[:, None, None] * omega
    )

    machine_epsilon = np.finfo(float).eps
    energy_scale = max(
        1.0,
        float(np.max(np.abs(energies))),
        float(np.max(np.abs(harmonic_orders * omega))),
    )
    coupling_scale = max(1.0, float(np.max(couplings)))
    detuning_tolerance = 64.0 * machine_epsilon * energy_scale
    coupling_tolerance = 64.0 * machine_epsilon * coupling_scale

    ratios = np.zeros_like(detunings, dtype=float)
    finite_detuning = detunings > detuning_tolerance
    np.divide(couplings, detunings, out=ratios, where=finite_detuning)
    exact_resonance = (~finite_detuning) & (couplings > coupling_tolerance)
    ratios[exact_resonance] = np.inf

    # The requested diagnostic only compares a band with other bands.
    diagonal = np.arange(dim)
    ratios[:, diagonal, diagonal] = -np.inf

    flattened_ratios = ratios.reshape(harmonic_orders.size * dim, dim)
    flattened_argmax = np.argmax(flattened_ratios, axis=0)
    strongest_ratios = flattened_ratios[flattened_argmax, np.arange(dim)]
    has_nonzero_coupling = strongest_ratios > 0.0

    harmonic_positions, coupled_bands = np.divmod(flattened_argmax, dim)
    max_ratios[has_nonzero_coupling] = strongest_ratios[has_nonzero_coupling]
    max_indices[has_nonzero_coupling, 0] = coupled_bands[has_nonzero_coupling]
    max_indices[has_nonzero_coupling, 1] = harmonic_orders[
        harmonic_positions[has_nonzero_coupling]
    ]
    return max_ratios, max_indices


def grid_eigenvalues_eigenfunctions(
    Hamiltonian,
    ki,
    kj,
    mesh_spacing,
    dim,
    kk=0,
    order="xyz",
    show_progress=True,
    max_l=10,
):
    """Diagonalize a 2D grid expressed in any supported coordinate order.

    ``ki`` and ``kj`` store the two varying input coordinates. They are mapped
    to physical Cartesian momenta by :func:`map_k_by_order` before the
    Hamiltonian is evaluated. Consequently, orders such as ``xpz`` may use
    ``ki=r`` and ``kj=phi`` while the Hamiltonian still receives ``kx, ky, kz``.

    When ``Hamiltonian.A0`` is nonzero, the routine also checks photon sectors
    ``l=-max_l,...,-1,1,...,max_l``.  The Fourier-zero eigenbasis defines the
    band axis of the two diagnostic outputs.  The six returned arrays are
    ``eigenvalues``, ``eigenfunctions``, ``H_array``, ``H_prime_array``,
    ``floquet_max_ratio_grid``, and ``floquet_max_ratio_indices_grid``.  The
    last axis of the index grid stores ``(coupled_band, l)``.
    """
    ki = np.asarray(ki)
    kj = np.asarray(kj)
    if ki.ndim != 2 or kj.ndim != 2 or ki.shape != kj.shape:
        raise ValueError("ki and kj must be 2D arrays with identical shapes")

    expected_spacing = int(mesh_spacing)
    if ki.shape != (expected_spacing, expected_spacing):
        raise ValueError(
            "ki and kj shapes must match "
            f"(mesh_spacing, mesh_spacing); received {ki.shape} for "
            f"mesh_spacing={expected_spacing}"
        )

    if isinstance(max_l, (bool, np.bool_)) or not isinstance(
        max_l,
        (int, np.integer),
    ):
        raise TypeError("max_l must be a positive integer")
    max_l = int(max_l)
    if max_l < 1:
        raise ValueError("max_l must be at least 1")
    if max_l >= _FLOQUET_FOURIER_SAMPLES // 2:
        raise ValueError(
            "max_l must be smaller than half the number of FFT samples "
            f"({_FLOQUET_FOURIER_SAMPLES // 2}) to avoid harmonic aliasing"
        )

    grid_shape = ki.shape
    eigenfunctions = np.zeros(grid_shape + (dim, dim), dtype=complex)
    eigenvalues = np.zeros(grid_shape + (dim,), dtype=float)
    H_array = np.zeros(grid_shape + (dim, dim), dtype=complex)
    H_prime_array = np.zeros(grid_shape + (dim, dim), dtype=complex)
    floquet_max_ratio_grid = np.zeros(grid_shape + (dim,), dtype=float)
    floquet_max_ratio_indices_grid = np.empty(
        grid_shape + (dim, 2),
        dtype=np.int32,
    )
    floquet_max_ratio_indices_grid[..., 0] = -1
    floquet_max_ratio_indices_grid[..., 1] = 0

    drive_active = float(getattr(Hamiltonian, "A0", 0.0)) != 0.0
    if drive_active and float(Hamiltonian.omega) == 0.0:
        raise ValueError(
            "Hamiltonian.omega must be nonzero when Hamiltonian.A0 is nonzero"
        )

    ratio_harmonic_orders = list(range(-max_l, 0)) + [0] + list(
        range(1, max_l + 1)
    )
    effective_harmonic_orders = set(ratio_harmonic_orders)
    if drive_active and int(getattr(Hamiltonian, "magnus_order", 0)) >= 2:
        effective_harmonic_orders.update((-2, 2))

    eigenvector = Eigenvectors(dim)

    total_points = int(np.prod(grid_shape))
    for idx in tqdm(
        range(total_points),
        desc="Diagonalizing Hamiltonians",
        disable=not show_progress,
    ):
        i, j = np.unravel_index(idx, grid_shape)

        # Map coordinates according to order
        kx, ky, kz = map_k_by_order(ki[i, j], kj[i, j], kk, order)

        # --- Measure Hamiltonian computation time ---
        t0 = time.time()
        harmonics = None
        if drive_active:
            # Sampling H(t) and taking its FFT dominates the Fourier work.  Get
            # every required harmonic in this one call and reuse the result.
            harmonics = Hamiltonian.fourier_components_fft(
                sorted(effective_harmonic_orders),
                kx,
                ky,
                kz,
                M=_FLOQUET_FOURIER_SAMPLES,
            )

        H, H_prime = get_Hamiltonian(
            Hamiltonian,
            kx,
            ky,
            kz=kz,
            harmonics=harmonics,
        )
        t1 = time.time()
        elapsed = t1 - t0
        if show_progress and elapsed > 0.1:
            print(f"Hamiltonian computed at point ({kx}, {ky}, {kz}) in {elapsed:.6f} s")

        # Diagonalize the already-computed matrix so H_eff (and its FFT) is not
        # evaluated a second time at the same point.
        vals, vecs = get_eigenvalues_and_eigenvectors(H)
        eigenvalue_order = np.argsort(np.real(vals))
        vals = vals[eigenvalue_order]
        vecs = vecs[eigenvalue_order, :]
        vecs = eigenvector.set_eigenvectors_eigenvalue_preordered(
            vecs,
            vals,
            kx,
            ky,
            kz=kz,
            ignore_small_phase_diff=False,
        )

        if drive_active:
            max_ratios, max_indices = max_floquet_perturbative_ratios(
                harmonics,
                Hamiltonian.omega,
                max_l,
            )
            floquet_max_ratio_grid[i, j] = max_ratios
            floquet_max_ratio_indices_grid[i, j] = max_indices

        # Store results
        eigenfunctions[i, j] = vecs
        eigenvalues[i, j]    = np.real(vals)
        H_array[i, j]        = H
        H_prime_array[i, j]  = H_prime

    return (
        eigenvalues,
        eigenfunctions,
        H_array,
        H_prime_array,
        floquet_max_ratio_grid,
        floquet_max_ratio_indices_grid,
    )


# & Calculations in a normal grid (Ordered by Eigenvectors Overlap)
def grid_eigenvalues_eigenfunctions_ordered(Hamiltonian, ki, kj, mesh_spacing, dim):
    # Initialize arrays
    eigenfunctions  = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)
    eigenvalues     = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)
    H_array         = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)
    H_prime_array   = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)

    eigenvector = Eigenvectors(dim)

    # Flatten indices for a single progress bar
    total_points = mesh_spacing * mesh_spacing
    for idx in tqdm(range(total_points), desc="Diagonalizing Hamiltonians (Ordered - Snake)"):
        # Snake Pattern: Even rows (0, 2, ...) go 0->Max, Odd rows go Max->0
        i = idx // mesh_spacing
        remaining = idx % mesh_spacing
        
        if i % 2 == 0:
            j = remaining
        else:
            j = (mesh_spacing - 1) - remaining

        # --- Measure Hamiltonian computation time ---
        t0 = time.time()
        H, H_prime = get_Hamiltonian(Hamiltonian, ki[i, j], kj[i, j])
        t1 = time.time()
        elapsed = t1 - t0
        if elapsed > 0.1:  # Only print if it takes more than 10 ms
            print(f"Hamiltonian computed at point ({ki[i, j]}, {kj[i, j]}) in {elapsed:.6f} s")

        # Eigen decomposition (with ordering + phase fix)
        vals, vecs = eigenvalues_and_vectors_eigenvector_ordering(
            Hamiltonian, ki[i, j], kj[i, j], eigenvector
        )

        # Store results
        eigenfunctions[i, j] = vecs
        eigenvalues[i, j]    = vals
        H_array[i, j]        = H
        H_prime_array[i, j]  = H_prime

    return eigenvalues, eigenfunctions, H_array, H_prime_array



def spiral_eigenvalues_eigenfunctions_nobar(Hamiltonian, kx, ky, mesh_spacing, dim, phase_correction = True):
    # Initialize arrays to store eigenfunctions and eigenvalues
    eigenfunctions = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)
    eigenvalues = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)
    phase_factors_array = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)

    spiral_indices = get_spiral_indices(mesh_spacing)

    eigenvector = Eigenvectors(dim)

    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            k,l = spiral_indices[i,j]
            if phase_correction: 
                vals, vecs = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx[k, l], ky[k, l], eigenvector=eigenvector)
            else:
                vals, vecs = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx[k, l], ky[k, l], eigenvector=None)
            phase_factors = eigenvector.get_phase_factors()
            
            eigenfunctions[k, l] = vecs
            eigenvalues[k, l] = vals
            phase_factors_array[k, l] = phase_factors
            
    neighbor_phase_array_after_calc = calculate_neighbor_phase_array(eigenfunctions, mesh_spacing, dim)

    return eigenvalues, eigenfunctions, phase_factors_array, neighbor_phase_array_after_calc

def spiral_eigenvalues_eigenfunctions(
    Hamiltonian,
    kx,
    ky,
    mesh_spacing,
    dim,
    phase_correction=True,
    calculate_phase_factors=False,
    calculating_magnus_terms=False
):
    """
    Calculate eigenvalues, eigenfunctions, and store Magnus terms on a spiral grid,
    skipping k-points that fail the Hamiltonian's .valid_k_point method.

    Returns:
        eigenvalues, eigenfunctions, phase_factors_array, neighbor_phase_array_after_calc,
        magnus_first_term, magnus_second_term
    """
    eigenfunctions = np.full((mesh_spacing, mesh_spacing, dim, dim), np.nan, dtype=complex)
    eigenvalues = np.full((mesh_spacing, mesh_spacing, dim), np.nan, dtype=float)
    phase_factors_array = np.full((mesh_spacing, mesh_spacing, dim), np.nan, dtype=float)
    magnus_first_term = np.full((mesh_spacing, mesh_spacing, dim, dim), np.nan, dtype=complex)
    magnus_second_term = np.full((mesh_spacing, mesh_spacing, dim, dim), np.nan, dtype=complex)

    spiral_indices = get_spiral_indices(mesh_spacing)
    eigenvector = Eigenvectors(dim)

    with tqdm(total=mesh_spacing * mesh_spacing, desc="Processing kx-ky grid", unit="point") as pbar:
        for i in range(mesh_spacing):
            for j in range(mesh_spacing):
                k, l = spiral_indices[i, j]
                kx_kl, ky_kl = kx[k, l], ky[k, l]

                if hasattr(Hamiltonian, "valid_k_point") and not Hamiltonian.valid_k_point(kx_kl, ky_kl):
                    pbar.update(1)
                    continue

                if phase_correction:
                    vals, vecs = eigenvalues_and_vectors_eigenvector_ordering(
                        Hamiltonian, kx_kl, ky_kl, eigenvector=eigenvector
                    )
                    if calculate_phase_factors:
                        phase_factors_array[k, l] = eigenvector.get_phase_factors()
                else:
                    if calculating_magnus_terms:
                        vals, vecs, m1, m2 = eigenvalues_and_vectors_eigenvalue_ordering(
                            Hamiltonian, kx_kl, ky_kl, eigenvector=None
                        )
                        magnus_first_term[k, l] = m1
                        magnus_second_term[k, l] = m2
                    else:
                        vals, vecs = eigenvalues_and_vectors_eigenvalue_ordering(
                            Hamiltonian, kx_kl, ky_kl, eigenvector=None
                        )

                eigenvalues[k, l] = vals
                eigenfunctions[k, l] = vecs
                pbar.update(1)

    neighbor_phase_array_after_calc = calculate_neighbor_phase_array(eigenfunctions, mesh_spacing, dim)

    return eigenvalues, eigenfunctions, phase_factors_array, neighbor_phase_array_after_calc, magnus_first_term, magnus_second_term



# * Miscellaenous 
def capping_eigenvalues(eigenvalues, z_limit):
    eigenvalues[eigenvalues > z_limit] = z_limit
    eigenvalues[eigenvalues < -z_limit] = -z_limit
    return eigenvalues

    
def compute_eigenvalues_3d(hamiltonian, kx_vals, ky_vals, kz_vals):
    """
    Compute 3D eigenvalues and eigenvectors on a grid defined by kx_vals, ky_vals, kz_vals.
    Uses vectorized computation slice-by-slice along kz.
    
    Parameters:
    - hamiltonian: The Hamiltonian object (must support compute_static_vectorized).
    - kx_vals, ky_vals, kz_vals: 1D arrays defining the grid.
    
    Returns:
    - eigenvalues_3d: 4D array [nkx, nky, nkz, dim]
    - eigenvectors_3d: 5D array [nkx, nky, nkz, dim, dim] (last dim is vector component, 2nd to last is band)
    """
    mesh_size_x = len(kx_vals)
    mesh_size_y = len(ky_vals)
    mesh_size_z = len(kz_vals)
    
    print("Computing 3D eigenvalues and eigenvectors (vectorized batching)...")
    # Initialize full 3D array
    # Shape: (nx, ny, nz, dim)    
    try:
        dim = hamiltonian.dim
    except AttributeError:
        # Compute one point to get dim
        dim = hamiltonian(0,0,0).shape[0]

    eigenvalues_3d = np.zeros((mesh_size_x, mesh_size_y, mesh_size_z, dim))
    eigenvectors_3d = np.zeros((mesh_size_x, mesh_size_y, mesh_size_z, dim, dim), dtype=complex)
    
    # Iterate over kz to save memory and process slice by slice
    # Use indexing='ij' to match (nx, ny) shape of arrays
    kx_grid_2d, ky_grid_2d = np.meshgrid(kx_vals, ky_vals, indexing='ij') 
    kx_flat = kx_grid_2d.flatten()
    ky_flat = ky_grid_2d.flatten()
    
    for i, kz in enumerate(kz_vals):
        # Compute for this z-slice
        kz_flat = np.full_like(kx_flat, kz)
        
        # Compute vectorized
        if hasattr(hamiltonian, 'compute_static_vectorized'):
            H_slice = hamiltonian.compute_static_vectorized(kx_flat, ky_flat, kz_flat) # (N*N, dim, dim)
        else:
             # Fallback to loop if needed, but here we assume vectorized for speed as per previous logic
             # Or implement fallback
            H_slice = np.zeros((len(kx_flat), dim, dim), dtype=complex)
            for idx in range(len(kx_flat)):
                H_slice[idx] = hamiltonian(kx_flat[idx], ky_flat[idx], kz)

        
        # Diagonalize
        # eigh for eigenvectors
        evals, evecs = np.linalg.eigh(H_slice) # (N*N, dim), (N*N, dim, dim)
        
        # Reshape back to 2D slice
        eigenvalues_3d[:, :, i, :] = evals.reshape(mesh_size_x, mesh_size_y, dim)
        

        evecs_reshaped = evecs.reshape(mesh_size_x, mesh_size_y, dim, dim) # [x, y, component, band]
        eigenvectors_3d[:, :, i, :, :] = np.swapaxes(evecs_reshaped, -1, -2) # [x, y, band, component]
        
        if i % 10 == 0:
            print(f"Processed slice {i}/{mesh_size_z}")
            
    return eigenvalues_3d, eigenvectors_3d
