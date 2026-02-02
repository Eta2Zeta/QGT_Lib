import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from itertools import permutations
from .indexing_lib import *
from .Hamiltonian.Hamiltonian_v2 import hamiltonian
from .Hamiltonian_helper import get_Hamiltonian
from .Eigenvector import *

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

# * Regional Calculations

def identify_regions(eigenfunctions, mesh_spacing, dim, phase_threshold=0.1):
    """
    Identify continuous regions in the kx, ky space where the phase difference between neighboring points
    is smaller than a specified threshold.
    """
    regions = np.zeros((mesh_spacing, mesh_spacing), dtype=int)
    current_region = 1
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def flood_fill(i, j):
        stack = [(i, j)]
        regions[i, j] = current_region
        while stack:
            x, y = stack.pop()
            for offset in neighbor_offsets:
                nx, ny = x + offset[0], y + offset[1]
                if 0 <= nx < mesh_spacing and 0 <= ny < mesh_spacing and regions[nx, ny] == 0:
                    phase_diff = np.angle(np.vdot(eigenfunctions[x, y, dim], eigenfunctions[nx, ny, dim]))
                    if abs(phase_diff) < phase_threshold:
                        regions[nx, ny] = current_region
                        stack.append((nx, ny))

    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            if regions[i, j] == 0:  # Unvisited point
                flood_fill(i, j)
                current_region += 1

    return regions, current_region - 1


def adjust_region_phases(eigenfunctions, regions, num_regions, dim):
    """
    Adjust the phase of each region to make the phase within each region consistent.
    """
    for region in range(1, num_regions + 1):
        region_indices = np.argwhere(regions == region)
        if len(region_indices) > 0:
            # Compute the average phase for the region
            avg_phase = 0
            for idx in region_indices:
                i, j = idx
                avg_phase += np.angle(eigenfunctions[i, j, dim])
            avg_phase /= len(region_indices)
            
            # Adjust all points in the region to align with the average phase
            for idx in region_indices:
                i, j = idx
                eigenfunctions[i, j, dim] *= np.exp(-1j * avg_phase)
    
    return eigenfunctions


# * Basic getting eigenvalues and eigenvectors 

def get_eigenvalues_and_eigenvectors(Hamiltonian):
    """
    Hamiltonian is any Matrix

    This solves the Hamiltonian for its spectrum and Eigenstates
    """
    eigenvalues, eigenvectors = np.linalg.eig(Hamiltonian)
    eigenvectors = np.transpose(eigenvectors)
    return eigenvalues, eigenvectors

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
        try:
            vals, vecs, pert, mon = eigenvalues_and_vectors_eigenvalue_ordering(
                Hamiltonian, kx, ky, eigenvector, band_index=band_index
            )
            phase_factors = eigenvector.get_phase_factors()

            eigenvalues[i] = vals
            eigenfunctions[i] = vecs
            phase_factors_array[i] = phase_factors
            perturbation_array[i] = pert
            magnus_operator_norm[i] = mon
        except Exception:
            # Just in case any numerical failure occurs
            continue

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
def grid_eigenvalues_eigenfunctions0(Hamiltonian, kx, ky, mesh_spacing, dim):
    # Initialize arrays
    eigenfunctions  = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)
    eigenvalues     = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)
    H_array         = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)
    H_prime_array   = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)

    eigenvector = Eigenvectors(dim)

    # Flatten indices for a single progress bar
    total_points = mesh_spacing * mesh_spacing
    for idx in tqdm(range(total_points), desc="Diagonalizing Hamiltonians"):
        i, j = divmod(idx, mesh_spacing)

        # Compute Hamiltonian and its derivative
        H, H_prime = get_Hamiltonian(Hamiltonian, kx[i, j], ky[i, j])

        # Eigen decomposition (with ordering + phase fix)
        vals, vecs = eigenvalues_and_vectors_eigenvalue_ordering(
            Hamiltonian, kx[i, j], ky[i, j], eigenvector
        )

        # Store results
        eigenfunctions[i, j] = vecs
        eigenvalues[i, j]    = vals
        H_array[i, j]        = H
        H_prime_array[i, j]  = H_prime

    return eigenvalues, eigenfunctions, H_array, H_prime_array


# & Calculations in a normal grid
def grid_eigenvalues_eigenfunctions(Hamiltonian, kx, ky, mesh_spacing, dim):
    # Initialize arrays
    eigenfunctions  = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)
    eigenvalues     = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)
    H_array         = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)
    H_prime_array   = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)

    eigenvector = Eigenvectors(dim)

    # Flatten indices for a single progress bar
    total_points = mesh_spacing * mesh_spacing
    for idx in tqdm(range(total_points), desc="Diagonalizing Hamiltonians"):
        i, j = divmod(idx, mesh_spacing)

        # --- Measure Hamiltonian computation time ---
        t0 = time.time()
        H, H_prime = get_Hamiltonian(Hamiltonian, kx[i, j], ky[i, j])
        t1 = time.time()
        elapsed = t1 - t0
        if elapsed > 0.1:  # Only print if it takes more than 10 ms
            print(f"Hamiltonian computed at point ({kx[i, j]}, {ky[i, j]}) in {elapsed:.6f} s")

        # Eigen decomposition (with ordering + phase fix)
        vals, vecs = eigenvalues_and_vectors_eigenvalue_ordering(
            Hamiltonian, kx[i, j], ky[i, j], eigenvector
        )

        # Store results
        eigenfunctions[i, j] = vecs
        eigenvalues[i, j]    = vals
        H_array[i, j]        = H
        H_prime_array[i, j]  = H_prime

    return eigenvalues, eigenfunctions, H_array, H_prime_array


# & Calculations in a normal grid (Ordered by Eigenvectors Overlap)
def grid_eigenvalues_eigenfunctions_ordered(Hamiltonian, kx, ky, mesh_spacing, dim):
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
        H, H_prime = get_Hamiltonian(Hamiltonian, kx[i, j], ky[i, j])
        t1 = time.time()
        elapsed = t1 - t0
        if elapsed > 0.1:  # Only print if it takes more than 10 ms
            print(f"Hamiltonian computed at point ({kx[i, j]}, {ky[i, j]}) in {elapsed:.6f} s")

        # Eigen decomposition (with ordering + phase fix)
        vals, vecs = eigenvalues_and_vectors_eigenvector_ordering(
            Hamiltonian, kx[i, j], ky[i, j], eigenvector
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
                    vals, vecs = eigenvalues_and_vectors_eigenvalue_ordering(
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
        
        # Reshape vectors
        # evecs is [pixel, component, band]
        # We want [x, y, band, component] (or [x, y, dimension_index(component), band_index])
        # "rows are vectors" -> band index is first?
        # In this lib, it seems "rows are vectors" usually means v[band, component].
        # numpy returns v[component, band].
        # So we transpose the last two dimensions.
        evecs_reshaped = evecs.reshape(mesh_size_x, mesh_size_y, dim, dim) # [x, y, component, band]
        eigenvectors_3d[:, :, i, :, :] = np.swapaxes(evecs_reshaped, -1, -2) # [x, y, band, component]
        
        if i % 10 == 0:
            print(f"Processed slice {i}/{mesh_size_z}")
            
    return eigenvalues_3d, eigenvectors_3d