import numpy as np
from indexing_lib import get_spiral_indices, order_grid_points_masked, plot_ordered_grid_3d, plot_ordered_grid_histogram, plot_ordered_grid_2d, unordered_grid_masked, order_grid_points_start_end
import matplotlib.pyplot as plt
from itertools import permutations

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


# def align_regions(eigenfunctions, regions, num_regions, dim):
#     """
#     Align the phases between different regions by matching the boundary phase of each region to a reference region.
#     """
#     reference_region = 1  # Choose the first region as the reference
    
#     for region in range(2, num_regions + 1):
#         found_boundary = False
#         boundary_phase_diff = None
        
#         # Find one boundary point between the current region and the reference region
#         for i in range(regions.shape[0]):
#             for j in range(regions.shape[1]):
#                 if regions[i, j] == region:
#                     # Check neighbors to see if they belong to the reference region
#                     for offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                         ni, nj = i + offset[0], j + offset[1]
#                         if 0 <= ni < regions.shape[0] and 0 <= nj < regions.shape[1]:
#                             if regions[ni, nj] == reference_region:
#                                 # Compute the phase difference between the current point and the reference region point
#                                 phase_diff = np.angle(np.vdot(eigenfunctions[i, j, dim], eigenfunctions[ni, nj, dim]))
                                
#                                 # Once the boundary phase difference is found, store it and break
#                                 boundary_phase_diff = phase_diff
#                                 found_boundary = True
#                                 break
#                     if found_boundary:
#                         break
        
#         # Apply the phase correction to the entire region
#         if boundary_phase_diff is not None:
#             for i, j in np.argwhere(regions == region):
#                 eigenfunctions[i, j, dim] *= np.exp(-1j * boundary_phase_diff)
    
#     return eigenfunctions


def align_regions(eigenfunctions, regions, num_regions, dim):
    """
    Align the phases between different regions by matching the largest boundary phase difference of each region 
    to a reference region.
    
    Parameters:
    - eigenfunctions: 4D array of eigenfunctions.
    - regions: 2D array of region indices.
    - num_regions: Total number of regions identified.
    - dim: The dimension (or band) to apply the phase correction to.
    """
    reference_region = 1  # Choose the first region as the reference
    
    for region in range(2, num_regions + 1):
        largest_boundary_phase_diff = None
        
        # Loop over all points to find the boundary between the current region and the reference region
        for i in range(regions.shape[0]):
            for j in range(regions.shape[1]):
                if regions[i, j] == region:
                    # Check neighbors to see if they belong to the reference region
                    for offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + offset[0], j + offset[1]
                        if 0 <= ni < regions.shape[0] and 0 <= nj < regions.shape[1]:
                            if regions[ni, nj] == reference_region:
                                # Compute the phase difference between the current point and the reference region point
                                phase_diff = np.angle(np.vdot(eigenfunctions[i, j, dim], eigenfunctions[ni, nj, dim]))
                                
                                # Track the largest boundary phase difference
                                if largest_boundary_phase_diff is None or abs(phase_diff) > abs(largest_boundary_phase_diff):
                                    largest_boundary_phase_diff = phase_diff
        
        # Apply the largest boundary phase correction to the entire region
        if largest_boundary_phase_diff is not None:
            for i, j in np.argwhere(regions == region):
                eigenfunctions[i, j, dim] *= np.exp(-1j * largest_boundary_phase_diff)
    
    return eigenfunctions


class Eigenvector:
    def __init__(self, dimension):
        self.dimension = dimension
        self.previous_eigenvector = None
        self.previous_kx = None
        self.previous_ky = None
        self.phase_factor = None

    def set_dimension(self, dim):
        self.dimension = dim

    def set_eigenvectors(self, new_eigenvector):
        if self.previous_eigenvector is not None:
            dot_product = np.vdot(self.previous_eigenvector, new_eigenvector)
            phase_diff = np.angle(dot_product)
            phase_factor = np.exp(-1j * phase_diff)
            new_eigenvector = new_eigenvector * phase_factor


        self.previous_eigenvector = new_eigenvector
        return new_eigenvector


class Eigenvectors:
    def __init__(self, dimension):
        self.dimension = dimension
        self.previous_eigenvectors = None
        self.previous_eigenvalues = None
        self.previous_kx = None
        self.previous_ky = None
        self.phase_factor = None

    def set_dimension(self, dim):
        self.dimension = dim

    # If the solutions are already ordered by eigenvalues, you can use this to just correct the phase factor of the eigenvalues
    def set_eigenvectors_eigenvalue_preordered(self, new_eigenvectors, new_eigenvalues, kx, ky, ignore_small_phase_diff=False, phase_diff_threshold=0.2):
        # Initialize phase_factor_array with the correct dimension
        phase_factor_array = np.zeros(self.dimension, dtype=complex)
        
        if self.previous_eigenvectors is not None:
            for i in range(len(new_eigenvectors)):
                dot_product = np.vdot(self.previous_eigenvectors[i], new_eigenvectors[i])
                phase_diff = np.angle(dot_product)
                
                if ignore_small_phase_diff and abs(phase_diff) < phase_diff_threshold:
                    # Ignore small phase differences if the option is set
                    phase_factor = 1.0  # No correction applied
                else:
                    phase_factor = np.exp(-1j * phase_diff)
                
                phase_factor_array[i] = phase_factor
                new_eigenvectors[i] = new_eigenvectors[i] * phase_factor
        else:
            # Sort by the real part of eigenvalues for the first set
            sorted_indices = np.argsort(-np.real(new_eigenvalues))
            new_eigenvectors = [new_eigenvectors[i] for i in sorted_indices]
            new_eigenvalues = [new_eigenvalues[i] for i in sorted_indices]
        
        self.previous_eigenvectors = new_eigenvectors
        self.previous_kx = kx
        self.previous_ky = ky
        self.phase_factor = phase_factor_array
        
        return new_eigenvectors


    # Eigenvector ordered
    def set_eigenvectors_eigenvector_ordered(self, new_eigenvectors, new_eigenvalues, kx, ky):
        # Initialize phase_factor_array with the correct dimension
        phase_factor_array = np.zeros(self.dimension, dtype=complex)
        
        if self.previous_eigenvectors is not None and self.previous_eigenvalues is not None:
            best_permutation = None
            min_phase_diff = np.inf
            
            # Check all permutations of the new eigenvectors and corresponding eigenvalues
            for perm in permutations(range(self.dimension)):
                total_phase_diff = 0

                for i in range(self.dimension):
                    previous_vector = self.previous_eigenvectors[i]
                    current_vector = new_eigenvectors[perm[i]]
                    dot_product = np.abs(np.vdot(previous_vector, current_vector))

                    # Calculate the phase difference
                    phase_diff = np.abs(1 - dot_product)
                    total_phase_diff += phase_diff
                
                # Update the best permutation if this one is better
                if total_phase_diff < min_phase_diff:
                    min_phase_diff = total_phase_diff
                    best_permutation = perm
            
            # Reorder the new eigenvectors and eigenvalues according to the best permutation
            new_eigenvectors = [new_eigenvectors[i] for i in best_permutation]
            new_eigenvalues = [new_eigenvalues[i] for i in best_permutation]
            phase_factor_array = np.array([np.vdot(self.previous_eigenvectors[i], new_eigenvectors[i]) for i in range(self.dimension)], dtype=complex)

            # Check for the correct sign alignment of the eigenvalues
            dot_eigenvalues = np.real(np.vdot(self.previous_eigenvalues, new_eigenvalues))
            if dot_eigenvalues < 0:
                new_eigenvectors = [-v for v in new_eigenvectors]
                new_eigenvalues = [-v for v in new_eigenvalues]

        else:
            # Sort by the real part of eigenvalues for the first set
            sorted_indices = np.argsort(-np.real(new_eigenvalues))
            new_eigenvectors = [new_eigenvectors[i] for i in sorted_indices]
            new_eigenvalues = [new_eigenvalues[i] for i in sorted_indices]

        self.previous_eigenvectors = new_eigenvectors
        self.previous_eigenvalues = new_eigenvalues
        self.previous_kx = kx
        self.previous_ky = ky
        self.phase_factor = phase_factor_array
        return new_eigenvalues, new_eigenvectors
    

    
    def set_eigenvectors_eigenvector_reordered(self, new_eigenvectors, new_eigenvalues, kx, ky):
        """
        Reorder only the third and fourth eigenvectors (and corresponding eigenvalues) based on the phase continuity condition.
        """
        if self.previous_eigenvectors is not None:
            # Extract relevant eigenvectors (3rd and 4th)
            previous_vectors = [self.previous_eigenvectors[0], self.previous_eigenvectors[1]]
            current_vectors = [new_eigenvectors[0], new_eigenvectors[1]]

            # Compute phase differences for the two possible reorderings
            phase_diff_33_44 = np.abs(1 - np.abs(np.vdot(previous_vectors[0], current_vectors[0]))) + \
                            np.abs(1 - np.abs(np.vdot(previous_vectors[1], current_vectors[1])))
            
            phase_diff_34_43 = np.abs(1 - np.abs(np.vdot(previous_vectors[0], current_vectors[1]))) + \
                            np.abs(1 - np.abs(np.vdot(previous_vectors[1], current_vectors[0])))
            if phase_diff_34_43 - phase_diff_33_44 < 1: 
                print(phase_diff_34_43 - phase_diff_33_44)
            # Choose the best reordering
            if phase_diff_34_43 < phase_diff_33_44:
                print("hello")
                # Swap the 3rd and 4th eigenvectors and their corresponding eigenvalues if it results in a smaller phase difference
                new_eigenvectors[0], new_eigenvectors[1] = new_eigenvectors[1], new_eigenvectors[0]
                new_eigenvalues[0], new_eigenvalues[1] = new_eigenvalues[1], new_eigenvalues[0]

        # Update previous eigenvectors and eigenvalues for the next iteration
        self.previous_eigenvectors = new_eigenvectors
        self.previous_eigenvalues = new_eigenvalues
        self.previous_kx = kx
        self.previous_ky = ky
        
        return new_eigenvectors, new_eigenvalues


    def get_phase_factors(self):
        return self.phase_factor

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
    H_k = Hamiltonian(kx, ky)
    
    eigenvalues, eigenvectors = get_eigenvalues_and_eigenvectors(H_k)

    # Set the new eigenvectors with phase correction
    eigenvalues, eigenvectors = eigenvector.set_eigenvectors_ordered(eigenvectors, eigenvalues, kx, ky)
    
    return eigenvalues, eigenvectors

def eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky, eigenvector: 'Eigenvectors' =None, zone_num=None):
    """
    Calculate eigenvalues and eigenvectors, with optional reordering based on the zone number.
    
    If the zone number is odd, the order between the 3rd and 4th eigenvalues and eigenvectors is reversed.

    Parameters:
    - Hamiltonian: A function that returns the Hamiltonian matrix given kx and ky.
    - kx: Just a number. The kx value for which to calculate the eigenvalues and eigenvectors.
    - ky: Just a number. The ky value for which to calculate the eigenvalues and eigenvectors.
    - eigenvector: An optional Eigenvector object for phase correction.
    - zone_num: An optional integer specifying the zone number. If odd, the 3rd and 4th eigenvalues/eigenvectors are swapped.

    Returns:
    - eigenvalues: The sorted eigenvalues.
    - eigenvectors: The sorted and possibly reordered eigenvectors.
    """
    # H_k = Hamiltonian(kx, ky)
    H_k = Hamiltonian.effective_hamiltonian(kx, ky)
    
    eigenvalues, eigenvectors = get_eigenvalues_and_eigenvectors(H_k)

    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[idx, :]

    # Check if zone_num is odd
    if zone_num is not None and zone_num % 2 == 1:
        # Swap the 3rd and 4th eigenvalues and eigenvectors
        eigenvalues[[2, 3]] = eigenvalues[[3, 2]]
        eigenvectors[[2, 3], :] = eigenvectors[[3, 2], :]
    
    # Set the new eigenvectors with phase correction
    if eigenvector is not None:
        eigenvectors = eigenvector.set_eigenvectors_eigenvalue_preordered(eigenvectors, eigenvalues, kx, ky, ignore_small_phase_diff=False)
    
    return eigenvalues, eigenvectors



# & Calculations in an anglee line

def line_eigenvalues_eigenfunctions(Hamiltonian, k_line, k_angle, dim):
    """
    Calculate eigenvalues and eigenvectors along a line in the kx-ky plane.

    Parameters:
    - Hamiltonian: Function to compute the Hamiltonian matrix.
    - k_line: 1D array of k-values along the line.
    - k_angle: The angle of the line in radians.
    - dim: The dimension of the system.

    Returns:
    - eigenvalues: 2D array of eigenvalues, shape (num_points, dim).
    - eigenfunctions: 3D array of eigenfunctions, shape (num_points, dim, dim).
    - phase_factors_array: 2D array of phase factors, shape (num_points, dim).
    """
    # Define kx and ky along the line
    line_kx = k_line * np.cos(k_angle)
    line_ky = k_line * np.sin(k_angle)
    
    # Initialize arrays to store results
    num_points = len(k_line)
    eigenvalues = np.zeros((num_points, dim), dtype=float)
    eigenfunctions = np.zeros((num_points, dim, dim), dtype=complex)
    phase_factors_array = np.zeros((num_points, dim), dtype=float)

    eigenvector = Eigenvectors(dim)

    # Loop through each point along the line
    for i, (kx, ky) in enumerate(zip(line_kx, line_ky)):
        vals, vecs = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky, eigenvector)
        phase_factors = eigenvector.get_phase_factors()
        
        eigenvalues[i] = vals
        eigenfunctions[i] = vecs
        phase_factors_array[i] = phase_factors

    return eigenvalues, eigenfunctions, phase_factors_array


# & Calculations in a normal grid

def grid_eigenvalues_eigenfunctions(Hamiltonian, kx, ky, mesh_spacing, dim):
    # Initialize arrays to store eigenfunctions and eigenvalues
    eigenfunctions = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)
    eigenvalues = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)
    phase_factors_array = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)

    eigenvector = Eigenvectors(dim)

    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            vals, vecs = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx[i, j], ky[i, j], eigenvector)
            phase_factors = eigenvector.get_phase_factors()
            
            eigenfunctions[i, j] = vecs
            eigenvalues[i, j] = vals
            phase_factors_array[i, j] = phase_factors
            
    return eigenvalues, eigenfunctions

def spiral_eigenvalues_eigenfunctions(Hamiltonian, kx, ky, mesh_spacing, dim, phase_correction = True):
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



def phase_corrected_spiral_eigenvalues_eigenfunctions(Hamiltonian, kx, ky, mesh_spacing, dim):
    # Initialize arrays to store eigenfunctions and eigenvalues
    eigenfunctions = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)
    eigenvalues = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)
    phase_factors_array = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)

    spiral_indices = get_spiral_indices(mesh_spacing)

    eigenvector = Eigenvectors(dim)

    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            k,l = spiral_indices[i,j]
            # ! This lines is important
            # vals, vecs = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx[k, l], ky[k, l], eigenvector=eigenvector)
            vals, vecs = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx[k, l], ky[k, l], eigenvector=None)
            phase_factors = eigenvector.get_phase_factors()
            
            eigenfunctions[k, l] = vecs
            eigenvalues[k, l] = vals
            phase_factors_array[k, l] = phase_factors
            
    neighbor_phase_array_after_calc = calculate_neighbor_phase_array(eigenfunctions, mesh_spacing, dim)

    #Identify continuous regions in the eigenfunctions for a specific dimension (band)
    regions, num_regions = identify_regions(eigenfunctions, mesh_spacing, dim=2, phase_threshold=0.07)  # dim=0 as an example

    # Adjust phases within each region to ensure internal consistency
    eigenfunctions = adjust_region_phases(eigenfunctions, regions, num_regions, dim=2)  # Correct for dim=0 as an example


    def plot_regions(regions, mesh_spacing):
        """
        Plots the identified continuous regions in the kx, ky space with different colors.
        
        Parameters:
        - regions: 2D array of region indices.
        - mesh_spacing: The size of the mesh (number of points along kx and ky directions).
        """
        # Create a plot
        plt.figure(figsize=(6, 6))
        
        # Use imshow to plot the regions, using a colormap with enough colors for all regions
        plt.imshow(regions, cmap='tab20', extent=[0, mesh_spacing, 0, mesh_spacing])
        
        # Add a color bar to show which color corresponds to which region
        plt.colorbar(label='Region Index')
        
        # Add plot titles and labels
        plt.title('Identified Continuous Regions in kx, ky Space')
        plt.xlabel('kx Index')
        plt.ylabel('ky Index')
        
        # Remove grid lines
        plt.grid(False)
        
        # Show the plot
        plt.show()
    
    # plot_regions(regions, mesh_spacing)

    # Align the phases between different regions
    eigenfunctions = align_regions(eigenfunctions, regions, num_regions, dim=2)  # Align regions for dim=0

    # Apply recursive phase correction
    # Example of calling recursive phase correction for dimension d = 2
    # recursive_phase_correction(eigenfunctions, neighbor_phase_array_after_calc, mesh_spacing, d=3, threshold=0.4)


    return eigenvalues, eigenfunctions, phase_factors_array, neighbor_phase_array_after_calc


# * Calculations in a zone

def eigenvalues_eigenfunctions_in_zone(Hamiltonian, kx, ky, mesh_spacing, dim, Zone, zone_num, touching_points):
    # Initialize arrays to store eigenfunctions and eigenvalues
    eigenfunctions = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)
    eigenvalues = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)
    phase_factors_array = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)

    eigenvector = Eigenvectors(dim)

    # Find the reference point in the zone from touching_points
    reference_point = touching_points[zone_num][0] if zone_num in touching_points else (0, 0)

    # Create a mask for the specified zone
    mask = Zone.create_mask_for_zone(zone_num)

    # Order the grid points in the zone by distance from the reference point using the mask
    ordered_grid_points, ordered2d = unordered_grid_masked(kx, ky, mask)
    
    plot_ordered_grid_2d(kx, ky, ordered2d)
    
    # plot_ordered_grid_3d(kx, ky, ordered_grid_points, reference_point)

    iteration = 0
    for i, j in ordered_grid_points:
        iteration += 1
        # Perform calculations only for points within the specified zone
        vals, vecs = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx[i, j], ky[i, j], eigenvector)
        phase_factors = eigenvector.get_phase_factors()

        eigenfunctions[i, j] = vecs
        eigenvalues[i, j] = vals
        phase_factors_array[i, j] = phase_factors

    ordered_grid_points, disntances2d, ordered2d = order_grid_points_masked(kx, ky, (0,0), mask)
    # ordered_grid_points, disntances2d, ordered2d = order_grid_points_masked(kx, ky, reference_point, mask)

    # # Second pass: Correct the phase of the already calculated eigenvectors and eigenvalues
    # eigenvector_second = Eigenvectors(dim)  # Reinitialize the Eigenvector object for phase correction


    # for i, j in ordered_grid_points:
    #     vecs = eigenfunctions[i, j]
    #     vals = eigenvalues[i, j]

    #     # Use set_eigenvectors to correct the phase of the eigenvectors
    #     _, corrected_vecs = eigenvector_second.set_eigenvectors_eigenvector_ordered(vecs, vals, kx[i, j], ky[i, j])
    #     phase_factors = eigenvector_second.get_phase_factors()

    #     # Update the eigenfunctions with the corrected eigenvectors
    #     eigenfunctions[i, j] = corrected_vecs
    #     phase_factors_array[i, j] = phase_factors

    neighbor_phase_array_after_calc = calculate_neighbor_phase_array(eigenfunctions, mesh_spacing, dim)

    # # Identify and print the first (i, j) with a non-zero neighbor phase
    # first_non_zero_found = False
    # for (i, j) in ordered_grid_points:
    #     # Check if any of the phase factors for this point are non-zero (within a small tolerance)
    #     if np.any(np.abs(neighbor_phase_array_after_calc[i, j, :]) > 1e-12):
    #         print(f"The first (i, j) with a non-zero neighbor phase is: ({i}, {j})")
    #         first_non_zero_found = True
    #         # break

    # if not first_non_zero_found:
    #     print("No (i, j) with a non-zero neighbor phase was found in the ordered grid points.")

    return eigenvalues, eigenfunctions, phase_factors_array, neighbor_phase_array_after_calc


def eigenvalues_eigenfunctions_in_connected_zone(Hamiltonian, kx, ky, mesh_spacing, dim, Zone, zone_num, eigenvector, smallest_angle_point, largest_angle_point):
    # * Trying to make a function so that the phase of different zones are connected
    # Initialize arrays to store eigenfunctions and eigenvalues
    eigenfunctions = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)
    eigenvalues = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)
    phase_factors_array = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)



    # Create a mask for the specified zone
    mask = Zone.create_mask_for_zone(zone_num)

    # Order the grid points in the zone by distance from the reference point using the mask
    ordered_grid_points, ordered2d = unordered_grid_masked(kx, ky, mask)

    for i, j in ordered_grid_points:
        # Perform calculations only for points within the specified zone
        vals, vecs = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx[i, j], ky[i, j], eigenvector, zone_num=None)
        eigenfunctions[i, j] = vecs
        eigenvalues[i, j] = vals

    ordered_grid_points, disntances2d, ordered2d = order_grid_points_start_end(kx, ky, smallest_angle_point, largest_angle_point, mask)

    plot_ordered_grid_2d(kx, ky, ordered2d)
    
    # plot_ordered_grid_3d(kx, ky, ordered_grid_points, reference_point)


    # Second pass: Correct the phase of the already calculated eigenvectors and eigenvalues

    for i, j in ordered_grid_points:
        vecs = eigenfunctions[i, j]
        vals = eigenvalues[i, j]

        # Use set_eigenvectors to correct the phase of the eigenvectors
        corrected_vecs = eigenvector.set_eigenvectors(vecs, vals, kx[i, j], ky[i, j])
        phase_factors = eigenvector.get_phase_factors()

        # Update the eigenfunctions with the corrected eigenvectors
        eigenfunctions[i, j] = corrected_vecs
        phase_factors_array[i, j] = phase_factors

    neighbor_phase_array_after_calc = calculate_neighbor_phase_array(eigenfunctions, mesh_spacing, dim)

    return eigenvalues, eigenfunctions, phase_factors_array, neighbor_phase_array_after_calc




def eigenvalues_eigenfunctions_in_zone_eigenvector_ordering(Hamiltonian, kx, ky, mesh_spacing, dim, Zone, zone_num, touching_points):
    # Initialize arrays to store eigenfunctions and eigenvalues
    eigenfunctions = np.zeros((mesh_spacing, mesh_spacing, dim, dim), dtype=complex)
    eigenvalues = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)
    phase_factors_array = np.zeros((mesh_spacing, mesh_spacing, dim), dtype=float)

    eigenvector = Eigenvector(dim)

    # Find the reference point in the zone from touching_points
    reference_point = touching_points[zone_num][0] if zone_num in touching_points else (0, 0)

    # Create a mask for the specified zone
    mask = Zone.create_mask_for_zone(zone_num)

    # Order the grid points in the zone by distance from the reference point using the mask
    ordered_grid_points, disntances2d, ordered2d = order_grid_points_masked(kx, ky, reference_point, mask)

    for i, j in ordered_grid_points:
        # Perform calculations only for points within the specified zone
        vals, vecs = eigenvalues_and_vectors_eigenvector_ordering(Hamiltonian, kx[i, j], ky[i, j], eigenvector)
        phase_factors = eigenvector.get_phase_factors()


        # # Check if the eigenvalues and eigenvectors are correct
        # if not check_eigen_solution(Hamiltonian, kx[i, j], ky[i, j], vals, vecs):
        #     print(f"Warning: Invalid eigenvalue-eigenvector pair found at (kx, ky) = ({kx[i, j]:.4f}, {ky[i, j]:.4f})")
        
        
        eigenfunctions[i, j] = vecs
        eigenvalues[i, j] = vals
        phase_factors_array[i, j] = phase_factors

    # * It turns out that this is never needed singe the permutations should have taken this into account
    # Second loop: Reorder the third and fourth eigenvectors and corresponding eigenvalues for continuity
    # reorder_eigenvector = Eigenvector(2)  # Object for reordering

    # for i, j in ordered_grid_points:
    #     # Extract the current third and fourth eigenvectors and eigenvalues
    #     third_vec, fourth_vec = eigenfunctions[i, j, 2], eigenfunctions[i, j, 3]
    #     third_val, fourth_val = eigenvalues[i, j, 2], eigenvalues[i, j, 3]

    #     # Reorder only the third and fourth eigenvectors and eigenvalues
    #     reordered_vectors, reordered_values = reorder_eigenvector.set_eigenvectors_reordered(
    #         [third_vec, fourth_vec], [third_val, fourth_val], kx[i, j], ky[i, j]
    #     )

    #     # Update the eigenfunctions array with the reordered eigenvectors
    #     eigenfunctions[i, j, 2], eigenfunctions[i, j, 3] = reordered_vectors[0], reordered_vectors[1]

    #     # Update the eigenvalues array with the reordered eigenvalues
    #     eigenvalues[i, j, 2], eigenvalues[i, j, 3] = reordered_values[0], reordered_values[1]

    return eigenvalues, eigenfunctions, phase_factors_array


# * Miscellaenous 

def capping_eigenvalues(eigenvalues, z_limit):
    eigenvalues[eigenvalues > z_limit] = z_limit
    eigenvalues[eigenvalues < -z_limit] = -z_limit
    return eigenvalues