import numpy as np
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
