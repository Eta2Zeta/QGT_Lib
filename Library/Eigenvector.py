import numpy as np
from itertools import permutations
class Eigenvector:
    def __init__(self, dimension):
        self.dimension = dimension
        self.previous_eigenvector = None
        self.previous_kx = None
        self.previous_ky = None
        self.previous_kz = None
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


def _find_degenerate_blocks(evals, gap_tol):
    """
    evals: 1D array/list of eigenvalues already in the *current ordering*.
    Returns a list of blocks, each block is a list of indices.
    """
    evals = np.asarray(evals)
    n = len(evals)
    blocks = []
    start = 0
    while start < n:
        end = start
        while end + 1 < n and abs(evals[end + 1] - evals[end]) < gap_tol:
            end += 1
        blocks.append(list(range(start, end + 1)))
        start = end + 1
    return blocks

def _align_block_svd(prev_vecs, new_vecs, block_inds, eps=1e-12):
    """
    prev_vecs, new_vecs: lists of eigenvectors; each eigenvector shape (dim,)
    block_inds: list of indices belonging to a degenerate/near-degenerate block
    Returns: new_vecs with that block rotated to best match prev_vecs.
    """
    # Build U0, U1 with columns = vectors in the block
    U0 = np.column_stack([prev_vecs[i] for i in block_inds])  # (dim, m)
    U1 = np.column_stack([new_vecs[i]  for i in block_inds])  # (dim, m)

    # Overlap matrix
    M = U0.conj().T @ U1  # (m, m)

    # SVD and polar unitary
    V, s, Wh = np.linalg.svd(M)
    R = V @ Wh  # closest unitary to M

    # Rotate new block: U1 <- U1 R^\dagger
    U1_aligned = U1 @ R.conj().T

    # Write back
    out = list(new_vecs)
    for col, idx in enumerate(block_inds):
        v = U1_aligned[:, col]
        nrm = np.linalg.norm(v)
        if nrm > eps:
            v = v / nrm
        out[idx] = v
    return out


class Eigenvectors:
    def __init__(self, dimension):
        self.dimension = dimension
        self.previous_eigenvectors = None
        self.previous_eigenvalues = None
        self.previous_kx = None
        self.previous_ky = None
        self.previous_kz = None
        self.phase_factor = None

    def set_dimension(self, dim):
        self.dimension = dim

    # If the solutions are already ordered by eigenvalues, you can use this to just correct the phase factor of the eigenvalues
    def set_eigenvectors_eigenvalue_preordered(self, new_eigenvectors, new_eigenvalues, kx, ky, kz=0, ignore_small_phase_diff=False, phase_diff_threshold=0.2):
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
            sorted_indices = np.argsort(np.real(new_eigenvalues))
            new_eigenvectors = [new_eigenvectors[i] for i in sorted_indices]
            new_eigenvalues = [new_eigenvalues[i] for i in sorted_indices]
        
        self.previous_eigenvectors = new_eigenvectors
        self.previous_kx = kx
        self.previous_ky = ky
        self.previous_kz = kz
        self.phase_factor = phase_factor_array
        
        return new_eigenvectors


    # Eigenvector ordered
    def set_eigenvectors_eigenvector_ordered(self, new_eigenvectors, new_eigenvalues, kx, ky, kz=0):
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
            sorted_indices = np.argsort(np.real(new_eigenvalues))
            new_eigenvectors = [new_eigenvectors[i] for i in sorted_indices]
            new_eigenvalues = [new_eigenvalues[i] for i in sorted_indices]

        self.previous_eigenvectors = new_eigenvectors
        self.previous_eigenvalues = new_eigenvalues
        self.previous_kx = kx
        self.previous_ky = ky
        self.previous_kz = kz
        self.phase_factor = phase_factor_array
        return new_eigenvalues, new_eigenvectors

    def set_eigenvectors_phase_corrected(self, new_eigenvectors, new_eigenvalues, kx, ky, kz=0, gap_tol=1e-6):
        """
        Robust ordering / gauge-fixing for (near-)degenerate bands:
        - Coarse sort by eigenvalue (real part)
        - Detect near-degenerate contiguous blocks by gap_tol
        - Do block SVD alignment to previous eigenvectors in each block (U(m) gauge fixing)
        """
        # --- Coarse sort by eigenvalue (keeps continuity away from degeneracies)
        idx = np.argsort(np.real(new_eigenvalues))
        new_eigenvectors = [new_eigenvectors[i] for i in idx]
        new_eigenvalues  = [new_eigenvalues[i]  for i in idx]

        # First call: just store and return
        if self.previous_eigenvectors is None:
            self.previous_eigenvectors = new_eigenvectors
            self.previous_eigenvalues  = new_eigenvalues
            self.previous_kx, self.previous_ky, self.previous_kz = kx, ky, kz
            self.phase_factor = np.ones(self.dimension, dtype=complex)
            return new_eigenvalues, new_eigenvectors

        # --- Align blocks
        blocks = _find_degenerate_blocks(new_eigenvalues, gap_tol=gap_tol)

        aligned = list(new_eigenvectors)

        for block in blocks:
            if len(block) == 1:
                i = block[0]
                dot = np.vdot(self.previous_eigenvectors[i], aligned[i])
                if abs(dot) > 0:
                    phase = np.exp(-1j * np.angle(dot))
                    aligned[i] = aligned[i] * phase
            else:
                aligned = _align_block_svd(self.previous_eigenvectors, aligned, block)

        # phase factors (optional bookkeeping)
        phase_factor_array = np.array(
            [np.vdot(self.previous_eigenvectors[i], aligned[i]) for i in range(self.dimension)],
            dtype=complex
        )

        # Update state
        self.previous_eigenvectors = aligned
        self.previous_eigenvalues  = new_eigenvalues
        self.previous_kx, self.previous_ky, self.previous_kz = kx, ky, kz
        self.phase_factor = phase_factor_array

        return new_eigenvalues, aligned

    


    def get_phase_factors(self):
        return self.phase_factor
