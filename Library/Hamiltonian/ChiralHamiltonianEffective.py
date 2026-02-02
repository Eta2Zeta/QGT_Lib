import numpy as np
from .Hamiltonian_v2 import hamiltonian
from .ChiralHamiltonian import ChiralHamiltonian

class ChiralHamiltonianEffective(hamiltonian):
    """
    Effective 2x2 Hamiltonian for n-layer rhombohedral graphene obtained by 
    downfolding the full 2n x 2n Chiral Hamiltonian using the Schrieffer-Wolff 
    transformation (perturbation theory to second order with epsilon=0).
    
    H_eff = H_LL - H_LH @ (H_HH)^(-1) @ H_HL
    
    where L denotes the low-energy subspace (A0, B_{n-1}) and H the high-energy subspace.
    """
    def __init__(self, n=2, vF=542.1, t1=355.16, V=0.0,
                 omega=2*np.pi, A0=0, **kwargs):
        """
        Initializes the effective Hamiltonian calculator.
        Args:
           n, vF, t1, V: Chiral model parameters.
           omega, A0: Drive parameters.
           **kwargs: Additional parameters passed to ChiralHamiltonian (e.g. magnus_order).
        """
        # The effective Hamiltonian is 2x2
        super().__init__(dim=2, omega=omega, A0=A0)
        
        # Create instance of the full Hamiltonian
        self.full_hamiltonian = ChiralHamiltonian(n=n, vF=vF, t1=t1, V=V,
                                                  omega=omega, A0=A0, **kwargs)
        self.n = n
        self.full_dim = 2 * n

    def _project_2x2(self, H_full):
        """
        Helper to project a 2n x 2n Hamiltonian to 2x2 effective Hamiltonian
        using Schrieffer-Wolff partitioning.
        """
        # 2. Define indices for Low (L) and High (H) subspaces
        # Low energy sites: A0 (index 0) and B_{n-1} (index 2n-1)
        idx_L = [0, self.full_dim - 1]
        
        # High energy sites: All others
        idx_H = [i for i in range(self.full_dim) if i not in idx_L]
        
        # 3. Partition the matrix
        # using numpy.ix_ to extract blocks
        H_LL = H_full[np.ix_(idx_L, idx_L)]
        H_LH = H_full[np.ix_(idx_L, idx_H)]
        H_HL = H_full[np.ix_(idx_H, idx_L)]
        H_HH = H_full[np.ix_(idx_H, idx_H)]
        
        # 4. Compute Effective Hamiltonian
        # H_eff = H_LL - H_LH @ (H_HH)^(-1) @ H_HL
        
        try:
            H_HH_inv = np.linalg.inv(H_HH)
            correction = H_LH @ H_HH_inv @ H_HL
            H_eff = H_LL - correction
        except np.linalg.LinAlgError:
            # Fallback if H_HH is singular (should not happen for gapped high-energy bands usually)
            # Retain H_LL or return NaNs
            H_eff = np.full((2, 2), np.nan, dtype=complex)
            
        return H_eff

    def compute_static(self, kx, ky, kz=0):
        """
        Computes the effective 2x2 Hamiltonian at (kx, ky) from the static full Hamiltonian.
        """
        # 1. Compute the full 2n x 2n Hamiltonian
        H_full = self.full_hamiltonian.compute_static(kx, ky, kz)
        return self._project_2x2(H_full)

    def compute_effective_hamiltonian(self, kx, ky, kz=0):
        """
        Computes the effective 2x2 Hamiltonian at (kx, ky) from the DRIVEN full Hamiltonian.
        This includes the Magnus correction (if A0 != 0 and magnus_order >= 1).
        
        Returns:
            H_eff_2x2 (ndarray): The projected 2x2 effective Hamiltonian.
        """
        # 1. Compute the full effective Hamiltonian (Static + Magnus)
        # Note: effective_hamiltonian returns (H_eff, H_prime), we only need H_eff
        H_full_eff, _ = self.full_hamiltonian.effective_hamiltonian(kx, ky, kz)
        
        # 2. Project down to 2x2 using the same logic
        return self._project_2x2(H_full_eff)
