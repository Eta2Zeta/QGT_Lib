import numpy as np
from .Hamiltonian import hamiltonian
from .ChiralHamiltonian import ChiralHamiltonian

class ChiralHamiltonianSWProjected(hamiltonian):
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

    def _project_2x2(self, H0, Hprime, order=2):
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
        H_AA = H0[np.ix_(idx_L, idx_L)]
        H_AB = Hprime[np.ix_(idx_L, idx_H)]
        H_BA = Hprime[np.ix_(idx_H, idx_L)]
        H_BB = H0[np.ix_(idx_H, idx_H)]
        H_prime_BB = Hprime[np.ix_(idx_H, idx_H)]
        
        
        try:
            H_BB_inv = np.linalg.inv(H_BB)
            H_eff = H_AA
            
            if order >= 2:
                # Second-order correction: - H_LH @ H_HH^(-1) @ H_HL
                second_order_correction = - (H_AB @ H_BB_inv @ H_BA)
                H_eff = H_eff + second_order_correction
            
            if order >= 3:
                # Third-order correction: + H_LH @ H_HH^(-1) @ H_prime_HH @ H_HH^(-1) @ H_HL
                third_order_correction = H_AB @ H_BB_inv @ H_prime_BB @ H_BB_inv @ H_BA
                H_eff = H_eff + third_order_correction
            
            if order >= 4:
                # Fourth-order correction: - H_LH @ H_HH^(-1) @ H_prime_HH @ H_HH^(-1) @ H_prime_HH @ H_HH^(-1) @ H_HL
                fourth_order_correction = - (H_AB @ H_BB_inv @ H_prime_BB @ H_BB_inv @ H_prime_BB @ H_BB_inv @ H_BA)
                H_eff = H_eff + fourth_order_correction
                
        except np.linalg.LinAlgError:
            # Fallback if H_HH is singular (should not happen for gapped high-energy bands usually)
            # Retain H_LL or return NaNs
            H_eff = np.full((2, 2), np.nan, dtype=complex)
            
        return H_eff

    def _project_2x2_test(self, H_full, order=2):
        """
        Original Schrieffer-Wolff partitioning using the full Hamiltonian (H_full)
        instead of explicitly separated H0 and Hprime blocks.
        
        Evaluates 2nd and 4th order SWT wavefunction renormalization corrections:
        2nd order: -H_AB * H_BB^(-1) * H_BA
        4th order: -0.5 * [ (H_AB * H_BB^(-1) * H_BA) * (H_AB * H_BB^(-2) * H_BA) 
                           + (H_AB * H_BB^(-2) * H_BA) * (H_AB * H_BB^(-1) * H_BA) ]
        """
        # Low energy sites: A0 (index 0) and B_{n-1} (index 2n-1)
        idx_L = [0, self.full_dim - 1]
        
        # High energy sites: All others
        idx_H = [i for i in range(self.full_dim) if i not in idx_L]
        
        # Partition the full matrix
        H_AA = H_full[np.ix_(idx_L, idx_L)]
        H_AB = H_full[np.ix_(idx_L, idx_H)]   # H_AB
        H_BA = H_full[np.ix_(idx_H, idx_L)]   # H_BA
        H_BB = H_full[np.ix_(idx_H, idx_H)]   # H_BB
        
        try:
            H_HH_inv = np.linalg.inv(H_BB)
            H_eff = H_AA
            
            if order >= 2:
                # 2nd order: -H_AB * H_BB^(-1) * H_BA
                second_order = - (H_AB @ H_HH_inv @ H_BA)
                H_eff = H_eff + second_order
                
            if order >= 4:
                H_HH_inv2 = H_HH_inv @ H_HH_inv
                # Canonical 4th order SWT wavefunction renormalization term:
                # -1/2 { H_eff^(2), H_AB H_BB^-2 H_BA }
                term_inv1 = -(H_AB @ H_HH_inv @ H_BA)
                term_inv2 = (H_AB @ H_HH_inv2 @ H_BA)
                
                fourth_order = -0.5 * (term_inv1 @ term_inv2 + term_inv2 @ term_inv1)
                
                H_eff = H_eff + fourth_order

        except np.linalg.LinAlgError:
            H_eff = np.full((2, 2), np.nan, dtype=complex)
            
        return H_eff

    def compute_static(self, kx, ky, kz=0, order=2):
        """
        Computes the effective 2x2 Hamiltonian at (kx, ky) from the static full Hamiltonian.
        """
        # # 1. Compute H0 and Hprime
        # H0 = self.full_hamiltonian.H0()
        # Hprime = self.full_hamiltonian.Hprime(kx, ky)
        # return self._project_2x2(H0, Hprime)
        
        # 1. Compute H_full
        H_full = self.full_hamiltonian.compute_static(kx, ky, kz)
        return self._project_2x2_test(H_full)

    def compute_static_analytic(self, kx, ky, kz=0, order=3):
        """
        Analytic effective 2x2 Hamiltonian via Schrieffer-Wolff.

        Zeroth order (H_AA):
            H_AA = diag(-V*(n-1)/2, +V*(n-1)/2)

        First-order correction (1/2) * [S1, V_od]_AA:
            [[  beta^2 * V,   beta^2 * V ],
             [ -beta^2 * V,  -beta^2 * V ]]
            multiplied by 1/2, where beta = vF * |k| / t1.
        """
        fh = self.full_hamiltonian
        n  = fh.n
        vF = fh.vF
        t1 = fh.t1
        V  = fh.V

        # |k|
        k = np.sqrt(kx**2 + ky**2)

        # beta = vF * |k| / t1
        beta2 = (vF * k / t1) ** 2

        # --- Zeroth order ---
        H_eff = np.array([
            [- V * (n - 1) / 2,  0],
            [0,                 + V * (n - 1) / 2]
        ], dtype=complex)

        if order >= 1:
            # --- First-order correction: (1/2) * [S1, V_od]_AA ---
            H_first = 0.5 * beta2 * V * np.array([
                [ 1,  0],
                [0, -1]
            ], dtype=complex)
            H_eff = H_eff + H_first

        if order >= 2:
            numerator   = 2 * V * (4 * V**4 + 2 * V**2 * t1**2 + t1**4)
            denominator = 3 * t1**6 * (t1**2 - 2 * V**2)
            H_second = (numerator / denominator) * (vF * k) ** 4 * np.array([
                [ 1,  0],
                [0, -1]
            ], dtype=complex)
            H_eff = H_eff + H_second

        if order >= 3:
            # --- Third-order correction ---
            H_third = (V * (V**2 + t1**2) / (3 * t1**6)) * (vF * k) ** 4 * np.array([
                [1,  0],
                [ 0,  -1]
            ], dtype=complex)
            H_eff = H_eff + H_third

        return H_eff

    def compute_effective_hamiltonian(self, kx, ky, kz=0, order=2):
        """
        Computes the effective 2x2 Hamiltonian at (kx, ky) from the DRIVEN full Hamiltonian.
        This includes the Magnus correction (if A0 != 0 and magnus_order >= 1).
        
        Returns:
            H_eff_2x2 (ndarray): The projected 2x2 effective Hamiltonian.
        """
        # # 1. Compute the Magnus perturbation (from the driven Hamiltonian)
        # # Note: effective_hamiltonian returns (H_eff, H_magnus), where the second term is the sum of Magnus corrections
        # _, H_magnus = self.full_hamiltonian.effective_hamiltonian(kx, ky, kz)
        
        # # 2. Compute static H0 and static Hprime
        # H0 = self.full_hamiltonian.H0() + H_magnus
        # Hprime = self.full_hamiltonian.Hprime(kx, ky) 
        
        # # 3. Project down to 2x2 using Schrieffer-Wolff
        # return self._project_2x2(H0, Hprime)

        # 1. Compute the full effective Hamiltonian (Static + Magnus)
        H_full_eff, _ = self.full_hamiltonian.effective_hamiltonian(kx, ky, kz)
        
        # 2. Project down to 2x2 using the test logic
        return self._project_2x2_test(H_full_eff)
