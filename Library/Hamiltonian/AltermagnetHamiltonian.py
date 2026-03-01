import numpy as np
from .Hamiltonian_v2 import hamiltonian

# Define Pauli matrices
sigma_0 = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

class AltermagnetHamiltonian(hamiltonian):
    """
    Low-energy model of electrons on a Lieb lattice coupled to an altermagnetic order parameter.
    
    Hamiltonian:
    H = \\sum_{k} c_k^dagger H_k c_k
    
    H_k = H_k^0 + H_k^N
    
    H_k^0 = -4 t_1 cos(kx/2) cos(ky/2) tau_x 
            - 2 t_2 (cos(kx) + cos(ky)) tau_0
            - 2 t_d (cos(kx) - cos(ky)) tau_z
            + lambda sin(kx/2) sin(ky/2) tau_y sigma_z
            
    H_k^N = J tau_z N_z sigma_z
    
    Basis: 4 component spinor (sublattice x spin)
    tau matrices act on sublattice indices (1,2)
    sigma matrices act on spin indices (up, down)
    
    The tensor product convention used here is tau \\otimes sigma.
    So a term like tau_x is tau_x \\otimes sigma_0.
    term like tau_y sigma_z is tau_y \\otimes sigma_z.
    """
    
    def __init__(self, t1=1.0, t2=0.5, td=0.1, lamb=0.2, J=1.0, Nz=0.1, omega=0.0, A0=0.0, magnus_order=1):
        """
        Initialize the Altermagnet Hamiltonian.
        
        Parameters:
        - t1: Nearest-neighbor hopping
        - t2: Average next-nearest-neighbor hopping t2 = (t2a + t2b) / 2
        - td: Anisotropic NNN hopping difference td = (t2a - t2b) / 2
        - lamb: Spin-orbit coupling strength
        - J: Coupling strength to altermagnetic order
        - Nz: Altermagnetic order parameter (z-component)
        - omega: Driving frequency (for time-dependent base class)
        - A0: Driving amplitude (for time-dependent base class)
        """
        super().__init__(dim=4, omega=omega, A0=A0, magnus_order=magnus_order)
        self.t1 = t1
        self.t2 = t2
        self.td = td
        self.lamb = lamb
        self.J = J
        self.Nz = Nz
        
        # Precompute basis matrices (4x4)
        # tau_mu \otimes sigma_nu
        
        # tau_x \otimes sigma_0
        self.tau_x = np.kron(sigma_x, sigma_0)
        
        # tau_0 \otimes sigma_0
        self.tau_0 = np.kron(sigma_0, sigma_0)
        
        # tau_z \otimes sigma_0
        self.tau_z = np.kron(sigma_z, sigma_0)
        
        # tau_y \otimes sigma_z
        self.tau_y_sigma_z = np.kron(sigma_y, sigma_z)
        
        # tau_z \otimes sigma_z (for magnetic term)
        self.tau_z_sigma_z = np.kron(sigma_z, sigma_z)

    def compute_static(self, kx, ky, kz=0):
        """
        Compute the static Hamiltonian H_k at a single k point.
        """
        # Trigonometric terms
        cx2 = np.cos(kx / 2)
        cy2 = np.cos(ky / 2)
        sx2 = np.sin(kx / 2)
        sy2 = np.sin(ky / 2)
        
        cx = np.cos(kx)
        cy = np.cos(ky)
        
        # H_k^0 terms
        term1 = -4 * self.t1 * cx2 * cy2 * self.tau_x
        term2 = -2 * self.t2 * (cx + cy) * self.tau_0
        term3 = -2 * self.td * (cx - cy) * self.tau_z
        term4 = self.lamb * sx2 * sy2 * self.tau_y_sigma_z
        
        # H_k^N term
        magnetic_term = self.J * self.Nz * self.tau_z_sigma_z
        
        H = term1 + term2 + term3 + term4 + magnetic_term
        return H
        
    def compute_static_vectorized(self, kx_arr, ky_arr, kz_arr=0):
        """
        Vectorized computation of the Hamiltonian for arrays of k points.
        Returns array of shape (N, 4, 4).
        """
        kx_arr = np.asarray(kx_arr)
        ky_arr = np.asarray(ky_arr)
        
        # Ensure shapes match
        if kx_arr.shape != ky_arr.shape:
             # Broadcast if one is scalar or compatible shapes
             kx_arr, ky_arr = np.broadcast_arrays(kx_arr, ky_arr)
             
        # Flatten for calculation, then reshape at the end
        original_shape = kx_arr.shape
        kx = kx_arr.flatten()
        ky = ky_arr.flatten()
        M = kx.size
        
        # Trigonometric terms
        cx2 = np.cos(kx / 2)
        cy2 = np.cos(ky / 2)
        sx2 = np.sin(kx / 2)
        sy2 = np.sin(ky / 2)
        
        cx = np.cos(kx)
        cy = np.cos(ky)
        
        # Initialize H array (M, 4, 4)
        H = np.zeros((M, 4, 4), dtype=complex)
        
        # Add terms
        # -4 t_1 cos(kx/2) cos(ky/2) tau_x
        coeff_1 = -4 * self.t1 * cx2 * cy2
        H += coeff_1[:, None, None] * self.tau_x[None, :, :]
        
        # -2 t_2 (cos k_x + cos k_y) tau_0
        coeff_2 = -2 * self.t2 * (cx + cy)
        H += coeff_2[:, None, None] * self.tau_0[None, :, :]
        
        # -2 t_d (cos k_x - cos k_y) tau_z
        coeff_3 = -2 * self.td * (cx - cy)
        H += coeff_3[:, None, None] * self.tau_z[None, :, :]
        
        # lambda sin(kx/2) sin(ky/2) tau_y sigma_z
        coeff_4 = self.lamb * sx2 * sy2
        H += coeff_4[:, None, None] * self.tau_y_sigma_z[None, :, :]
        
        # H_k^N = J N_z tau_z sigma_z
        # This is constant in k, but needs to be added to every k-point
        coeff_N = self.J * self.Nz
        H += coeff_N * self.tau_z_sigma_z[None, :, :]
        
        # Reshape if input was not 1D
        return H
        
    def compute_qgt_analytic(self, kx_arr, ky_arr, band_index):
        """
        Compute the Quantum Geometric Tensor (QGT) analytically for the specified band.
        Uses the block-diagonal structure in sigma_z basis.
        
        Returns:
            g_xx, g_xy_real, g_xy_imag, g_yy, trace
        """
        kx = np.asarray(kx_arr)
        ky = np.asarray(ky_arr)
        
        # 1. Calculate d-vectors elements
        cx2 = np.cos(kx / 2)
        cy2 = np.cos(ky / 2)
        sx2 = np.sin(kx / 2)
        sy2 = np.sin(ky / 2)
        cx = np.cos(kx)
        cy = np.cos(ky)
        sx = np.sin(kx)
        sy = np.sin(ky)
        
        d0 = -2 * self.t2 * (cx + cy)
        dx = -4 * self.t1 * cx2 * cy2
        dz = -2 * self.td * (cx - cy)
        
        # dy depends on spin s = +/- 1
        # term: lambda sin(kx/2)sin(ky/2) + J Nz
        base_dy = self.lamb * sx2 * sy2 + self.J * self.Nz
        
        # 2. Derivatives
        # d_x derivatives
        # dx = -4 t1 cos(x/2)cos(y/2)
        ddx_dkx = 2 * self.t1 * sx2 * cy2
        ddx_dky = 2 * self.t1 * cx2 * sy2
        
        # d_z derivatives
        # dz = -2 td (cos x - cos y)
        ddz_dkx = 2 * self.td * sx
        ddz_dky = -2 * self.td * sy
        
        # base_dy derivatives
        # dy = lambda sin(x/2)sin(y/2) + const
        ddy_dkx_base = 0.5 * self.lamb * cx2 * sy2
        ddy_dky_base = 0.5 * self.lamb * sx2 * cy2
        
        # 3. Determine which spin block and which sign (+/-) corresponds to band_index
        # Energies: E = d0 +/- |d|
        # We need to evaluate energies to sort them and find which one matches band_index
        # Since this might vary across the BZ (crossings), we do it per point?
        # Vectorized approach: calculate all 4 energies, sort, and pick the mask.
        
        
        # Helper to compute gradients of normalized vector d_hat
        def compute_gradients_dhat(d_vec, dd_dkx, dd_dky):
            # d_vec: (3, N)
            # dd_dkx: (3, N)
            d_norm = np.linalg.norm(d_vec, axis=0) # (N,)
            d_norm_sq = d_norm**2
            d_norm_cubed = d_norm**3
            
            # d d_norm / dk = (d . dd/dk) / d_norm
            dnorm_dkx = np.sum(d_vec * dd_dkx, axis=0) / d_norm
            dnorm_dky = np.sum(d_vec * dd_dky, axis=0) / d_norm
            
            # d (d_vec / d_norm) / dk = (dd/dk * d_norm - d_vec * dnorm/dk) / d_norm^2
            dhat_dkx = (dd_dkx * d_norm - d_vec * dnorm_dkx) / d_norm_sq
            dhat_dky = (dd_dky * d_norm - d_vec * dnorm_dky) / d_norm_sq
            
            return dhat_dkx, dhat_dky, d_norm


        # --- Prepare Block Calculations ---
        # Helper to compute QGT components for a 2x2 block
        # Returns: (gxx, gyy, gxy_real, gxy_imag_plus, tr, d_norm)
        # Note: gxy_imag_plus is for the UPPER band (sign -), LOWER band has opposite sign
        def compute_block_geom(d_vec, dd_dkx, dd_dky):
            dhat_dkx, dhat_dky, d_norm = compute_gradients_dhat(d_vec, dd_dkx, dd_dky)
            
            # Quantum Metric (Symmetric part) - Same for both bands
            g_xx = 0.25 * np.sum(dhat_dkx * dhat_dkx, axis=0)
            g_yy = 0.25 * np.sum(dhat_dky * dhat_dky, axis=0)
            g_xy_real = 0.25 * np.sum(dhat_dkx * dhat_dky, axis=0)
            trace = g_xx + g_yy
            
            # Berry Curvature (Antisymmetric part)
            # Omega_+ = - (1/2) * d_hat . (d_hat_x x d_hat_y)
            cross_prod = np.cross(dhat_dkx, dhat_dky, axis=0)
            d_hat = d_vec / d_norm
            omega_plus = -0.5 * np.sum(d_hat * cross_prod, axis=0)
            
            # For QGT imag part: g_xy_imag = -0.5 * Omega
            g_xy_imag_plus = -0.5 * omega_plus
            
            return g_xx, g_xy_real, g_xy_imag_plus, g_yy, trace, d_norm

        # --- Block 1: s = +1 (Spin Up) ---
        dy_only = self.lamb * sx2 * sy2
        dz_kinetic = -2 * self.td * (cx - cy)
        d_mag = self.J * self.Nz
        
        # Block p (s=+1) => sigma_z = +1
        d_vec_p = np.array([dx, dy_only, dz_kinetic + d_mag])
        dd_dkx_p = np.array([ddx_dkx, ddy_dkx_base, ddz_dkx])
        dd_dky_p = np.array([ddx_dky, ddy_dky_base, ddz_dky])
        
        gxx_p, gxy_r_p, gxy_i_p_plus, gyy_p, tr_p, norm_p = compute_block_geom(d_vec_p, dd_dkx_p, dd_dky_p)
        
        # Block m (s=-1) => sigma_z = -1
        # d_x = dx, d_y = -dy_only, d_z = dz_kinetic - d_mag
        d_vec_m = np.array([dx, -dy_only, dz_kinetic - d_mag])
        dd_dkx_m = np.array([ddx_dkx, -ddy_dkx_base, ddz_dkx])
        dd_dky_m = np.array([ddx_dky, -ddy_dky_base, ddz_dky])
        
        gxx_m, gxy_r_m, gxy_i_m_plus, gyy_m, tr_m, norm_m = compute_block_geom(d_vec_m, dd_dkx_m, dd_dky_m)
        
        # --- Construct Energies and G Components ---
        E_p_up = d0 + norm_p
        E_p_down = d0 - norm_p
        
        E_m_up = d0 + norm_m
        E_m_down = d0 - norm_m
        
        # Construct result tuples for gather function
        # Format: (gxx, gxy_r, gxy_i, gyy, tr)
        
        g_p_up = (gxx_p, gxy_r_p, gxy_i_p_plus, gyy_p, tr_p)
        g_p_down = (gxx_p, gxy_r_p, -gxy_i_p_plus, gyy_p, tr_p)
        g_m_up = (gxx_m, gxy_r_m, gxy_i_m_plus, gyy_m, tr_m)
        g_m_down = (gxx_m, gxy_r_m, -gxy_i_m_plus, gyy_m, tr_m)
        
        # 4. Selection based on band_index
        all_E = np.stack([E_p_down, E_m_down, E_m_up, E_p_up], axis=0)
        
        # Get indices that would sort these energies (argsort along axis 0)
        sorted_indices = np.argsort(all_E, axis=0) # (4, ...)
        target_indices = sorted_indices[band_index] # (...) array of 0,1,2,3
        
        # Now we need to gather the QGT components from the corresponding blocks.
        # Arrays of G components: (4, ...)
        # 0: p_down, 1: m_down, 2: m_up, 3: p_up (arbitrary order matching all_E stack)
        
        def gather(arr0, arr1, arr2, arr3, idxs):
            # Selects elements from arr0..3 based on idxs
            # Use choose?
            return np.choose(idxs, [arr0, arr1, arr2, arr3])
        
        # Extract components from the tuples
        # g_p_up is (gxx, gxy_r, gxy_i, gyy, tr, norm)
        
        gxx = gather(g_p_down[0], g_m_down[0], g_m_up[0], g_p_up[0], target_indices)
        gxy_r = gather(g_p_down[1], g_m_down[1], g_m_up[1], g_p_up[1], target_indices)
        gxy_i = gather(g_p_down[2], g_m_down[2], g_m_up[2], g_p_up[2], target_indices)
        gyy = gather(g_p_down[3], g_m_down[3], g_m_up[3], g_p_up[3], target_indices)
        trace = gather(g_p_down[4], g_m_down[4], g_m_up[4], g_p_up[4], target_indices)
        
        # Reshape to original kx shape
        if kx_arr.ndim > 1:
            gxx = gxx.reshape(kx_arr.shape)
            gxy_r = gxy_r.reshape(kx_arr.shape)
            gxy_i = gxy_i.reshape(kx_arr.shape)
            gyy = gyy.reshape(kx_arr.shape)
            trace = trace.reshape(kx_arr.shape)
            
        return gxx, gxy_r, gxy_i, gyy, trace

