import numpy as np
from .Hamiltonian import hamiltonian

# Define Pauli matrices
sigma_0 = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

class RuO2Hamiltonian(hamiltonian):
    """
    Hamiltonian for RuO2 altermagnet (3D).
    
    H = epsilon_{0, k} + t_{x, k} tau_x + t_{z, k} tau_z + tau_y (vec{lambda}_k . vec{sigma}) + tau_z (vec{J} . vec{sigma})
    
    Default parameters for RuO2:
    t1 = -0.05, t2 = 0.7, t3 = 0.5, t4 = -0.15, t5 = -0.4, t6 = -0.6, t7 = 0.3, t8 = 1.7
    mu = 0.25
    J = 0.2
    lambda = 0.1
    lambda_z = 0.1
    """
    def __init__(self, t1=-0.05, t2=0.7, t3=0.5, t4=-0.15, t5=-0.4, t6=-0.6, t7=0.3, t8=1.7,
                 mu=0.25, Jx=0.2, Jy=0, Jz=0, lamb=0.1, lamb_z=0.1, omega=0.0, A0=0.0, magnus_order=1):
        super().__init__(dim=4, omega=omega, A0=A0, magnus_order=magnus_order)
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.t5 = t5
        self.t6 = t6
        self.t7 = t7
        self.t8 = t8
        self.mu = mu
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.lamb = lamb
        self.lamb_z = lamb_z
        
        # Basis matrices (dim=4)
        # Tensor product structure: tau (sublattice) x sigma (spin)
        # tau matrices
        self.tau_x = np.kron(sigma_x, sigma_0)
        self.tau_y = np.kron(sigma_y, sigma_0)
        self.tau_z = np.kron(sigma_z, sigma_0)
        self.tau_0 = np.kron(sigma_0, sigma_0)
        
        # Combined matrices needed for terms
        # tau_y \otimes sigma_x
        self.tau_y_sigma_x = np.kron(sigma_y, sigma_x)
        self.tau_y_sigma_y = np.kron(sigma_y, sigma_y)
        self.tau_y_sigma_z = np.kron(sigma_y, sigma_z)
        self.tau_z_sigma_z = np.kron(sigma_z, sigma_z)
        self.tau_z_sigma_x = np.kron(sigma_z, sigma_x)
        self.tau_z_sigma_y = np.kron(sigma_z, sigma_y)


    def get_eps0(self, kx, ky, kz=0):
        """
        Calculate the baseline scalar dispersion eps0 for a given k-point or array of k-points.
        """
        ckx = np.cos(kx)
        cky = np.cos(ky)
        ckz = np.cos(kz)
        
        return (self.t1 * (ckx + cky) - self.mu + self.t2 * ckz + 
                self.t3 * ckx * cky + 
                self.t4 * (ckx + cky) * ckz + 
                self.t5 * ckx * cky * ckz)
        
    def compute_static(self, kx, ky, kz=0):
        """
        Compute H(k) at a single k point.
        """
        # Trigonometric functions
        ckx = np.cos(kx)
        cky = np.cos(ky)
        ckz = np.cos(kz)
        
        skx = np.sin(kx)
        sky = np.sin(ky)
        # skz = np.sin(kz) # not needed for full, but half angles are needed
        
        ckx2 = np.cos(kx/2)
        cky2 = np.cos(ky/2)
        ckz2 = np.cos(kz/2)
        
        skx2 = np.sin(kx/2)
        sky2 = np.sin(ky/2)
        skz2 = np.sin(kz/2)
        
        # epsilon_{0, k}
        eps0 = self.get_eps0(kx, ky, kz)
        
        # t_{x, k}
        tx = self.t8 * ckx2 * cky2 * ckz2
        
        # t_{z, k}
        tz = self.t6 * skx * sky + self.t7 * skx * sky * ckz
        
        # lambda components
        lam_x = self.lamb * skz2 * skx2 * cky2
        lam_y = -self.lamb * skz2 * sky2 * ckx2
        lam_z = self.lamb_z * ckz2 * ckx2 * cky2 * (ckx - cky)
        
        # Construct H
        
        # Term 1: eps0 * I
        H = eps0 * self.tau_0
        
        # Term 2: tx * tau_x
        H += tx * self.tau_x
        
        # Term 3: tz * tau_z
        H += tz * self.tau_z
        
        # Term 4: tau_y (lambda . sigma)
        H += lam_x * self.tau_y_sigma_x
        H += lam_y * self.tau_y_sigma_y
        H += lam_z * self.tau_y_sigma_z
        
        # Term 5: tau_z (J . sigma)
        # Assuming J is along z
        H += self.Jx * self.tau_z_sigma_x
        H += self.Jy * self.tau_z_sigma_y
        H += self.Jz * self.tau_z_sigma_z
        
        return H

    def compute_static_vectorized(self, kx_arr, ky_arr, kz_arr=0):
        """
        Vectorized H(k).
        """
        kx = np.asarray(kx_arr)
        ky = np.asarray(ky_arr)
        
        # Broadcast kz if necesssary
        if np.ndim(kz_arr) == 0:
            kz = np.full_like(kx, kz_arr, dtype=float)
        else:
            kz = np.asarray(kz_arr)
            
        # Ensure shapes match (simple broadcasting handling)
        # We assume kx, ky, kz are broadcastable to the same shape
        
        # Flatten for operations
        shape = kx.shape
        kx = kx.flatten()
        ky = ky.flatten()
        kz = kz.flatten()
        M = kx.size
        
        # Trigonometric functions
        ckx = np.cos(kx)
        cky = np.cos(ky)
        ckz = np.cos(kz)
        
        skx = np.sin(kx)
        sky = np.sin(ky)
        
        ckx2 = np.cos(kx/2)
        cky2 = np.cos(ky/2)
        ckz2 = np.cos(kz/2)
        
        skx2 = np.sin(kx/2)
        sky2 = np.sin(ky/2)
        skz2 = np.sin(kz/2)
        
        # epsilon_{0, k}
        eps0 = self.get_eps0(kx, ky, kz)
        
        # t_{x, k}
        tx = self.t8 * ckx2 * cky2 * ckz2
        
        # t_{z, k}
        tz = self.t6 * skx * sky + self.t7 * skx * sky * ckz
        
        # lambda components
        lam_x = self.lamb * skz2 * skx2 * cky2
        lam_y = -self.lamb * skz2 * sky2 * ckx2
        lam_z = self.lamb_z * ckz2 * ckx2 * cky2 * (ckx - cky)
        
        # Initialize H
        H = np.zeros((M, 4, 4), dtype=complex)
        
        # Add terms using broadcasting
        # eps0 * I
        H += eps0[:, None, None] * self.tau_0[None, :, :]
        
        # tx * tau_x
        H += tx[:, None, None] * self.tau_x[None, :, :]
        
        # tz * tau_z
        H += tz[:, None, None] * self.tau_z[None, :, :]
        
        # lambda terms
        H += lam_x[:, None, None] * self.tau_y_sigma_x[None, :, :]
        H += lam_y[:, None, None] * self.tau_y_sigma_y[None, :, :]
        H += lam_z[:, None, None] * self.tau_y_sigma_z[None, :, :]
        
        # J term
        H += self.Jx * self.tau_z_sigma_x[None, :, :]
        H += self.Jy * self.tau_z_sigma_y[None, :, :]
        H += self.Jz * self.tau_z_sigma_z[None, :, :]
        
        # Reshape to (..., 4, 4) if input was shaped
        if len(shape) > 1:
            H = H.reshape(shape + (4, 4))
            
        return H

    def g_xz_imag(self, kx, ky, kz=0.0, *, band=None, energy=None):
        """
        Imag part of QGT component g_{xz} from the paper's analytic Berry curvature:

            Ω_xz = [1 / (8 E^3)] * λ * t8 * J * cos^2(ky/2) * (cos kz - cos kx)

        Convention used throughout your code:
            g_{xz}^{Im} = -1/2 * Ω_xz

        Parameters
        ----------
        kx, ky, kz : float
        energy : float (REQUIRED)
            The band energy E(kx,ky,kz) for the band you're evaluating.

        Returns
        -------
        float
            g_xz_imag(kx,ky,kz; energy)
        """
        if energy is None:
            raise ValueError("g_xz_imag requires energy=E(kx,ky,kz) for the chosen band.")

        # 1. Re-calculate the scalar baseline energy (eps0) at this k-point using the new helper method
        eps0 = float(self.get_eps0(kx, ky, kz))

        # 2. Extract the true band splitting 'E_split'
        E_split = abs(float(energy) - eps0)

        # Prevent division by absolute zero exactly at the nodal points
        if E_split < 1e-12:
            return 0.0

        lam = float(self.lamb)
        t8  = float(self.t8)

        # Use the magnitude of J unless you prefer a specific component:
        J = float(np.sqrt(self.Jx**2 + self.Jy**2 + self.Jz**2))

        cky2 = np.cos(ky / 2.0)
        Omega_xz = (lam * t8 * J * (cky2**2) * (np.cos(kz) - np.cos(kx))) / (8.0 * (E_split**3))

        return float(-0.5 * Omega_xz)
