import numpy as np
from .Hamiltonian_v2 import hamiltonian

# Define Pauli matrices
sigma_0 = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Basis matrices (dim=4)
tau_x = np.kron(sigma_x, sigma_0)
tau_y = np.kron(sigma_y, sigma_0)
tau_z = np.kron(sigma_z, sigma_0)
tau_0 = np.kron(sigma_0, sigma_0)

# Combined matrices needed for terms
# tau_y \otimes sigma
tau_y_sigma_x = np.kron(sigma_y, sigma_x)
tau_y_sigma_y = np.kron(sigma_y, sigma_y)
tau_y_sigma_z = np.kron(sigma_y, sigma_z)

# tau_z \otimes sigma (for J term)
tau_z_sigma_x = np.kron(sigma_z, sigma_x)
tau_z_sigma_y = np.kron(sigma_z, sigma_y)
tau_z_sigma_z = np.kron(sigma_z, sigma_z)


class gWaveAltermagnetHamiltonian(hamiltonian):
    """
    Hamiltonian for g-wave altermagnet (CrSb/MnTe).
    
    H = epsilon_{0, k} + t_{x, k} tau_x + t_{z, k} tau_z + tau_y (vec{lambda}_k . vec{sigma}) + tau_z (vec{J} . vec{sigma})
    
    Parameters per user request:
    t1 = 0.8 (Main Hopping)
    t2 = 0.0 (Interlayer? Not specified, default 0)
    t3 = 0.3 (Inter-layer Hopping)
    t4 = 0.3 (Altermagnetism)
    mu = 0.0
    J = [0.2, 0, 0] (Jx=0.2)
    lambda = 0.1
    lamb_z = 0.1 (Assume same as lambda if not specified, or use lambda for all)
    """
    def __init__(self, t1=0.8, t2=0.2, t3=0.3, t4=0.3,
                 mu=0.0, Jx=0.2, Jy=0.0, Jz=0.0, lamb=0.1, lamb_z=0.1, 
                 omega=0.0, A0=0.0, magnus_order=1):
        super().__init__(dim=4, omega=omega, A0=A0, magnus_order=magnus_order)
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.mu = mu
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.lamb = lamb
        self.lamb_z = lamb_z
        
        
    def compute_static(self, kx, ky, kz=0):
        # Trigonometric functions
        ckx = np.cos(kx)
        arg_kx_2 = kx / 2.0
        arg_sq3ky_2 = np.sqrt(3) * ky / 2.0
        arg_kz_2 = kz / 2.0
        
        ckx = np.cos(kx)
        ckz = np.cos(kz)
        
        ckx_2 = np.cos(arg_kx_2)
        skx_2 = np.sin(arg_kx_2)
        
        csq3ky_2 = np.cos(arg_sq3ky_2)
        ssq3ky_2 = np.sin(arg_sq3ky_2)
        
        ckz_2 = np.cos(arg_kz_2)
        skz_2 = np.sin(arg_kz_2)
        
        skx = np.sin(kx)
        skz = np.sin(kz)
        
        fx = skx + skx_2 * csq3ky_2 
        fy = np.sqrt(3) * ckx_2 * ssq3ky_2
        
        term1 = self.t1 * (ckx + 2 * ckx_2 * csq3ky_2)
        eps0 = term1 + self.t2 * ckz - self.mu
        
        tx = self.t3 * ckz_2
        
        tz = self.t4 * skz * fy * (fy**2 - 3 * fx**2)
        
        # lambda_{x, k} = lambda cos(kz/2) (fx^2 - fy^2)
        lam_xk = self.lamb * ckz_2 * (fx**2 - fy**2)
        
        # lambda_{y, k} = -2 lambda cos(kz/2) fx fy
        lam_yk = -2 * self.lamb * ckz_2 * fx * fy
        
        # lambda_{z, k} = lambda_z sin(kz/2) fx (fx^2 - 3 fy^2)
        lam_zk = self.lamb_z * skz_2 * fx * (fx**2 - 3 * fy**2)
        
        # Construct H
        
        # Term 1: eps0 * I
        H = eps0 * tau_0
        
        # Term 2: tx * tau_x
        H += tx * tau_x
        
        # Term 3: tz * tau_z
        H += tz * tau_z
        
        # Term 4: tau_y (lambda . sigma)
        H += lam_xk * tau_y_sigma_x
        H += lam_yk * tau_y_sigma_y
        H += lam_zk * tau_y_sigma_z
        
        # Term 5: tau_z (vec{J} . vec{sigma})
        H += self.Jx * tau_z_sigma_x
        H += self.Jy * tau_z_sigma_y
        H += self.Jz * tau_z_sigma_z
        
        return H

    def compute_static_vectorized(self, kx_arr, ky_arr, kz_arr=0):
        kx = np.asarray(kx_arr)
        ky = np.asarray(ky_arr)
        if np.ndim(kz_arr) == 0:
            kz = np.full_like(kx, kz_arr, dtype=float)
        else:
            kz = np.asarray(kz_arr)
            
        shape = kx.shape
        kx = kx.flatten()
        ky = ky.flatten()
        kz = kz.flatten()
        M = kx.size
        
        # Arguments
        arg_kx_2 = kx / 2.0
        arg_sq3ky_2 = np.sqrt(3) * ky / 2.0
        arg_kz_2 = kz / 2.0
        
        ckx = np.cos(kx)
        ckz = np.cos(kz)
        
        ckx_2 = np.cos(arg_kx_2)
        skx_2 = np.sin(arg_kx_2)
        
        csq3ky_2 = np.cos(arg_sq3ky_2)
        ssq3ky_2 = np.sin(arg_sq3ky_2)
        
        ckz_2 = np.cos(arg_kz_2)
        skz_2 = np.sin(arg_kz_2)
        
        skx = np.sin(kx)
        skz = np.sin(kz)
        
        # f_x, f_y
        fx = skx + skx_2 * csq3ky_2
        fy = np.sqrt(3) * ckx_2 * ssq3ky_2
        
        # eps0
        term1 = self.t1 * (ckx + 2 * ckx_2 * csq3ky_2)
        eps0 = term1 + self.t2 * ckz - self.mu
        
        # tx, tz
        tx = self.t3 * ckz_2
        tz = self.t4 * skz * fy * (fy**2 - 3 * fx**2)
        
        # lambda
        lam_xk = self.lamb * ckz_2 * (fx**2 - fy**2)
        lam_yk = -2 * self.lamb * ckz_2 * fx * fy
        lam_zk = self.lamb_z * skz_2 * fx * (fx**2 - 3 * fy**2)
        
        # Initialize H
        H = np.zeros((M, 4, 4), dtype=complex)
        
        # Add terms
        H += eps0[:, None, None] * tau_0[None, :, :]
        H += tx[:, None, None] * tau_x[None, :, :]
        H += tz[:, None, None] * tau_z[None, :, :]
        
        H += lam_xk[:, None, None] * tau_y_sigma_x[None, :, :]
        H += lam_yk[:, None, None] * tau_y_sigma_y[None, :, :]
        H += lam_zk[:, None, None] * tau_y_sigma_z[None, :, :]
        
        H += self.Jx * tau_z_sigma_x[None, :, :]
        H += self.Jy * tau_z_sigma_y[None, :, :]
        H += self.Jz * tau_z_sigma_z[None, :, :]
        
        if len(shape) > 1:
            H = H.reshape(shape + (4, 4))
            
        return H
