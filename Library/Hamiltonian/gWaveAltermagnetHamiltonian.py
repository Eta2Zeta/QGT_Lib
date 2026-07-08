import numpy as np
from .Hamiltonian import hamiltonian

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

    def get_sym_path(self, path="ALHABHPLUADUPDCPBCAPKDGMKGEKUMDECK"):
        """
        High-symmetry path for the hexagonal g-wave altermagnet Brillouin zone.

        The path is supplied as a string of single-character point labels. Useful
        historical choices:

            Standard hexagonal path:
                "GMKGALHA"          # G -> M -> K -> G -> A -> L -> H -> A

            Old requested path:
                "LHALMHKMGKAGL"     # L -> H -> A -> L -> M -> H -> K -> M -> G -> K -> A -> G -> L

            Old requested path:
                "LAHLUHPUDPKUMGKMADG"

            Newest requested path, used by default:
                "ALHABHPLUADUPDCPBCAPKDGMKGEKUMDECK"

        Returns
        -------
        sym_points : dict
            Mapping of label -> np.ndarray of shape (3,).
        path_labels : list of str
            Ordered labels defining the path.
        """
        kM_y = 2.0 * np.pi / np.sqrt(3.0)
        kK_x = 2.0 * np.pi / 3.0
        kz_A = np.pi

        kB_x = np.pi
        kB_y = np.pi / np.sqrt(3.0)

        sym_points = {
            "G": np.array([0.0, 0.0, 0.0]),
            "M": np.array([0.0, kM_y, 0.0]),
            "K": np.array([kK_x, kM_y, 0.0]),
            "A": np.array([0.0, 0.0, kz_A]),
            "L": np.array([0.0, kM_y, kz_A]),
            "H": np.array([kK_x, kM_y, kz_A]),
            "U": np.array([0.0, kM_y, kz_A / 2.0]),
            "D": np.array([0.0, 0.0, kz_A / 2.0]),
            "P": np.array([kK_x, kM_y, kz_A / 2.0]),
            "B": np.array([kB_x, kB_y, kz_A]),
            "C": np.array([kB_x, kB_y, kz_A / 2.0]),
            "E": np.array([kB_x, kB_y, 0.0]),
        }

        path_labels = list(path)
        unknown_labels = [label for label in path_labels if label not in sym_points]
        if unknown_labels:
            valid_labels = "".join(sorted(sym_points))
            raise ValueError(
                f"Unknown g-wave symmetry labels {unknown_labels}. Valid labels are: {valid_labels}."
            )

        return sym_points, path_labels
        
        
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
        
    def get_analytical_eigenvalues(self, kx_arr, ky_arr, kz_arr=0):
        """
        Compute the analytical eigenvalues for the g-wave altermagnet Hamiltonian.
        """
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
        
        # lambda components
        lam_xk = self.lamb * ckz_2 * (fx**2 - fy**2)
        lam_yk = -2 * self.lamb * ckz_2 * fx * fy
        lam_zk = self.lamb_z * skz_2 * fx * (fx**2 - 3 * fy**2)
        
        # Squared magnitudes
        J2 = self.Jx**2 + self.Jy**2 + self.Jz**2
        lam2 = lam_xk**2 + lam_yk**2 + lam_zk**2
        
        # Cross product (lambda x J)
        cross_x = lam_yk * self.Jz - lam_zk * self.Jy
        cross_y = lam_zk * self.Jx - lam_xk * self.Jz
        cross_z = lam_xk * self.Jy - lam_yk * self.Jx
        cross2 = cross_x**2 + cross_y**2 + cross_z**2
        
        # E_{\alpha= \pm, \beta= \pm} = \varepsilon_{0, k} + \alpha (t_x^2 + t_z^2 + \lambda^2 + J^2 + \beta * 2 \sqrt{t_z^2 J^2 + (\lambda \times J)^2})^{1/2}
        X = tx**2 + tz**2 + lam2 + J2
        Y = np.sqrt(tz**2 * J2 + cross2)
        
        # The 4 bands in ascending order
        E1 = eps0 - np.sqrt(np.maximum(X + 2 * Y, 0))
        E2 = eps0 - np.sqrt(np.maximum(X - 2 * Y, 0))
        E3 = eps0 + np.sqrt(np.maximum(X - 2 * Y, 0))
        E4 = eps0 + np.sqrt(np.maximum(X + 2 * Y, 0))
        
        E = np.stack([E1, E2, E3, E4], axis=-1)
        
        if len(shape) > 1:
            E = E.reshape(shape + (4,))
            
        return E
        
    def get_spin_operator(self, component='z'):
        """
        Returns the spin operator matrix tau_0 x sigma_i for the requested component ('x', 'y', or 'z').
        Returns a 4x4 complex matrix.
        """
        if component == 'x':
            return np.kron(sigma_0, sigma_x)
        elif component == 'y':
            return np.kron(sigma_0, sigma_y)
        elif component == 'z':
            return np.kron(sigma_0, sigma_z)
        else:
            raise ValueError(f"Unknown spin component: {component}")
