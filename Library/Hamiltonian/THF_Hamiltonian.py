import numpy as np
from .Hamiltonian import hamiltonian


class THF_Hamiltonian_Legacy(hamiltonian):
    """
    Hamiltonian for the THF model. This is not optimized for the Magnus expansion yet and it is 
    with the frequency term included as G = A0^2/omega. 
    """
    def __init__(self, nu_star=-50, nu_star_prime=13.0, gamma=-25.0, M=5, G=0.001, omega = np.pi, A0 = 0):
        super().__init__(dim=6, omega=omega, A0=A0)  # THF model has a 6x6 matrix
        self.nu_star = nu_star
        self.nu_star_prime = nu_star_prime
        self.gamma = gamma
        self.M = M
        self.G = G
    
    def compute_static(self, kx, ky, kz = 0):
        k = np.sqrt(kx**2 + ky**2)
        theta = np.arctan2(ky, kx)
        
        H_k = np.array([
            [self.G * (self.nu_star**2 - self.nu_star_prime**2), 0, self.nu_star * k * np.exp(1j * theta), 0, self.gamma, self.nu_star_prime * k * np.exp(-1j * theta)],
            [0, -self.G * (self.nu_star**2 - self.nu_star_prime**2), 0, self.nu_star * k * np.exp(-1j * theta), self.nu_star_prime * k * np.exp(1j * theta), self.gamma],
            [self.nu_star * k * np.exp(-1j * theta), 0, -self.G * self.nu_star**2, self.M, 0, 0],
            [0, self.nu_star * k * np.exp(1j * theta), self.M, self.G * self.nu_star**2, 0, 0],
            [self.gamma, self.nu_star_prime * k * np.exp(-1j * theta), 0, 0, -self.G * self.nu_star_prime**2, 0],
            [self.nu_star_prime * k * np.exp(1j * theta), self.gamma, 0, 0, 0, self.G * self.nu_star_prime**2]
        ])
        
        return H_k

    def get_sym_path(self, const_fact=3):
        """
        High-symmetry path for the hexagonal (THF) Brillouin zone.
        K → Γ → M → K
        """
        kK_x = const_fact * np.sqrt(3) * np.pi / 2          # K corner
        kK_y = const_fact * 1/2 * np.pi

        sym_points = {
            "G": np.array([0.0,   0.0  ]),
            "K": np.array([kK_x,  kK_y  ]),
            "M": np.array([kK_x,  0 ]),
        }
        path = ["K", "G", "M", "K"]
        return sym_points, path


class THF_Hamiltonian(hamiltonian):
    """
    Six-band THF Hamiltonian with Gaussian c-f form factor.

    Energies are returned in meV. The BM velocities are supplied in eV Angstrom
    and converted to meV Angstrom internally so they are consistent with gamma
    and M.
    """
    def __init__(
        self,
        eta=1,
        a_M1=1.0,
        lambda_factor=0.3375,
        gamma=-24.75,
        M=3.697,
        v_star=-4.303,
        v_star_prime=1.623,
        v_star_double_prime=-0.0332,
        omega=np.pi,
        A0=0,
        polarization='left',
        magnus_order=1,
        analytic_magnus=False,
    ):
        super().__init__(
            dim=6,
            omega=omega,
            A0=A0,
            polarization=polarization,
            magnus_order=magnus_order,
            analytic_magnus=analytic_magnus,
        )
        if eta not in (-1, 1):
            raise ValueError("eta must be +1 or -1")

        self.eta = int(eta)
        self.a_M1 = float(a_M1)
        self.lambda_factor = float(lambda_factor)
        self.lambda_ = self.lambda_factor * self.a_M1

        self.gamma = float(gamma)
        self.M = float(M)

        # Convert eV Angstrom to meV Angstrom.
        self.v_star = 1000.0 * float(v_star)
        self.v_star_prime = 1000.0 * float(v_star_prime)
        self.v_star_double_prime = 1000.0 * float(v_star_double_prime)

    def gaussian_form_factor(self, kx, ky):
        return np.exp(-0.5 * (kx**2 + ky**2) * self.lambda_**2)

    def compute_static(self, kx, ky, kz=0):
        eta_kx = self.eta * kx
        kp = eta_kx + 1j * ky
        km = eta_kx - 1j * ky
        gk = self.gaussian_form_factor(kx, ky)

        v = self.v_star
        vp = self.v_star_prime
        vpp = self.v_star_double_prime
        gamma_g = gk * self.gamma
        vp_g = gk * vp
        vpp_g = gk * vpp

        return np.array([
            [0,          0,          v * kp,     0,          gamma_g,    vp_g * km],
            [0,          0,          0,          v * km,     vp_g * kp,  gamma_g],
            [v * km,     0,          0,          self.M,     0,          vpp_g * kp],
            [0,          v * kp,     self.M,     0,          vpp_g * km, 0],
            [gamma_g,    vp_g * km,  0,          vpp_g * kp, 0,          0],
            [vp_g * kp,  gamma_g,    vpp_g * km, 0,          0,          0],
        ], dtype=complex)

    def get_sym_path(self, const_fact=3):
        """
        High-symmetry path for the hexagonal (THF) Brillouin zone.
        K -> Gamma -> M -> K
        """
        kK_x = const_fact * np.sqrt(3) * np.pi / 2
        kK_y = const_fact * 0.5 * np.pi

        sym_points = {
            "G": np.array([0.0, 0.0]),
            "K": np.array([kK_x, kK_y]),
            "M": np.array([kK_x, 0.0]),
        }
        path = ["K", "G", "M", "K"]
        return sym_points, path
