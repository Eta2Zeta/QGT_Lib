import numpy as np
from .Hamiltonian import hamiltonian


class THF_Hamiltonian_Legacy(hamiltonian):
    """
    Hamiltonian for the THF model. This is not optimized for the Magnus expansion yet and it is 
    with the frequency term included as G = A0^2/omega. 
    """
    def __init__(self, nu_star=-50, 
                 nu_star_prime=13.0, 
                 gamma=-25.0, 
                 M=5, 
                 G=0.001, 
                 omega = np.pi, 
                 A0 = 0):
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

    The defaults reproduce the single-particle parameters quoted below
    Eqs. (A31)-(A33) of ``Flat_band_quantum_geometry.pdf`` for twist angle
    1.05 degrees and w0/w1 = 0.8, using the corrected M sigma_x convention.
    Momenta must be supplied in inverse Angstrom, lengths are stored in
    Angstrom, and energies are returned in meV. The BM velocities are supplied
    in eV Angstrom and converted to meV Angstrom.
    """
    def __init__(
        self,
        eta=1,
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
        V=0.0,
        theta_deg=1.05,
        K_plus=1.703,
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
        self.theta_deg = float(theta_deg)
        self.K_plus = float(K_plus)
        self.k_theta = 2.0 * self.K_plus * np.sin(
            0.5 * np.deg2rad(self.theta_deg)
        )
        self.q1 = self.k_theta * np.array([0.0, -1.0])
        self.q2 = self.k_theta * np.array([np.sqrt(3.0) / 2.0, 0.5])
        self.q3 = self.k_theta * np.array([-np.sqrt(3.0) / 2.0, 0.5])
        self.b1 = self.q2 - self.q1
        self.b2 = self.q3 - self.q1

        # Eq. (A2): |a_M1| = 4*pi/(3*k_theta).
        self.a_M1 = 4.0 * np.pi / (3.0 * self.k_theta)

        self.lambda_factor = float(lambda_factor)
        self.lambda_ = self.lambda_factor * self.a_M1

        self.gamma = float(gamma)
        self.M = float(M)
        self.V = float(V)

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

        H_k = np.array([
            [0,          0,          v * kp,     0,          gamma_g,    vp_g * km],
            [0,          0,          0,          v * km,     vp_g * kp,  gamma_g],
            [v * km,     0,          0,          self.M,     0,          vpp_g * kp],
            [0,          v * kp,     self.M,     0,          vpp_g * km, 0],
            [gamma_g,    vp_g * km,  0,          vpp_g * kp, 0,          0],
            [vp_g * kp,  gamma_g,    vpp_g * km, 0,          0,          0],
        ], dtype=complex)

        potential_test = self.V * np.diag([-1, 1, 1, -1, 1, -1])
        return H_k + potential_test

    def get_sym_path(self):
        """
        Physical high-symmetry path for the hexagonal moire Brillouin zone.

        The K point is q2 from Eq. (A1), and M is the midpoint of the
        adjacent edge connecting q2 and -q1. Coordinates are in inverse
        Angstrom.

        K -> Gamma -> M -> K
        """
        sym_points = {
            "G": np.array([0.0, 0.0]),
            "K": self.q2,
            "M": 0.5 * self.b1,
        }
        path = ["K", "G", "M", "K"]
        return sym_points, path
