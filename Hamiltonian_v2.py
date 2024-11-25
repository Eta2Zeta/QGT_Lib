import numpy as np
from scipy.integrate import quad_vec


class Hamiltonian_Obj:
    """
    Base class for defining time-dependent Hamiltonians with driven terms.
    """
    def __init__(self, dim, omega, A0):
        """
        Initialize the Hamiltonian with its dimension, driving frequency (omega), and driving amplitude (A0).
        """
        self.dim = dim  # Dimension of the Hamiltonian matrix
        self.omega = omega  # Driving frequency
        self.A0 = A0  # Driving amplitude
    
    def compute_static(self, kx, ky):
        """
        Compute the static Hamiltonian matrix for a given (kx, ky).
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'compute_static' method.")
    
    def compute_time_dependent(self, t, kx, ky):
        """
        Compute the time-dependent Hamiltonian matrix for a given time t and (kx, ky).
        Default implementation assumes driving terms are applied to kx and ky.
        """
        # Apply the driven terms to kx and ky
        kx_t = kx + self.A0 * np.cos(self.omega * t)
        ky_t = ky + self.A0 * np.sin(self.omega * t)
        
        # Return the static Hamiltonian evaluated at the transformed kx, ky
        return self.compute_static(kx_t, ky_t)
    
    def numerical_fourier_component(self, n, kx, ky):
        """
        Compute the nth Fourier component of the time-dependent Hamiltonian.
        """
        integral = quad_vec(
            lambda t: self.compute_time_dependent(t, kx, ky) * np.exp(-1j * n * self.omega * t), 
            0, 2 * np.pi / self.omega, 
            epsrel=1e-8
        )
        return integral[0]
    
    def magnus_first_term(self, kx, ky):
        """
        Compute the first term of the Magnus expansion:
        (1/omega) * [H1, H-1]
        """
        # Compute H1 and H-1 Fourier components
        H1 = self.numerical_fourier_component(1, kx, ky)
        Hm1 = self.numerical_fourier_component(-1, kx, ky)
        
        # Compute the commutator [H1, H-1]
        comm = self.commutator_static(H1, Hm1)
        
        # Return the first Magnus term
        return (1 / self.omega) * comm

    @staticmethod
    def commutator_static(H1, H2):
        """
        Static method to compute the commutator of two Hamiltonians.
        """
        return np.dot(H1, H2) - np.dot(H2, H1)

    def effective_hamiltonian(self, kx, ky):
        """
        Compute the total effective Hamiltonian:
        H_eff = H_0 + (1/omega) * [H1, H-1]
        If A0 = 0, return H_0 directly as the effective Hamiltonian.
        """
        # Original static Hamiltonian (H_0)
        H_0 = self.compute_static(kx, ky)
        
        if self.A0 == 0:
            # If A0 is 0, no driving terms exist, return static Hamiltonian
            return H_0
        
        # First Magnus term
        magnus_term = self.magnus_first_term(kx, ky)
        
        # Effective Hamiltonian
        H_eff = H_0 + magnus_term
        return H_eff

class TwoOrbitalSpinfulHamiltonian(Hamiltonian_Obj):
    """
    Hamiltonian for the two-orbital spinful model.
    """
    def __init__(self, t=1, mu=0, zeta=0, a=1, omega = np.pi/2, A0 = 0):
        """
        Initialize the two-orbital spinful Hamiltonian.
        Parameters:
        - t: Hopping parameter
        - mu: Chemical potential
        - zeta: Parameter for alpha_k
        - a: Lattice spacing
        - omega: Driving frequency
        - A0: Driving amplitude
        """
        super().__init__(dim=4, omega=omega, A0=A0)
        self.t = t
        self.mu = mu
        self.zeta = zeta
        self.a = a

    def compute_static(self, kx, ky):
        """
        Compute the static Hamiltonian for the two-orbital spinful model.
        """
        # Compute alpha_k
        alpha_k = self.zeta * (np.cos(kx * self.a) + np.cos(ky * self.a))
        
        # Define matrix elements
        H = np.zeros((4, 4), dtype=complex)
        sin_alpha = np.sin(alpha_k)
        cos_alpha = np.cos(alpha_k)
        
        # Fill the Hamiltonian matrix
        H[0, 0] = -self.t * self.mu
        H[1, 1] = -self.t * self.mu
        H[2, 2] = -self.t * self.mu
        H[3, 3] = -self.t * self.mu
        
        H[0, 2] = -self.t * (sin_alpha - 1j * cos_alpha)
        H[2, 0] = -self.t * (sin_alpha + 1j * cos_alpha)
        
        H[1, 3] = -self.t * (sin_alpha + 1j * cos_alpha)
        H[3, 1] = -self.t * (sin_alpha - 1j * cos_alpha)
        
        return H

class SquareLatticeHamiltonian(Hamiltonian_Obj):
    """
    Hamiltonian for a square lattice model.
    """
    def __init__(self, t1=1, t2=1/np.sqrt(2), t5=-0.1, omega=2 * np.pi, A0=1.0):
        super().__init__(dim=2, omega=omega, A0=A0)  # Pass omega and A0 to the base class
        self.t1 = t1
        self.t2 = t2
        self.t5 = t5
    
    def compute_static(self, kx, ky):
        """
        Compute the static Hamiltonian for the square lattice.
        """
        H11 = -2 * self.t5 * (np.cos(2 * (kx + ky)) + np.cos(2 * (kx - ky))) - 2 * self.t2 * (np.cos(kx + ky) - np.cos(kx - ky))
        H12 = -2 * self.t1 * np.exp(1j * np.pi / 4) * np.exp(1j * ky) * np.cos(ky) - 2 * self.t1 * np.exp(-1j * np.pi / 4) * np.exp(1j * ky) * np.cos(kx)
        H21 = np.conj(H12)  # H21 is the Hermitian conjugate of H12
        H22 = -2 * self.t5 * (np.cos(2 * (kx + ky)) + np.cos(2 * (kx - ky))) + 2 * self.t2 * (np.cos(kx + ky) - np.cos(kx - ky))
        
        H_k = np.array([
            [H11, H12],
            [H21, H22]
        ], dtype=complex)
        
        return H_k



    
class THF_Hamiltonian(Hamiltonian_Obj):
    """
    Hamiltonian for the THF model. This is not optimized for the Magnus expansion yet and it is 
    with the frequency term included as G. 
    """
    def __init__(self, nu_star=-50, nu_star_prime=13.0, gamma=-25.0, M=5, G=0.001, omega = np.pi, A0 = 0):
        super().__init__(dim=6, omega=omega, A0=A0)  # THF model has a 6x6 matrix
        self.nu_star = nu_star
        self.nu_star_prime = nu_star_prime
        self.gamma = gamma
        self.M = M
        self.G = G
    
    def compute(self, kx, ky):
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


class RashbaHamiltonian(Hamiltonian_Obj):
    """
    Rashba Hamiltonian for a 2D system with spin-orbit coupling.
    """
    def __init__(self, m=1, alpha=1, omega=2 * np.pi, A0=0.0):
        """
        Initialize the Rashba Hamiltonian.
        
        Parameters:
        - m: Effective mass of the particle
        - alpha: Rashba spin-orbit coupling strength
        - omega: Driving frequency
        - A0: Driving amplitude
        """
        super().__init__(dim=2, omega=omega, A0=A0)
        self.m = m
        self.alpha = alpha

    def compute_static(self, kx, ky):
        """
        Compute the static Rashba Hamiltonian for a given (kx, ky).
        """
        k_squared = kx**2 + ky**2
        H11 = k_squared / (2 * self.m)
        H22 = k_squared / (2 * self.m)
        H12 = self.alpha * (ky + 1j * kx)
        H21 = self.alpha * (ky - 1j * kx)

        H_k = np.array([
            [H11, H12],
            [H21, H22]
        ], dtype=complex)
        
        return H_k



class SquareLattice1Hamiltonian(Hamiltonian_Obj):
    """
    Hamiltonian for the modified square lattice model (Equation 3 from PhysRevLett.106.236804). Not used anymore
    """
    def __init__(self, t1=1, t2=1/np.sqrt(2)):
        super().__init__(dim=2)  # Square lattice model has a 2x2 matrix
        self.t1 = t1
        self.t2 = t2
    
    def compute(self, kx, ky):
        M11 = 2 * self.t2 * (np.cos(kx) - np.cos(ky))
        M12 = self.t1 * np.exp(1j * np.pi / 4) * (1 + np.exp(-1j * (ky - kx))) + self.t1 * np.exp(-1j * np.pi / 4) * (np.exp(1j * kx) + np.exp(-1j * ky))
        M21 = np.conj(M12)  # M21 is the Hermitian conjugate of M12
        M22 = -2 * self.t2 * (np.cos(kx) - np.cos(ky))
        
        H_k = np.array([
            [M11, M12],
            [M21, M22]
        ], dtype=complex)
        
        return H_k
