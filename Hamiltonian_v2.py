import numpy as np
from scipy.integrate import quad_vec


class Hamiltonian:
    """
    Base class for defining Hamiltonians.
    """
    def __init__(self, dim):
        self.dim = dim  # Dimension of the Hamiltonian matrix
    
    def compute(self, kx, ky):
        """
        Compute the Hamiltonian matrix for a given (kx, ky).
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'compute' method.")
    
    def commutator(self, H2, kx, ky):
        """
        Compute the commutator of this Hamiltonian with another Hamiltonian H2.
        """
        H1_k = self.compute(kx, ky)
        H2_k = H2.compute(kx, ky)
        return np.dot(H1_k, H2_k) - np.dot(H2_k, H1_k)
    
    def numerical_fourier_component(self, n, kx, ky, omega):
        """
        Compute the nth Fourier component of the time-dependent Hamiltonian.
        """
        integral = quad_vec(lambda t: self.compute(kx, ky) * np.exp(-1j * n * omega * t), 
                            0, 2 * np.pi / omega, epsrel=1e-8)
        return integral[0]
    
    def magnus_first_term(self, kx, ky, omega):
        """
        Compute the first term of the Magnus expansion:
        (1/omega) * [H1, H-1]
        """
        # Compute H1 and H-1 Fourier components
        H1 = self.numerical_fourier_component(1, kx, ky, omega)
        Hm1 = self.numerical_fourier_component(-1, kx, ky, omega)
        
        # Compute the commutator [H1, H-1]
        comm = self.commutator_static(H1, Hm1)
        
        # Return the first Magnus term
        return (1 / omega) * comm

    @staticmethod
    def commutator_static(H1, H2):
        """
        Static method to compute the commutator of two Hamiltonians.
        """
        return np.dot(H1, H2) - np.dot(H2, H1)

    def effective_hamiltonian(self, kx, ky, omega):
        """
        Compute the total effective Hamiltonian:
        H_eff = H_0 + (1/omega) * [H1, H-1]
        """
        # Original static Hamiltonian (H_0)
        H_0 = self.compute(kx, ky)
        
        # First Magnus term
        magnus_term = self.magnus_first_term(kx, ky, omega)
        
        # Effective Hamiltonian
        H_eff = H_0 + magnus_term
        return H_eff
    

    
class THF_Hamiltonian(Hamiltonian):
    """
    Hamiltonian for the THF model.
    """
    def __init__(self, nu_star=-50, nu_star_prime=13.0, gamma=-25.0, M=5, G=0.001):
        super().__init__(dim=6)  # THF model has a 6x6 matrix
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


class SquareLatticeHamiltonian(Hamiltonian):
    """
    Hamiltonian for a square lattice model.
    """
    def __init__(self, t1=1, t2=1/np.sqrt(2), t5=-0.1):
        super().__init__(dim=2)  # Square lattice model has a 2x2 matrix
        self.t1 = t1
        self.t2 = t2
        self.t5 = t5
    
    def compute(self, kx, ky):
        H11 = -2 * self.t5 * (np.cos(2 * (kx + ky)) + np.cos(2 * (kx - ky))) - 2 * self.t2 * (np.cos(kx + ky) - np.cos(kx - ky))
        H12 = -2 * self.t1 * np.exp(1j * np.pi / 4) * np.exp(1j * ky) * np.cos(ky) - 2 * self.t1 * np.exp(-1j * np.pi / 4) * np.exp(1j * ky) * np.cos(kx)
        H21 = np.conj(H12)  # H21 is the Hermitian conjugate of H12
        H22 = -2 * self.t5 * (np.cos(2 * (kx + ky)) + np.cos(2 * (kx - ky))) + 2 * self.t2 * (np.cos(kx + ky) - np.cos(kx - ky))
        
        H_k = np.array([
            [H11, H12],
            [H21, H22]
        ], dtype=complex)
        
        return H_k


class SquareLattice1Hamiltonian(Hamiltonian):
    """
    Hamiltonian for the modified square lattice model (Equation 3 from PhysRevLett.106.236804).
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
