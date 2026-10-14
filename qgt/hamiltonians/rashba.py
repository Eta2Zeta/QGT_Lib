import numpy as np

from qgt.hamiltonians.base import hamiltonian


class RashbaHamiltonian(hamiltonian):
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

    def compute_static(self, kx, ky, kz=0):
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
    
    def Hp1(self, kx, ky, kz=0):
        H11 = (self.A0 / (2 * self.m)) * (kx - 1j*ky)
        H12 = 0
        H21 = -1j * self.alpha * self.A0
        H22 = H11


        H_k = np.array([
            [H11, H12],
            [H21, H22]
        ], dtype=complex)
        
        return H_k
    
    def Hp2(self, kx, ky, kz=0):
        H11 = 0
        H12 = 0
        H21 = 0
        H22 = 0

        H_k = np.array([
            [H11, H12],
            [H21, H22]
        ], dtype=complex)
        
        return H_k
