import numpy as np
from .Hamiltonian import hamiltonian

class THF_Hamiltonian(hamiltonian):
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
    
    def compute_static(self, kx, ky):
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
