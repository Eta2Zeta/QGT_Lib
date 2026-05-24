import numpy as np
from scipy.integrate import quad_vec


# Define the new Hamiltonian function
def H_THF(kx, ky, nu_star=-50, nu_star_prime=13.0, gamma=-25.0, M=5, G=0.001):
    k = np.sqrt(kx**2 + ky**2)
    theta = np.arctan2(ky, kx)
    
    H_k = np.array([
        [G*(nu_star**2 - nu_star_prime**2), 0, nu_star * k * np.exp(1j * theta), 0, gamma, nu_star_prime * k * np.exp(-1j * theta)],
        [0, -G*(nu_star**2 - nu_star_prime**2), 0, nu_star * k * np.exp(-1j * theta), nu_star_prime * k * np.exp(1j * theta), gamma],
        [nu_star * k * np.exp(-1j * theta), 0, -G * nu_star**2, M, 0, 0],
        [0, nu_star * k * np.exp(1j * theta), M, G * nu_star**2, 0, 0],
        [gamma, nu_star_prime * k * np.exp(-1j * theta), 0, 0, -G * nu_star_prime**2, 0],
        [nu_star_prime * k * np.exp(1j * theta), gamma, 0, 0, 0, G * nu_star_prime**2]
    ])
    
    return H_k

def H_THF_factory(G):
    def H_THF(kx, ky, nu_star=-50, nu_star_prime=13.0, gamma=-25.0, M=5):
        k = np.sqrt(kx**2 + ky**2)
        theta = np.arctan2(ky, kx)
        
        H_k = np.array([
            [G*(nu_star**2 - nu_star_prime**2), 0, nu_star * k * np.exp(1j * theta), 0, gamma, nu_star_prime * k * np.exp(-1j * theta)],
            [0, -G*(nu_star**2 - nu_star_prime**2), 0, nu_star * k * np.exp(-1j * theta), nu_star_prime * k * np.exp(1j * theta), gamma],
            [nu_star * k * np.exp(-1j * theta), 0, -G * nu_star**2, M, 0, 0],
            [0, nu_star * k * np.exp(1j * theta), M, G * nu_star**2, 0, 0],
            [gamma, nu_star_prime * k * np.exp(-1j * theta), 0, 0, -G * nu_star_prime**2, 0],
            [nu_star_prime * k * np.exp(1j * theta), gamma, 0, 0, 0, G * nu_star_prime**2]
        ])
        
        return H_k
    return H_THF


# Compute numerical Fourier components
def compute_numerical_Hn(H, n, kx, ky, omega):
    integral = quad_vec(lambda t: H(t, kx, ky) * np.exp(-1j * n * omega * t), 0, 2 * np.pi / omega, epsrel=1e-8)
    return integral[0]

