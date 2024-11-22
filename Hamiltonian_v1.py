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


# Define the corrected Hamiltonian function with constants t1, t2, and t5
def H_Square_Lattice(kx, ky, t1=1, t2=1/np.sqrt(2), t5=-0.1):
    # Compute the matrix elements
    H11 = -2 * t5 * (np.cos(2 * (kx + ky)) + np.cos(2 * (kx - ky))) - 2 * t2 * (np.cos(kx + ky) - np.cos(kx - ky))
    H12 = -2 * t1 * np.exp(1j * np.pi / 4) * np.exp(1j * ky) * np.cos(ky) - 2 * t1 * np.exp(-1j * np.pi / 4) * np.exp(1j * ky) * np.cos(kx)
    H21 = -2 * t1 * np.exp(-1j * np.pi / 4) * np.exp(-1j * ky) * np.cos(ky) - 2 * t1 * np.exp(1j * np.pi / 4) * np.exp(-1j * ky) * np.cos(kx)
    H22 = -2 * t5 * (np.cos(2 * (kx + ky)) + np.cos(2 * (kx - ky))) + 2 * t2 * (np.cos(kx + ky) - np.cos(kx - ky))
    
    # Construct the Hamiltonian matrix
    H_k = np.array([
        [H11, H12],
        [H21, H22]
    ], dtype=complex)
    
    return H_k



# Define the Square_Lattice_1 function with constants t1 and t2
# Equation 3 from PhysRevLett.106.236804
def H_Square_Lattice_1(kx, ky, t1=1, t2=1/np.sqrt(2)):
    # Compute the matrix elements
    M11 = 2 * t2 * (np.cos(kx) - np.cos(ky))
    M12 = t1 * np.exp(1j * np.pi / 4) * (1 + np.exp(-1j * (ky - kx))) + t1 * np.exp(-1j * np.pi / 4) * (np.exp(1j * kx) + np.exp(-1j * ky))
    M21 = t1 * np.exp(-1j * np.pi / 4) * (1 + np.exp(1j * (ky - kx))) + t1 * np.exp(1j * np.pi / 4) * (np.exp(-1j * kx) + np.exp(1j * ky))
    M22 = -2 * t2 * (np.cos(kx) - np.cos(ky))
    
    # Construct the matrix
    H_k = np.array([
        [M11, M12],
        [M21, M22]
    ], dtype=complex)
    
    return H_k


# Hamiltonian Utility Functions


# Compute numerical Fourier components
def compute_numerical_Hn(H, n, kx, ky, omega):
    integral = quad_vec(lambda t: H(t, kx, ky) * np.exp(-1j * n * omega * t), 0, 2 * np.pi / omega, epsrel=1e-8)
    return integral[0]

def commutator(H1, H2):
    return np.dot(H1, H2) - np.dot(H2, H1)