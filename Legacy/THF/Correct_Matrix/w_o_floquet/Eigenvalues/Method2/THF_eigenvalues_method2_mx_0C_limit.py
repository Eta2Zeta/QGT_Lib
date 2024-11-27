import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cmath

# Constants
nu_star = -4.303  # Adjust as needed
nu_star_prime = 1.622  # Adjust as needed
nu_star_prime_prime = nu_star_prime / 10  # Adjust as needed
lambda_val = 0.03  # Adjust as needed
M = 3.697  # Adjust as needed
gamma = -24.75
z_limit = 5  # Set the z-axis limit

# Define parameters
mesh_spacing = 200
k_range = 1
k_max = k_range * np.pi  # Adjust based on the realistic Brillouin zone
kx = np.linspace(-k_max, k_max, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)
k = np.sqrt(kx**2 + ky**2)
theta = np.arctan2(ky, kx)

# Define the coefficients based on kx, ky, and other constants
def get_coefficients(k, theta):
    C = np.exp(-k**2 * lambda_val**2 / 2) * gamma
    D = np.exp(-k**2 * lambda_val**2 / 2) * nu_star_prime * k
    A = nu_star * k

    return A, C, D

# Define Q based on the coefficients
def compute_Q(A, D, M):
    return 9 * A**2 * M - 18 * D**2 * M + 2 * M**3 + cmath.sqrt(
        4 * (-3 * A**2 - 3 * D**2 - M**2)**3 + (9 * A**2 * M - 18 * D**2 * M + 2 * M**3)**2)

# Analytical solutions for eigenvalues
def analytical_eigenvalue_1(A, D, M, Q):
    return M / 3 - (2**(1/3) * (-3 * A**2 - 3 * D**2 - M**2)) / (
        3 * Q**(1/3)) + Q**(1/3) / (3 * 2**(1/3))

def analytical_eigenvalue_2(A, D, M, Q):
    return M / 3 + ((1 + 1j * np.sqrt(3)) * (-3 * A**2 - 3 * D**2 - M**2)) / (
        3 * 2**(2/3) * Q**(1/3)) - ((1 - 1j * np.sqrt(3)) * Q**(1/3)) / (6 * 2**(1/3))

def analytical_eigenvalue_3(A, D, M, Q):
    return M / 3 + ((1 - 1j * np.sqrt(3)) * (-3 * A**2 - 3 * D**2 - M**2)) / (
        3 * 2**(2/3) * Q**(1/3)) - ((1 + 1j * np.sqrt(3)) * Q**(1/3)) / (6 * 2**(1/3))

def analytical_eigenvalue_4(A, D, M, Q):
    return -M / 3 + (2**(1/3) * (-3 * (A**2 + D**2) - M**2)) / (
        3 * Q**(1/3)) - Q**(1/3) / (3 * 2**(1/3))

def analytical_eigenvalue_5(A, D, M, Q):
    return -M / 3 - ((1 + 1j * np.sqrt(3)) * (-3 * (A**2 + D**2) - M**2)) / (
        3 * 2**(2/3) * Q**(1/3)) + ((1 - 1j * np.sqrt(3)) * Q**(1/3)) / (6 * 2**(1/3))

def analytical_eigenvalue_6(A, D, M, Q):
    return -M / 3 - ((1 - 1j * np.sqrt(3)) * (-3 * (A**2 + D**2) - M**2)) / (
        3 * 2**(2/3) * Q**(1/3)) + ((1 + 1j * np.sqrt(3)) * Q**(1/3)) / (6 * 2**(1/3))

# Initialize arrays to store eigenvalues
eigenvalues = np.zeros((mesh_spacing, mesh_spacing, 6), dtype=complex)

for i in range(mesh_spacing):
    for j in range(mesh_spacing):
        k_ij = k[i, j]
        theta_ij = theta[i, j]
        A, C, D = get_coefficients(k_ij, theta_ij)
        Q = compute_Q(A, D, M)
        
        eigenvalues_1 = analytical_eigenvalue_1(A, D, M, Q)
        eigenvalues_2 = analytical_eigenvalue_2(A, D, M, Q)
        eigenvalues_3 = analytical_eigenvalue_3(A, D, M, Q)
        eigenvalues_4 = analytical_eigenvalue_4(A, D, M, Q)
        eigenvalues_5 = analytical_eigenvalue_5(A, D, M, Q)
        eigenvalues_6 = analytical_eigenvalue_6(A, D, M, Q)
        
        eigenvalues[i, j, :] = [eigenvalues_1, eigenvalues_2, eigenvalues_3, eigenvalues_4, eigenvalues_5, eigenvalues_6]

# Apply z-limit to the real part of the eigenvalues
eigenvalues_real = np.clip(np.real(eigenvalues), -z_limit, z_limit)

# Plot the eigenvalues in the same 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
stride_size = 3

# Plot all six sets of eigenvalues in the same plot
ax.plot_surface(kx, ky, eigenvalues_real[:, :, 0], cmap='viridis', alpha=0.5, rstride=stride_size, cstride=stride_size)
ax.plot_surface(kx, ky, eigenvalues_real[:, :, 1], cmap='plasma', alpha=0.5, rstride=stride_size, cstride=stride_size)
ax.plot_surface(kx, ky, eigenvalues_real[:, :, 2], cmap='inferno', alpha=0.5, rstride=stride_size, cstride=stride_size)
ax.plot_surface(kx, ky, eigenvalues_real[:, :, 3], cmap='magma', alpha=0.5, rstride=stride_size, cstride=stride_size)
ax.plot_surface(kx, ky, eigenvalues_real[:, :, 4], cmap='cividis', alpha=0.5, rstride=stride_size, cstride=stride_size)
ax.plot_surface(kx, ky, eigenvalues_real[:, :, 5], cmap='coolwarm', alpha=0.5, rstride=stride_size, cstride=stride_size)

ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.set_zlabel('Eigenvalues')
ax.set_zlim(-z_limit, z_limit)
plt.title('Analytical Eigenvalues of the Characteristic Polynomial')
plt.tight_layout()
plt.show()
