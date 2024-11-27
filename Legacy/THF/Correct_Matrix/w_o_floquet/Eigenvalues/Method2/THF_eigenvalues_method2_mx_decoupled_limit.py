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
z_limit = 6  # Set the z-axis limit

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

# Analytical solutions for eigenvalues
def analytical_eigenvalues(A, C, D, M):
    eigenvalue_1 = 0
    eigenvalue_2 = 0
    eigenvalue_3 = 0.5 * (-M - np.sqrt(4 * A**2 + M**2))
    eigenvalue_4 = 0.5 * (M - np.sqrt(4 * A**2 + M**2))
    eigenvalue_5 = 0.5 * (-M + np.sqrt(4 * A**2 + M**2))
    eigenvalue_6 = 0.5 * (M + np.sqrt(4 * A**2 + M**2))

    return eigenvalue_1, eigenvalue_2, eigenvalue_3, eigenvalue_4, eigenvalue_5, eigenvalue_6

# Initialize arrays to store eigenvalues
eigenvalues = np.zeros((mesh_spacing, mesh_spacing, 6), dtype=complex)

for i in range(mesh_spacing):
    for j in range(mesh_spacing):
        k_ij = k[i, j]
        theta_ij = theta[i, j]
        A, C, D = get_coefficients(k_ij, theta_ij)
        
        ev1, ev2, ev3, ev4, ev5, ev6 = analytical_eigenvalues(A, C, D, M)
        eigenvalues[i, j, :] = [ev1, ev2, ev3, ev4, ev5, ev6]

# Apply z-limit to the real part of the eigenvalues
eigenvalues_real = np.clip(np.real(eigenvalues), -z_limit, z_limit)

# Plot the eigenvalues in the same 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
stride_size = 4

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
