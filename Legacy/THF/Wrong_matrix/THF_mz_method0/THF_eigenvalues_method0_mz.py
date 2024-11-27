import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
a = 1.0  # Some constant, adjust as needed
nu_star = 1.0  # Adjust as needed
nu_star_prime = nu_star / 10  # Adjust as needed
nu_star_prime_prime = nu_star_prime / 10  # Adjust as needed
lambda_val = 0.0  # Adjust as needed
M = 1.0  # Adjust as needed
gamma = 1


# Define parameters
mesh_spacing = 200
k_range = 1e-1
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

    # Coefficients of the first cubic polynomial in lambda
    a_cubic_1 = 1
    b_cubic_1 = -np.exp(1j * theta) * M
    c_cubic_1 = (-A**2 * np.exp(2j * theta) - C**2 * np.exp(2j * theta) - 2 * C * D * np.exp(2j * theta) - D**2 * np.exp(2j * theta))
    d_cubic_1 = (C**2 * np.exp(3j * theta) * M + 2 * C * D * np.exp(3j * theta) * M + D**2 * np.exp(3j * theta) * M)

    # Coefficients of the second cubic polynomial in lambda
    a_cubic_2 = 1
    b_cubic_2 = -np.exp(1j * theta) * M
    c_cubic_2 = (-A**2 * np.exp(2j * theta) - C**2 * np.exp(2j * theta) + 2 * C * D * np.exp(2j * theta) - D**2 * np.exp(2j * theta))
    d_cubic_2 = (C**2 * np.exp(3j * theta) * M - 2 * C * D * np.exp(3j * theta) * M + D**2 * np.exp(3j * theta) * M)

    return (a_cubic_1, b_cubic_1, c_cubic_1, d_cubic_1), (a_cubic_2, b_cubic_2, c_cubic_2, d_cubic_2)

# Solve the cubic equation for each kx, ky
def solve_cubic_equation(a, b, c, d):
    coefficients = [a, b, c, d]
    roots = np.roots(coefficients)
    return roots

# Initialize arrays to store eigenvalues
eigenvalues_1 = np.zeros((mesh_spacing, mesh_spacing, 3), dtype=complex)
eigenvalues_2 = np.zeros((mesh_spacing, mesh_spacing, 3), dtype=complex)

for i in range(mesh_spacing):
    for j in range(mesh_spacing):
        k_ij = k[i, j]
        theta_ij = theta[i, j]
        coefficients_1, coefficients_2 = get_coefficients(k_ij, theta_ij)
        
        roots_1 = solve_cubic_equation(*coefficients_1)
        roots_2 = solve_cubic_equation(*coefficients_2)
        
        eigenvalues_1[i, j, :] = roots_1
        eigenvalues_2[i, j, :] = roots_2

# Plot the eigenvalues
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
stride_size = 3

# Plot eigenvalues from the first set of coefficients
ax.plot_surface(kx, ky, np.real(eigenvalues_1[:, :, 0]), cmap='viridis', alpha=0.5, rstride=stride_size, cstride=stride_size)
# ax.plot_surface(kx, ky, np.real(eigenvalues_1[:, :, 1]), cmap='viridis', alpha=0.5, rstride=stride_size, cstride=stride_size)
# ax.plot_surface(kx, ky, np.real(eigenvalues_1[:, :, 2]), cmap='viridis', alpha=0.5, rstride=stride_size, cstride=stride_size)

# Plot eigenvalues from the second set of coefficients
# ax.plot_surface(kx, ky, np.real(eigenvalues_2[:, :, 0]), cmap='plasma', alpha=0.5, rstride=stride_size, cstride=stride_size)
# ax.plot_surface(kx, ky, np.real(eigenvalues_2[:, :, 1]), cmap='plasma', alpha=0.5, rstride=stride_size, cstride=stride_size)
# ax.plot_surface(kx, ky, np.real(eigenvalues_2[:, :, 2]), cmap='plasma', alpha=0.5, rstride=stride_size, cstride=stride_size)

ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.set_zlabel('Eigenvalues')
plt.title('Eigenvalues of the Cubic Equation')
plt.show()
