import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
nu_star = -50  # Adjust as needed
nu_star_prime = 13  # Adjust as needed
lambda_val = 0.0  # Adjust as needed
M = 3.2  # Adjust as needed
gamma = -25  # Adjust as needed
z_limit = 70  # Set the z-axis limit

# Define parameters
mesh_spacing = 200
k_range = 1
k_max = k_range * np.pi  # Adjust based on the realistic Brillouin zone
kx = np.linspace(-k_max, k_max, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)
k = np.sqrt(kx**2 + ky**2)
theta = np.arctan2(ky, kx)

# Define the coefficients based on k and theta
def get_coefficients(k, theta):
    C = np.exp(-k**2 * lambda_val**2 / 2) * gamma
    D = np.exp(-k**2 * lambda_val**2 / 2) * nu_star_prime * k
    A = nu_star * k

    # Coefficients of the sixth-order polynomial in T
    c6 = 1
    c4 = -2 * (A**2 + C**2 + D**2) - M**2
    c2 = A**4 + 2 * A**2 * C**2 + C**4 + 2 * A**2 * D**2 - 2 * C**2 * D**2 + D**4 + 2 * C**2 * M**2 + 2 * D**2 * M**2
    c1 = -2 * A**2 * C * D * (np.exp(-3j * theta) + np.exp(3j * theta)) * M
    c0 = -C**4 * M**2 + 2 * C**2 * D**2 * M**2 - D**4 * M**2

    return [c6, 0, c4, 0, c2, c1, c0]

# Solve the sixth-order polynomial equation for each kx, ky
def solve_sixth_order_polynomial(coefficients):
    roots = np.roots(coefficients)
    return np.sort(roots)

# Initialize array to store eigenvalues
eigenvalues = np.zeros((mesh_spacing, mesh_spacing, 6), dtype=complex)

for i in range(mesh_spacing):
    for j in range(mesh_spacing):
        k_ij = k[i, j]
        theta_ij = theta[i, j]
        coefficients = get_coefficients(k_ij, theta_ij)
        roots = solve_sixth_order_polynomial(coefficients)
        eigenvalues[i, j, :] = roots

# Apply z-limit to the real part of the eigenvalues
eigenvalues_real = np.real(eigenvalues)
eigenvalues_real[np.abs(eigenvalues_real) > z_limit] = np.nan

# Plot the eigenvalues
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
stride_size = 3

# Plot all six sets of eigenvalues in the same plot with different color maps
color_maps = ['viridis', 'magma', 'plasma', 'inferno', 'cividis', 'coolwarm']
for n in range(6):
    ax.plot_surface(kx, ky, eigenvalues_real[:, :, n], cmap=color_maps[n], alpha=0.5, rstride=stride_size, cstride=stride_size)

ax.set_xlabel(r'$k_x/\pi$')
ax.set_ylabel(r'$k_y/\pi$')
ax.set_zlabel('Eigenvalues')
ax.set_zlim(-z_limit, z_limit)
plt.title('Eigenvalues of the Sixth-Order Polynomial Equation')
plt.tight_layout()
plt.show()
