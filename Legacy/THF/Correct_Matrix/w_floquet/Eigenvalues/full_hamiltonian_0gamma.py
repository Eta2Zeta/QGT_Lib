import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
nu_star = -50  # Adjust as needed
nu_star_prime = 13  # Adjust as needed
lambda_val = 0.0  # Adjust as needed
M = 6  # Adjust as needed
gamma = -25  # Adjust as needed
G = 0.04  # Adjust as needed
z_limit = 150  # Set the z-axis limit

# Define parameters
mesh_spacing = 200
k_range = 1
k_max = k_range * np.pi  # Adjust based on the realistic Brillouin zone
kx = np.linspace(-k_max, k_max, mesh_spacing) / np.pi  # Scale kx by pi
ky = np.linspace(-k_max, k_max, mesh_spacing) / np.pi  # Scale ky by pi
kx, ky = np.meshgrid(kx, ky)
k = np.sqrt((kx * np.pi)**2 + (ky * np.pi)**2)
theta = np.arctan2(ky * np.pi, kx * np.pi)

# Define the coefficients based on k and theta
def get_coefficients(k, theta):
    a = nu_star
    b = nu_star_prime
    A = a * k
    C = np.exp(-k**2 * lambda_val**2 / 2) * gamma
    D = np.exp(-k**2 * lambda_val**2 / 2) * b * k

    # Coefficients of the sixth-order polynomial in T
    c6 = 1
    c5 = 0
    c4 = (-2 * (A**2 + C**2 + D**2) - M**2 - 2 * a**4 * G**2 + 2 * a**2 * b**2 * G**2 - 2 * b**4 * G**2)
    c3 = 0
    c2 = (A**4 + 2 * A**2 * C**2 + C**4 + 2 * A**2 * D**2 - 2 * C**2 * D**2 + D**4 +
          2 * a**4 * A**2 * G**2 - 2 * a**2 * A**2 * b**2 * G**2 + 2 * A**2 * b**4 * G**2 +
          2 * a**4 * C**2 * G**2 + 2 * a**2 * b**2 * C**2 * G**2 - 2 * b**4 * C**2 * G**2 +
          2 * a**4 * D**2 * G**2 - 2 * a**2 * b**2 * D**2 * G**2 + 2 * b**4 * D**2 * G**2 +
          a**8 * G**4 - 2 * a**6 * b**2 * G**4 + 3 * a**4 * b**4 * G**4 - 2 * a**2 * b**6 * G**4 + b**8 * G**4 +
          2 * C**2 * M**2 + 2 * D**2 * M**2 + a**4 * G**2 * M**2 - 2 * a**2 * b**2 * G**2 * M**2 + 2 * b**4 * G**2 * M**2)
    c1 = (-2 * A**2 * C * D * np.exp(-3j * theta) * M - 2 * A**2 * C * D * np.exp(3j * theta) * M)
    c0 = (-A**4 * b**4 * G**2 - 2 * a**2 * A**2 * b**2 * C**2 * G**2 - a**4 * C**4 * G**2 +
          2 * a**2 * A**2 * b**2 * D**2 * G**2 + 2 * a**4 * C**2 * D**2 * G**2 - a**4 * D**4 * G**2 -
          2 * a**4 * A**2 * b**4 * G**4 + 2 * a**2 * A**2 * b**6 * G**4 - 2 * a**6 * b**2 * C**2 * G**4 +
          2 * a**4 * b**4 * C**2 * G**4 + 2 * a**6 * b**2 * D**2 * G**4 - 2 * a**4 * b**4 * D**2 * G**4 -
          a**8 * b**4 * G**6 + 2 * a**6 * b**6 * G**6 - a**4 * b**8 * G**6 - C**4 * M**2 +
          2 * C**2 * D**2 * M**2 - D**4 * M**2 - 2 * a**2 * b**2 * C**2 * G**2 * M**2 +
          2 * b**4 * C**2 * G**2 * M**2 + 2 * a**2 * b**2 * D**2 * G**2 * M**2 -
          2 * b**4 * D**2 * G**2 * M**2 - a**4 * b**4 * G**4 * M**2 + 2 * a**2 * b**6 * G**4 * M**2 - b**8 * G**4 * M**2)

    return [c6, c5, c4, c3, c2, c1, c0]

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
eigenvalues_real = np.clip(np.real(eigenvalues), -z_limit, z_limit)

# Plot the eigenvalues
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
stride_size = 5

# Plot all six sets of eigenvalues in the same plot with different color maps
color_maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm']
for n in range(6):
    ax.plot_surface(kx, ky, eigenvalues_real[:, :, n], cmap=color_maps[n], alpha=0.5, rstride=stride_size, cstride=stride_size)

ax.set_xlabel(r'$k_x/\pi$')
ax.set_ylabel(r'$k_y/\pi$')
ax.set_zlabel('Eigenvalues')
ax.set_zlim(-z_limit, z_limit)
plt.title('Eigenvalues of the Sixth-Order Polynomial Equation')
plt.tight_layout()
plt.show()
