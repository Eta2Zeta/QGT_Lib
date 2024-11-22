import numpy as np
import matplotlib.pyplot as plt

# Constants
nu_star = -4.303  # Adjust as needed
nu_star_prime = 1.622  # Adjust as needed
lambda_val = 0.0  # Adjust as needed
M = 3.2  # Adjust as needed
gamma = -24.75  # Adjust as needed
z_limit = 70  # Set the z-axis limit

# Define parameters
mesh_spacing = 200
k_range = 6
k_max = k_range * np.pi  # Adjust based on the realistic Brillouin zone
kx = np.linspace(-k_max, k_max, mesh_spacing)
ky = np.zeros(mesh_spacing)
k = np.abs(kx)  # Since ky = 0, k = |kx|
theta = np.zeros(mesh_spacing)  # Since ky = 0, theta = 0

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

# Solve the sixth-order polynomial equation for each kx
def solve_sixth_order_polynomial(coefficients):
    roots = np.roots(coefficients)
    return np.sort(roots)

# Initialize array to store eigenvalues
eigenvalues = np.zeros((mesh_spacing, 6), dtype=complex)

for i in range(mesh_spacing):
    k_i = k[i]
    theta_i = theta[i]
    coefficients = get_coefficients(k_i, theta_i)
    roots = solve_sixth_order_polynomial(coefficients)
    eigenvalues[i, :] = roots

# Apply z-limit to the real part of the eigenvalues
eigenvalues_real = np.clip(np.real(eigenvalues), -z_limit, z_limit)

# Plot the eigenvalues in 2D
plt.figure(figsize=(12, 8))
for n in range(6):
    plt.plot(kx, eigenvalues_real[:, n], label=f'Eigenvalue {n+1}')

plt.xlabel('kx')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalues of the Sixth-Order Polynomial Equation for ky = 0')
plt.legend()
plt.grid(True)
plt.show()
