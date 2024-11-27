import numpy as np
import matplotlib.pyplot as plt

# Constants
nu_star = -50  # Adjust as needed
nu_star_prime = 13  # Adjust as needed
lambda_val = 0.0  # Adjust as needed
M = 3.2  # Adjust as needed
gamma = -25  # Adjust as needed
G = 0.01  # Adjust as needed
z_limit = 50  # Set the z-axis limit

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

# Solve the sixth-order polynomial equation for each k
def solve_sixth_order_polynomial(coefficients):
    roots = np.roots(coefficients)
    return np.sort(roots)

# Function to calculate and plot band structure along an arbitrary line
def plot_band_structure(angle, k_max=2*np.pi, mesh_spacing=198):
    # Calculate kx and ky along the specified direction
    kx = np.linspace(-k_max, k_max, mesh_spacing) * np.cos(angle)
    ky = np.linspace(-k_max, k_max, mesh_spacing) * np.sin(angle)

    k = np.sqrt(kx**2 + ky**2)
    theta = np.arctan2(ky, kx)

    # Initialize array to store eigenvalues
    eigenvalues = np.zeros((mesh_spacing, 6), dtype=complex)

    for i in range(mesh_spacing):
        k_i = k[i]
        theta_i = theta[i]
        coefficients = get_coefficients(k_i, theta_i)
        roots = solve_sixth_order_polynomial(coefficients)
        eigenvalues[i, :] = roots

    # Apply z-limit to the real part of the eigenvalues
    eigenvalues_real = np.real(eigenvalues)
    eigenvalues_real[np.abs(eigenvalues_real) > z_limit] = np.nan

    # Plot the eigenvalues along the line
    plt.figure(figsize=(8, 8))
    for n in range(6):
        plt.plot(kx, eigenvalues_real[:, n], label=f'Eigenvalue {n+1}')

    plt.xlabel('kx')
    plt.ylabel('Eigenvalues')
    plt.title(f'Eigenvalues along kx-ky line at angle {angle} radians')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage: plot band structure along a line at 45 degrees
angle = np.pi / 6  # 45 degrees
plot_band_structure(angle)
