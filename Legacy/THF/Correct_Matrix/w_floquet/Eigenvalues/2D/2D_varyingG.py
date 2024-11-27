import numpy as np
import matplotlib.pyplot as plt

# Constants
nu_star = -50  # Adjust as needed
nu_star_prime = 13  # Adjust as needed
lambda_val = 0.0  # Adjust as needed
M = 3.2  # Adjust as needed
gamma = -25  # Adjust as needed
z_limit = 55  # Set the z-axis limit

# Define parameters
mesh_spacing = 198  # Adjusted to match the range
k_points = {
    'K': (2*np.pi/3, 2*np.pi/(3*np.sqrt(3))),
    'Gamma': (0, 0),
    'M': (2*np.pi/3, 0),
}

# Generate k-path
kx_K_Gamma = np.linspace(k_points['K'][0], k_points['Gamma'][0], mesh_spacing//3)
ky_K_Gamma = np.linspace(k_points['K'][1], k_points['Gamma'][1], mesh_spacing//3)

kx_Gamma_M = np.linspace(k_points['Gamma'][0], k_points['M'][0], mesh_spacing//3)
ky_Gamma_M = np.linspace(k_points['Gamma'][1], k_points['M'][1], mesh_spacing//3)

kx_M_K = np.linspace(k_points['M'][0], k_points['K'][0], mesh_spacing//3)
ky_M_K = np.linspace(k_points['M'][1], k_points['K'][1], mesh_spacing//3)

kx = np.concatenate((kx_K_Gamma, kx_Gamma_M, kx_M_K))
ky = np.concatenate((ky_K_Gamma, ky_Gamma_M, ky_M_K))

k = np.sqrt(kx**2 + ky**2)
theta = np.arctan2(ky, kx)

# Define the coefficients based on k and theta
def get_coefficients(k, theta, G):
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

# Function to plot eigenvalues for a list of G values
def plot_eigenvalues(G_values):
    # Initialize figure for subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))

    # Loop over G values and plot in subplots
    for idx, G in enumerate(G_values):
        row, col = divmod(idx, 3)
        eigenvalues = np.zeros((mesh_spacing, 6), dtype=complex)

        for i in range(mesh_spacing):
            k_i = k[i]
            theta_i = theta[i]
            coefficients = get_coefficients(k_i, theta_i, G)
            roots = solve_sixth_order_polynomial(coefficients)
            eigenvalues[i, :] = roots

        # Apply z-limit to the real part of the eigenvalues
        eigenvalues_real = np.real(eigenvalues)
        eigenvalues_real[np.abs(eigenvalues_real) > z_limit] = np.nan

        # Predefine distance array for x-axis proportional to the actual k-space distances
        d1 = 4 * np.pi / (3 * np.sqrt(3))  # Distance from K to Gamma
        d2 = 2 * np.pi / 3  # Distance from Gamma to M
        d3 = 2 * np.pi / (3 * np.sqrt(3))  # Distance from M to K

        distances = np.zeros(mesh_spacing)
        distances[:mesh_spacing//3] = np.linspace(0, d1, mesh_spacing//3)
        distances[mesh_spacing//3:2*mesh_spacing//3] = np.linspace(d1, d1 + d2, mesh_spacing//3)
        distances[2*mesh_spacing//3:] = np.linspace(d1 + d2, d1 + d2 + d3, mesh_spacing//3)

        ax = axes[row, col]
        for n in range(6):
            ax.plot(distances, eigenvalues_real[:, n], label=f'Eigenvalue {n+1}')

        ax.set_xticks([0, d1, d1 + d2, d1 + d2 + d3])
        ax.set_xticklabels(['K', 'Γ', 'M', 'K'])
        ax.set_ylim(-z_limit, z_limit)
        ax.set_xlabel('Wave Vector Path')
        ax.set_ylabel('Eigenvalues')
        ax.set_title(f'G = {G:.4f}')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

# Custom list of G values
custom_G_values = [0, 0.001, 0.003, 0.006, 0.01, 0.015]

# Call the plotting function with custom G values
plot_eigenvalues(custom_G_values)
