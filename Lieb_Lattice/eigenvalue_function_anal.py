import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Brillouin zone vectors
a = 1.0  # Lattice constant
t = 1.0  # Hopping parameter

# Define parameters
mesh_spacing = 100
k_max = 2* (np.pi / a)  # Maximum k value for the first Brillouin zone
k_dirac_cone = k_max / 2
k_range = 0.1

# Create kx and ky arrays
kx = np.linspace((1 - k_range) * k_dirac_cone, (1 + k_range) * k_dirac_cone, mesh_spacing)
ky = np.linspace((1 - k_range) * k_dirac_cone, (1 + k_range) * k_dirac_cone, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)

# Analytical eigenvalues
def analytical_eigenvalues(kx, ky):
    term = 4 + 2 * (np.cos(kx) + np.cos(ky))
    return np.array([t * np.sqrt(term), 0, -t * np.sqrt(term)])

# Analytical eigenfunctions
def analytical_eigenfunctions(kx, ky):
    denom0 = np.sqrt(np.cos(kx / 2)**2 + np.cos(ky / 2)**2)
    U0 = np.array([0, -np.cos(kx / 2), np.cos(ky / 2)]) / denom0
    
    term = np.sqrt(2 + np.cos(kx) + np.cos(ky))
    U_plus = np.array([1 / np.sqrt(2), np.cos(ky / 2) / term, np.cos(kx / 2) / term])
    U_minus = np.array([-1 / np.sqrt(2), np.cos(ky / 2) / term, np.cos(kx / 2) / term])
    
    return np.array([U_plus, U0, U_minus])

# Initialize arrays to store eigenfunctions and eigenvalues
eigenfunctions = np.zeros((mesh_spacing, mesh_spacing, 3, 3), dtype=complex)
eigenvalues = np.zeros((mesh_spacing, mesh_spacing, 3), dtype=float)

# Calculate eigenfunctions and eigenvalues for each point in the Brillouin zone
for i in range(mesh_spacing):
    for j in range(mesh_spacing):
        eigenvalue = analytical_eigenvalues(kx[i, j], ky[i, j])
        eigenvalues[i, j] = eigenvalue
        eigenvector = analytical_eigenfunctions(kx[i, j], ky[i, j])
        eigenfunctions[i, j] = eigenvector

# Plot the eigenvalues and eigenfunctions
fig = plt.figure(figsize=(24, 8))

X, Y = kx, ky

for band in range(3):
    ax_eigenvalue = fig.add_subplot(3, 4, band * 4 + 1, projection='3d')
    Z_eigenvalue = eigenvalues[:, :, band]
    ax_eigenvalue.plot_surface(X, Y, Z_eigenvalue, cmap='viridis')
    ax_eigenvalue.set_title(f'Eigenvalue {band + 1}')
    ax_eigenvalue.set_xlabel('kx')
    ax_eigenvalue.set_ylabel('ky')
    ax_eigenvalue.set_zlabel('Eigenvalue')

for band in range(3):
    for component in range(3):
        ax_eigenfunction = fig.add_subplot(3, 4, band * 4 + component + 2, projection='3d')
        Z_eigenfunction = eigenfunctions[:, :, band, component]  # Plot the magnitude of the component of the eigenfunction
        ax_eigenfunction.plot_surface(X, Y, Z_eigenfunction, cmap='viridis')
        ax_eigenfunction.set_title(f'Eigenfunction Component {component + 1}, Band {band + 1}')
        ax_eigenfunction.set_xlabel('kx')
        ax_eigenfunction.set_ylabel('ky')
        ax_eigenfunction.set_zlabel('Magnitude')

plt.tight_layout()
plt.show()
