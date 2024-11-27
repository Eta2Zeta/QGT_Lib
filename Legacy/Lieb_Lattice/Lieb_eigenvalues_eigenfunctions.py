import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eigenvalue_calc_lib import spiral_eigenvalues_eigenfunctions
from eigenvalue_calc_lib import Eigenvector



# Brillouin zone vectors
a = 1.0  # Lattice constant

# Define parameters
mesh_spacing = 100
k_max = 2* (np.pi / a)  # Maximum k value for the first Brillouin zone
k_dirac_cone = k_max / 2
k_range = 0.1
ky_range = 0.1

# Create kx and ky arrays
kx = np.linspace((1 - k_range) * k_dirac_cone, (1 + k_range) * k_dirac_cone, mesh_spacing)
ky = np.linspace((1 - ky_range) * k_dirac_cone, (1 + ky_range) * k_dirac_cone, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)


def get_spiral_indices(n):
    indices = []
    left, right = 0, n - 1
    top, bottom = 0, n - 1
    spiral_indices_matrix = np.empty((n, n), dtype=object)


    while left <= right and top <= bottom:
        for i in range(left, right + 1):
            indices.append((top, i))
        top += 1

        for i in range(top, bottom + 1):
            indices.append((i, right))
        right -= 1

        for i in range(right, left - 1, -1):
            indices.append((bottom, i))
        bottom -= 1

        for i in range(bottom, top - 1, -1):
            indices.append((i, left))
        left += 1
    
    i_index = 0
    for i in range(n): 
        for j in range (n):
            spiral_indices_matrix[i,j] = indices[i_index]
            i_index += 1

    return spiral_indices_matrix


    
# Hamiltonian for the Lieb lattice
def H_Lieb(kx, ky, a=1):
    return 2 * np.array([
        [0, np.cos(ky * a/2), np.cos(kx * a/2)],
        [np.cos(ky * a/2), 0, 0],
        [np.cos(kx * a/2), 0, 0]
    ])


dim = 3

eigenvalues, eigenfunctions, _, _ = spiral_eigenvalues_eigenfunctions(H_Lieb, kx, ky, mesh_spacing, dim)

# Plot the eigenvalues and eigenfunctions
fig = plt.figure(figsize=(24, 8))

X, Y = kx, ky

for band in range(3):
    # Plot eigenvalues
    ax_eigenvalue = fig.add_subplot(3, 4, band * 4 + 1, projection='3d')
    Z_eigenvalue = eigenvalues[:, :, band]
    ax_eigenvalue.plot_surface(X, Y, Z_eigenvalue, cmap='viridis')
    ax_eigenvalue.set_title(f'Eigenvalue {band + 1}')
    ax_eigenvalue.set_xlabel('kx')
    ax_eigenvalue.set_ylabel('ky')
    ax_eigenvalue.set_zlabel('Eigenvalue')


    # Plot corresponding eigenfunction components
    for component in range(3):
        ax_eigenfunction = fig.add_subplot(3, 4, band * 4 + component + 2, projection='3d')
        Z_eigenfunction = eigenfunctions[:, :, band, component]
        ax_eigenfunction.plot_surface(X, Y, Z_eigenfunction, cmap='viridis')
        ax_eigenfunction.set_title(f'Eigenfunction Component {component + 1}, Band {band + 1}')
        ax_eigenfunction.set_xlabel('kx')
        ax_eigenfunction.set_ylabel('ky')
        # ax_eigenfunction.set_zlabel('Magnitude')

plt.tight_layout()
plt.show()
