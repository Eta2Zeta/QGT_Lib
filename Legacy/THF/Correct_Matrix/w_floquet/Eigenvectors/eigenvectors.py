import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from eigenvalue_calc_lib import spiral_eigenvalues_eigenfunctions
from QGT_lib import *



# Define parameters
mesh_spacing = 100
dim = 6

diff_para = 1  # How much smaller the dk is compare to the grid size of the k space
k_max = 1 * (np.pi)  # Maximum k value for the first Brillouin zone
k_range = 2e-1
delta_k = k_max / mesh_spacing / diff_para  # Small step for numerical differentiation
z_cutoff = 1e1 #where to cutoff the plot for the z axis when singularties occur

# Create kx and ky arrays

kx = np.linspace(-k_max, k_max, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)


# Define the new Hamiltonian function
def H_THF(kx, ky, nu_star=-50, nu_star_prime=13.0, gamma=-25.0, M=5, G=0.01):
    k = np.sqrt(kx**2 + ky**2)
    theta = np.arctan2(ky, kx)
    
    H_k = np.array([
        [G*(nu_star**2 - nu_star_prime**2), 0, nu_star * k * np.exp(1j * theta), 0, gamma, nu_star_prime * k * np.exp(-1j * theta)],
        [0, -G*(nu_star**2 - nu_star_prime**2), 0, nu_star * k * np.exp(-1j * theta), nu_star_prime * k * np.exp(1j * theta), gamma],
        [nu_star * k * np.exp(-1j * theta), 0, -G * nu_star**2, M, 0, 0],
        [0, nu_star * k * np.exp(1j * theta), M, G * nu_star**2, 0, 0],
        [gamma, nu_star_prime * k * np.exp(-1j * theta), 0, 0, -G * nu_star_prime**2, 0],
        [nu_star_prime * k * np.exp(1j * theta), gamma, 0, 0, 0, G * nu_star_prime**2]
    ])
    
    return H_k

eigenvalues, eigenfunctions, _, _ = spiral_eigenvalues_eigenfunctions(H_THF, kx, ky, mesh_spacing, dim)

# Plot the eigenvalues and eigenfunctions
fig = plt.figure(figsize=(24, 8.5))
X, Y = kx, ky

for band in range(6):
    if band == 0:
        # Plot eigenvalues
        ax_eigenvalue = fig.add_subplot(1, 4, band * 7 + 1, projection='3d')
        Z_eigenvalue = eigenvalues[:, :, band]
        ax_eigenvalue.plot_surface(X, Y, Z_eigenvalue, cmap='viridis')
        ax_eigenvalue.set_title(f'Eigenvalue {band + 1}')
        ax_eigenvalue.set_xlabel('kx')
        ax_eigenvalue.set_ylabel('ky')
        ax_eigenvalue.set_zlabel('Eigenvalue')

        # Plot corresponding eigenfunction components
        for component in range(6):
            if (component == 0) or (component == 1) or (component == 2):
                ax_eigenfunction = fig.add_subplot(1, 4, band * 7 + component + 2, projection='3d')
                Z_eigenfunction = eigenfunctions[:, :, band, component]
                ax_eigenfunction.plot_surface(X, Y, Z_eigenfunction, cmap='viridis')
                ax_eigenfunction.set_title(f'Eigenfunction Component {component + 1}, Band {band + 1}')
                ax_eigenfunction.set_xlabel('kx')
                ax_eigenfunction.set_ylabel('ky')

plt.tight_layout()
plt.show()