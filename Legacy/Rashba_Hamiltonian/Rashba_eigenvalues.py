import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
m = 1.0  # Effective mass, adjust as needed
alpha = 2  # Rashba coupling constant, adjust as needed

# Define parameters
mesh_spacing = 200
k_max = 2* np.pi   # Adjust based on the realistic Brillouin zone
kx = np.linspace(-k_max, k_max, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)
k = np.sqrt(kx**2 + ky**2)

# Eigenvalue calculation for Rashba Hamiltonian
def rashba_eigenvalues(k, m, alpha):
    term1 = (k**2) / (2 * m)
    term2 = 2 * alpha * k
    eigenvalues_positive = (term1 + term2)
    eigenvalues_negative = (term1 - term2) 
    return eigenvalues_positive, eigenvalues_negative

# Compute eigenvalues
positive_eigenvalues_rashba, negative_eigenvalues_rashba = rashba_eigenvalues(k, m, alpha)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot both positive and negative eigenvalues in the same plot
ax.plot_surface(kx, ky, positive_eigenvalues_rashba, cmap='viridis', label='$E_{+}$')
ax.plot_surface(kx, ky, negative_eigenvalues_rashba, cmap='plasma', label='$E_{-}$')

ax.set_title('Rashba Hamiltonian Eigenvalues $E_{\pm}$')
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('$E(k)$')

plt.tight_layout()
plt.show()
plt.close()
