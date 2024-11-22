import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
a = 1.0  # Lattice constant, you can adjust as per actual parameters
t = 2.7  # hopping parameter in eV, typical for graphene
M = 0.1  # Mass term, adjust to vary the size of the bandgap


# Define parameters
mesh_spacing = 200
k_max = 1.5* 2 * np.pi / (3 * a)  # Adjust based on the realistic Brillouin zone
kx = np.linspace(-k_max, k_max, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)

# Eigenvalue calculation
def graphene_eigenvalues(kx, ky, a, t):
    term1 = 1 + 4 * np.cos(3/2 * kx * a) * np.cos(np.sqrt(3)/2 * ky * a)
    term2 = 4 * np.cos(np.sqrt(3)/2 * ky * a)**2
    eigenvalues = t * np.sqrt(term1 + term2)
    return eigenvalues

# Eigenvalue calculation for bandgap model
def bandgap_eigenvalues(kx, ky, M):
    eigenvalues = np.sqrt(kx**2 + ky**2 + M**2)
    return eigenvalues

# Compute eigenvalues
eigenvalues = graphene_eigenvalues(kx, ky, a, t)
positive_eigenvalues = eigenvalues
negative_eigenvalues = -eigenvalues

# Compute eigenvalues with bandgap
eigenvalues_with_gap = bandgap_eigenvalues(kx, ky, M)
positive_eigenvalues_with_gap = eigenvalues_with_gap
negative_eigenvalues_with_gap = -eigenvalues_with_gap


# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot both positive and negative eigenvalues in the same plot
ax.plot_surface(kx, ky, positive_eigenvalues, cmap='viridis', label='$E_+$')
ax.plot_surface(kx, ky, negative_eigenvalues, cmap='plasma', label='$E_-$')

ax.set_title('Graphene Eigenvalues $E_{\pm}$')
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('$E(k)$')

plt.tight_layout()
plt.show()
plt.close()


# Set the background to dark
plt.style.use('dark_background')

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot both positive and negative eigenvalues in the same plot
ax.plot_surface(kx, ky, positive_eigenvalues_with_gap, cmap='viridis', label='$E_{M,+}$')
ax.plot_surface(kx, ky, negative_eigenvalues_with_gap, cmap='plasma', label='$E_{M,-}$')

ax.set_title('Energy Bands with Bandgap $E_{M, \pm}$')
ax.set_xlabel('$q_x$')
ax.set_ylabel('$q_y$')
ax.set_zlabel('$E_M(q)$')

plt.tight_layout()
plt.show()
