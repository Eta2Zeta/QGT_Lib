import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
m = 1.0  # Effective mass, adjust as needed
alpha = 5  # Rashba coupling constant, adjust as needed
v = 1.0  # Velocity, adjust as needed
e = 1.0  # Charge, adjust as needed
A0 = 1.0  # Amplitude of the Floquet drive, adjust as needed
omega = 1.0  # Frequency of the Floquet drive, adjust as needed
z_limit = 70  # Set the z-axis limit

# Define parameters
mesh_spacing = 200
k_max = 2 * np.pi   # Adjust based on the realistic Brillouin zone
kx = np.linspace(-k_max, k_max, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)
k = np.sqrt(kx**2 + ky**2)

# Eigenvalue calculation for Rashba Hamiltonian with Floquet terms
def rashba_floquet_eigenvalues(k, m, alpha, v, e, A0, omega):
    term1 = (k**2) / (2 * m)
    term2 = np.sqrt((v * e * A0)**4 / omega**2 + alpha**2 * k**2)
    eigenvalues_positive = term1 + term2
    eigenvalues_negative = term1 - term2
    return eigenvalues_positive, eigenvalues_negative

# Compute eigenvalues
positive_eigenvalues_rashba, negative_eigenvalues_rashba = rashba_floquet_eigenvalues(k, m, alpha, v, e, A0, omega)

# Apply z-limit: Set values exceeding z_limit to NaN
positive_eigenvalues_rashba = np.where(positive_eigenvalues_rashba > z_limit, np.nan, positive_eigenvalues_rashba)
negative_eigenvalues_rashba = np.where(negative_eigenvalues_rashba > z_limit, np.nan, negative_eigenvalues_rashba)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

stride_size = 2

# Plot both positive and negative eigenvalues in the same plot
ax.plot_surface(kx, ky, positive_eigenvalues_rashba, cmap='viridis', label='$E_{+}$', rstride=stride_size, cstride=stride_size)
ax.plot_surface(kx, ky, negative_eigenvalues_rashba, cmap='plasma', label='$E_{-}$', rstride=stride_size, cstride=stride_size)

ax.set_title('Rashba Eigenvalues with Floquet Terms $E_{\pm}$')
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('$E(k)$')
ax.set_zlim(-20, z_limit)

plt.tight_layout()
plt.show()
plt.close()
