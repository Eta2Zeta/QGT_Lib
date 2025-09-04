import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from QGT_lib import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Brillouin zone vectors
a = 1.0  # Lattice constant
A = -1 / np.sqrt(3)
B = 4 * np.pi / (3 * np.sqrt(3) * a)

# Define parameters
M = 1  # Example value for M
mesh_spacing = 200
diff_para = 1e2 # How much smaller the dk is compare to the grid size of the k space
k_max = 1 * (2 * np.pi / (3 * a))  # Maximum k value for the first Brillouin zone
delta_k = k_max / mesh_spacing / diff_para  # Small step for numerical differentiation

# Create kx and ky arrays
kx = np.linspace(-k_max, k_max, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)

# Define the eigenfunction
def psi(kx, ky, prev_psi = None):
    epsilon = np.sqrt(M**2 + kx**2 + ky**2)
    phi = np.arctan2(ky, kx)
    psi_1 = np.sqrt((epsilon + M) / epsilon)
    psi_2 = np.sqrt((epsilon - M) / epsilon) * np.exp(1j * phi)
    return np.array([psi_1, psi_2]) / np.sqrt(2)


# Identity matrix
I = np.eye(2)


# Analytic solution for g_xy (real part)
def analytic_g_xy_real(kx, ky, M):
    k = np.sqrt(kx**2 + ky**2)
    epsilon = np.sqrt(M**2 + kx**2 + ky**2)
    return -kx * ky / (4  * epsilon**4)

# Analytic solution for g_xx
def analytic_g_xx(kx, ky, M):
    epsilon = np.sqrt(M**2 + kx**2 + ky**2)
    k = np.sqrt(kx**2 + ky**2)
    term1 = (M**2 * kx**2) / (4 * epsilon**4 * k**2)
    term2 = (ky**2 * (epsilon - M)) / (2 * k**4 * epsilon)
    term3 = (ky**2 * (epsilon - M)**2) / (4 * k**4 * epsilon**2)
    return term1 + term2 - term3

# Analytic solution for g_xy (imaginary part)
def analytic_g_xy_imag(kx, ky, M):
    epsilon = np.sqrt(M**2 + kx**2 + ky**2)
    return M / (4*epsilon**3)

# Function to check if a point is within the hexagonal Brillouin zone
def in_brillouin_zone(kx, ky):
    return True
    return (np.abs(kx) <= 2 * np.pi / (3 * a)) & (np.abs(ky) <= A * np.abs(kx) + B)


# Initialize arrays to store tensor components
g_xx_array = np.zeros(kx.shape)
g_xy_real_array = np.zeros(kx.shape)
g_xy_imag_array = np.zeros(kx.shape)
g_yy_array = np.zeros(kx.shape)
analytic_g_xx_array = np.zeros(kx.shape)
analytic_g_xy_real_array = np.zeros(kx.shape)
analytic_g_xy_imag_array = np.zeros(kx.shape)

# Calculate tensor components for each point in the Brillouin zone
for i in range(kx.shape[0]):
    for j in range(kx.shape[1]):
        if in_brillouin_zone(kx[i, j], ky[i, j]):
            g_xx, g_xy_real, g_xy_imag, g_yy = quantum_geometric_tensor_semi_num(psi, I, kx[i, j], ky[i, j], delta_k,)
            g_xx_array[i, j] = g_xx
            g_xy_real_array[i, j] = g_xy_real
            g_xy_imag_array[i, j] = g_xy_imag
            g_yy_array[i, j] = g_yy
            analytic_g_xx_array[i, j] = analytic_g_xx(kx[i, j], ky[i, j], M)
            analytic_g_xy_real_array[i, j] = analytic_g_xy_real(kx[i, j], ky[i, j], M)
            analytic_g_xy_imag_array[i, j] = analytic_g_xy_imag(kx[i, j], ky[i, j], M)
        else:
            g_xx_array[i, j] = np.nan
            g_xy_real_array[i, j] = np.nan
            g_xy_imag_array[i, j] = np.nan
            g_yy_array[i, j] = np.nan
            analytic_g_xx_array[i, j] = np.nan
            analytic_g_xy_real_array[i, j] = np.nan
            analytic_g_xy_imag_array[i, j] = np.nan

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(10, 8))

# Plot g_xx (Numerical)
ax1 = axes[0, 0]
c1 = ax1.imshow(g_xx_array, extent=(-k_max, k_max, -k_max, k_max), origin='lower', cmap='viridis')
ax1.set_title('$g_{xx}$')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
fig.colorbar(c1, ax=ax1)

# Plot g_xy_real (Numerical)
ax2 = axes[0, 1]
c2 = ax2.imshow(g_xy_real_array, extent=(-k_max, k_max, -k_max, k_max), origin='lower', cmap='plasma')
ax2.set_title('$g_{xy}$ (Real part)')
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
fig.colorbar(c2, ax=ax2)

# Plot g_xy_imag (Numerical)
ax3 = axes[0, 2]
c3 = ax3.imshow(g_xy_imag_array, extent=(-k_max, k_max, -k_max, k_max), origin='lower', cmap='inferno')
ax3.set_title('$g_{xy}$ (Imag part)')
ax3.set_xlabel('kx')
ax3.set_ylabel('ky')
fig.colorbar(c3, ax=ax3)

# Plot g_xx (Analytical)
ax4 = axes[1, 0]
c4 = ax4.imshow(analytic_g_xx_array, extent=(-k_max, k_max, -k_max, k_max), origin='lower', cmap='viridis')
ax4.set_title('$g_{xx}$ (Analytical)')
ax4.set_xlabel('kx')
ax4.set_ylabel('ky')
fig.colorbar(c4, ax=ax4)

# Plot g_xy_real (Analytical)
ax5 = axes[1, 1]
c5 = ax5.imshow(analytic_g_xy_real_array, extent=(-k_max, k_max, -k_max, k_max), origin='lower', cmap='plasma')
ax5.set_title('$g_{xy}$ (Analytical, real part)')
ax5.set_xlabel('kx')
ax5.set_ylabel('ky')
fig.colorbar(c5, ax=ax5)

# Plot g_xy_imag (Analytical)
ax6 = axes[1, 2]
c6 = ax6.imshow(analytic_g_xy_imag_array, extent=(-k_max, k_max, -k_max, k_max), origin='lower', cmap='inferno')
ax6.set_title('$g_{xy}$ (Analytical, imag part)')
ax6.set_xlabel('kx')
ax6.set_ylabel('ky')
fig.colorbar(c6, ax=ax6)

plt.tight_layout()
plt.show()
plt.close()


# Plotting
fig = plt.figure(figsize=(12, 8))

# Plot g_xy_real
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(kx, ky, g_xx_array, cmap='plasma')
ax1.set_title('Numerical $g_{xx}$ (real part)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xx}$')

# Plot analytic g_xy_real
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(kx, ky, analytic_g_xx_array, cmap='viridis')
ax2.set_title('Analytic $g_{xx}$ (real part)')
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
ax2.set_zlabel('$g_{xx}$')


plt.tight_layout()
# plt.show()
plt.close()

# Plotting
fig = plt.figure(figsize=(12, 8))

# Plot g_xy_real
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(kx, ky, g_xy_real_array, cmap='plasma')
ax1.set_title('Numerical $g_{xy}$ (real part)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xy}$ (real)')

# Plot analytic g_xy_real
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(kx, ky, analytic_g_xy_real_array, cmap='viridis')
ax2.set_title('Analytic $g_{xy}$ (real part)')
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
ax2.set_zlabel('$g_{xy}$ (real)')

plt.tight_layout()
# plt.show()
plt.close()

# Plotting
fig = plt.figure(figsize=(12, 8))
# Plot numerical g_xx
ax1 = fig.add_subplot(231, projection='3d')
ax1.plot_surface(kx, ky, g_xx_array, cmap='viridis')
ax1.set_title('Numerical $g_{xx}$')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xx}$')

# Plot numerical g_xy_real
ax2 = fig.add_subplot(232, projection='3d')
ax2.plot_surface(kx, ky, g_xy_real_array, cmap='plasma')
ax2.set_title('Numerical $g_{xy}$ (real part)')
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
ax2.set_zlabel('$g_{xy}$ (real)')

# Plot numerical g_xy_imag
ax3 = fig.add_subplot(233, projection='3d')
ax3.plot_surface(kx, ky, g_xy_imag_array, cmap='inferno')
ax3.set_title('Numerical $g_{xy}$ (imaginary part)')
ax3.set_xlabel('kx')
ax3.set_ylabel('ky')
ax3.set_zlabel('$g_{xy}$ (imag)')

# Plot analytic g_xx
ax4 = fig.add_subplot(234, projection='3d')
ax4.plot_surface(kx, ky, analytic_g_xx_array, cmap='viridis')
ax4.set_title('Analytic $g_{xx}$')
ax4.set_xlabel('kx')
ax4.set_ylabel('ky')
ax4.set_zlabel('$g_{xx}$')

# Plot analytic g_xy_real
ax5 = fig.add_subplot(235, projection='3d')
ax5.plot_surface(kx, ky, analytic_g_xy_real_array, cmap='plasma')
ax5.set_title('Analytic $g_{xy}$ (real part)')
ax5.set_xlabel('kx')
ax5.set_ylabel('ky')
ax5.set_zlabel('$g_{xy}$ (real)')

# Plot analytic g_xy_imag
ax6 = fig.add_subplot(236, projection='3d')
ax6.plot_surface(kx, ky, analytic_g_xy_imag_array, cmap='inferno')
ax6.set_title('Analytic $g_{xy}$ (imaginary part)')
ax6.set_xlabel('kx')
ax6.set_ylabel('ky')
ax6.set_zlabel('$g_{xy}$ (imag)')

plt.tight_layout()
plt.show()
plt.close()



# Plotting
fig = plt.figure(figsize=(10, 8))

# Plot g_xx
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(kx, ky, g_xx_array, cmap='viridis')
ax1.set_title('g_{xx}')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('g_{xx}')

# Plot g_xy_real
ax2 = fig.add_subplot(222, projection='3d')
ax2.plot_surface(kx, ky, g_xy_real_array, cmap='plasma')
ax2.set_title('g_{xy} (real part)')
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
ax2.set_zlabel('g_{xy} (real)')

# Plot g_xy_imag
ax3 = fig.add_subplot(223, projection='3d')
ax3.plot_surface(kx, ky, g_xy_imag_array, cmap='plasma')
ax3.set_title('g_{xy} (imaginary part)')
ax3.set_xlabel('kx')
ax3.set_ylabel('ky')
ax3.set_zlabel('g_{xy} (imag)')

# Plot g_yy
ax4 = fig.add_subplot(224, projection='3d')
ax4.plot_surface(kx, ky, g_yy_array, cmap='inferno')
ax4.set_title('g_{yy}')
ax4.set_xlabel('kx')
ax4.set_ylabel('ky')
ax4.set_zlabel('g_{yy}')

plt.tight_layout()
# plt.show()
