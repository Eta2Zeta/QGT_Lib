import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from QMT_lib import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Define parameters
mesh_spacing = 100
diff_para = 1e2  # How much smaller the dk is compare to the grid size of the k space
k_max = (np.pi)  # Maximum k value for the first Brillouin zone
k_dirac_cone = k_max
k_range = 2e-1
delta_k = k_max / mesh_spacing / diff_para  # Small step for numerical differentiation
z_cutoff = 1e1 #where to cutoff the plot for the z axis when singularties occur

# Create kx and ky arrays
kx = np.linspace((1-k_range)*k_dirac_cone, (1+k_range)*k_dirac_cone, mesh_spacing)
ky = np.linspace((1-k_range)*k_dirac_cone, (1+k_range)*k_dirac_cone, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)


# Define the eigenfunction for the Lieb lattice
def psi(kx, ky, prev_psi=None):
    norm = np.sqrt(np.cos(kx/2)**2 + np.cos(ky/2)**2)
    psi_1 = 0
    psi_2 = -np.cos(kx/2) / norm
    psi_3 = np.cos(ky/2) / norm
    return np.array([psi_1, psi_2, psi_3])


# Identity matrix
I = np.eye(3)

# Initialize arrays to store tensor components
g_xx_array = np.zeros(kx.shape)
g_xy_real_array = np.zeros(kx.shape)
g_xy_imag_array = np.zeros(kx.shape)
g_yy_array = np.zeros(kx.shape)

trace_array = np.zeros(kx.shape)


# Calculate tensor components for each point in the Brillouin zone
for i in range(kx.shape[0]):
    for j in range(kx.shape[1]):
        g_xx, g_xy_real, g_xy_imag, g_yy = quantum_geometric_tensor(psi, I, kx[i, j], ky[i, j], delta_k)
        g_xx_array[i, j] = g_xx
        g_xy_real_array[i, j] = g_xy_real
        g_yy_array[i, j] = g_yy
        trace_array[i,j] = g_xx + g_yy
        
# Apply the cutoff using np.clip in one line
g_xx_array = np.clip(g_xx_array, None, z_cutoff)
g_xy_real_array = np.clip(g_xy_real_array, None, z_cutoff)
g_xy_imag_array = np.clip(g_xy_imag_array, None, z_cutoff)
g_yy_array = np.clip(g_yy_array, None, z_cutoff)
trace_array = np.clip(trace_array, None, z_cutoff)

# Plotting
fig = plt.figure(figsize=(12, 8))

# Plot g_xy_real
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(kx, ky, g_xx_array, cmap='plasma', rstride=1, cstride=1, antialiased=True)
ax1.set_title('Numerical $g_{xx}$ (real part)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xx}$')


plt.tight_layout()
plt.show()
plt.close()

# Plotting
fig = plt.figure(figsize=(12, 8))

# Plot g_xy_real
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(kx, ky, g_yy_array, cmap='plasma')
ax1.set_title('Numerical $g_{yy}$ (real part)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xx}$')


plt.tight_layout()
plt.show()
plt.close()


# Plotting
fig = plt.figure(figsize=(12, 8))

# Plot g_xy_real
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(kx, ky, trace_array, cmap='plasma')
ax1.set_title('Numerical trace (real part)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xx}$')


plt.tight_layout()
plt.show()
plt.close()




# Plotting Num vs Anal of gxx, gxy
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
ax3.plot_surface(kx, ky, g_yy_array, cmap='inferno')
ax3.set_title('Numerical $g_{yy}$')
ax3.set_xlabel('kx')
ax3.set_ylabel('ky')
ax3.set_zlabel('$g_{yy}$')


plt.tight_layout()
# plt.show()
plt.close()



# Plotting numerical results in 3d
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
plt.close()

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(10, 8))

# Plot g_xx
ax1 = axes[0]
c1 = ax1.imshow(g_xx_array, extent=(0, k_max, 0, k_max), origin='lower', cmap='viridis')
ax1.set_title('g_{xx} (Numerical)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
fig.colorbar(c1, ax=ax1)

# Plot g_xy_real
ax2 = axes[1]
c2 = ax2.imshow(g_yy_array, extent=(0, k_max, 0, k_max), origin='lower', cmap='plasma')
ax2.set_title('g_{yy} (Numerical)')
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
fig.colorbar(c2, ax=ax2)

# Plot both positive and negative eigenvalues
ax3 = axes[2]
c3 = ax3.imshow(trace_array, extent=(0, k_max, 0, k_max), origin='lower', cmap='plasma')
ax3.set_title('Trace (Numerical)')
ax3.set_xlabel('kx')
ax3.set_ylabel('ky')
fig.colorbar(c3, ax=ax3)


plt.tight_layout()
plt.show()
plt.close()