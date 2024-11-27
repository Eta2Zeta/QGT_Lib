import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eigenvalue_calc_lib import spiral_eigenvalues_eigenfunctions
from QGT_lib import *
from Hamiltonian_v1 import H_THF



# Define parameters
mesh_spacing = 200
dim = 6

k_max = 1 * (np.pi)  # Maximum k value for the first Brillouin zone
delta_k = k_max / mesh_spacing # Small step for numerical differentiation
z_cutoff = 1e1 #where to cutoff the plot for the z axis when singularties occur

# Create kx and ky arrays
kx = np.linspace(-k_max, k_max, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)

# File paths for loading the data
eigenvalues_file = "eigenvalues.npy"
eigenfunctions_file = "eigenfunctions.npy"

# Load the eigenvalues and eigenfunctions from files
if os.path.exists(eigenvalues_file) and os.path.exists(eigenfunctions_file):
    eigenvalues = np.load(eigenvalues_file)
    eigenfunctions = np.load(eigenfunctions_file)
    print("Loaded eigenvalues and eigenfunctions from files.")
else:
    print("Eigenvalues or eigenfunctions files not found. Please ensure they are available at the specified paths.")
    sys.exit(1)

# Initialize arrays to store tensor components
g_xx_array = np.zeros(kx.shape)
g_xy_real_array = np.zeros(kx.shape)
g_xy_imag_array = np.zeros(kx.shape)
g_yy_array = np.zeros(kx.shape)

trace_array = np.zeros(kx.shape)

band = 3

# Calculate tensor components for each point in the Brillouin zone
for i in range(kx.shape[0]):
    for j in range(kx.shape[1]):
        # Get the eigenfunction at the spot of i and j
        eigenfunction = eigenfunctions[i,j]
        eigenvalue = eigenvalues[i,j]
        g_xx, g_xy_real, g_xy_imag, g_yy = quantum_geometric_tensor_num(H_THF, kx[i, j], ky[i, j], delta_k, eigenvalue, eigenfunction, dim, band_index=band)
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

def replace_zeros_with_nan(Z):
    Z_nan = np.where(Z == 0, np.nan, Z)
    return Z_nan
X, Y = kx, ky
stride_size = 3

# Plot eigenfunction components for each band
for band in range(6):

    # if (band == 0) or (band == 2):
        fig, axes = plt.subplots(2, 3, figsize=(18, 8), subplot_kw={'projection': '3d'})
        fig.suptitle(f'Band {band + 1} Eigenfunction Components', fontsize=16)

        for component in range(6):
            row = component // 3
            col = component % 3
            ax = axes[row, col]
            
            Z_eigenfunction = eigenfunctions[:, :, band, component].flatten()
            Z_eigenfunction = replace_zeros_with_nan(Z_eigenfunction)  # Replace zeros with NaN

            # Use the magnitude of Z_eigenfunction for color mapping
            Z_magnitude = np.real(Z_eigenfunction)

            # Use ax.scatter for 3D scatter plot
            sc = ax.scatter(X, Y, Z_magnitude, c=Z_magnitude, cmap='viridis', s=3)

            ax.set_title(f'Component {component + 1}')
            ax.set_xlabel('kx')
            ax.set_ylabel('ky')
            ax.set_zlabel('Magnitude')

            # Add a color bar
            fig.colorbar(sc, ax=ax, shrink=0.6, aspect=5)

        plt.tight_layout()
        # plt.show()
        plt.close()

# Plotting
fig = plt.figure(figsize=(12, 8))
stride_size = 1


# Plot g_xy_real
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(kx, ky, g_xx_array, cmap='plasma', rstride=stride_size, cstride=stride_size, antialiased=True)
ax1.set_title('Numerical $g_{xx}$ (real part)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xx}$')


# Plot g_xy_real
ax1 = fig.add_subplot(132, projection='3d')
ax1.plot_surface(kx, ky, g_yy_array, cmap='plasma', rstride=stride_size, cstride=stride_size)
ax1.set_title('Numerical $g_{yy}$ (real part)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xx}$')

# Plot g_xy_real
ax1 = fig.add_subplot(133, projection='3d')
ax1.plot_surface(kx, ky, trace_array, cmap='plasma', rstride=stride_size, cstride=stride_size)
ax1.set_title('Numerical trace (real part)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xx}$')


plt.tight_layout()
plt.show()
plt.close()




# Plotting Num vs Anal of gxx, gxy
fig = plt.figure(figsize=(12, 9))
# Plot numerical g_xx
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(kx, ky, g_xx_array, cmap='viridis')
ax1.set_title('Numerical $g_{xx}$')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xx}$')

# Plot numerical g_xy_real
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(kx, ky, g_xy_real_array, cmap='plasma')
ax2.set_title('Numerical $g_{xy}$ (real part)')
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
ax2.set_zlabel('$g_{xy}$ (real)')

# Plot numerical g_xy_imag
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(kx, ky, g_yy_array, cmap='inferno')
ax3.set_title('Numerical $g_{yy}$')
ax3.set_xlabel('kx')
ax3.set_ylabel('ky')
ax3.set_zlabel('$g_{yy}$')


plt.tight_layout()
plt.show()
plt.close()



# Plotting
fig, axes = plt.subplots(1, 3, figsize=(14, 8))

# Plot g_xx
ax1 = axes[0]
c1 = ax1.imshow(g_xx_array, extent=(-k_max, k_max, -k_max, k_max), origin='lower', cmap='viridis')
ax1.set_title('g_{xx} (Numerical)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
fig.colorbar(c1, ax=ax1)

# Plot g_xy_real
ax2 = axes[1]
c2 = ax2.imshow(g_yy_array, extent=(-k_max, k_max, -k_max, k_max), origin='lower', cmap='plasma')
ax2.set_title('g_{yy} (Numerical)')
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
fig.colorbar(c2, ax=ax2)

# Plot both positive and negative eigenvalues
ax3 = axes[2]
c3 = ax3.imshow(trace_array, extent=(-k_max, k_max, -k_max, k_max), origin='lower', cmap='plasma')
ax3.set_title('Trace (Numerical)')
ax3.set_xlabel('kx')
ax3.set_ylabel('ky')
fig.colorbar(c3, ax=ax3)


plt.tight_layout()
plt.show()
plt.close()