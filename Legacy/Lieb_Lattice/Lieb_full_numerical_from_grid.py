import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from QGT_lib import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from eigenvalue_calc_lib import spiral_eigenvalues_eigenfunctions



# Define parameters
mesh_spacing = 150

k_max = (np.pi)  # Maximum k value for the first Brillouin zone
k_dirac_cone = k_max
k_range = 0.4
delta_k = k_max / mesh_spacing  # Small step for numerical differentiation
z_cutoff = 1e1 #where to cutoff the plot for the z axis when singularties occur

# Create kx and ky arrays
kx = np.linspace((1-k_range)*k_dirac_cone, (1+k_range)*k_dirac_cone, mesh_spacing)
ky = np.linspace((1-k_range)*k_dirac_cone, (1+k_range)*k_dirac_cone, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)


# Modify dpsi_dx_num and dpsi_dy_num to select the appropriate band for the finite difference calculation
def dpsi_dx_num_from_grid(eigenfunctions, i, j, delta_k, band_index):
    if i == 0 or i == eigenfunctions.shape[0] - 1:
        return np.zeros_like(eigenfunctions[i, j, band_index])  # Edge handling: return zero at the boundary
    # Centered difference approximation for the selected band
    return (eigenfunctions[i + 1, j, band_index] - eigenfunctions[i - 1, j, band_index]) / (2 * delta_k)

def dpsi_dy_num_from_grid(eigenfunctions, i, j, delta_k, band_index):
    if j == 0 or j == eigenfunctions.shape[1] - 1:
        return np.zeros_like(eigenfunctions[i, j, band_index])  # Edge handling: return zero at the boundary
    # Centered difference approximation for the selected band
    return (eigenfunctions[i, j + 1, band_index] - eigenfunctions[i, j - 1, band_index]) / (2 * delta_k)

# Hamiltonian for the Lieb lattice
def H_Lieb(kx, ky, a=1):
    return 2 * np.array([
        [0, np.cos(ky * a/2), np.cos(kx * a/2)],
        [np.cos(ky * a/2), 0, 0],
        [np.cos(kx * a/2), 0, 0]
    ])


eigenvalues, eigenfunctions, _, _ = spiral_eigenvalues_eigenfunctions(H_Lieb, kx, ky, mesh_spacing, dim =3 )

# Initialize arrays to store tensor components
g_xx_array = np.zeros(kx.shape)
g_xy_real_array = np.zeros(kx.shape)
g_xy_imag_array = np.zeros(kx.shape)
g_yy_array = np.zeros(kx.shape)

trace_array = np.zeros(kx.shape)

band = 1

# Calculate tensor components using existing eigenfunctions on the grid, for a specific band
for i in range(1, kx.shape[0] - 1):  # Exclude the edges
    for j in range(1, kx.shape[1] - 1):  # Exclude the edges
        eigenfunction = eigenfunctions[i, j]
        eigenvalue = eigenvalues[i, j]
        
        # Use finite difference approximation for dpsi/dx and dpsi/dy for the chosen band
        dpsi_dx_val = dpsi_dx_num_from_grid(eigenfunctions, i, j, delta_k, band)
        dpsi_dy_val = dpsi_dy_num_from_grid(eigenfunctions, i, j, delta_k, band)
        
        psi_val = eigenfunction[band]  # Get the eigenfunction for the specified band
        
        I = np.eye(3)
        P = projection_operator(psi_val)
        
        g_xx = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dx_val).real
        g_xy_real = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).real
        g_xy_imag = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).imag
        g_yy = np.vdot(dpsi_dy_val, (I - P) @ dpsi_dy_val).real
        
        g_xx_array[i, j] = g_xx
        g_xy_real_array[i, j] = g_xy_real
        g_yy_array[i, j] = g_yy
        trace_array[i, j] = g_xx + g_yy

    
# Apply the cutoff using np.clip in one line
g_xx_array = np.clip(g_xx_array, None, z_cutoff)
g_xy_real_array = np.clip(g_xy_real_array, None, z_cutoff)
g_xy_imag_array = np.clip(g_xy_imag_array, None, z_cutoff)
g_yy_array = np.clip(g_yy_array, None, z_cutoff)
trace_array = np.clip(trace_array, None, z_cutoff)


extent = [(1 - k_range) * k_dirac_cone, (1 + k_range) * k_dirac_cone, 
          (1 - k_range) * k_dirac_cone, (1 + k_range) * k_dirac_cone]
# Plotting
# fig = plt.figure(figsize=(12, 8))
X, Y = kx, ky
stride_size = 1


def replace_zeros_with_nan(Z):
    Z_nan = np.where(Z == 0, np.nan, Z)
    return Z_nan


# Plot eigenfunction components for each band
for band in range(3):

    # if (band == 0) or (band == 2):
        fig, axes = plt.subplots(1, 3, figsize=(18, 8), subplot_kw={'projection': '3d'})
        fig.suptitle(f'Band {band + 1} Eigenfunction Components', fontsize=16)

        for component in range(3):
            col = component % 3
            ax = axes[col]
            
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

# Plot g_xy_real
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(kx, ky, g_xx_array, cmap='plasma', rstride=1, cstride=1, antialiased=True)
ax1.set_title('Numerical $g_{xx}$ (real part)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xx}$')


# Plot g_xy_real
ax1 = fig.add_subplot(132, projection='3d')
ax1.plot_surface(kx, ky, g_yy_array, cmap='plasma')
ax1.set_title('Numerical $g_{yy}$ (real part)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xx}$')

# Plot g_xy_real
ax1 = fig.add_subplot(133, projection='3d')
ax1.plot_surface(kx, ky, trace_array, cmap='plasma')
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
# plt.show()
plt.close()



# Plotting numerical results in 3d
fig = plt.figure(figsize=(12, 9))

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
fig, axes = plt.subplots(2, 3, figsize=(10, 8))

# Plot g_xx
ax1 = axes[0, 0]
c1 = ax1.imshow(g_xx_array, extent=extent, origin='lower', cmap='viridis')
ax1.set_title('g_{xx} (Numerical)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
fig.colorbar(c1, ax=ax1)

# Plot g_xy_real
ax2 = axes[0, 1]
c2 = ax2.imshow(g_yy_array, extent=extent, origin='lower', cmap='plasma')
ax2.set_title('g_{yy} (Numerical)')
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
fig.colorbar(c2, ax=ax2)

# Plot both positive and negative eigenvalues
ax3 = axes[0, 2]
c3 = ax3.imshow(trace_array, extent=extent, origin='lower', cmap='plasma')
ax3.set_title('Trace (Numerical)')
ax3.set_xlabel('kx')
ax3.set_ylabel('ky')
fig.colorbar(c3, ax=ax3)


plt.tight_layout()
plt.show()
plt.close()