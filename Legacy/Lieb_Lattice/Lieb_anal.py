import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from QGT_lib import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Define parameters
mesh_spacing = 200
diff_para = 1e3  # How much smaller the dk is compare to the grid size of the k space
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

# Analytical derivative of psi w.r.t kx
def dpsi_dx_analytic(kx, ky):
    norm = (np.cos(kx/2)**2 + np.cos(ky/2)**2)**(3/2)
    dpsi_dx_1 = 0
    dpsi_dx_2 = np.sin(kx/2) * np.cos(ky/2)**2 / norm
    dpsi_dx_3 = np.sin(kx/2) * np.cos(kx/2) * np.cos(ky/2) / norm
    return np.array([dpsi_dx_1, dpsi_dx_2, dpsi_dx_3])


# Identity matrix
I = np.eye(3)

# Analytic solution for g_xy (real part)
def analytic_g_xy_real(kx, ky, M):
    k = np.sqrt(kx**2 + ky**2)
    epsilon = np.sqrt(M**2 + kx**2 + ky**2)
    return -kx * ky / (4 * epsilon**4)

# Analytic solution for g_xx
def analytic_g_xx(kx, ky):
    numerator = (np.sin(kx/2)**2) * (np.cos(ky/2)**2)
    denominator = 4*(np.cos(kx/2)**2+np.cos(ky/2)**2)**2
    
    return numerator/denominator

# Analytic solution for g_yy
def analytic_g_yy(kx, ky):
    numerator = (np.cos(kx/2) ** 2) * (np.sin(ky/2) ** 2)
    denominator = 4*(np.cos(kx/2) ** 2 + np.cos(ky/2) ** 2) ** 2
    return numerator / denominator



# Initialize arrays to store tensor components
g_xx_array = np.zeros(kx.shape)
g_xy_real_array = np.zeros(kx.shape)
g_xy_imag_array = np.zeros(kx.shape)
g_yy_array = np.zeros(kx.shape)
analytic_g_xx_array = np.zeros(kx.shape)
analytic_g_xy_real_array = np.zeros(kx.shape)
analytic_g_yy_array = np.zeros(kx.shape)
trace_array = np.zeros(kx.shape)
analytic_trace_array = np.zeros(kx.shape)

eigenvalue_pos_array = np.zeros(kx.shape)
eigenvalue_neg_array = np.zeros(kx.shape)

# Calculate tensor components for each point in the Brillouin zone
for i in range(kx.shape[0]):
    for j in range(kx.shape[1]):
        g_xx, g_xy_real, g_xy_imag, g_yy = quantum_geometric_tensor(psi, I, kx[i, j], ky[i, j], delta_k)
        # if False: 
        if g_xx > z_cutoff:
            g_xx_array[i, j] = z_cutoff
        else: 
            g_xx_array[i, j] = g_xx
        if g_xy_real > z_cutoff:
            g_xy_real_array[i, j] = z_cutoff
        else:
            g_xy_real_array[i, j] = g_xy_real
        if g_yy > z_cutoff:
            g_yy_array[i, j] = z_cutoff
        else:
            g_yy_array[i, j] = g_yy
        trace = g_xx + g_yy
        if trace > z_cutoff: 
            trace_array[i,j] = z_cutoff
        else: 
            trace_array[i,j] = trace
        g_xx_anal = analytic_g_xx(kx[i, j], ky[i, j])
        g_yy_anal = analytic_g_yy(kx[i, j], ky[i, j])
        trace_anal = g_xx_anal + g_yy_anal
        if g_xx_anal > z_cutoff:
            analytic_g_xx_array[i, j] = z_cutoff
        else: 
            analytic_g_xx_array[i, j] = g_xx_anal
        if g_yy_anal > z_cutoff:
            analytic_g_yy_array[i, j] = z_cutoff
        else: 
            analytic_g_yy_array[i, j] = g_yy_anal
        if trace_anal > z_cutoff: 
            analytic_trace_array[i, j] = z_cutoff
        else:
            analytic_trace_array[i, j] = trace_anal

        
        eigenvalue = np.sqrt(np.cos(kx[i, j])**2 + np.cos(ky[i, j])**2)
        eigenvalue_pos_array[i, j] = eigenvalue
        eigenvalue_neg_array[i, j] = -eigenvalue


# Plotting
fig = plt.figure(figsize=(12, 8))

# Plot g_xy_real
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(kx, ky, g_xx_array, cmap='plasma', rstride=1, cstride=1, antialiased=True)
ax1.set_title('Numerical $g_{xx}$ (real part)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xx}$')

# Plot analytic g_xy_real
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(kx, ky, analytic_g_xx_array, cmap='viridis', rstride=1, cstride=1)
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
ax1.plot_surface(kx, ky, g_yy_array, cmap='plasma')
ax1.set_title('Numerical $g_{yy}$ (real part)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xx}$')

# Plot analytic g_xy_real
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(kx, ky, analytic_g_yy_array, cmap='viridis')
ax2.set_title('Analytic $g_{yy}$ (real part)')
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
ax1.plot_surface(kx, ky, trace_array, cmap='plasma')
ax1.set_title('Numerical trace (real part)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('$g_{xx}$')

# Plot analytic g_xy_real
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(kx, ky, analytic_trace_array, cmap='viridis')
ax2.set_title('Analytic trace (real part)')
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
ax2.set_zlabel('$g_{xx}$')


plt.tight_layout()
# plt.show()
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

# Plot analytic g_xx
ax4 = fig.add_subplot(234, projection='3d')
ax4.plot_surface(kx, ky, analytic_g_xx_array, cmap='viridis')
ax4.set_title('Analytic $g_{xx}$')
ax4.set_xlabel('kx')
ax4.set_ylabel('ky')
ax4.set_zlabel('$g_{xx}$')

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
fig, axes = plt.subplots(2, 3, figsize=(10, 8))

# Plot g_xx
ax1 = axes[0, 0]
c1 = ax1.imshow(g_xx_array, extent=(0, k_max, 0, k_max), origin='lower', cmap='viridis')
ax1.set_title('g_{xx} (Numerical)')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
fig.colorbar(c1, ax=ax1)

# Plot g_xy_real
ax2 = axes[0, 1]
c2 = ax2.imshow(g_yy_array, extent=(0, k_max, 0, k_max), origin='lower', cmap='plasma')
ax2.set_title('g_{yy} (Numerical)')
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
fig.colorbar(c2, ax=ax2)

# Plot both positive and negative eigenvalues
ax3 = axes[0, 2]
c3 = ax3.imshow(trace_array, extent=(0, k_max, 0, k_max), origin='lower', cmap='inferno')
ax3.set_title('Trace (Numerical)')
ax3.set_xlabel('kx')
ax3.set_ylabel('ky')
fig.colorbar(c3, ax=ax3)


# Plot g_yy
ax4 = axes[1, 0]
c4 = ax4.imshow(analytic_g_xx_array, extent=(0, k_max, 0, k_max), origin='lower', cmap='inferno')
ax4.set_title('g_{xx} (Analytical)')
ax4.set_xlabel('kx')
ax4.set_ylabel('ky')
fig.colorbar(c4, ax=ax4)

# Plot g_yy
ax5 = axes[1, 1]
c4 = ax5.imshow(analytic_g_yy_array, extent=(0, k_max, 0, k_max), origin='lower', cmap='inferno')
ax5.set_title('g_{yy} (Analytical)')
ax5.set_xlabel('kx')
ax5.set_ylabel('ky')
fig.colorbar(c4, ax=ax5)

# Plot g_yy
ax6 = axes[1, 2]
c4 = ax6.imshow(analytic_trace_array, extent=(0, k_max, 0, k_max), origin='lower', cmap='inferno')
ax6.set_title('Trace (Analytical)')
ax6.set_xlabel('kx')
ax6.set_ylabel('ky')
fig.colorbar(c4, ax=ax6)


plt.tight_layout()
plt.show()
plt.close()