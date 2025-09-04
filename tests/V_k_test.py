import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters from the paper
vF = 542.1
t1 = 355.16
V = 30.0
n = 4

# Define V(k) function
def V_k(k):
    vk_t1 = (vF * k / t1)
    num = (n - 1) * vk_t1 ** (2 * n + 2) + vk_t1 ** 2 - n * vk_t1 ** (2 * n)
    denom = (1 - vk_t1 ** 2) * (1 - vk_t1 ** (2 * n))
    return V * (-0.5 * (n - 1) + num / denom)

# Define kx, ky grid
mesh_points = 300
kx = np.linspace(-np.pi, np.pi, mesh_points)
ky = np.linspace(-np.pi, np.pi, mesh_points)
kx_grid, ky_grid = np.meshgrid(kx, ky)
k_mag = np.sqrt(kx_grid**2 + ky_grid**2)

# Avoid dividing by zero by setting k=0 to a small value
k_mag[k_mag == 0] = 1e-10

# Evaluate V(k) over the grid
V_grid = V_k(k_mag)

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(kx_grid, ky_grid, V_grid, cmap='viridis', edgecolor='none')

ax.set_title(r'$V(k)$ in 2D $k$-space', fontsize=16)
ax.set_xlabel(r'$k_x$')
ax.set_ylabel(r'$k_y$')
ax.set_zlabel(r'$V(k)$')
fig.colorbar(surf, shrink=0.6, aspect=10)

plt.tight_layout()
plt.show()
