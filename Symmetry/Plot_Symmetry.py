import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 1. Set up the grid of angles (spherical coordinates)
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 200)
theta, phi = np.meshgrid(theta, phi)

# 2. Define the unit direction vector (k_hat)
kx = np.sin(theta) * np.cos(phi)
ky = np.sin(theta) * np.sin(phi)
kz = np.cos(theta)

# 3. Calculate the function value (d-wave: x^2 - y^2)
func_value = kx**2 - ky**2

# 4. Define the radius r (Magnitude of the function)
# The paper plotted squared magnitude, but standard orbital plots use absolute value.
# You can use func_value**2 if you want sharper lobes like the paper.
r = np.abs(func_value)

# 5. Convert back to Cartesian coordinates for the surface (r * direction)
x = r * kx
y = r * ky
z = r * kz

# 6. Set up coloring based on the sign (+ is Red, - is Blue)
# Normalize the colors to map -1 (blue) to +1 (red)
norm = plt.Normalize(-1, 1)
colors = cm.bwr(norm(np.sign(func_value)))

# 7. Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface with the calculated colors
surf = ax.plot_surface(x, y, z, facecolors=colors, 
                       rstride=1, cstride=1, linewidth=0, antialiased=False, shade=True)

# Styling
ax.set_title(r'3D Polar Plot of $d_{x^2-y^2}$ Symmetry', fontsize=14)
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('$k_z$')

# Set view angle to see the 4 lobes clearly
ax.view_init(elev=30, azim=45)

# Equal aspect ratio hack for matplotlib 3D
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()
