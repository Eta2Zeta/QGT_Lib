import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from Hamiltonian import H_THF
from eigenvalue_calc_lib import grid_eigenvalues_eigenfunctions, capping_eigenvalues, eigenvalues_eigenfunctions_in_zone, spiral_eigenvalues_eigenfunctions, eigenvalues_eigenfunctions_in_zone_eigenvector_ordering, calculate_neighbor_phase_array
from Geometry.zones import ZoneDivider
from plotting_lib import plot_neighbor_phases, plot_eigenvalues_surface, plot_eigenfunction_components
from utilities import replace_zeros_with_nan

# Brillouin zone vectors

# Define parameters
mesh_spacing = 200
k_max = 1 * (np.pi)  # Maximum k value for the first Brillouin zone

# Create kx and ky arrays
kx = np.linspace(-k_max, k_max, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)
z_limit = 300

dim = 6

# File paths for saving and loading data
eigenvalues_file = "eigenvalues.npy"
eigenfunctions_file = "eigenfunctions.npy"
phasefactors_file = "phasefactors.npy"

useExisting = False

# Check if the data already exists
if useExisting:
    # Load the data
    eigenvalues = np.load(eigenvalues_file)
    eigenfunctions = np.load(eigenfunctions_file)
    phasefactors = np.load(phasefactors_file)
    print("Loaded eigenvalues, eigenfunctions, and phasefactors from files.")
else:
    # Initialize arrays to store eigenfunctions and eigenvalues with NaNs
    eigenfunctions = np.full((mesh_spacing, mesh_spacing, 6, 6), np.nan, dtype=complex)
    eigenvalues = np.full((mesh_spacing, mesh_spacing, 6), np.nan, dtype=float)
    phasefactors = np.full((mesh_spacing, mesh_spacing, 6), np.nan, dtype=float)
    overall_neighbor_phase_array = np.full((mesh_spacing, mesh_spacing, 6), np.nan, dtype=float)


    # Create a ZoneDivider object with 6 zones
    zone_divider = ZoneDivider(kx, ky, num_zones=6, start_angle = 0)
    zone_divider.calculate_zones()


    # Calculate the eigenvalues and eigenfunctions
    eigenvalues, eigenfunctions, phasefactors, overall_neighbor_phase_array = spiral_eigenvalues_eigenfunctions(H_THF, kx, ky, mesh_spacing, dim=6)


    # Save the data to files
    np.save(eigenvalues_file, eigenvalues)
    np.save(eigenfunctions_file, eigenfunctions)
    np.save(phasefactors_file, phasefactors)
    np.save('neighbor_phase_array.npy', overall_neighbor_phase_array)
    print("Saved eigenvalues, eigenfunctions, phasefactors, and neighbor phase array to files.")

# Now you can continue using eigenvalues, eigenfunctions, and phasefactors as needed.

# If desired, you can also visualize the touching points in the plot

eigenvalues = capping_eigenvalues(eigenvalues=eigenvalues, z_limit=z_limit)



plot_eigenvalues_surface(kx, ky, eigenvalues, dim=dim, z_limit=z_limit, color_maps="bwr")

color_maps = ['viridis', 'magma', 'coolwarm', 'plasma', 'inferno', 'cividis']

# Plot the phases for different bands
fig, axes = plt.subplots(2, 3, figsize=(18, 8), subplot_kw={'projection': '3d'})
fig.suptitle('Phases', fontsize=16)

for band in range(6):
    row = band // 3
    col = band % 3
    ax = axes[row, col]
    
    Z_phasefactor = phasefactors[:, :, band].flatten()
    Z_phasefactor = replace_zeros_with_nan(Z_phasefactor)  # Replace zeros with NaN

    # Create a scatter plot
    sc = ax.scatter(kx, ky, Z_phasefactor, c=Z_phasefactor, cmap=color_maps[band], marker='o', s=1)

    
    
    ax.set_title(f'Phase factor {band + 1}')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('Phasefactor')
    ax.set_zlim(-2, 2)

plt.tight_layout()
# plt.show()
plt.close()

# Plot the overall neighbor phase array for different bands
fig, axes = plt.subplots(2, 3, figsize=(18, 8), subplot_kw={'projection': '3d'})
fig.suptitle('Overall Neighbor Phase Factors', fontsize=16)

for band in range(6):
    row = band // 3
    col = band % 3
    ax = axes[row, col]
    
    Z_neighbor_phase = overall_neighbor_phase_array[:, :, band].flatten()
    Z_neighbor_phase = replace_zeros_with_nan(Z_neighbor_phase)  # Replace zeros with NaN

    # Create a scatter plot
    sc = ax.scatter(kx, ky, Z_neighbor_phase, c=Z_neighbor_phase, cmap=color_maps[band], marker='o', s=1)


    ax.set_title(f'Neighbor Phase Factor {band + 1}')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('Neighbor Phase Factor')
    ax.set_zlim(-2, 2)

plt.tight_layout()
# plt.show()
plt.close()

plot_neighbor_phases(kx, ky, overall_neighbor_phase_array)  

# # Iterate over each component and plot them one by one
# for component in range(6):
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     fig.suptitle(f'Eigenvector Component {component + 1} with All Bands', fontsize=16)

#     # Plot the same component for all six bands on the same plot
#     for band in range(6):
#         # if (band == 3) or (band == 2):
#             Z_eigenfunction = eigenfunctions[:, :, band, component].flatten()
#             Z_eigenfunction = replace_zeros_with_nan(Z_eigenfunction)  # Replace zeros with NaN
            
#             # Use the magnitude of Z_eigenfunction for color mapping
#             Z_magnitude = np.imag(Z_eigenfunction)

#             # Use ax.scatter for 3D scatter plot
#             ax.scatter(X, Y, Z_magnitude, c=Z_magnitude, cmap=color_maps[band], s=5, label=f'Band {band + 1}')

#     ax.set_title(f'Eigenvector Component {component + 1}')
#     ax.set_xlabel('kx')
#     ax.set_ylabel('ky')
#     ax.set_zlabel('Magnitude')
#     ax.legend(loc='upper right')

#     plt.tight_layout()
#     plt.show()
#     plt.close()


# # Define fixed colors for each band
# fixed_colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']

# # Iterate over each component and plot them one by one
# for component in range(6):
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     fig.suptitle(f'Eigenvector Component {component + 1} with All Bands', fontsize=16)

#     # Plot the same component for all six bands on the same plot
#     for band in range(6):
#         Z_eigenfunction = eigenfunctions[:, :, band, component].flatten()
#         Z_eigenfunction = replace_zeros_with_nan(Z_eigenfunction)  # Replace zeros with NaN
        
#         # Use the magnitude or real/imaginary part of Z_eigenfunction for Z-axis
#         Z_magnitude = np.imag(Z_eigenfunction)  # or np.real(Z_eigenfunction) depending on what you want to plot

#         # Use ax.scatter for 3D scatter plot with a fixed color for each band
#         ax.scatter(X, Y, Z_magnitude, c=fixed_colors[band], s=5, label=f'Band {band + 1}')

#     ax.set_title(f'Eigenvector Component {component + 1}')
#     ax.set_xlabel('kx')
#     ax.set_ylabel('ky')
#     ax.set_zlabel('Magnitude')
#     ax.legend(loc='upper right')

#     plt.tight_layout()
#     plt.show()
#     plt.close()
