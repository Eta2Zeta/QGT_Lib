import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from eigenvalue_calc_lib import eigenvalues_eigenfunctions_in_connected_zone, grid_eigenvalues_eigenfunctions, capping_eigenvalues, eigenvalues_eigenfunctions_in_zone, eigenvalues_eigenfunctions_in_zone_eigenvector_ordering, calculate_neighbor_phase_array
from Geometry.zones import ZoneDivider
from eigenvalue_calc_lib import Eigenvector

# Brillouin zone vectors
a = 1.0  # Lattice constant

# Define parameters
mesh_spacing = 150
k_max = 0.8 * (np.pi / a)  # Maximum k value for the first Brillouin zone

# Create kx and ky arrays
kx = np.linspace(-k_max, k_max, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)
z_limit = 2

dim = 6


# Define the new Hamiltonian function
def H_THF(kx, ky, nu_star=-50, nu_star_prime=13.0, gamma=-25.0, M=5, G=0.00):
    k = np.sqrt(kx**2 + ky**2)
    theta = np.arctan2(ky, kx)
    
    H_k = np.array([
        [G*(nu_star**2 - nu_star_prime**2), 0, nu_star * k * np.exp(1j * theta), 0, gamma, nu_star_prime * k * np.exp(-1j * theta)],
        [0, -G*(nu_star**2 - nu_star_prime**2), 0, nu_star * k * np.exp(-1j * theta), nu_star_prime * k * np.exp(1j * theta), gamma],
        [nu_star * k * np.exp(-1j * theta), 0, -G * nu_star**2, M, 0, 0],
        [0, nu_star * k * np.exp(1j * theta), M, G * nu_star**2, 0, 0],
        [gamma, nu_star_prime * k * np.exp(-1j * theta), 0, 0, -G * nu_star_prime**2, 0],
        [nu_star_prime * k * np.exp(1j * theta), gamma, 0, 0, 0, G * nu_star_prime**2]
    ])
    
    return H_k

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
    eigenvalues, _ = grid_eigenvalues_eigenfunctions(H_THF, kx, ky, mesh_spacing, dim=6)

    # Initialize the touching_points dictionary with zone numbers as keys
    touching_points = {}

    # Initialize the smallest differences for each zone with a large value (z_limit)
    for zone in range(6):
        touching_points[zone] = ((None, None), z_limit)  # Initially, no point and max difference

    # Find the points where bands are touching
    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            for band1 in range(6):
                for band2 in range(band1 + 1, 6):
                    difference = abs(eigenvalues[i, j, band1] - eigenvalues[i, j, band2])
                    kx_val, ky_val = kx[i, j], ky[i, j]
                    zone_number = zone_divider.get_zone(kx_val, ky_val)

                    # Update the dictionary if a smaller difference is found in this zone
                    if difference < touching_points[zone_number][1]:
                        touching_points[zone_number] = ((kx_val, ky_val), difference)

    # Print the results
    for zone, ((kx_val, ky_val), difference) in touching_points.items():
        if kx_val is not None and ky_val is not None:
            print(f"Zone {zone}: The smallest difference is {difference:.4e} at point (kx, ky) = ({kx_val:.4f}, {ky_val:.4f})")

    # zone_divider = ZoneDivider(kx, ky, num_zones=6, start_angle = np.pi/6)
    zone_divider = ZoneDivider(kx, ky, num_zones=6, start_angle = 0)
    zone_divider.calculate_zones()
    
    eigenvector = Eigenvector(dim)    

    # Calculate the eigenvalues and eigenfunctions for each zone
    for zone_num in range(6):
        smallest_angle_point, largest_angle_point = zone_divider.get_farthest_points_in_zone(zone_num)
        eigenvalues_zone, eigenfunctions_zone, phasefactors_zone, neighbor_phase_array_after_calc = eigenvalues_eigenfunctions_in_connected_zone(
            H_THF, kx, ky, mesh_spacing, dim=6, Zone=zone_divider, zone_num=zone_num, eigenvector=eigenvector, smallest_angle_point=smallest_angle_point, largest_angle_point= largest_angle_point
        )

        # Stitch the results into the combined arrays
        mask = zone_divider.create_mask_for_zone(zone_num)
        eigenvalues[mask] = eigenvalues_zone[mask]
        eigenfunctions[mask] = eigenfunctions_zone[mask]
        phasefactors[mask] = phasefactors_zone[mask]
        overall_neighbor_phase_array[mask] = neighbor_phase_array_after_calc[mask]


    # Save the data to files
    np.save(eigenvalues_file, eigenvalues)
    np.save(eigenfunctions_file, eigenfunctions)
    np.save(phasefactors_file, phasefactors)
    np.save('neighbor_phase_array.npy', overall_neighbor_phase_array)
    print("Saved eigenvalues, eigenfunctions, phasefactors, and neighbor phase array to files.")

# Now you can continue using eigenvalues, eigenfunctions, and phasefactors as needed.

# If desired, you can also visualize the touching points in the plot

eigenvalues = capping_eigenvalues(eigenvalues=eigenvalues, z_limit=z_limit)



def replace_zeros_with_nan(Z):
    Z_nan = np.where(Z == 0, np.nan, Z)
    return Z_nan

color_maps = ['viridis', 'magma', 'coolwarm', 'plasma', 'inferno', 'cividis']
# # Use the original kx and ky arrays (non-flattened)
X, Y = kx, ky
stride_size = 3

# Plot the eigenvalues and eigenfunctions
fig = plt.figure(figsize=(24, 8))
ax_eigenvalue = fig.add_subplot(111, projection='3d')

# Highlight the touching points using the new dictionary format
for zone, ((kx_val, ky_val), difference) in touching_points.items():
    if kx_val is not None and ky_val is not None:
        ax_eigenvalue.scatter(kx_val, ky_val, 0, color='red', s=50, label=f'Touching Point in Zone {zone} (Diff: {difference:.4e})')



# Plot all six sets of eigenvalues as surface plots with different color maps
for band in range(6):
    if (band == 2) or (band == 3):
        Z_eigenvalue = eigenvalues[:, :, band]
        Z_eigenvalue = replace_zeros_with_nan(Z_eigenvalue)  # Replace zeros with NaN
        ax_eigenvalue.plot_surface(X, Y, Z_eigenvalue, cmap=color_maps[band], rstride=stride_size, cstride=stride_size, alpha=0.8)

ax_eigenvalue.set_title('Eigenvalues for All Bands with Touching Points')
ax_eigenvalue.set_xlabel('kx')
ax_eigenvalue.set_ylabel('ky')
ax_eigenvalue.set_zlabel('Eigenvalue')
ax_eigenvalue.set_zlim(-z_limit,z_limit)


plt.tight_layout()
plt.show()
plt.close()


# Plot the eigenvalues and eigenfunctions as scatter plots
fig = plt.figure(figsize=(24, 8))
ax_eigenvalue = fig.add_subplot(111, projection='3d')

# Highlight the touching points using the new dictionary format
for zone, ((kx_val, ky_val), difference) in touching_points.items():
    if kx_val is not None and ky_val is not None:
        ax_eigenvalue.scatter(kx_val, ky_val, 0, color='red', s=50, label=f'Touching Point in Zone {zone} (Diff: {difference:.4e})')

# Plot all six sets of eigenvalues as scatter plots with different color maps
for band in range(6):
    if (band == 2) or (band == 3):
        Z_eigenvalue = eigenvalues[:, :, band].flatten()
        Z_eigenvalue = replace_zeros_with_nan(Z_eigenvalue)  # Replace zeros with NaN
        ax_eigenvalue.scatter(X.flatten(), Y.flatten(), Z_eigenvalue, c=Z_eigenvalue, cmap=color_maps[band], s=5)

ax_eigenvalue.set_title('Eigenvalues for All Bands with Touching Points')
ax_eigenvalue.set_xlabel('kx')
ax_eigenvalue.set_ylabel('ky')
ax_eigenvalue.set_zlabel('Eigenvalue')
ax_eigenvalue.set_zlim(-z_limit, z_limit)

plt.tight_layout()
plt.show()
plt.close()


# Plot eigenvalues for different bands
fig, axes = plt.subplots(2, 3, figsize=(18, 8), subplot_kw={'projection': '3d'})
fig.suptitle('Eigenvalues for Different Bands', fontsize=16)

# Iterate over each band and plot the eigenvalues as surface plots
for band in range(6):
    row = band // 3
    col = band % 3
    ax = axes[row, col]
    
    # Get the Z data for the eigenvalues
    Z_eigenvalue = eigenvalues[:, :, band]
    Z_eigenvalue = replace_zeros_with_nan(Z_eigenvalue)  # Replace zeros with NaN
    
    # Plot the surface plot for each band with transparency
    ax.plot_surface(X, Y, Z_eigenvalue, cmap=color_maps[band], rstride=stride_size, cstride=stride_size, alpha=0.6)
    
    # Highlight the touching points
    for zone, ((kx_val, ky_val), difference) in touching_points.items():
        if kx_val is not None and ky_val is not None:
            ax.scatter(kx_val, ky_val, 0, color='red', s=50, label=f'Touching Point in Zone {zone} (Diff: {difference:.4e})')

    ax.set_title(f'Eigenvalue {band + 1}')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('Eigenvalue')
    ax.set_zlim(-3, 3)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


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
    sc = ax.scatter(X, Y, Z_phasefactor, c=Z_phasefactor, cmap=color_maps[band], marker='o', s=1)

        
    # Highlight the touching points
    for zone, ((kx_val, ky_val), difference) in touching_points.items():
        if kx_val is not None and ky_val is not None:
            ax.scatter(kx_val, ky_val, 0, color='red', s=50, label=f'Touching Point in Zone {zone} (Diff: {difference:.4e})')

    
    ax.set_title(f'Phase factor {band + 1}')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('Phasefactor')
    ax.set_zlim(-z_limit, z_limit)

plt.tight_layout()
plt.show()
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
    sc = ax.scatter(X, Y, Z_neighbor_phase, c=Z_neighbor_phase, cmap=color_maps[band], marker='o', s=1)

    # Highlight the touching points
    for zone, ((kx_val, ky_val), difference) in touching_points.items():
        if kx_val is not None and ky_val is not None:
            ax.scatter(kx_val, ky_val, 0, color='red', s=50, label=f'Touching Point in Zone {zone} (Diff: {difference:.4e})')

    ax.set_title(f'Neighbor Phase Factor {band + 1}')
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('Neighbor Phase Factor')
    ax.set_zlim(-z_limit, z_limit)

plt.tight_layout()
plt.show()
plt.close()

# Plot eigenfunction components for each band
for band in range(6):

    if (band == 2) or (band == 3):
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

                
            # Highlight the touching points
            for zone, ((kx_val, ky_val), difference) in touching_points.items():
                if kx_val is not None and ky_val is not None:
                    ax.scatter(kx_val, ky_val, 0, color='red', s=50, label=f'Touching Point in Zone {zone} (Diff: {difference:.4e})')

            ax.set_title(f'Component {component + 1}')
            ax.set_xlabel('kx')
            ax.set_ylabel('ky')
            ax.set_zlabel('Magnitude')

            # Add a color bar
            fig.colorbar(sc, ax=ax, shrink=0.6, aspect=5)

        plt.tight_layout()
        plt.show()
        plt.close()


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
