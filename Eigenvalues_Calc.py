import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotting_lib import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from Hamiltonian_v1 import *
from eigenvalue_calc_lib import *
from Geometry.zones import ZoneDivider
from Hamiltonian_v2 import * 


# Brillouin zone vectors

# Define parameters
mesh_spacing = 200
k_max = 1 * (np.pi)  # Maximum k value for the first Brillouin zone

# Create kx and ky arrays
kx = np.linspace(-k_max, k_max, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)
z_limit = 5

dim = 6



# File paths for saving and loading data
eigenvalues_file = "eigenvalues.npy"
eigenfunctions_file = "eigenfunctions.npy"
phasefactors_file = "phasefactors.npy"

# Initialize arrays to store eigenfunctions and eigenvalues with NaNs
eigenfunctions = np.full((mesh_spacing, mesh_spacing, dim, dim), np.nan, dtype=complex)
eigenvalues = np.full((mesh_spacing, mesh_spacing, dim), np.nan, dtype=float)
phasefactors = np.full((mesh_spacing, mesh_spacing, dim), np.nan, dtype=float)
overall_neighbor_phase_array = np.full((mesh_spacing, mesh_spacing, dim), np.nan, dtype=float)

# Calculate the eigenvalues and eigenfunctions
eigenvalues, eigenfunctions, phasefactors, overall_neighbor_phase_array = spiral_eigenvalues_eigenfunctions(H_THF, kx, ky, mesh_spacing, dim=dim)
# eigenvalues, eigenfunctions = grid_eigenvalues_eigenfunctions(H_Square_Lattice, kx, ky, mesh_spacing, dim=dim)
# Save the data to files
np.save(eigenvalues_file, eigenvalues)
np.save(eigenfunctions_file, eigenfunctions)
np.save(phasefactors_file, phasefactors)
np.save('neighbor_phase_array.npy', overall_neighbor_phase_array)
print("Saved eigenvalues, eigenfunctions, phasefactors, and neighbor phase array to files.")


def experiment():
    # Experiment Changing the phase of one zone
    num_zones = 4
    zone_to_modify = 1  # Index of the zone to modify (0-based)
    phase_factor = np.exp(0.8j * np.pi)  # Example phase factor

    # Initialize the ZoneDivider
    zone_divider = ZoneDivider(kx, ky, num_zones)
    # Calculate zones
    zone_divider.calculate_zones()
    # Create a mask for the selected zone
    zone_mask = zone_divider.create_mask_for_zone(zone_to_modify)

    # Apply the phase factor to eigenfunctions within the selected zone
    modified_eigenfunctions = eigenfunctions.copy()
    modified_eigenfunctions[:, :, 0, 0][zone_mask] *= phase_factor



eigenvalues = capping_eigenvalues(eigenvalues=eigenvalues, z_limit=z_limit)

plot_eigenvalues_surface(kx, ky, eigenvalues, dim=dim, z_limit=z_limit, color_maps="bwr")


plot_eigenfunction_components(kx, ky, eigenfunctions, band_index=0, components_to_plot=[0])




# plot_phases(kx, ky, phasefactors, dim=2)

# plot_neighbor_phases(kx, ky, overall_neighbor_phase_array, dim=2)