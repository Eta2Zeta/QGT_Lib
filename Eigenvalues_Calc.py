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
mesh_spacing = 100
k_max = 1 * (np.pi)  # Maximum k value for the first Brillouin zone

# Create kx and ky arrays
kx = np.linspace(-k_max, k_max, mesh_spacing)
# ky = np.linspace(-(3/2)*k_max, k_max/2, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)
z_limit = 5



Hamiltonian_Obj = TwoOrbitalSpinfulHamiltonian(zeta=0.2, A0=0)
dim = Hamiltonian_Obj.dim

UseExisting = False


# File paths for saving and loading data
eigenvalues_file = "eigenvalues.npy"
eigenfunctions_file = "eigenfunctions.npy"
phasefactors_file = "phasefactors.npy"
meta_info_file = "meta_info.npy"  # New file for meta information

if UseExisting: 
    # Load the eigenvalues and eigenfunctions from files
    if os.path.exists(eigenvalues_file) and os.path.exists(eigenfunctions_file) and os.path.exists(meta_info_file):
        eigenvalues = np.load(eigenvalues_file)
        eigenfunctions = np.load(eigenfunctions_file)
        meta_info = np.load(meta_info_file, allow_pickle=True).item()
        kx = meta_info["kx"]
        ky = meta_info["ky"]
        mesh_spacing = meta_info["mesh_spacing"]
        print("Loaded eigenvalues, eigenfunctions, and meta information from files.")
    else:
        print("Required files not found. Please ensure eigenvalues, eigenfunctions, and meta information are available.")
        sys.exit(1)
else: 
    # Initialize arrays to store eigenfunctions and eigenvalues with NaNs
    eigenfunctions = np.full((mesh_spacing, mesh_spacing, dim, dim), np.nan, dtype=complex)
    eigenvalues = np.full((mesh_spacing, mesh_spacing, dim), np.nan, dtype=float)
    phasefactors = np.full((mesh_spacing, mesh_spacing, dim), np.nan, dtype=float)
    overall_neighbor_phase_array = np.full((mesh_spacing, mesh_spacing, dim), np.nan, dtype=float)

    # Calculate the eigenvalues and eigenfunctions
    eigenvalues, eigenfunctions, phasefactors, overall_neighbor_phase_array = spiral_eigenvalues_eigenfunctions(
        Hamiltonian_Obj, kx, ky, mesh_spacing, dim=dim, phase_correction=False
    )

    # Save the data to files
    np.save(eigenvalues_file, eigenvalues)
    np.save(eigenfunctions_file, eigenfunctions)
    np.save(phasefactors_file, phasefactors)
    np.save('neighbor_phase_array.npy', overall_neighbor_phase_array)

    # Save meta information
    meta_info = {
        "kx": kx,
        "ky": ky,
        "mesh_spacing": mesh_spacing
    }
    np.save(meta_info_file, meta_info)
    print("Saved eigenvalues, eigenfunctions, phasefactors, neighbor phase array, and meta information to files.")

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


# plot_eigenfunction_components(kx, ky, eigenfunctions, band_index=0, components_to_plot=[0])




# plot_phases(kx, ky, phasefactors, dim=2)

# plot_neighbor_phases(kx, ky, overall_neighbor_phase_array, dim=2)