import sys
import os
import numpy as np
import pickle

# from Library import * 
from Library.plotting_lib import *
from Library.Hamiltonian_v1 import *
from Library.Hamiltonian_v2 import * 
from Library.eigenvalue_calc_lib import *
from Library.Geometry.zones import ZoneDivider



# Define parameters
mesh_spacing = 150
k_max = 1 * (np.pi)  # Maximum k value for the first Brillouin zone

# Create kx and ky arrays
kx = np.linspace(-k_max, k_max, mesh_spacing)
# ky = np.linspace(-(3/2)*k_max, k_max/2, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)
dkx = np.abs(kx[0, 1] - kx[0, 0])  # Spacing in the x-direction (constant for a uniform grid)
dky = np.abs(ky[1, 0] - ky[0, 0])  # Spacing in the y-direction (constant for a uniform grid)
z_limit = 10

# Hamiltonian_Obj = THF_Hamiltonian(A0=0)
# Hamiltonian_Obj = TwoOrbitalUnspinfulHamiltonian(zeta=0.5, A0=0, mu=2)
Hamiltonian_Obj = SquareLatticeHamiltonian(A0=1, omega=5e0)
dim = Hamiltonian_Obj.dim



UseExisting = False

# Define the temp directory for storing .npy files
temp_dir = os.path.join(os.getcwd(), "temp")

# Create the temp directory if it doesn't exist
os.makedirs(temp_dir, exist_ok=True)

# File paths for saving and loading data
eigenvalues_file = os.path.join(temp_dir, "eigenvalues.npy")
eigenfunctions_file = os.path.join(temp_dir, "eigenfunctions.npy")
phasefactors_file = os.path.join(temp_dir, "phasefactors.npy")
meta_info_file = os.path.join(temp_dir, "meta_info.npy")  # New file for meta information
neighbor_phase_array_file = os.path.join(temp_dir, "neighbor_phase_array.npy")


if UseExisting: 
    # Load the eigenvalues and eigenfunctions from files
    if os.path.exists(eigenvalues_file) and os.path.exists(eigenfunctions_file) and os.path.exists(meta_info_file):
        eigenvalues = np.load(eigenvalues_file)
        eigenfunctions = np.load(eigenfunctions_file)
        with open(meta_info_file, "rb") as meta_file:
            meta_info = pickle.load(meta_file)
            kx = meta_info["kx"]
            ky = meta_info["ky"]
            mesh_spacing = meta_info["mesh_spacing"]
            Hamiltonian_Obj = meta_info["Hamiltonian_Obj"]
        print("Loaded eigenvalues, eigenfunctions, and meta information from files.")
    else:
        print("Required files not found. Please ensure eigenvalues, eigenfunctions, and meta information are available in the 'temp' directory.")
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
    np.save(neighbor_phase_array_file, overall_neighbor_phase_array)

    # Save meta information
    meta_info = {
        "kx": kx,
        "ky": ky,
        "dkx": dkx, 
        "dky": dky,
        "mesh_spacing": mesh_spacing,
        "Hamiltonian_Obj": Hamiltonian_Obj  # Include the Hamiltonian object
    }

    # Save the metadata using pickle
    with open(meta_info_file, "wb") as meta_file:
        pickle.dump(meta_info, meta_file)
    print(f"Saved eigenvalues, eigenfunctions, phasefactors, neighbor phase array, and meta information to '{temp_dir}'.")




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

plot_eigenvalues_surface_colorbar(kx, ky, eigenvalues, dim=dim, z_limit=z_limit, color_maps='bwr', norm=None)

# plot_eigenfunction_components(kx, ky, eigenfunctions, band_index=0, components_to_plot=[0])

# plot_phases(kx, ky, phasefactors, dim=2)

# plot_neighbor_phases(kx, ky, overall_neighbor_phase_array, dim=2)





# Define the line parameters
angle_deg = 90  # Line angle in degrees
k_angle = np.deg2rad(angle_deg) # Convert into Radians
kx_shift = 0
# ky_shift = -np.pi/2
ky_shift = 0
num_points = 100  # Number of points along the line
# k_max = 1 * (np.pi)
k_max = np.sqrt(2) * (np.pi)
k_line = np.linspace(-k_max, k_max, num_points)
line_kx = k_line * np.cos(k_angle) + kx_shift
line_ky = k_line * np.sin(k_angle) + ky_shift
# plot_k_line_on_grid(line_kx, line_ky, k_max)

eigenvalues, eigenfunctions, _ = line_eigenvalues_eigenfunctions(Hamiltonian_Obj, line_kx, line_ky)

plot_eigenvalues_line(k_line, eigenvalues)

