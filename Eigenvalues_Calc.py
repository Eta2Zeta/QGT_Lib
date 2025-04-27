import os
import numpy as np
import pickle
import shutil


# from Library import * 
from Library.plotting_lib import *
from Library.Hamiltonian_v1 import *
from Library.Hamiltonian_v2 import * 
from Library.eigenvalue_calc_lib import *
from Library.Geometry.zones import ZoneDivider
from Library.utilities import *

# There were four zones of the eigenfunction with distinct phases and
# I was trying to test if changing the phases of one can make it match 
# to the phase another but it could not. Forgot which Hamiltonian it
# was for
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

# Ensure the temp directory exists
temp_dir = os.path.join(os.getcwd(), "temp")
os.makedirs(temp_dir, exist_ok=True)

# Hamiltonian_Obj = THF_Hamiltonian(A0=0)
hamiltonian = TwoOrbitalUnspinfulHamiltonian(zeta=1.0, omega = 10.0, A0=0.1, mu=0, magnus_order = 1)
# hamiltonian = SquareLatticeHamiltonian(A0=0, omega=5e0, t1=1, t2=1/np.sqrt(2), t5=0)
# hamiltonian = SquareLatticeHamiltonian(A0=0, omega=5e0, t1=1, t2=1/np.sqrt(2), t5=(1-np.sqrt(2))/4)
dim = hamiltonian.dim

def calculation_2d(hamiltonian = hamiltonian):
    # Does the calculation on 2d 


    # Define parameters
    k_max = np.pi
    kx_min, kx_max = -k_max, k_max
    ky_min, ky_max = -k_max, k_max

    # Create kx and ky arrays
    kx_range = (kx_min, kx_max)
    ky_range = (ky_min, ky_max)
    mesh_spacing = 150

    kx = np.linspace(kx_min, kx_max, mesh_spacing)
    ky = np.linspace(ky_min, ky_max, mesh_spacing)
    kx, ky = np.meshgrid(kx, ky)
    dkx = np.abs(kx[0, 1] - kx[0, 0])  # Spacing in the x-direction (constant for a uniform grid)
    dky = np.abs(ky[1, 0] - ky[0, 0])  # Spacing in the y-direction (constant for a uniform grid)
    z_limit = 10

    # Create the results directory
    file_paths, use_existing, results_subdir = setup_results_directory(hamiltonian, kx_range, ky_range, mesh_spacing)

    if use_existing:
        # Load existing data
        eigenvalues = np.load(file_paths["eigenvalues"])
        eigenfunctions = np.load(file_paths["eigenfunctions"])
        phasefactors = np.load(file_paths["phasefactors"])
        magnus_first_term = np.load(file_paths["magnus_first"])
        magnus_second_term = np.load(file_paths["magnus_second"])

        with open(file_paths["meta_info"], "rb") as meta_file:
            meta_info = pickle.load(meta_file)
            kx, ky, mesh_spacing, hamiltonian = meta_info["kx"], meta_info["ky"], meta_info["mesh_spacing"], meta_info["Hamiltonian_Obj"]

        print("Loaded eigenvalues, eigenfunctions, and metadata from files.")

        # Copy files to temp directory
        for key, file_path in file_paths.items():
            shutil.copy(file_path, os.path.join(temp_dir, os.path.basename(file_path)))

        print(f"Copied existing results to temp directory: {temp_dir}")
    else: 
        # Initialize arrays to store eigenfunctions, eigenvalues, and Magnus terms
        eigenfunctions = np.full((mesh_spacing, mesh_spacing, dim, dim), np.nan, dtype=complex)
        eigenvalues = np.full((mesh_spacing, mesh_spacing, dim), np.nan, dtype=float)
        phasefactors = np.full((mesh_spacing, mesh_spacing, dim), np.nan, dtype=float)
        overall_neighbor_phase_array = np.full((mesh_spacing, mesh_spacing, dim), np.nan, dtype=float)
        magnus_first_term = np.full((mesh_spacing, mesh_spacing, dim, dim), np.nan, dtype=complex)
        magnus_second_term = np.full((mesh_spacing, mesh_spacing, dim, dim), np.nan, dtype=complex)
        
        # Calculate the eigenvalues and eigenfunctions
        eigenvalues, eigenfunctions, phasefactors, overall_neighbor_phase_array, magnus_first_term, magnus_second_term = spiral_eigenvalues_eigenfunctions(
            hamiltonian, kx, ky, mesh_spacing, dim=dim, phase_correction=False
        )

        # Save results
        for key, array in {
            "eigenvalues": eigenvalues,
            "eigenfunctions": eigenfunctions,
            "phasefactors": phasefactors,
            "neighbor_phase_array": overall_neighbor_phase_array,
            "magnus_first": magnus_first_term,
            "magnus_second": magnus_second_term,
        }.items():
            np.save(file_paths[key], array)
            np.save(os.path.join(temp_dir, os.path.basename(file_paths[key])), array)  # Save to temp directory


        # Save meta information
        meta_info = {
            "kx": kx,
            "ky": ky,
            "dkx": dkx, 
            "dky": dky,
            "mesh_spacing": mesh_spacing,
            "Hamiltonian_Obj": hamiltonian, 
            "kx_range": kx_range,
            "ky_range": ky_range,
            "mesh_spacing": mesh_spacing
        }

        # Save the metadata using pickle
        with open(file_paths["meta_info"], "wb") as meta_file:
            pickle.dump(meta_info, meta_file)
        print(f"Saved all results to '{results_subdir}'.")

        with open(os.path.join(temp_dir, "meta_info.pkl"), "wb") as meta_file:
            pickle.dump(meta_info, meta_file)  # Save to temp directory as well
            
        print(f"Saved all results to '{results_subdir}' and copied to temp directory: {temp_dir}")




    eigenvalues = capping_eigenvalues(eigenvalues=eigenvalues, z_limit=z_limit)

    plot_eigenvalues_surface_colorbar(kx, ky, eigenvalues, dim=dim, z_limit=z_limit, color_maps='bwr', norm=None)

    plot_individual_eigenvalues(kx, ky, eigenvalues, dim=dim, z_limit=None)

    # plot_eigenfunction_components(kx, ky, eigenfunctions, band_index=0, components_to_plot=[0])

    # plot_phases(kx, ky, phasefactors, dim=2)

    # plot_neighbor_phases(kx, ky, overall_neighbor_phase_array, dim=2)


def calculation_1d(hamiltonian=hamiltonian):
    # Does the calculation on a line
    band_index = 1

    # Define the line parameters
    angle_deg = 30  # For the Two Orbital Hamiltonian
    # angle_deg = 45  # Line angle in degrees for the Square Lattice Hamiltonian
    k_angle = np.deg2rad(angle_deg)  # Convert into Radians
    kx_shift = 0
    ky_shift = 0
    # ky_shift = - np.pi / 2
    num_points = 100  # Number of points along the line
    k_max = np.sqrt(2) * np.pi
    k_line = np.linspace(-k_max, k_max, num_points)
    line_kx = k_line * np.cos(k_angle) + kx_shift
    line_ky = k_line * np.sin(k_angle) + ky_shift

    # Create the results directory
    file_paths, use_existing, results_subdir = setup_results_directory_1d(
        hamiltonian, angle_deg, kx_shift, ky_shift, num_points, k_max
    )

    if use_existing:
        # Load existing data
        eigenvalues = np.load(file_paths["eigenvalues"])
        eigenfunctions = np.load(file_paths["eigenfunctions"])

        with open(file_paths["meta_info"], "rb") as meta_file:
            meta_info = pickle.load(meta_file)
            hamiltonian = meta_info["Hamiltonian_Obj"]

        print("Loaded eigenvalues and eigenfunctions from files.")
    else:
        # Calculate eigenvalues and eigenfunctions
        eigenvalues, eigenfunctions, _, _ = line_eigenvalues_eigenfunctions(hamiltonian, line_kx, line_ky, band_index)

        # Save results
        np.save(file_paths["eigenvalues"], eigenvalues)
        np.save(file_paths["eigenfunctions"], eigenfunctions)

        # Save meta information
        meta_info = {
            "kx_line": line_kx,
            "ky_line": line_ky,
            "num_points": num_points,
            "Hamiltonian_Obj": hamiltonian  
        }

        # Save metadata using pickle
        with open(file_paths["meta_info"], "wb") as meta_file:
            pickle.dump(meta_info, meta_file)
        print(f"Saved all results to '{results_subdir}'.")

    plot_eigenvalues_line(k_line, eigenvalues, dim = None, bands_to_plot=(0,))


# calculation_1d()
calculation_2d()