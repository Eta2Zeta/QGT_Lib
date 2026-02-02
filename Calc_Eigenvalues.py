import os
import numpy as np
import pickle
import shutil


# from Library import * 
from Library.plotting_lib import *
from Library.Hamiltonian_v1 import *
from Library.Hamiltonian.Hamiltonian_v2 import * 
from Library.Hamiltonian.Chiral_Hamiltonian_Projected import *
from Library.Hamiltonian.Altermagnet_Hamiltonian import *
from Library.Hamiltonian.RuO2Hamiltonian import *
from Library.plotting_lib_3d import *
from Library.eigenvalue_calc_lib import *
from Library.Geometry.zones import ZoneDivider
from Library.utilities import *


# Ensure the temp directory exists
temp_dir = os.path.join(os.getcwd(), "temp")
os.makedirs(temp_dir, exist_ok=True)

# Hamiltonian_Obj = THF_Hamiltonian(A0=0)
# hamiltonian = TwoOrbitalUnspinfulHamiltonian(zeta=1.0, omega = 10.0, A0=0.1, mu=0, magnus_order = 1)
# hamiltonian = SquareLatticeHamiltonian(A0=0, omega=5e0, t1=1, t2=1/np.sqrt(2), t5=0)
# hamiltonian = SquareLatticeHamiltonian(A0=0, omega=5e0, t1=1, t2=1/np.sqrt(2), t5=(1-np.sqrt(2))/4)
# hamiltonian = ChiralHamiltonianProjected(n=5, V=30, A0=0.1, omega=1000)
# bands = (0,1)
# hamiltonian = ChiralHamiltonian(n=5, V=30)
# bands = (4,5)
# hamiltonian = AltermagnetHamiltonian(t1=1.0, t2=0.5, td=2, lamb=2, J=1.0, Nz=4)
# k_max = np.pi #This is for AltermagnetHamiltonian
# bands = (0,1)
# hamiltonian = HaldaneHamiltonian(psi = -np.pi/2, M=0)
# hamiltonian = GrapheneHamiltonian(A0=0)
hamiltonian = RuO2Hamiltonian()
k_max = np.pi
bands = (0,1)
dim = hamiltonian.dim

def calculation_2d(hamiltonian = hamiltonian, force_new=True):
    # Does the calculation on 2d 


    # Create kx and ky arrays
    kx_range = (-k_max, k_max)
    ky_range = (-k_max, k_max)
    mesh_spacing = 100

    kx = np.linspace(-k_max, k_max, mesh_spacing)
    ky = np.linspace(-k_max, k_max, mesh_spacing)
    kx, ky = np.meshgrid(kx, ky)
    dkx = np.abs(kx[0, 1] - kx[0, 0])  # Spacing in the x-direction (constant for a uniform grid)
    dky = np.abs(ky[1, 0] - ky[0, 0])  # Spacing in the y-direction (constant for a uniform grid)
    z_limit = 1000

    # Create the results directory
    file_paths, use_existing, results_subdir = setup_results_directory(hamiltonian, kx_range, ky_range, mesh_spacing, force_new=force_new)

    if use_existing:
        # Load existing data
        eigenvalues = np.load(file_paths["eigenvalues"])
        eigenfunctions = np.load(file_paths["eigenfunctions"])

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
        
        # Calculate the eigenvalues and eigenfunctions
        # eigenvalues, eigenfunctions, _, _ = grid_eigenvalues_eigenfunctions_ordered(
        #     hamiltonian, kx, ky, mesh_spacing, dim=dim
        # )

        eigenvalues, eigenfunctions, _, _ = grid_eigenvalues_eigenfunctions(
            hamiltonian, kx, ky, mesh_spacing, dim=dim
        )

        # eigenvalues = analytic_eigenvalues_2d(hamiltonian, kx, ky, mesh_spacing, dim)


        # Save results
        for key, array in {
            "eigenvalues": eigenvalues,
            "eigenfunctions": eigenfunctions
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

    # plot_eigenvalues_surface_colorbar(kx, ky, eigenvalues, dim=dim, z_limit=z_limit, stride_size=2, color_maps='bwr', norm=None, bands_to_plot=None)
    plot_eigenvalues_surface_colorbar(kx, ky, eigenvalues, dim=dim, z_limit=z_limit, stride_size=2, color_maps='bwr', norm=None, bands_to_plot=1)

    # plot_individual_eigenvalues(kx, ky, eigenvalues, dim=dim, z_limit=None)

    # plot_eigenfunction_components(kx, ky, eigenfunctions, band_index=0, components_to_plot=[0,1,2,3])

    # plot_phases(kx, ky, phasefactors, dim=2)

    # plot_neighbor_phases(kx, ky, overall_neighbor_phase_array, dim=2)


def calculation_1d(hamiltonian=hamiltonian):
    print("Currently performing 1D calculation")
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


def calculation_3d(hamiltonian=hamiltonian, force_new=True):
    print("Performing 3D calculation...")
    
    # 3D Grid Parameters
    mesh_size = 100
    k_range = np.pi
    
    kx_vals = np.linspace(-k_range, k_range, mesh_size)
    ky_vals = np.linspace(-k_range, k_range, mesh_size)
    kz_vals = np.linspace(-k_range, k_range, mesh_size)
    
    # Setup results path
    subfolder = f"{hamiltonian.name}_3D_mesh_{mesh_size}"
    results_dir = os.path.join(os.getcwd(), "results", subfolder)
    os.makedirs(results_dir, exist_ok=True)
    
    filename_eig = os.path.join(results_dir, "eigenvalues_3d.npy")
    
    # Check if calculation exists
    if not force_new and os.path.exists(filename_eig):
        print("Loading existing 3D results...")
        eigenvalues_3d = np.load(filename_eig)
    else:
        # Use the new function from library
        eigenvalues_3d, _ = compute_eigenvalues_3d(hamiltonian, kx_vals, ky_vals, kz_vals)
        np.save(filename_eig, eigenvalues_3d)
        print("Calculation complete and saved.")

    # --- Plotting ---
    print("Generating plots...")
    
    band_idx = 0
    eig_band = eigenvalues_3d[:, :, :, band_idx] # [x, y, z]
    
    # 1. Stacked Slices
    # Use the new helper function
    # plot_3d_stacked_slices_from_volume(eigenvalues_3d, kx_vals, ky_vals, kz_vals, 
    #                                    band_index=band_idx, num_slices=10, 
    #                                    title=f"RuO2 Band {band_idx} Eigenvalues (10 Z-Slices)")

    # print("Generating isosurface plot...")
    # 2. Isosurface
    min_e = np.min(eig_band)
    max_e = np.max(eig_band)
    avg_e = (min_e + max_e) / 2
    
    print(f"Band {band_idx} Energy Range: [{min_e:.3f}, {max_e:.3f}]")
    
    
    plot_isosurface(eig_band, avg_e, kx_vals, ky_vals, kz_vals, band_index=band_idx, 
                    title=f"RuO2 Band {band_idx} Isosurface (E={avg_e:.3f})", step_size=3)

    print("Generating arbitrary slice plots...")
    orientation = 'x'
    # Define shifts: for xyz, max range is larger (sqrt(3)*pi), for others pi or sqrt(2)*pi
    # Let's just pick a shift = 0 for central slice and one offset
    
    shift_val = np.pi/3
    plot_arbitrary_slice(eigenvalues_3d, orientation, shift_val, kx_vals, ky_vals, kz_vals, 
                            title=f"Slice {orientation} (shift=pi/3)")

    print("Generating volumetric cloud plot...")
    # 4. Volumetric Cloud
    cloud_filename = os.path.join(results_dir, f"volumetric_cloud_band_{band_idx}.html")
    plot_volumetric_cloud(eigenvalues_3d, kx_vals, ky_vals, kz_vals, band_index=band_idx, 
                          opacity=0.1, surface_count=20, 
                          title=f"RuO2 Band {band_idx} Cloud", filename=cloud_filename)


# calculation_1d()
# calculation_2d(force_new=True)


calculation_3d(hamiltonian, force_new=False)
