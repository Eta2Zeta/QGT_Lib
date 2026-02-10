import os
import numpy as np
import pickle
import shutil
import hashlib
import json

# from Library import * 
from Library.plotting_lib_2d import *
from Library.Hamiltonian_v1 import *
from Library.Hamiltonian.Hamiltonian_v2 import * 
from Library.Hamiltonian.Chiral_Hamiltonian_Projected import *
from Library.Hamiltonian.gWaveAltermagnetHamiltonian import *
from Library.Hamiltonian.RuO2Hamiltonian import *
from Library.plotting_lib_3d import *
from Library.eigenvalue_calc_lib import *
from Library.Geometry.zones import ZoneDivider
from Library.utilities import setup_results_directory, centered_kvals
from Library.data_management_utils import setup_3D_Eigen_results_directory


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
# hamiltonian = RuO2Hamiltonian()
hamiltonian = gWaveAltermagnetHamiltonian(t1=0.8, t2=0.0, t3=0.3, t4=0.3, mu=0, Jx=0.0, Jy=0.0, Jz=0.2, lamb=0.1, lamb_z=0.1)
k_max = np.pi
bands = (0,1)
dim = hamiltonian.dim

def calculation_2d(hamiltonian = hamiltonian, force_new=True, include_end_points=True, kz=0):
    # Create kx and ky arrays
    
    mesh_spacing = 100

    if include_end_points:
        kx = np.linspace(-k_max, k_max, mesh_spacing)
        ky = np.linspace(-k_max, k_max, mesh_spacing)
        kx_range = (-k_max, k_max)
        ky_range = (-k_max, k_max)
    else:
        kx = np.linspace(-k_max, k_max, mesh_spacing + 2)[1:-1]
        ky = np.linspace(-k_max, k_max, mesh_spacing + 2)[1:-1]
        kx_range = (kx[0], kx[-1])
        ky_range = (ky[0], ky[-1])
    
    kx, ky = np.meshgrid(kx, ky)
    dkx = np.abs(kx[0, 1] - kx[0, 0])  # Spacing in the x-direction (constant for a uniform grid)
    dky = np.abs(ky[1, 0] - ky[0, 0])  # Spacing in the y-direction (constant for a uniform grid)
    z_limit = 1000

    # Create the results directory
    # file_paths, use_existing, results_subdir = setup_results_directory(hamiltonian, kx_range, ky_range, mesh_spacing, force_new=force_new)
    from Library.data_management_utils_2D import setup_2D_Eigen_results_directory

    kvals_mode = "endpoints" if include_end_points else "centered"
    
    file_paths, use_existing, results_subdir, meta_target = setup_2D_Eigen_results_directory(
        hamiltonian, 
        kx_range, 
        ky_range, 
        mesh_spacing, 
        include_endpoints=include_end_points, 
        force_new=force_new,
        kvals_mode=kvals_mode
    )

    if use_existing:
        # Load existing data
        eigenvalues = np.load(file_paths["eigenvalues"])
        eigenfunctions = np.load(file_paths["eigenfunctions"])

        with open(file_paths["meta_pkl"], "rb") as meta_file:
            meta_info_pkl = pickle.load(meta_file)
            
            # Extract what we need from the pickle (usually objects or things not in JSON)
            # e.g. hamiltonian object if we stored it
            if "Hamiltonian_Obj" in meta_info_pkl:
                hamiltonian = meta_info_pkl["Hamiltonian_Obj"]
            
            # We can also trust the parameters we passed in, or read from JSON
            
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
            hamiltonian, kx, ky, mesh_spacing, dim=dim, kz=kz
        )

        # eigenvalues = analytic_eigenvalues_2d(hamiltonian, kx, ky, mesh_spacing, dim)


        # Save results
        for key, array in {
            "eigenvalues": eigenvalues,
            "eigenfunctions": eigenfunctions
        }.items():
            np.save(file_paths[key], array)
            np.save(os.path.join(temp_dir, os.path.basename(file_paths[key])), array)  # Save to temp directory

        ham_name = getattr(hamiltonian, "name", "Hamiltonian")

        meta_info_json = meta_target.copy()
        meta_info_json.update({
             "hamiltonian_name": ham_name,
             "kx_range": kx_range,
             "ky_range": ky_range,
             "mesh_spacing": mesh_spacing,
             "dkx": dkx, 
             "dky": dky,
             "kz": kz, # kz was passed in
             "include_endpoints": include_end_points,
             
        })
        
        with open(file_paths["meta_json"], "w") as f:
            json.dump(meta_info_json, f, indent=2, sort_keys=True)

        # Save meta information (Pickle) - for objects
        meta_info_pkl = {
            "kx": kx,
            "ky": ky,
            "kz": kz,
            "dkx": dkx, 
            "dky": dky,
            "mesh_spacing": mesh_spacing,
            "Hamiltonian_Obj": hamiltonian, 
            "kx_range": kx_range,
            "ky_range": ky_range,
            "kvals_mode": kvals_mode,
            "include_endpoints": include_end_points
        }

        # Save the metadata using pickle
        with open(file_paths["meta_pkl"], "wb") as meta_file:
            pickle.dump(meta_info_pkl, meta_file)
            
        print(f"Saved all results to '{results_subdir}'.")

        with open(os.path.join(temp_dir, "meta_info.pkl"), "wb") as meta_file:
            pickle.dump(meta_info_pkl, meta_file)  # Save to temp directory as well
        
        shutil.copy(file_paths["meta_json"], os.path.join(temp_dir, "meta.json"))

        print(f"Saved all results to '{results_subdir}' and copied to temp directory: {temp_dir}")




    eigenvalues = capping_eigenvalues(eigenvalues=eigenvalues, z_limit=z_limit)

    plot_eigenvalues_surface_colorbar(kx, ky, eigenvalues, dim=dim, z_limit=z_limit, stride_size=2, color_maps='bwr', norm=None, bands_to_plot=None)
    # plot_eigenvalues_surface_colorbar(kx, ky, eigenvalues, dim=dim, z_limit=z_limit, stride_size=2, color_maps='bwr', norm=None, bands_to_plot=(0,1))
    # plot_eigenvalues_surface_colorbar(kx, ky, eigenvalues, dim=dim, z_limit=z_limit, stride_size=2, color_maps='bwr', norm=None, bands_to_plot=0)
    # plot_eigenvalues_surface_colorbar(kx, ky, eigenvalues, dim=dim, z_limit=z_limit, stride_size=2, color_maps='bwr', norm=None, bands_to_plot=1)
    # plot_eigenvalues_surface_colorbar(kx, ky, eigenvalues, dim=dim, z_limit=z_limit, stride_size=2, color_maps='bwr', norm=None, bands_to_plot=2)
    # plot_eigenvalues_surface_colorbar(kx, ky, eigenvalues, dim=dim, z_limit=z_limit, stride_size=2, color_maps='bwr', norm=None, bands_to_plot=3)

    # plot_individual_eigenvalues(kx, ky, eigenvalues, dim=dim, z_limit=None)
    
    # --- New: Plot Eigenvalues along a diagonal cut ---
    print("Plotting eigenvalues along diagonal cut...")
    extract_and_plot_eigenvalues_along_line(kx, ky, eigenvalues, start_k=(-np.pi, 0), end_k=(np.pi, 0), num_points=100)
    extract_and_plot_eigenvalues_along_line(kx, ky, eigenvalues, start_k=(-np.pi, -np.pi/np.sqrt(3)), end_k=(np.pi, np.pi/np.sqrt(3)), num_points=100)
    extract_and_plot_eigenvalues_along_line(kx, ky, eigenvalues, start_k=(0, -np.pi), end_k=(0, np.pi), num_points=100)

    # plot_eigenfunction_components(kx, ky, eigenfunctions, band_index=0, components_to_plot=[0,1,2,3])

    # plot_phases(kx, ky, phasefactors, dim=2)

    # plot_neighbor_phases(kx, ky, overall_neighbor_phase_array, dim=2)


def calculation_1d(hamiltonian=hamiltonian):
    #TODO: make the definition for the end points just be two points
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


def calculation_3d(hamiltonian=hamiltonian, force_new=True, include_end_points=True):
    print("Performing 3D calculation...")
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    k_range = 0.95*np.pi
    mesh_nx, mesh_ny, mesh_nz = 100, 100, 1


    if include_end_points: 
        kx_vals = np.linspace(-k_range, k_range, mesh_nx)
        ky_vals = np.linspace(-k_range, k_range, mesh_ny)
        kz_vals = np.linspace(-k_range, k_range, mesh_nz)
        kvals_mode = "endpoints"
    else: 
        kx_vals = centered_kvals(k_range, mesh_nx)
        ky_vals = centered_kvals(k_range, mesh_ny)
        kz_vals = centered_kvals(k_range, mesh_nz)
        kvals_mode = "centered"
    
    # spacings (uniform by construction)
    dkx = float(kx_vals[1] - kx_vals[0]) if mesh_nx > 1 else (2*k_range)  # fallback for N=1
    dky = float(ky_vals[1] - ky_vals[0]) if mesh_ny > 1 else (2*k_range)
    dkz = float(kz_vals[1] - kz_vals[0]) if mesh_nz > 1 else (2*k_range)
    mesh_shape = (len(kx_vals), len(ky_vals), len(kz_vals))


    file_paths, use_existing, results_dir, meta_target = setup_3D_Eigen_results_directory(
        hamiltonian, 
        [kx_vals[0], kx_vals[-1]], 
        [ky_vals[0], ky_vals[-1]], 
        [kz_vals[0], kz_vals[-1]], 
        mesh_shape=mesh_shape,
        include_endpoints=include_end_points,
        force_new=force_new,
        kvals_mode=kvals_mode
    )
    
    filename_eig = file_paths["eigenvalues"] # os.path.join(results_dir, "eigenvalues_3d.npy")
    
    # Check if calculation exists
    if use_existing:
        print("Loading existing 3D results...")
        eigenvalues_3d = np.load(filename_eig)
        # Load meta info if needed, but we don't strictly need it for plotting right here
    else:
        # Use the new function from library
        eigenvalues_3d, eigenvectors_3d = compute_eigenvalues_3d(hamiltonian, kx_vals, ky_vals, kz_vals)
        np.save(filename_eig, eigenvalues_3d)
        np.save(file_paths["eigenfunctions"], eigenvectors_3d) # Saving eigenvectors too!
        
        ham_name = getattr(hamiltonian, "name", "Hamiltonian")

        meta_info = {
            "hamiltonian_name": ham_name,
            "k_range": float(k_range),
            "kvals_mode": kvals_mode,
            "kx_range": [float(kx_vals[0]), float(kx_vals[-1])],
            "ky_range": [float(ky_vals[0]), float(ky_vals[-1])],
            "kz_range": [float(kz_vals[0]), float(kz_vals[-1])],
            "mesh_shape": list(mesh_shape),
            "dk": [dkx, dky, dkz],
            "include_endpoints": bool(include_end_points),
        }


        # build a JSON-safe meta dict (see #2)
        with open(file_paths["meta_json"], "w") as f:
            json.dump(meta_info, f, indent=2, sort_keys=True)

        meta_pkl = {
            "kx_vals": kx_vals,
            "ky_vals": ky_vals,
            "kz_vals": kz_vals,
            "mesh_shape": mesh_shape,
            "dk": (dkx, dky, dkz),
            "include_endpoints": bool(include_end_points),
            "kvals_mode": kvals_mode,
            "Hamiltonian_Obj": hamiltonian,
        }
        with open(file_paths["meta_pkl"], "wb") as f:
            pickle.dump(meta_pkl, f)

        

        print("Calculation complete and saved.")


    print(f"Copying 3D results to temp directory: {temp_dir}")
    shutil.copy(file_paths["meta_json"], os.path.join(temp_dir, "meta.json"))
    shutil.copy(file_paths["meta_pkl"],  os.path.join(temp_dir, "meta_info.pkl"))
    shutil.copy(file_paths["eigenvalues"], os.path.join(temp_dir, "eigenvalues_3d.npy"))
    shutil.copy(file_paths["eigenfunctions"], os.path.join(temp_dir, "eigenvectors_3d.npy"))
    print("3D results copied to temp.")

        
    # --- Plotting ---
    print("Generating plots...")
    
    band_idx = 0
    eig_band = eigenvalues_3d[:, :, :, band_idx] # [x, y, z]
    
    # 1. Stacked Slices
    # Use the new helper function
    # plot_3d_stacked_slices_from_volume(eigenvalues_3d, kx_vals, ky_vals, kz_vals, 
    #                                    band_index=band_idx, num_slices=3, 
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
    orientation = 'z'
    shift_val = 0
    plot_arbitrary_slice_no_interp(eigenvalues_3d, orientation, shift_val, kx_vals, ky_vals, kz_vals, 
                            title=f"Slice {orientation} (shift={shift_val})")

    print("Generating volumetric cloud plot...")
    # 4. Volumetric Cloud
    cloud_filename = os.path.join(results_dir, f"volumetric_cloud_band_{band_idx}.html")
    plot_volumetric_cloud(eigenvalues_3d, kx_vals, ky_vals, kz_vals, band_index=band_idx, 
                          opacity=0.1, surface_count=20, 
                          title=f"RuO2 Band {band_idx} Cloud", filename=cloud_filename)


# calculation_1d()
calculation_2d(hamiltonian, force_new=True, include_end_points=False, kz=0.01*np.pi) 
# calculation_3d(hamiltonian, force_new=False, include_end_points=False)

