import os
import numpy as np
import pickle
import shutil
import hashlib
import json

# from Library import * 
from Library.plotting_lib_2d import *
from Library.Hamiltonian.Hamiltonian import * 
from Library.Hamiltonian.THF_Hamiltonian import *
from Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected import *
from Library.Hamiltonian.SquareLatticeHamiltonian import *
from Library.Hamiltonian.gWaveAltermagnetHamiltonian import *
from Library.Hamiltonian.AltermagnetHamiltonian import *
from Library.Hamiltonian.RuO2Hamiltonian import *
from Library.plotting_lib_3d import plot_degeneracy_3d, plot_degeneracy_on_path_3d, plot_arbitrary_slice_no_interp, plot_volumetric_cloud
from Library.plotting_lib_2d import *
from Library.plotting_lib_2d import plot_degeneracy_2d
from Library.plotting_lib_1d import *
from Library.eigenvalue_calc_lib import *
from Library.utilities import centered_kvals, generate_1d_lines_at_angles
from Library.data_management_utils_2d import setup_2D_Eigen_results_directory



# Ensure the temp directory exists
temp_dir = os.path.join(os.getcwd(), "temp")
os.makedirs(temp_dir, exist_ok=True)

# Hamiltonian_Obj = THF_Hamiltonian(A0=0)
# Hamiltonian_Obj = TwoOrbitalUnspinfulHamiltonian(zeta=1.0, omega = 10.0, A0=0.1, mu=0, magnus_order = 1)
Hamiltonian_Obj = SquareLatticeHamiltonian(A0=0, omega=5e0, t1=1, t2=1/np.sqrt(2), t5=0)
# Hamiltonian_Obj = SquareLatticeHamiltonian(A0=0, omega=5e0, t1=1, t2=1/np.sqrt(2), t5=(1-np.sqrt(2))/4)
# Hamiltonian_Obj = ChiralHamiltonianProjected(n=5, V=30, A0=0.1, omega=1000)
# bands = (0,1)
# Hamiltonian_Obj = ChiralHamiltonian(n=5, V=30)
# bands = (4,5)
# Hamiltonian_Obj = AltermagnetHamiltonian(t1=1.0, t2=0.5, td=2, lamb=2, J=1.0, Nz=4)
# k_max = np.pi #This is for AltermagnetHamiltonian
# bands = (0,1)
# Hamiltonian_Obj = HaldaneHamiltonian(psi = -np.pi/2, M=0)
# Hamiltonian_Obj = GrapheneHamiltonian(A0=0)
# Hamiltonian_Obj = RuO2Hamiltonian(lamb_z=0)
# Hamiltonian_Obj = gWaveAltermagnetHamiltonian(t1=0.3, t2=0.3, t3=0.3, t4=0.3, mu=0, Jx=0.0, Jy=0.0, Jz=0.2, lamb=0.1, lamb_z=0.1)
k_max = 2*np.pi
bands = (2,3)
dim = Hamiltonian_Obj.dim

def calculation_2d(Hamiltonian_Obj = Hamiltonian_Obj, force_new=True, include_end_points=True, kk=0, order="xyz"):
    # Create ki and kj arrays
    
    mesh_spacing = 150

    if include_end_points:
        ki = np.linspace(-k_max, k_max, mesh_spacing)
        kj = np.linspace(-k_max, k_max, mesh_spacing)
        ki_range = (-k_max, k_max)
        kj_range = (-k_max, k_max)
    else:
        ki = np.linspace(-k_max, k_max, mesh_spacing + 2)[1:-1]
        kj = np.linspace(-k_max, k_max, mesh_spacing + 2)[1:-1]
        ki_range = (ki[0], ki[-1])
        kj_range = (kj[0], kj[-1])
    
    ki, kj = np.meshgrid(ki, kj)
    dki = 2*k_max/mesh_spacing
    dkj = 2*k_max/mesh_spacing
    z_limit = 1000


    kvals_mode = "endpoints" if include_end_points else "centered"

    ham_params = Hamiltonian_Obj.get_parameters_dict(parameter="2D")
    
    Hamiltonian_name = getattr(Hamiltonian_Obj, "name", "Hamiltonian_Obj")
    
    meta_params = {
        "hamiltonian_name": Hamiltonian_name,
        "ki_range": [float(ki_range[0]), float(ki_range[1])],
        "kj_range": [float(kj_range[0]), float(kj_range[1])],
        "mesh_spacing": int(mesh_spacing),
        "dki": float(dki), 
        "dkj": float(dkj),
        "kk": float(kk) if kk is not None else None,
        "include_endpoints": bool(include_end_points),
        "kvals_mode": str(kvals_mode),
        "order": str(order),
        "hamiltonian_params": ham_params
    }

    file_paths, use_existing, results_subdir, meta_target = setup_2D_Eigen_results_directory(
        meta_params=meta_params,
        force_new=force_new
    )

    if use_existing:
        # Load existing data
        eigenvalues = np.load(file_paths["eigenvalues"])
        eigenfunctions = np.load(file_paths["eigenfunctions"])

        with open(file_paths["meta_pkl"], "rb") as meta_file:
            meta_info_pkl = pickle.load(meta_file)
            
            # Extract what we need from the pickle (usually objects or things not in JSON)
            # e.g. Hamiltonian_Obj object if we stored it
            if "Hamiltonian_Obj" in meta_info_pkl:
                Hamiltonian_Obj = meta_info_pkl["Hamiltonian_Obj"]
            
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
        #     Hamiltonian_Obj, ki, kj, mesh_spacing, dim=dim
        # )

        eigenvalues, eigenfunctions, _, _ = grid_eigenvalues_eigenfunctions(
            Hamiltonian_Obj, ki, kj, mesh_spacing, dim=dim, kk=kk, order=order
        )

        # eigenvalues, eigenfunctions, _, _, _, _ = spiral_eigenvalues_eigenfunctions(
        #     Hamiltonian_Obj, ki, kj, mesh_spacing, dim=dim
        # )

        # eigenvalues = analytic_eigenvalues_2d(Hamiltonian_Obj, ki, kj, mesh_spacing, dim)


        # Save results
        for key, array in {
            "eigenvalues": eigenvalues,
            "eigenfunctions": eigenfunctions
        }.items():
            np.save(file_paths[key], array)
            np.save(os.path.join(temp_dir, os.path.basename(file_paths[key])), array)  # Save to temp directory

        meta_info_json = meta_target.copy()
        
        with open(file_paths["meta_json"], "w") as f:
            json.dump(meta_info_json, f, indent=2, sort_keys=True)

        # Save meta information (Pickle) - for objects
        meta_info_pkl = meta_target
        meta_info_pkl["Hamiltonian_Obj"] = Hamiltonian_Obj
        meta_info_pkl["ki"] = ki
        meta_info_pkl["kj"] = kj

        # Save the metadata using pickle
        with open(file_paths["meta_pkl"], "wb") as meta_file:
            pickle.dump(meta_info_pkl, meta_file)
            
        print(f"Saved all results to '{results_subdir}'.")

        with open(os.path.join(temp_dir, "meta_info.pkl"), "wb") as meta_file:
            pickle.dump(meta_info_pkl, meta_file)  # Save to temp directory as well
        
        shutil.copy(file_paths["meta_json"], os.path.join(temp_dir, "meta.json"))

        print(f"Saved all results to '{results_subdir}' and copied to temp directory: {temp_dir}")




    eigenvalues = capping_eigenvalues(eigenvalues=eigenvalues, z_limit=z_limit)

    plot_eigenvalues_surface_colorbar(ki, kj, eigenvalues, dim=dim, z_limit=z_limit, stride_size=2, color_maps='default', norm=None, bands_to_plot=None, results_dir=results_subdir, save_fig=True)
    
    # --- New: Plot 2D Degeneracy Map ---
    print("Plotting 2D Degeneracy Map...")
    plot_degeneracy_2d(ki, kj, eigenvalues, threshold=0.02, title=f"Band Degeneracy Map ({Hamiltonian_Obj.name})", results_dir=results_subdir, save_fig=True)

def calculation_1d(Hamiltonian_Obj=Hamiltonian_Obj):
    #TODO: make the definition for the end points just be two points
    print("Currently performing 1D calculation")
    # Does the calculation on a line
    band_index = 1

    # Define the line parameters
    angle_deg = 30  # For the Two Orbital Hamiltonian_Obj
    # angle_deg = 45  # Line angle in degrees for the Square Lattice Hamiltonian_Obj
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
        Hamiltonian_Obj, angle_deg, kx_shift, ky_shift, num_points, k_max
    )

    if use_existing:
        # Load existing data
        eigenvalues = np.load(file_paths["eigenvalues"])
        eigenfunctions = np.load(file_paths["eigenfunctions"])

        with open(file_paths["meta_info"], "rb") as meta_file:
            meta_info = pickle.load(meta_file)
            Hamiltonian_Obj = meta_info["Hamiltonian_Obj"]

        print("Loaded eigenvalues and eigenfunctions from files.")
    else:
        # Calculate eigenvalues and eigenfunctions
        eigenvalues, eigenfunctions, _, _ = line_eigenvalues_eigenfunctions(Hamiltonian_Obj, line_kx, line_ky, band_index)

        # Save results
        np.save(file_paths["eigenvalues"], eigenvalues)
        np.save(file_paths["eigenfunctions"], eigenfunctions)

        # Save meta information
        meta_info = {
            "kx_line": line_kx,
            "ky_line": line_ky,
            "num_points": num_points,
            "Hamiltonian_Obj": Hamiltonian_Obj  
        }

        # Save metadata using pickle
        with open(file_paths["meta_info"], "wb") as meta_file:
            pickle.dump(meta_info, meta_file)
        print(f"Saved all results to '{results_subdir}'.")

    plot_eigenvalues_line(k_line, eigenvalues, dim = None, bands_to_plot=(0,))


def calculation_3d(Hamiltonian_Obj=Hamiltonian_Obj, force_new=True, include_end_points=True):
    print("Performing 3D calculation...")
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    k_range = 1*np.pi
    mesh = 120
    mesh_nx, mesh_ny, mesh_nz = mesh, mesh, mesh

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
    dkx = 2*k_range/mesh_nx
    dky = 2*k_range/mesh_ny
    dkz = 2*k_range/mesh_nz
    mesh_shape = (len(kx_vals), len(ky_vals), len(kz_vals))


    file_paths, use_existing, results_dir, meta_target = setup_3D_Eigen_results_directory(
        Hamiltonian_Obj, 
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
        eigenvalues_3d, eigenvectors_3d = compute_eigenvalues_3d(Hamiltonian_Obj, kx_vals, ky_vals, kz_vals)
        np.save(filename_eig, eigenvalues_3d)
        np.save(file_paths["eigenfunctions"], eigenvectors_3d) # Saving eigenvectors too!
        
        ham_name = getattr(Hamiltonian_Obj, "name", "Hamiltonian_Obj")

        meta_info = meta_target.copy()
        meta_info.update({
             "hamiltonian_name": ham_name,
             "dk": [dkx, dky, dkz],
        })

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
            "Hamiltonian_Obj": Hamiltonian_Obj,
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
    
    band_idx = 2
    eig_band = eigenvalues_3d[:, :, :, band_idx] # [x, y, z]
    
    plot_volumetric_cloud(eig_band, kx_vals, ky_vals, kz_vals,
                          opacity=0.2, levels=[0],
                          results_dir=results_dir, save_fig=True)
    

    orientation = 'z'
    shift_val = 0
    plot_arbitrary_slice_no_interp(eigenvalues_3d, orientation, shift_val, kx_vals, ky_vals, kz_vals, 
                            title=f"Slice {orientation} (shift={shift_val})",
                            results_dir=results_dir, save_fig=True)

    # 5. 3D Degeneracy Map
    print("Plotting 3D Degeneracy Map...")
    plot_degeneracy_3d(kx_vals, ky_vals, kz_vals, eigenvalues_3d, threshold=0.05,
                       title=f"3D Band Degeneracy Map ({Hamiltonian_Obj.name})",
                       results_dir=results_dir, save_fig=True)


def calculation_sym_points(Hamiltonian_Obj=Hamiltonian_Obj, force_new=True, use_analytical=False,
                           num_points_per_segment=100, bands_to_plot=None):
    """
    Calculate and plot the band structure along the Hamiltonian's own high-symmetry k-path.

    The path is defined by Hamiltonian_Obj.get_sym_path(), which returns a dict of
    labelled k-points and an ordered list of labels. Works for both 2D and 3D paths.

    Parameters
    ----------
    bands_to_plot : list of int, optional
        Indices of bands to include in the plot. None plots all bands.
    """
    print(f"Performing symmetry-path band structure calculation ({Hamiltonian_Obj.name})...")

    # --- Get path from the Hamiltonian itself ---
    sym_points, path_labels = Hamiltonian_Obj.get_sym_path()
    nodes = [np.array(sym_points[label]) for label in path_labels]

    # Determine dimensionality from the first node
    k_dim = nodes[0].shape[0]  # 2 or 3

    # --- Build interpolated k-path (same logic as generate_3d_sym_lines) ---
    all_k_points = [nodes[0]]
    all_k_dist   = [0.0]
    node_indices = [0]
    cum_dist = 0.0

    for i in range(len(nodes) - 1):
        start = nodes[i]
        end   = nodes[i + 1]
        dist  = np.linalg.norm(end - start)
        pts   = np.linspace(start, end, num_points_per_segment + 1)[1:]
        dists = np.linspace(0, dist,    num_points_per_segment + 1)[1:]
        for p, d in zip(pts, dists):
            all_k_points.append(p)
            all_k_dist.append(cum_dist + d)
        cum_dist += dist
        node_indices.append(len(all_k_points) - 1)

    k_path      = np.array(all_k_points)  # (N, k_dim)
    k_dist      = np.array(all_k_dist)
    path_points = np.array(nodes)

    # Pad to 3 columns so eigenvalues_along_path always gets (N, 3)
    if k_dim == 2:
        k_path_3d = np.column_stack([k_path, np.zeros(len(k_path))])
    else:
        k_path_3d = k_path

    # --- Results directory ---
    file_paths, use_existing, results_dir, meta_target = setup_sym_points_results_directory(
        Hamiltonian_Obj,
        path_points,
        path_labels,
        num_points_per_segment,
        force_new=force_new
    )

    if use_existing:
        print("Loading existing results...")
        eigenvalues = np.load(file_paths["eigenvalues"])
    else:
        print("Calculating eigenvalues along path...")
        eigenvalues = eigenvalues_along_path(Hamiltonian_Obj, k_path_3d, use_analytical=use_analytical)

        np.save(file_paths["eigenvalues"], eigenvalues)
        with open(file_paths["meta_json"], "w") as f:
            json.dump(meta_target, f, indent=2)
        with open(file_paths["meta_pkl"], "wb") as f:
            pickle.dump(meta_target, f)
        print("Calculation complete and saved.")

    # --- Plot ---
    plot_band_structure_sym(
        k_dist,
        eigenvalues,
        node_indices,
        path_labels,
        bands_to_plot=bands_to_plot,
        title=f"Band Structure along symmetry path ({Hamiltonian_Obj.name})",
        results_dir=results_dir,
        save_fig=True,
        use_analytical=use_analytical
    )

    plot_degeneracy_on_path_3d(
        k_path, 
        eigenvalues, 
        threshold=0.02, 
        title=f"Degeneracy along Path ({Hamiltonian_Obj.name})",
        results_dir=results_dir,
        save_fig=True
    )

def calculation_1d_at_angles(
    Hamiltonian_Obj=Hamiltonian_Obj,
    k_max=2*np.pi,
    num_angles=10,
    num_points_per_line=1000,
    force_new=True,
    use_analytical=False,
    bands_to_plot=None
):
    """
    Calculate and plot the band structure along 1D lines through the origin at various angles.
    Generates an interactive Plotly plot with a slider for the angle.
    """
    print(f"Performing 1D angled band structure calculation ({Hamiltonian_Obj.name})...")
    
    # Generate the paths
    k_path, k_vals, angles = generate_1d_lines_at_angles(k_max, num_angles, num_points_per_line)
    
    # Pad to 3 columns (N, 3) for eigenvalues_along_path which expects up to 3D k-vectors
    k_path_3d = np.column_stack([k_path, np.zeros(len(k_path))])
    
    # Setup directory
    file_paths, use_existing, results_dir, meta_target = setup_1D_angles_results_directory(
        Hamiltonian_Obj, k_max, num_angles, num_points_per_line, force_new=force_new
    )
    
    if use_existing:
        print("Loading existing 1D angles results...")
        eigenvalues_flat = np.load(file_paths["eigenvalues"])
        num_bands = eigenvalues_flat.shape[1]
    else:
        print("Calculating eigenvalues along all angled paths...")
        eigenvalues_flat = eigenvalues_along_path(Hamiltonian_Obj, k_path_3d, use_analytical=use_analytical)
        num_bands = eigenvalues_flat.shape[1]
        
        np.save(file_paths["eigenvalues"], eigenvalues_flat)
        with open(file_paths["meta_json"], "w") as f:
            json.dump(meta_target, f, indent=2)
        with open(file_paths["meta_pkl"], "wb") as f:
            pickle.dump(meta_target, f)
        print("Calculation complete and saved.")
        
    # Reshape eigenvalues back to (num_angles, num_points_per_line, num_bands)
    eigenvalues = eigenvalues_flat.reshape((num_angles, num_points_per_line, num_bands))
    
    # Plot using our new interactive slider function
    plot_band_structure_angles_slider(
        k_vals=k_vals,
        eigenvalues=eigenvalues,
        angles=angles,
        bands_to_plot=bands_to_plot,
        title=f"1D Band Structure vs Angle ({Hamiltonian_Obj.name})",
        results_dir=results_dir,
        save_fig=True,
        show=True
    )


# calculation_1d()
calculation_2d(Hamiltonian_Obj, force_new=False, include_end_points=False, kk=0.0*np.pi)
# calculation_3d(Hamiltonian_Obj, force_new=False, include_end_points=True)
# calculation_sym_points(Hamiltonian_Obj, force_new=True)
# calculation_1d_at_angles(Hamiltonian_Obj, k_max=k_max, num_angles=200, bands_to_plot=bands)
