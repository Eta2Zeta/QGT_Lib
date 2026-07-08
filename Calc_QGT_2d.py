import sys
import os
import numpy as np
import pickle
import shutil
from tqdm import tqdm  # Import tqdm for progress bar
from multiprocessing import Pool, cpu_count
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

# from Library import * 
from Library.Hamiltonian_v1 import *
from Library.Hamiltonian.Hamiltonian import * 
from Library.eigenvalue_calc_lib import *
from Library.eigenvalue_calc_lib_1d import eigenvalues_along_path
from Library.QGT_lib import *
from Library.topology import *
from Library.utilities import *
from Library.plotting_lib_2d import *
from Library.Hamiltonian.RuO2Hamiltonian import *
from Library.Hamiltonian.gWaveAltermagnetHamiltonian import *
from Library.data_management_utils_2d import setup_2D_QGT_results_directory


def _save_sym_line_eigenvalues(Hamiltonian_Obj, results_subdir, *, num_points_per_segment=100):
    """
    Save band energies along the Hamiltonian's own 2D high-symmetry path.
    """
    output_path = os.path.join(results_subdir, "eigenvalues_sym_line.npz")
    if os.path.exists(output_path):
        return output_path

    if not hasattr(Hamiltonian_Obj, "get_sym_path"):
        print("Hamiltonian has no get_sym_path(); skipping symmetry-line eigenvalues.")
        return None

    try:
        k_path, k_dist, node_indices, path_labels, path_points = generate_2d_sym_lines(
            Hamiltonian_Obj,
            num_points_per_segment=num_points_per_segment,
        )
    except (NotImplementedError, AttributeError, KeyError, ValueError) as exc:
        print(f"Could not generate symmetry path; skipping symmetry-line eigenvalues: {exc}")
        return None

    k_path_3d = np.column_stack([k_path, np.zeros(len(k_path))])
    eigenvalues_sym, _ = eigenvalues_along_path(Hamiltonian_Obj, k_path_3d)

    np.savez(
        output_path,
        eigenvalues=eigenvalues_sym,
        k_path=k_path,
        k_dist=k_dist,
        node_indices=np.asarray(node_indices, dtype=int),
        path_labels=np.asarray(path_labels, dtype=object),
        path_points=path_points,
    )
    print(f"Saved symmetry-line eigenvalues to: {output_path}")
    return output_path


def _normalize_bands_to_compute(bands_to_compute, dim):
    if bands_to_compute is None:
        return list(range(dim))
    if isinstance(bands_to_compute, int):
        bands = [bands_to_compute]
    else:
        bands = list(bands_to_compute)
    bad = [band for band in bands if not (0 <= band < dim)]
    if bad:
        raise IndexError(f"bands_to_compute contains invalid bands {bad}; valid range is [0, {dim - 1}]")
    return bands


def _sym_path_qgt_output_path(results_subdir, method_name, bands):
    band_label = "bands_" + "_".join(str(band) for band in bands)
    return os.path.join(results_subdir, f"qgt_sym_path_{method_name}_{band_label}.npz")


def _qgt_function_for_method(method_name):
    if method_name == "numerical":
        return quantum_geometric_tensor_3d_num
    if method_name == "numerical_phase_corrected":
        return quantum_geometric_tensor_3d_num_phase_corrected
    if method_name == "analytic":
        return quantum_geometric_tensor_analytic
    raise ValueError(f"Symmetry-path QGT does not support method_name={method_name!r}.")


def _calculate_qgt_on_sym_path(
    Hamiltonian_Obj,
    method_name,
    bands_to_compute,
    *,
    delta_k=1e-5,
    kk=0,
    order="xyz",
    z_cutoff=None,
    num_points_per_segment=100,
):
    k_path, k_dist, node_indices, path_labels, path_points = generate_2d_sym_lines(
        Hamiltonian_Obj,
        num_points_per_segment=num_points_per_segment,
    )
    eigenvalues_path, eigenfunctions_path = eigenvalues_along_path(Hamiltonian_Obj, k_path)

    ki_path = k_path[:, 0].reshape(-1, 1)
    kj_path = k_path[:, 1].reshape(-1, 1)
    eigenvalues_grid = eigenvalues_path.reshape(-1, 1, Hamiltonian_Obj.dim)
    eigenfunctions_grid = eigenfunctions_path.reshape(-1, 1, Hamiltonian_Obj.dim, Hamiltonian_Obj.dim)
    qgt_func = _qgt_function_for_method(method_name)

    components = {
        "g_xx": [],
        "g_yy": [],
        "g_zz": [],
        "g_xy_real": [],
        "g_xy_imag": [],
        "g_xz_real": [],
        "g_xz_imag": [],
        "g_yz_real": [],
        "g_yz_imag": [],
        "trace": [],
    }

    for band in bands_to_compute:
        if method_name in ("numerical", "numerical_phase_corrected"):
            (g_xx, g_yy, g_zz,
             g_xy_real, g_xy_imag,
             g_xz_real, g_xz_imag,
             g_yz_real, g_yz_imag) = QGT_grid_num(
                ki_path,
                kj_path,
                eigenvalues_grid,
                eigenfunctions_grid,
                qgt_func,
                Hamiltonian_Obj,
                delta_k=delta_k,
                band_index=band,
                progress_label=f"sym path band {band}",
                kk=kk,
                order=order,
            )
        elif method_name == "analytic":
            eigenvalues_band = eigenvalues_grid[..., band]
            (g_xx, g_yy, g_zz,
             g_xy_real, g_xy_imag,
             g_xz_real, g_xz_imag,
             g_yz_real, g_yz_imag) = QGT_grid_analytic(
                ki_path,
                kj_path,
                qgt_func,
                Hamiltonian_Obj,
                kk=kk,
                z_cutoff=z_cutoff,
                eigenvalues=eigenvalues_band,
                band_index=band,
                order=order,
            )

        components["g_xx"].append(g_xx[:, 0])
        components["g_yy"].append(g_yy[:, 0])
        components["g_zz"].append(g_zz[:, 0])
        components["g_xy_real"].append(g_xy_real[:, 0])
        components["g_xy_imag"].append(g_xy_imag[:, 0])
        components["g_xz_real"].append(g_xz_real[:, 0])
        components["g_xz_imag"].append(g_xz_imag[:, 0])
        components["g_yz_real"].append(g_yz_real[:, 0])
        components["g_yz_imag"].append(g_yz_imag[:, 0])
        components["trace"].append(g_xx[:, 0] + g_yy[:, 0] + g_zz[:, 0])

    return {
        key: np.stack(value, axis=0)
        for key, value in components.items()
    } | {
        "bands": np.asarray(bands_to_compute, dtype=int),
        "k_path": k_path,
        "k_dist": k_dist,
        "node_indices": np.asarray(node_indices, dtype=int),
        "path_labels": np.asarray(path_labels, dtype=object),
        "path_points": path_points,
        "eigenvalues": eigenvalues_path,
    }


def _save_qgt_sym_path(
    Hamiltonian_Obj,
    results_subdir,
    method_name,
    *,
    bands_to_compute=None,
    delta_k=1e-5,
    kk=0,
    order="xyz",
    z_cutoff=None,
    num_points_per_segment=100,
    force_new=False,
):
    bands = _normalize_bands_to_compute(bands_to_compute, Hamiltonian_Obj.dim)
    output_path = _sym_path_qgt_output_path(results_subdir, method_name, bands)
    if os.path.exists(output_path) and not force_new:
        print(f"Using existing symmetry-path QGT data: {output_path}")
        return output_path

    data = _calculate_qgt_on_sym_path(
        Hamiltonian_Obj,
        method_name,
        bands,
        delta_k=delta_k,
        kk=kk,
        order=order,
        z_cutoff=z_cutoff,
        num_points_per_segment=num_points_per_segment,
    )
    np.savez(output_path, **data)
    print(f"Saved symmetry-path QGT data to: {output_path}")
    return output_path


def calculate_2d(
    Force_new=True,
    include_sym_path_qgt=False,
    sym_path_bands=None,
    sym_path_points_per_segment=100,
):
    # Define parameters
    band =5 # Which band to calculate your QMT on, starting from 0
    z_cutoff = .3 #where to cutoff the plot for the z axis when singularties occur
    z_percentile = 99 # percentile to cut off the plot

    # Define the temp directory for storing .npy files
    temp_dir = os.path.join(os.getcwd(), "temp")

    # File paths for loading the data
    eigenvalues_file = os.path.join(temp_dir, "eigenvalues.npy")
    eigenfunctions_file = os.path.join(temp_dir, "eigenfunctions.npy")
    meta_info_file = os.path.join(temp_dir, "meta_info.pkl")  # New file for meta information

    # Load the eigenvalues and eigenfunctions from files
    if os.path.exists(eigenvalues_file) and os.path.exists(eigenfunctions_file) and os.path.exists(meta_info_file):
        eigenvalues = np.load(eigenvalues_file)
        eigenfunctions = np.load(eigenfunctions_file)
        with open(meta_info_file, "rb") as meta_file:
            meta_info = pickle.load(meta_file)
            Hamiltonian_Obj = meta_info["Hamiltonian_Obj"] # ALWAYS REQUIRED
            ki = meta_info["ki"] # Required for 1D
            kj = meta_info["kj"] # Required for 1D
            kk = meta_info["kk"]
            dki = meta_info["dki"]
            dkj = meta_info["dkj"]
            mesh_spacing = meta_info["mesh_spacing"]
            ki_range = meta_info["ki_range"]
            kj_range = meta_info["kj_range"]
            order = meta_info["order"]
        print("Loaded eigenvalues, eigenfunctions, and meta information from files.")
        print(f"Current Hamiltonian: {Hamiltonian_Obj.name}")
    else:
        print("Eigenvalues or eigenfunctions files not found. Please ensure they are available at the specified paths.")
        sys.exit(1)

    # Define method name for directory naming ("analytic", "numerical", etc.)
    # method_name = "numerical"
    method_name = "numerical_phase_corrected"
    # method_name = "analytic"

    Hamiltonian_name = getattr(Hamiltonian_Obj, "name", "Hamiltonian")
    hamiltonian_params = Hamiltonian_Obj.get_parameters_dict(parameter="2D")
    
    # Construct metadata parameters dictionary with ALL info
    meta_params = {
        "hamiltonian_name": Hamiltonian_name,
        "kk": kk,
        "ki_range": ki_range,
        "kj_range": kj_range,
        "mesh_spacing": mesh_spacing,
        "method_name": method_name,
        "include_endpoints": True,
        "hamiltonian_params": hamiltonian_params,
        "dki": dki,
        "dkj": dkj,
        "order": order
    }
    
    file_paths, use_existing, results_subdir, meta_target = setup_2D_QGT_results_directory(
        meta_params=meta_params,
        force_new=Force_new
    )
    
    if use_existing:
        # Load existing QGT data
        g_xx_array = np.load(file_paths["g_xx"])
        g_yy_array = np.load(file_paths["g_yy"])
        g_zz_array = np.load(file_paths["g_zz"])
        g_xy_real_array = np.load(file_paths["g_xy_real"])
        g_xy_imag_array = np.load(file_paths["g_xy_imag"])
        g_xz_real_array = np.load(file_paths["g_xz_real"])
        g_xz_imag_array = np.load(file_paths["g_xz_imag"])
        g_yz_real_array = np.load(file_paths["g_yz_real"])
        g_yz_imag_array = np.load(file_paths["g_yz_imag"])
        trace_array = np.load(file_paths["trace"])

        with open(file_paths["meta_pkl"], "rb") as meta_file:
            qgt_meta_info = pickle.load(meta_file)

        print("Loaded QGT data from existing files.")


    else:
        # Select QGT calculation method
        if method_name == "numerical":            
            (g_xx_array, g_yy_array, g_zz_array, 
             g_xy_real_array, g_xy_imag_array, 
             g_xz_real_array, g_xz_imag_array, 
             g_yz_real_array, g_yz_imag_array) = QGT_grid_num(
                ki, kj, eigenvalues, eigenfunctions, quantum_geometric_tensor_3d_num, 
                Hamiltonian_Obj, delta_k=1e-5, band_index=band, kk=kk
            )
            trace_array = g_xx_array + g_yy_array + g_zz_array

        
        elif method_name == "numerical_phase_corrected":
            (g_xx_array, g_yy_array, g_zz_array, 
             g_xy_real_array, g_xy_imag_array, 
             g_xz_real_array, g_xz_imag_array, 
             g_yz_real_array, g_yz_imag_array) = QGT_grid_num(
                ki, kj, eigenvalues, eigenfunctions, quantum_geometric_tensor_3d_num_phase_corrected, 
                Hamiltonian_Obj, delta_k=1e-5, band_index=band, kk=kk
            )
            trace_array = g_xx_array + g_yy_array + g_zz_array
        
        elif method_name == "array_analytic":
            print("Using Analytic QGT Calculation (Block Diagonalization)...")
            # I believe this is calculating analytic QGT on the entire k grid using the funtion already built into the Hamiltonian Class. Should probably update this to just calcualtion at a single point for uniformity
            g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array = Hamiltonian_Obj.compute_qgt_analytic(
                ki, kj, band_index=band
            )
            # Analytic QG computes 2D subsets natively, use zeros for z
            g_zz_array = np.zeros_like(g_xx_array)
            trace_array = g_xx_array + g_yy_array + g_zz_array

        elif method_name == "analytic":
            print("Using Array Analytic QGT Calculation...")
            eigenvalues_band = eigenvalues[..., band]
            g_xx_array, g_yy_array, g_zz_array, \
            g_xy_real_array, g_xy_imag_array, g_xz_real_array, \
            g_xz_imag_array, g_yz_real_array, g_yz_imag_array = QGT_grid_analytic(
                ki, kj, quantum_geometric_tensor_analytic, 
                Hamiltonian_Obj, kk=kk, z_cutoff=z_cutoff, eigenvalues=eigenvalues_band,
                band_index=band, order=order
            )
            trace_array = g_xx_array + g_yy_array + g_zz_array

        elif method_name == "semi_numerical":
            print("Using Semi-Numerical QGT Calculation...")
            g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array = QGT_grid_semi_num(
                ki, kj,
                quantum_geometric_tensor_semi_num,
                hamiltonian=Hamiltonian_Obj,
                delta_k=dki,
                band_index=band,
                z_cutoff=z_cutoff
            )
        
        else:
            raise ValueError(f"Unknown QGT calculation method: {method_name}")


        # Save QGT results
        # We now have 9 components + trace
        save_dict = {
            "g_xx": g_xx_array,
            "g_yy": g_yy_array,
            "g_zz": g_zz_array,
            "g_xy_real": g_xy_real_array,
            "g_xy_imag": g_xy_imag_array,
            "g_xz_real": g_xz_real_array,
            "g_xz_imag": g_xz_imag_array,
            "g_yz_real": g_yz_real_array,
            "g_yz_imag": g_yz_imag_array,
            "trace": trace_array
        }

        for key, array in save_dict.items():
            if key in file_paths:
                np.save(file_paths[key], array)
                np.save(os.path.join(temp_dir, os.path.basename(file_paths[key])), array)
            else:
                np.save(os.path.join(temp_dir, f"{key}.npy"), array)

        if eigenvalues is not None:
            np.save(os.path.join(results_subdir, "eigenvalues.npy"), eigenvalues)

        print(f"Saved QGT results to '{results_subdir}' and copied to temp directory: {temp_dir}")



    shutil.copy2(meta_info_file, file_paths["meta_pkl"])
    _save_sym_line_eigenvalues(Hamiltonian_Obj, results_subdir)
    if include_sym_path_qgt:
        bands_for_sym_path = [band] if sym_path_bands is None else sym_path_bands
        _save_qgt_sym_path(
            Hamiltonian_Obj,
            results_subdir,
            method_name,
            bands_to_compute=bands_for_sym_path,
            delta_k=1e-5,
            kk=kk,
            order=order,
            z_cutoff=z_cutoff,
            num_points_per_segment=sym_path_points_per_segment,
            force_new=Force_new,
        )


    # b1, b2 = Hamiltonian_Obj.b1, Hamiltonian_Obj.b2
    # chern_number = compute_chern_number(
    #     g_xy_imag_array,
    #     dki, dkj,
    #     ki, kj,
    #     b1, b2
    # )
    # print("Chern number is: ", chern_number)


    # plot_QGT_components_3d(ki, kj, g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, stride_size=1)

    # plot_g_components_2d(g_xx_array, g_yy_array, trace_array, k_max=k_max)

    # plot_trace_w_eigenvalue(ki, kj, g_xx_array, g_yy_array, eigenvalues, trace_array, eigenvalue_band=band)


    # --- FHS Method (Commented out effectively by not using its result) ---
    # flux_field = berry_flux_FHS(eigenfunctions, dim_band=band)
    # berry_curvature_fhs = flux_field / (dki * dkj)
    # g_xy_imag_fhs = -0.5 * berry_curvature_fhs



    plot_eigen_and_all_berry_2d(
        ki, kj, eigenvalues, 
        g_xy_imag_array, g_xz_imag_array, g_yz_imag_array,
        eigenvalue_band=band,
        zlim_berry=z_cutoff,
        zlim_percentile=z_percentile,
        results_dir=results_subdir,
        save_fig=True
    )

    # 1. 3D Components
    print(f"Plotting QGT Components (3D) for band {band}...")
    plot_QGT_components_3d(
        ki, kj, g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array,
        stride_size=2,
        results_dir=results_subdir,
        save_fig=True,
        filename=f"QGT_components_3d_band_{band}.html",
        show=False
    )

    # 2. Combined Plots
    print(f"Plotting QMT/Eig/Berry/Trace (3D) for band {band}...")
    plot_qmt_eig_berry_trace_3d(
        ki, kj, eigenvalues, g_xy_imag_array, trace_array,
        eigenvalue_band=band,
        title=f"3D Results: {Hamiltonian_Obj.name if Hamiltonian_Obj else ''} (Band {band})",
        results_dir=results_subdir,
        save_fig=True,
        filename=f"qmt_eig_berry_trace_3d_band_{band}.html",
        show=False
    )

    print(f"Plotting QMT/Eig/Berry/Trace (2D Heatmaps) for band {band}...")
    plot_qmt_eig_berry_trace_2d(
        ki, kj, eigenvalues, g_xy_imag_array, trace_array,
        eigenvalue_band=band,
        title=f"2D Results: {Hamiltonian_Obj.name if Hamiltonian_Obj else ''} (Band {band})",
        results_dir=results_subdir,
        save_fig=True
    )


def _qgt_one_band_worker(payload: dict):
    """
    Runs in a separate process.
    Loads eigenvalues/eigenfunctions via mmap (no giant pickles).
    Computes QGT for ONE band.
    """
    # ---- unpack minimal payload ----
    band = payload["band"]
    temp_dir = payload["temp_dir"]
    z_cutoff = payload["z_cutoff"]
    delta_k = payload["delta_k"]
    method_name = payload["method_name"]
    kk = payload["kk"]
    order = payload["order"]

    # Load meta + Hamiltonian object (pickled) inside worker
    meta_info_file = os.path.join(temp_dir, "meta_info.pkl")
    with open(meta_info_file, "rb") as f:
        meta_info = pickle.load(f)
    H = meta_info["Hamiltonian_Obj"]

    ki = meta_info["ki"]
    kj = meta_info["kj"]

    # mmap the big arrays
    eigenvalues = np.load(os.path.join(temp_dir, "eigenvalues.npy"), mmap_mode="r")
    eigenfunctions = np.load(os.path.join(temp_dir, "eigenfunctions.npy"), mmap_mode="r")

    # If you need these set:
    H.A0 = 0.0
    H.omega = 5e3

    if method_name == "numerical":
        func_target = quantum_geometric_tensor_3d_num
    elif method_name == "numerical_phase_corrected":
        func_target = quantum_geometric_tensor_3d_num_phase_corrected
    elif method_name == "analytic":
        func_target = quantum_geometric_tensor_analytic
    else:
        raise ValueError(f"Method '{method_name}' not supported by _qgt_one_band_worker yet.")

    # --- compute band QGT ---
    if method_name in ["numerical", "numerical_phase_corrected"]:
        (g_xx, g_yy, g_zz,
         g_xy_r, g_xy_i,
         g_xz_r, g_xz_i,
         g_yz_r, g_yz_i) = QGT_grid_num(
            ki, kj,
            eigenvalues, eigenfunctions,
            func_target,
            H,
            delta_k=delta_k,
            band_index=band,
            kk=kk, 
            order=order
        )
        trace = g_xx + g_yy + g_zz
    elif method_name == "analytic":
        eigenvalues_band = eigenvalues[..., band]
        g_xx, g_yy, g_zz, g_xy_r, g_xy_i, g_xz_r, g_xz_i, g_yz_r, g_yz_i = QGT_grid_analytic(
            ki, kj, func_target, 
            H, kk=kk, z_cutoff=z_cutoff, eigenvalues=eigenvalues_band,
            band_index=band, order=order
        )
        trace = g_xx + g_yy + g_zz

    # return numpy arrays to parent (these are much smaller than the eigenfunctions)
    return {
        "band": band,
        "g_xx": np.array(g_xx),
        "g_yy": np.array(g_yy),
        "g_zz": np.array(g_zz),
        "g_xy_real": np.array(g_xy_r),
        "g_xy_imag": np.array(g_xy_i),
        "g_xz_real": np.array(g_xz_r),
        "g_xz_imag": np.array(g_xz_i),
        "g_yz_real": np.array(g_yz_r),
        "g_yz_imag": np.array(g_yz_i),
        "trace": np.array(trace),
    }

def calculate_2d_all_bands(
    Force_new=True,
    method_name="numerical_phase_corrected",
    include_sym_path_qgt=False,
    sym_path_bands=None,
    sym_path_points_per_segment=100,
):
    # ---- your existing setup/load block ----
    z_cutoff = 1000
    z_percentile = 95
    temp_dir = os.path.join(os.getcwd(), "temp")

    eigenvalues_file = os.path.join(temp_dir, "eigenvalues.npy")
    eigenfunctions_file = os.path.join(temp_dir, "eigenfunctions.npy")
    meta_info_file = os.path.join(temp_dir, "meta_info.pkl")

    if not (os.path.exists(eigenvalues_file) and os.path.exists(eigenfunctions_file) and os.path.exists(meta_info_file)):
        raise FileNotFoundError("Missing temp eigenvalues/eigenfunctions/meta_info.pkl")

    # mmap in parent only for nbands inference + plotting ki/kj later
    eigenvalues = np.load(eigenvalues_file, mmap_mode="r")

    with open(meta_info_file, "rb") as meta_file:
        meta_info = pickle.load(meta_file)
        Hamiltonian_Obj = meta_info["Hamiltonian_Obj"]
        ki = meta_info["ki"]
        kj = meta_info["kj"]
        kk = meta_info["kk"]
        dki = meta_info["dki"]
        dkj = meta_info["dkj"]
        mesh_spacing = meta_info["mesh_spacing"]
        ki_range = meta_info["ki_range"]
        kj_range = meta_info["kj_range"]
        order = meta_info["order"]

    delta_k = 1e-5

    n_bands = Hamiltonian_Obj.dim # Number of bands
    n_cores = os.cpu_count() or 1
    n_workers = min(n_cores, n_bands)  # <-- the correct "use each core for each band" rule

    print(f"Detected n_bands={n_bands}, cpu_cores={n_cores} -> using n_workers={n_workers}")

    # ---- results dir: you can either (A) one dir with stacked arrays, or (B) per-band dirs ----

    Hamiltonian_name = getattr(Hamiltonian_Obj, "name", "Hamiltonian")
    hamiltonian_params = Hamiltonian_Obj.get_parameters_dict(parameter="2D")
    meta_params = {
        "hamiltonian_name": Hamiltonian_name,
        "kk": kk,
        "ki_range": ki_range,
        "kj_range": kj_range,
        "mesh_spacing": mesh_spacing,
        "method_name": method_name,
        "include_endpoints": True,
        "order": order,
        "hamiltonian_params": hamiltonian_params,
        "dki": dki,
        "dkj": dkj,
        
        # store that this run includes all bands
        "band_index": "ALL",
        "n_bands": int(n_bands),
    }

    file_paths, use_existing, results_subdir, meta_target_json = setup_2D_QGT_results_directory(
        meta_params=meta_params,
        force_new=Force_new
    )

    # ---- if existing stacked results, load them and skip compute ----
    # (you will need to ensure setup_2D_QGT_results_directory defines these filenames)
    stacked_keys = ["g_xx","g_yy","g_zz","g_xy_real","g_xy_imag","g_xz_real","g_xz_imag","g_yz_real","g_yz_imag","trace"]

    if use_existing and all(k in file_paths and os.path.exists(file_paths[k]) for k in stacked_keys):
        print("Loading existing stacked QGT arrays...")
        loaded = {k: np.load(file_paths[k]) for k in stacked_keys}
        # shapes should be (n_bands, Nx, Ny)
        g_xx_all = loaded["g_xx"]; g_yy_all = loaded["g_yy"]; g_zz_all = loaded["g_zz"]
        g_xy_imag_all = loaded["g_xy_imag"]; g_xy_real_all = loaded["g_xy_real"]
        g_xz_imag_all = loaded["g_xz_imag"]; g_xz_real_all = loaded["g_xz_real"]
        g_yz_imag_all = loaded["g_yz_imag"]; g_yz_real_all = loaded["g_yz_real"]
        trace_all = loaded["trace"]
    else:
        # ---- parallel compute over bands ----
        payloads = [
            {
                "band": b,
                "temp_dir": temp_dir,
                "z_cutoff": z_cutoff,
                "delta_k": delta_k,
                "method_name": method_name,
                "kk": kk,
                "order": order
            }
            for b in range(n_bands)
        ]

        results_by_band = [None] * n_bands

        # IMPORTANT: ProcessPool on mac uses "spawn" by default.
        # This design keeps payload small and mmap-loads big arrays inside each worker.
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_qgt_one_band_worker, p) for p in payloads]
            for fut in as_completed(futures):
                out = fut.result()
                results_by_band[out["band"]] = out
                print(f"Finished band {out['band']}")

        # ---- stack into (n_bands, Nx, Ny) ----
        def stack(key):
            return np.stack([results_by_band[b][key] for b in range(n_bands)], axis=0)

        g_xx_all = stack("g_xx")
        g_yy_all = stack("g_yy")
        g_zz_all = stack("g_zz")
        g_xy_real_all = stack("g_xy_real")
        g_xy_imag_all = stack("g_xy_imag")
        g_xz_real_all = stack("g_xz_real")
        g_xz_imag_all = stack("g_xz_imag")
        g_yz_real_all = stack("g_yz_real")
        g_yz_imag_all = stack("g_yz_imag")
        trace_all = stack("trace")

        # ---- save stacked ----
        stacked_save = {
            "g_xx": g_xx_all,
            "g_yy": g_yy_all,
            "g_zz": g_zz_all,
            "g_xy_real": g_xy_real_all,
            "g_xy_imag": g_xy_imag_all,
            "g_xz_real": g_xz_real_all,
            "g_xz_imag": g_xz_imag_all,
            "g_yz_real": g_yz_real_all,
            "g_yz_imag": g_yz_imag_all,
            "trace": trace_all,
        }

        for k, arr in stacked_save.items():
            if k in file_paths:
                np.save(file_paths[k], arr)
            # also copy to temp for convenience
            np.save(os.path.join(temp_dir, f"{k}.npy"), arr)

        if eigenvalues is not None:
            np.save(os.path.join(results_subdir, "eigenvalues.npy"), eigenvalues)

        print(f"Saved STACKED QGT results for all bands to '{results_subdir}' (and temp/)")

    shutil.copy2(meta_info_file, file_paths["meta_pkl"])
    _save_sym_line_eigenvalues(Hamiltonian_Obj, results_subdir)
    if include_sym_path_qgt:
        _save_qgt_sym_path(
            Hamiltonian_Obj,
            results_subdir,
            method_name,
            bands_to_compute=sym_path_bands,
            delta_k=delta_k,
            kk=kk,
            order=order,
            z_cutoff=z_cutoff,
            num_points_per_segment=sym_path_points_per_segment,
            force_new=Force_new,
        )

    # ---- example plotting: choose a band to view ----
    for band_to_plot in range(n_bands): # pick which band you want to visualize
        plot_eigen_and_all_berry_2d(
            ki, kj, eigenvalues, 
            g_xy_imag_all[band_to_plot], g_xz_imag_all[band_to_plot], g_yz_imag_all[band_to_plot],
            eigenvalue_band=band_to_plot,
            zlim_berry=z_cutoff,
            zlim_percentile=z_percentile,
            results_dir=results_subdir,
            save_fig=True
        )

        # 1. 3D Components
        print(f"Plotting QGT Components (3D) for band {band_to_plot}...")
        plot_QGT_components_3d(
            ki, kj, g_xx_all[band_to_plot], g_xy_real_all[band_to_plot], g_xy_imag_all[band_to_plot], g_yy_all[band_to_plot],
            stride_size=2,
            results_dir=results_subdir,
            save_fig=True,
            filename=f"QGT_components_3d_band_{band_to_plot}.html",
            show=False
        )

        # 2. Combined Plots
        print(f"Plotting QMT/Eig/Berry/Trace (3D) for band {band_to_plot}...")
        plot_qmt_eig_berry_trace_3d(
            ki, kj, eigenvalues, g_xy_imag_all[band_to_plot], trace_all[band_to_plot],
            eigenvalue_band=band_to_plot,
            title=f"3D Results: {Hamiltonian_Obj.name if Hamiltonian_Obj else ''} (Band {band_to_plot})",
            results_dir=results_subdir,
            save_fig=True,
            filename=f"qmt_eig_berry_trace_3d_band_{band_to_plot}.html",
            show=False
        )

        print(f"Plotting QMT/Eig/Berry/Trace (2D Heatmaps) for band {band_to_plot}...")
        plot_qmt_eig_berry_trace_2d(
            ki, kj, eigenvalues, g_xy_imag_all[band_to_plot], trace_all[band_to_plot],
            eigenvalue_band=band_to_plot,
            title=f"2D Results: {Hamiltonian_Obj.name if Hamiltonian_Obj else ''} (Band {band_to_plot})",
            results_dir=results_subdir,
            save_fig=True
        )



if __name__ == '__main__':
    calculate_2d_all_bands(Force_new=False, method_name="numerical_phase_corrected", include_sym_path_qgt=True)
    # calculate_2d(Force_new=False)                        
