import sys
import os
import numpy as np
import pickle
from tqdm import tqdm  # Import tqdm for progress bar
from multiprocessing import Pool, cpu_count
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

# from Library import * 
from Library.Hamiltonian_v1 import *
from Library.Hamiltonian.Hamiltonian_v2 import * 
from Library.eigenvalue_calc_lib import *
from Library.QGT_lib import *
from Library.QGT_Calc_3d_lib import *
from Library.topology import *
from Library.utilities import *
from Library.plotting_lib_2d import *
from Library.Hamiltonian.RuO2Hamiltonian import *
from Library.Hamiltonian.gWaveAltermagnetHamiltonian import *
from Library.data_management_utils import setup_3D_QGT_results_directory



def calculate_3d():
    print(f"Starting 3D QGT Calculation...")
    band = 0
    z_cutoff = 20
    # --- Check Temp Directory First (User Request) ---
    temp_dir = os.path.join(os.getcwd(), "temp")
    temp_eig = os.path.join(temp_dir, "eigenvalues_3d.npy")
    temp_vec = os.path.join(temp_dir, "eigenvectors_3d.npy")
    temp_meta = os.path.join(temp_dir, "meta_info.pkl")
    
    
    if os.path.exists(temp_eig) and os.path.exists(temp_vec) and os.path.exists(temp_meta):
        print("Found results in temp directory. Attempting to load from there...")
        try:
            with open(temp_meta, "rb") as f:
                temp_info = pickle.load(f)
            
            # Check if it looks like what we expect
            # We trust the user wants to use this
            hamiltonian = temp_info["Hamiltonian_Obj"]
            kx_vals = temp_info["kx_vals"]
            ky_vals = temp_info["ky_vals"]
            kz_vals = temp_info["kz_vals"]
            mesh_shape = temp_info["mesh_shape"]
            kvals_mode = temp_info["kvals_mode"]
            include_endpoints = temp_info["include_endpoints"]
            eigenvalues_3d = np.load(temp_eig)
            eigenvectors_3d = np.load(temp_vec)
            print(f"Successfully loaded data from temp. Hamiltonian: {hamiltonian.name}, Mesh: {mesh_size}")
            
        except Exception as e:
            print(f"Failed to load from temp: {e}. Falling back to standard calculation.")


    delta_k = 1e-5 # Use small step for numerical derivative, matching 2D calculation
    
    qgt_file_paths, qgt_use_existing, qgt_results_dir, qgt_meta_target = setup_3D_QGT_results_directory(
        hamiltonian,
        [kx_vals[0], kx_vals[-1]],
        [ky_vals[0], ky_vals[-1]],
        [kz_vals[0], kz_vals[-1]],
        mesh_shape=mesh_shape,
        include_endpoints=include_endpoints,
        force_new=True,
        kvals_mode=kvals_mode,  # if your function supports it
    )

    if qgt_use_existing and all(os.path.exists(p) for p in qgt_file_paths.values()):
         print("Loading existing QGT results...")
         g_xx_arr = np.load(qgt_file_paths["g_xx"])
         g_yy_arr = np.load(qgt_file_paths["g_yy"])
         g_zz_arr = np.load(qgt_file_paths["g_zz"])
         g_xy_real_arr = np.load(qgt_file_paths["g_xy_real"])
         g_xy_imag_arr = np.load(qgt_file_paths["g_xy_imag"])
         g_xz_real_arr = np.load(qgt_file_paths["g_xz_real"])
         g_xz_imag_arr = np.load(qgt_file_paths["g_xz_imag"])
         g_yz_real_arr = np.load(qgt_file_paths["g_yz_real"])
         g_yz_imag_arr = np.load(qgt_file_paths["g_yz_imag"])
    else:
        # We use QGT_grid_3d_num
        print("Computing 3D QGT Grid...")
        results = QGT_grid_3d_num(
            kx_vals, ky_vals, kz_vals, eigenvalues_3d, eigenvectors_3d, 
            quantum_geometric_tensor_3d_num_eigenvector_ordered, # Use ordered version
            hamiltonian, delta_k=delta_k, band_index=band, z_cutoff=z_cutoff
        )
        
        (g_xx_arr, g_yy_arr, g_zz_arr, 
         g_xy_real_arr, g_xy_imag_arr, 
         g_xz_real_arr, g_xz_imag_arr, 
         g_yz_real_arr, g_yz_imag_arr) = results
     
        # 3. Save Results
        print("Saving 3D QGT Results...")
        save_dict = {
            "g_xx": g_xx_arr, "g_yy": g_yy_arr, "g_zz": g_zz_arr,
            "g_xy_real": g_xy_real_arr, "g_xy_imag": g_xy_imag_arr,
            "g_xz_real": g_xz_real_arr, "g_xz_imag": g_xz_imag_arr,
            "g_yz_real": g_yz_real_arr, "g_yz_imag": g_yz_imag_arr
        }
        
        for key, val in save_dict.items():
            np.save(qgt_file_paths[key], val)
            
        # JSON
        qgt_meta = dict(qgt_meta_target)
        qgt_meta.update({"dk": delta_k, "band_index": band, "z_cutoff": z_cutoff})
        with open(qgt_file_paths["meta_json"], "w") as f:
            json.dump(qgt_meta, f, indent=2, sort_keys=True)

        # Pickle (loadable)
        qgt_meta_pkl = {
            "kx_vals": kx_vals, "ky_vals": ky_vals, "kz_vals": kz_vals,
            "mesh_shape": mesh_shape,
            "dk": delta_k,
            "include_endpoints": include_endpoints,
            "kvals_mode": kvals_mode,
            "Hamiltonian_Obj": hamiltonian,
        }
        with open(qgt_file_paths["meta_pkl"], "wb") as f:
            pickle.dump(qgt_meta_pkl, f)

        
    print(f"3D QGT calculation complete. Results saved to {qgt_results_dir}")

def _qgt3d_one_band_worker(payload: dict):
    """
    Compute 3D QGT for ONE band in a separate process.
    Big arrays are mmap-loaded from temp to avoid huge pickles.
    """
    band      = payload["band"]
    temp_dir  = payload["temp_dir"]
    delta_k   = payload["delta_k"]
    method    = payload["method"]

    # load meta (includes Hamiltonian + k grids)
    meta_path = os.path.join(temp_dir, "meta_info.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    H = meta["Hamiltonian_Obj"]
    kx_vals = meta["kx_vals"]
    ky_vals = meta["ky_vals"]
    kz_vals = meta["kz_vals"]

    # mmap big arrays
    eig_path = os.path.join(temp_dir, "eigenvalues_3d.npy")
    vec_path = os.path.join(temp_dir, "eigenvectors_3d.npy")
    eigenvalues_3d  = np.load(eig_path, mmap_mode="r")
    eigenvectors_3d = np.load(vec_path, mmap_mode="r")

    
    if method == "numerical":
        # compute for one band
        (g_xx, g_yy, g_zz,
         g_xy_r, g_xy_i,
         g_xz_r, g_xz_i,
         g_yz_r, g_yz_i) = QGT_3d_num(
            kx_vals, ky_vals, kz_vals,
        eigenvalues_3d, eigenvectors_3d,
        # quantum_geometric_tensor_3d_num_eigenvector_ordered,
        quantum_geometric_tensor_3d_num_phase_corrected,
        H,
        delta_k=delta_k,
        band_index=band
    )
    elif method == "analytic":
        # compute for one band
        (g_xx, g_yy, g_zz,
        g_xy_r, g_xy_i,
        g_xz_r, g_xz_i,
        g_yz_r, g_yz_i) = QGT_grid_3d_analytic(
            kx_vals, ky_vals, kz_vals, eigenvalues_3d,
            quantum_geometric_tensor_3d_analytic,
            H,
            band_index=band,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    trace = g_xx + g_yy + g_zz

    # return *small* arrays only
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

def calculate_3d_all_bands_parallel(force_new=True, method = "numerical"):
    print("Starting PARALLEL 3D QGT (one core per band)...")

    temp_dir  = os.path.join(os.getcwd(), "temp")
    temp_eig  = os.path.join(temp_dir, "eigenvalues_3d.npy")
    temp_vec  = os.path.join(temp_dir, "eigenvectors_3d.npy")
    temp_meta = os.path.join(temp_dir, "meta_info.pkl")

    if not (os.path.exists(temp_eig) and os.path.exists(temp_vec) and os.path.exists(temp_meta)):
        raise FileNotFoundError("Missing temp eigenvalues_3d.npy / eigenvectors_3d.npy / meta_info.pkl")

    # load meta in parent (small)
    with open(temp_meta, "rb") as f:
        meta = pickle.load(f)

    hamiltonian = meta["Hamiltonian_Obj"]
    kx_vals     = meta["kx_vals"]
    ky_vals     = meta["ky_vals"]
    kz_vals     = meta["kz_vals"]
    mesh_shape  = meta["mesh_shape"]
    kvals_mode  = meta.get("kvals_mode", None)
    include_endpoints = meta.get("include_endpoints", True)

    # band count: prefer Hamiltonian dimension
    n_bands = int(getattr(hamiltonian, "dim", None) or getattr(hamiltonian, "n_bands", None) or 0)
    if n_bands <= 0:
        # fallback: infer from eigenvalues file shape
        ev = np.load(temp_eig, mmap_mode="r")
        n_bands = int(ev.shape[-1])

    n_cores   = os.cpu_count() or 1
    n_workers = min(n_cores, n_bands)

    print(f"Detected n_bands={n_bands}, cpu_cores={n_cores} -> using n_workers={n_workers}")

    # ---- results dir + filenames ----
    qgt_file_paths, use_existing, qgt_results_dir, qgt_meta_target = setup_3D_QGT_results_directory(
        hamiltonian,
        [kx_vals[0], kx_vals[-1]],
        [ky_vals[0], ky_vals[-1]],
        [kz_vals[0], kz_vals[-1]],
        mesh_shape=mesh_shape,
        include_endpoints=include_endpoints,
        force_new=force_new,
        kvals_mode=kvals_mode,
        # IMPORTANT: ideally your setup function should incorporate these into the meta match:
        band_index="ALL",
        n_bands=int(n_bands),
        method_name="numerical"
    )

    stacked_keys = [
        "g_xx","g_yy","g_zz",
        "g_xy_real","g_xy_imag",
        "g_xz_real","g_xz_imag",
        "g_yz_real","g_yz_imag",
        "trace"
    ]

    # If you already have stacked results saved, load and return
    if use_existing and all((k in qgt_file_paths) and os.path.exists(qgt_file_paths[k]) for k in stacked_keys):
        print("Loading existing STACKED 3D QGT arrays...")
        loaded = {k: np.load(qgt_file_paths[k]) for k in stacked_keys}
        print(f"Loaded stacked arrays from {qgt_results_dir}")
        return loaded, qgt_results_dir

    # ---- parallel compute ----
    delta_k  = 1e-5

    payloads = [
        {"band": b, "temp_dir": temp_dir, "delta_k": delta_k, "method": method}
        for b in range(n_bands)
    ]

    results_by_band = [None] * n_bands

    # macOS uses spawn: this design is spawn-safe (payload small; mmap loads inside worker).
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_qgt3d_one_band_worker, p) for p in payloads]
        for fut in as_completed(futures):
            out = fut.result()
            results_by_band[out["band"]] = out
            print(f"Finished 3D band {out['band']}")

    # ---- stack to (n_bands, Nx, Ny, Nz) ----
    def stack(key):
        return np.stack([results_by_band[b][key] for b in range(n_bands)], axis=0)

    stacked = {k: stack(k) for k in stacked_keys}

    # ---- save stacked arrays ----
    for k, arr in stacked.items():
        if k in qgt_file_paths:
            np.save(qgt_file_paths[k], arr)
        # also mirror to temp for convenience (like your 2D)
        np.save(os.path.join(temp_dir, f"{k}_3d.npy"), arr)

    # ---- meta save ----
    # JSON meta
    meta_json = dict(qgt_meta_target)
    meta_json.update({
        "dk": delta_k,
        "band_index": "ALL",
        "n_bands": int(n_bands),
        "method_name": "numerical",
        "include_endpoints": include_endpoints,
        "kvals_mode": kvals_mode,
        "mesh_shape": mesh_shape,
    })
    with open(qgt_file_paths["meta_json"], "w") as f:
        json.dump(meta_json, f, indent=2, sort_keys=True)

    # Pickle meta (loadable objects)
    meta_pkl = {
        "Hamiltonian_Obj": hamiltonian,
        "kx_vals": kx_vals, "ky_vals": ky_vals, "kz_vals": kz_vals,
        "mesh_shape": mesh_shape,
        "dk": delta_k,
        "band_index": "ALL",
        "n_bands": int(n_bands),
        "include_endpoints": include_endpoints,
        "kvals_mode": kvals_mode,
        "method_name": "numerical",
    }
    with open(qgt_file_paths["meta_pkl"], "wb") as f:
        pickle.dump(meta_pkl, f)

    print(f"Saved STACKED 3D QGT results for all bands to {qgt_results_dir}")
    return stacked, qgt_results_dir

if __name__ == '__main__':
    # calculate_3d()
    calculate_3d_all_bands_parallel(method="numerical")