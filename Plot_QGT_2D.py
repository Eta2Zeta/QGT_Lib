
import sys
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Import plotting functions
from Library.plotting_lib_2d import (
    plot_QGT_components_3d,
    plot_qmt_eig_berry_trace_3d,
    plot_qmt_eig_berry_trace_2d,
    plot_g_components_2d
)

def plot_qgt_from_directory(target_dir):
    """
    Loads QGT results and metadata from the specified directory and generates plots.
    """
    print(f"Loading data from: {target_dir}")

    # Define file paths
    meta_info_file = os.path.join(target_dir, "meta_info.pkl") # try loading qgt_meta_info.pkl first if it exists? 

    if os.path.exists(os.path.join(target_dir, "qgt_meta_info.pkl")):
        meta_path = os.path.join(target_dir, "qgt_meta_info.pkl")
    elif os.path.exists(os.path.join(target_dir, "meta_info.pkl")):
        meta_path = os.path.join(target_dir, "meta_info.pkl")
    else:
        print(f"Error: No meta info file found in {target_dir}")
        sys.exit(1)

    # Load Metadata
    try:
        with open(meta_path, "rb") as f:
            meta_info = pickle.load(f)
        print("Loaded metadata.")
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        sys.exit(1)

    # Extract info from metadata
    try:
        kx = meta_info["kx"]
        ky = meta_info["ky"]
        Hamiltonian_Obj = meta_info.get("Hamiltonian_Obj", None)
        mesh_spacing = meta_info.get("mesh_spacing", "Unknown")
        print(f"Grid loaded. Mesh: {mesh_spacing}")
        if Hamiltonian_Obj:
            print(f"Hamiltonian: {Hamiltonian_Obj.name}")
    except KeyError as e:
        print(f"Error: Missing key in metadata: {e}")
        sys.exit(1)

    # Load QGT Data Arrays
    # Try loading from the standard names
    try:
        g_xx = np.load(os.path.join(target_dir, "g_xx.npy"))
        g_xy_real = np.load(os.path.join(target_dir, "g_xy_real.npy"))
        g_xy_imag = np.load(os.path.join(target_dir, "g_xy_imag.npy"))
        g_yy = np.load(os.path.join(target_dir, "g_yy.npy"))
        trace = np.load(os.path.join(target_dir, "trace.npy"))
        print("Loaded QGT arrays.")
    except FileNotFoundError as e:
        print(f"Error: Missing QGT data file: {e}")
        sys.exit(1)

    # Load Eigenvalues (Optional but needed for combined plots)
    eigenvalues = None
    eig_path = os.path.join(target_dir, "eigenvalues.npy")
    if os.path.exists(eig_path):
        eigenvalues = np.load(eig_path)
        print("Loaded eigenvalues.")
    else:
        print("Warning: 'eigenvalues.npy' not found. Some plots will be skipped.")

    # Plot Parameters
    band = 1 # Default, maybe overwrite if in meta?
    z_cutoff = 1

    # --- Generate Plots ---

    # 1. 3D Components
    print("Plotting QGT Components (3D)...")
    plot_QGT_components_3d(kx, ky, g_xx, g_xy_real, g_xy_imag, g_yy, stride_size=2)

    # 2. Combined Plots
    print("Plotting QMT/Eig/Berry/Trace (3D)...")
    plot_qmt_eig_berry_trace_3d(
        kx, ky, eigenvalues, g_xy_imag, trace,
        eigenvalue_band=band,
        zlims=(None, (-z_cutoff, z_cutoff), (-z_cutoff, z_cutoff)),
        title=f"3D Results: {Hamiltonian_Obj.name if Hamiltonian_Obj else ''}"
    )

    print("Plotting QMT/Eig/Berry/Trace (2D Heatmaps)...")
    plot_qmt_eig_berry_trace_2d(
        kx, ky, eigenvalues, g_xy_imag, trace,
        eigenvalue_band=band,
        zlims=(None, (-z_cutoff, z_cutoff), (-z_cutoff, z_cutoff)),
        title=f"2D Results: {Hamiltonian_Obj.name if Hamiltonian_Obj else ''}"
    )
    
    print("Done.")

if __name__ == "__main__":
    target_dir = "/Users/home/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/2D_QGT_results/gWaveAltermagnetHamiltonian/data_set_1"
        
    plot_qgt_from_directory(target_dir)
