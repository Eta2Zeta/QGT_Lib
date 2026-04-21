
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
    plot_g_components_2d,
    plot_eigen_and_all_berry_2d
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
        kx = meta_info.get("kx", meta_info.get("ki", None))
        ky = meta_info.get("ky", meta_info.get("kj", None))
        if kx is None or ky is None:
            raise KeyError("Neither 'kx'/'ky' nor 'ki'/'kj' found in metadata.")
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

    # Check if we have stacked bands
    num_bands = 1
    if g_xx.ndim == 3:
        num_bands = g_xx.shape[0]

    for b in range(num_bands):
        # Extract band slice if 3D, else use arrays as is
        b_g_xx = g_xx[b] if g_xx.ndim == 3 else g_xx
        b_g_xy_real = g_xy_real[b] if g_xy_real.ndim == 3 else g_xy_real
        b_g_xy_imag = g_xy_imag[b] if g_xy_imag.ndim == 3 else g_xy_imag
        b_g_yy = g_yy[b] if g_yy.ndim == 3 else g_yy
        b_trace = trace[b] if trace.ndim == 3 else trace

        print(f"--- Generating Plots for Band {b} ---")

        # 1. 3D Components
        print(f"Plotting QGT Components (3D) for band {b}...")
        plot_QGT_components_3d(
            kx, ky, b_g_xx, b_g_xy_real, b_g_xy_imag, b_g_yy,
            stride_size=2,
            results_dir=target_dir,
            save_fig=True,
            filename=f"QGT_components_3d_band_{b}.html",
            show=False
        )

        # 2. Combined Plots
        print(f"Plotting QMT/Eig/Berry/Trace (3D) for band {b}...")
        plot_qmt_eig_berry_trace_3d(
            kx, ky, eigenvalues, b_g_xy_imag, b_trace,
            eigenvalue_band=b,

            title=f"3D Results: {Hamiltonian_Obj.name if Hamiltonian_Obj else ''} (Band {b})",
            results_dir=target_dir,
            save_fig=True,
            filename=f"qmt_eig_berry_trace_3d_band_{b}.html",
            show=False
        )

        print(f"Plotting QMT/Eig/Berry/Trace (2D Heatmaps) for band {b}...")
        plot_qmt_eig_berry_trace_2d(
            kx, ky, eigenvalues, b_g_xy_imag, b_trace,
            eigenvalue_band=b,
            title=f"2D Results: {Hamiltonian_Obj.name if Hamiltonian_Obj else ''} (Band {b})",
            results_dir=target_dir,
            save_fig=True
        )
    print("Done.")

def plot_all_2d_berries_from_directory(target_dir):
    """
    Loads QGT results and optionally plotted the new 2D 1x4 horizontal plot 
    for Eigenvalues, and all three Berry curvature components.
    """
    print(f"Loading data for all-Berry plots from: {target_dir}")

    # Metadata
    meta_path = os.path.join(target_dir, "meta_info.pkl")
    if os.path.exists(os.path.join(target_dir, "qgt_meta_info.pkl")):
        meta_path = os.path.join(target_dir, "qgt_meta_info.pkl")

    if not os.path.exists(meta_path):
        print(f"Error: No meta info file found in {target_dir}")
        return

    try:
        with open(meta_path, "rb") as f:
            meta_info = pickle.load(f)
        kx = meta_info.get("kx", meta_info.get("ki", None))
        ky = meta_info.get("ky", meta_info.get("kj", None))
        if kx is None or ky is None:
            raise KeyError("Neither 'kx'/'ky' nor 'ki'/'kj' found in metadata.")
    except Exception as e:
        print(f"Failed to load metadata/grid: {e}")
        return

    # Arrays
    try:
        g_xy_imag = np.load(os.path.join(target_dir, "g_xy_imag.npy"))
        g_xz_imag = np.load(os.path.join(target_dir, "g_xz_imag.npy"))
        g_yz_imag = np.load(os.path.join(target_dir, "g_yz_imag.npy"))
    except FileNotFoundError as e:
        print(f"Error: Missing QGT data file: {e}")
        return

    # Eigenvalues
    eigenvalues = None
    eig_path = os.path.join(target_dir, "eigenvalues.npy")
    if os.path.exists(eig_path):
        eigenvalues = np.load(eig_path)
    
    # Parameters matches Calc_QGT
    band = 1
    z_cutoff = 1000
    z_percentile = 95

    print("Plotting all-Berry 2D heatmaps...")

    if g_xy_imag.ndim > 2:
        num_bands = g_xy_imag.shape[0]
        for b in range(num_bands):
            print(f"Plotting all-Berry 2D for band {b}...")
            plot_eigen_and_all_berry_2d(
                kx, ky, eigenvalues, 
                g_xy_imag[b], g_xz_imag[b], g_yz_imag[b],
                eigenvalue_band=b,
                zlim_berry=z_cutoff,
                zlim_percentile=z_percentile,
                results_dir=target_dir,
                save_fig=True
            )
    else:
        print("Plotting all-Berry 2D for single band...")
        plot_eigen_and_all_berry_2d(
            kx, ky, eigenvalues, 
            g_xy_imag, g_xz_imag, g_yz_imag,
            eigenvalue_band=band,
            zlim_berry=z_cutoff,
            zlim_percentile=z_percentile,
            results_dir=target_dir,
            save_fig=True
        )

    print("Done all-Berry plots.")

if __name__ == "__main__":
    import os
    # base_dir = "/Users/home/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/2D_QGT_results/gWaveAltermagnetHamiltonian"
    # base_dir = "/Users/home/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/2D_QGT_results/THF_Hamiltonian"
    base_dir = "/Users/home/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/2D_QGT_results/TwoOrbitalUnspinfulHamiltonian"

    for subdir in os.listdir(base_dir):
        target_dir = os.path.join(base_dir, subdir)
        if os.path.isdir(target_dir) and os.path.exists(os.path.join(target_dir, "meta_info.pkl")):
            print(f"\n--- Processing: {subdir} ---")
            # plot_all_2d_berries_from_directory(target_dir)
            # You can also uncomment the next line to run plot_qgt_from_directory
            plot_qgt_from_directory(target_dir)
