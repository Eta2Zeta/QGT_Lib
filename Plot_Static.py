
import numpy as np
import os
import pickle
import sys

# Ensure Library is importable
# sys.path.append(os.getcwd()) # Usually typically handled by running from root

from Library.plotting_lib_2d import plot_qmt_eig_berry_trace_3d
import Library.Hamiltonian.Hamiltonian_v2
import Library.Hamiltonian.ChiralHamiltonian
# Apply backward compatibility patch if needed for old pickles
try:
    sys.modules["Library.Hamiltonian_v2"] = Library.Hamiltonian.Hamiltonian_v2
    sys.modules["Library.Hamiltonian_v2.ChiralHamiltonian"] = Library.Hamiltonian.ChiralHamiltonian
except ImportError:
    pass


def plot_static_from_file_3d(eig_dir, qgt_dir, band_index=0, z_cutoff=20):
    """
    Load pre-calculated QGT results from directories and plot them in 3D.
    
    Args:
        eig_dir: Directory containing eigenvalues.npy, eigenfunctions.npy, and meta_info.pkl
        qgt_dir: Directory containing g_xy_imag.npy and trace.npy
        band_index: Band to plot
        z_cutoff: Cutoff for z-axis
    """
    
    if not os.path.exists(eig_dir):
        print(f"Error: Eigenvalue Directory not found: {eig_dir}")
        return
    if not os.path.exists(qgt_dir):
        print(f"Error: QGT Directory not found: {qgt_dir}")
        return

    # 1. Load Metadata (from eig_dir usually containing the Hamiltonian info)
    meta_path = os.path.join(eig_dir, "meta_info.pkl")
    if not os.path.exists(meta_path):
        # Fallback: try qgt_meta_info.pkl
        meta_path = os.path.join(eig_dir, "qgt_meta_info.pkl")
        
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file not found in {eig_dir}")
        return

    with open(meta_path, "rb") as f:
        meta_info = pickle.load(f)

    kx = meta_info["kx"]
    ky = meta_info["ky"]
    Hamiltonian_Obj = meta_info.get("Hamiltonian_Obj") 

    # 2. Load Arrays
    try:
        # Check if eigenvalues exist
        eig_path = os.path.join(eig_dir, "eigenvalues.npy")
        if os.path.exists(eig_path):
            eigenvalues = np.load(eig_path)
            print("Eigenvalues loaded.")
        else:
            print(f"Error: eigenvalues.npy not found in {eig_dir}")
            return

        # Check if eigenfunctions exist
        eig_func_path = os.path.join(eig_dir, "eigenfunctions.npy")
        if os.path.exists(eig_func_path):
            eigenfunctions = np.load(eig_func_path)
            print("Eigenfunctions loaded.")
        else:
            print(f"Warning: eigenfunctions.npy not found in {eig_dir}")
            eigenfunctions = None

        # We need g_xy_imag (Berry curvature related) and trace (Quantum Metric trace)
        # Check filenames: usually g_xy_imag.npy and trace.npy
        g_xy_imag_array = np.load(os.path.join(qgt_dir, "g_xy_imag.npy"))
        trace_array = np.load(os.path.join(qgt_dir, "trace.npy"))
        
    except FileNotFoundError as e:
        print(f"Error loading data arrays: {e}")
        return

    print(f"Loaded eigenvalues from {eig_dir}")
    print(f"Loaded QGT data from {qgt_dir}")
    print(f"  Grid Shape: {kx.shape}")
    print(f"  Eigenvalues Shape: {eigenvalues.shape}")
    print(f"  QGT Components Loaded.")

    # 3. Plot
    print(f"Plotting Band {band_index} with z_cutoff={z_cutoff}...")
    
    if z_cutoff is None:
        qgt_zlim = None
    else:
        qgt_zlim = (-z_cutoff, z_cutoff)

    plot_qmt_eig_berry_trace_3d(
        kx, ky, eigenvalues, g_xy_imag_array, trace_array,
        eigenvalue_band=band_index,
        zlims=(None, qgt_zlim, qgt_zlim),
    )

if __name__ == "__main__":
    # --- Configuration ---
    
    eig_dir = "results/2D_Eigen_results/AltermagnetHamiltonian/2D_A0_0.00-J_1.00-Nz_4-analytic_magnus_False-lamb_2-magnus_order_1-omega_0.00-polarization_left-t1_1.00-t2_0.50-td_2_kx-3.14_3.14_ky-3.14_3.14_mesh100_data_set1"
    qgt_dir = "results/2D_QGT_results/AltermagnetHamiltonian/QGT_numerical_A0_0.00-J_1.00-Nz_4-analytic_magnus_False-lamb_2-magnus_order_1-omega_0.00-polarization_left-t1_1.00-t2_0.50-td_2_kx-3.14_3.14_ky-3.14_3.14_mesh100_data_set1"
    
    band = 3
    z_cutoff = None
    
    plot_static_from_file_3d(eig_dir, qgt_dir, band_index=band, z_cutoff=z_cutoff)
