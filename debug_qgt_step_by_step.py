import numpy as np
import os
import pickle
import sys

# Add current directory to path to import libraries
sys.path.append(os.getcwd())

from Library.QGT_lib import quantum_geometric_tensor_num_eigenvector_ordered, dpsi_dx_num_eigenvector_ordered, dpsi_dy_num_eigenvector_ordered, projection_operator
from Library.Eigenvector import Eigenvectors, Eigenvector
from Library.Hamiltonian.Altermagnet_Hamiltonian import AltermagnetHamiltonian
from Library.eigenvalue_calc_lib import eigenvalues_and_vectors_eigenvalue_ordering

# Paths for the two datasets
dir_cont = "results/2D_Eigen_results/AltermagnetHamiltonian/2D_A0_0.00-J_1.00-Nz_4-analytic_magnus_False-lamb_2-magnus_order_1-omega_0.00-polarization_left-t1_1.00-t2_0.50-td_2_kx-3.14_3.14_ky-3.14_3.14_mesh100_data_set6"
dir_energ = "results/2D_Eigen_results/AltermagnetHamiltonian/2D_A0_0.00-J_1.00-Nz_4-analytic_magnus_False-lamb_2-magnus_order_1-omega_0.00-polarization_left-t1_1.00-t2_0.50-td_2_kx-3.14_3.14_ky-3.14_3.14_mesh100_data_set7"

# Initialize Hamiltonian
Hamiltonian_Obj = AltermagnetHamiltonian(t1=1.0, t2=0.5, td=2.0, lamb=2.0, J=1.0, Nz=4.0)
target_idx = (0, 24)
band_idx = 1
delta_k = 1e-5

print(f"--- Debugging QGT Calculation at Local Index {target_idx}, Band {band_idx} ---")

def debug_method(dir_path, label):
    print(f"\n[{label} Method] Loading from: {dir_path}")
    
    # Load Data
    eigenvalues = np.load(os.path.join(dir_path, "eigenvalues.npy"))
    eigenfunctions = np.load(os.path.join(dir_path, "eigenfunctions.npy"))
    
    # Load Meta Info to get kx, ky arrays
    with open(os.path.join(dir_path, "meta_info.pkl"), "rb") as f:
        meta = pickle.load(f)
        kx_grid = meta["kx"]
        ky_grid = meta["ky"]
    
    i, j = target_idx
    kx_val = kx_grid[i, j]
    ky_val = ky_grid[i, j]
    
    print(f"Target Point: kx={kx_val:.4f}, ky={ky_val:.4f}")
    
    # Get stored values
    eig_val_stored = eigenvalues[i, j, :]
    eig_vec_stored = eigenfunctions[i, j, :, :]
    psi_target = eig_vec_stored[band_idx]
    
    print(f"Stored Eigenvalues (All Bands): {eig_val_stored}")
    print(f"Stored Eigenvector (Band {band_idx}): {psi_target[0]:.4f}...")
    
    # Trace QGT Calculation Steps
    print("\n--- Tracing QGT Components ---")
    
    # 1. dpsi/dx
    # Internally, dpsi_dx_num_eigenvector_ordered calls eigenvalues_and_vectors_eigenvalue_ordering at k+dk and k-dk
    # AND it uses `set_eigenvectors_eigenvector_ordered` with psi_target as `previous_eigenvector`
    
    print("Calculating dpsi/dx...")
    # Replicate logic manually to print intermediates
    
    # Setup Eigenvectors class with target as 'previous'
    eigenvector_poly = Eigenvectors(4)
    # We fake "previous" by setting it to current target (as done in dpsi function)
    eigenvector_poly.set_eigenvectors_eigenvector_ordered(eig_vec_stored, eig_val_stored, kx_val, ky_val)
    
    # Calc k+dk
    vals_plus, vecs_plus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian_Obj, kx_val + delta_k, ky_val)
    # Reorder
    v_plus_ord, psi_plus_ord = eigenvector_poly.set_eigenvectors_eigenvector_ordered(vecs_plus, vals_plus, kx_val + delta_k, ky_val)
    
    # print what was found at +dk
    print(f"  At k+dk: Eigenvalues found: {vals_plus}")
    print(f"  At k+dk: Reordered Vals: {v_plus_ord}")
    print(f"  At k+dk: Overlap with target Band {band_idx}: {np.abs(np.vdot(psi_target, psi_plus_ord[band_idx])):.6f}")
    
    # Calc k-dk
    # Reset helper? No, dpsi function creates NEW helper instances for plus and minus
    eigenvector_poly_minus = Eigenvectors(4)
    eigenvector_poly_minus.set_eigenvectors_eigenvector_ordered(eig_vec_stored, eig_val_stored, kx_val, ky_val)
    
    vals_minus, vecs_minus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian_Obj, kx_val - delta_k, ky_val)
    v_minus_ord, psi_minus_ord = eigenvector_poly_minus.set_eigenvectors_eigenvector_ordered(vecs_minus, vals_minus, kx_val - delta_k, ky_val)
    
    # Finite Difference
    dpsi_dx_val = (psi_plus_ord[band_idx] - psi_minus_ord[band_idx]) / (2 * delta_k)
    print(f"  dpsi/dx norm: {np.linalg.norm(dpsi_dx_val):.6f}")
    
    # 2. dpsi/dy (Briefly)
    print("Calculating dpsi/dy...")
    dpsi_dy_val = dpsi_dy_num_eigenvector_ordered(Hamiltonian_Obj, kx_val, ky_val, delta_k, eig_val_stored, eig_vec_stored, band_idx)
    print(f"  dpsi/dy norm: {np.linalg.norm(dpsi_dy_val):.6f}")

    # 3. Final QGT
    I = np.eye(4)
    P = projection_operator(psi_target)
    
    g_xx = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dx_val).real
    g_xy_real = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).real
    g_xy_imag = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).imag
    g_yy = np.vdot(dpsi_dy_val, (I - P) @ dpsi_dy_val).real
    
    print(f"\n[{label} RESULTS]")
    print(f"  g_xx: {g_xx:.6f}")
    print(f"  g_yy: {g_yy:.6f}")
    print(f"  g_xy (imag): {g_xy_imag:.6f}")
    print(f"  Trace: {g_xx + g_yy:.6f}")

# Run for both
debug_method(dir_cont, "CONTINUITY/SNAKE (Set 6)")
print("\n" + "="*50 + "\n")
debug_method(dir_energ, "ENERGY/RASTER (Set 7)")
