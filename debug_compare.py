import numpy as np
import os

# Paths
dir1 = "results/2D_Eigen_results/AltermagnetHamiltonian/2D_A0_0.00-J_1.00-Nz_4-analytic_magnus_False-lamb_2-magnus_order_1-omega_0.00-polarization_left-t1_1.00-t2_0.50-td_2_kx-3.14_3.14_ky-3.14_3.14_mesh100_data_set6"
dir2 = "results/2D_Eigen_results/AltermagnetHamiltonian/2D_A0_0.00-J_1.00-Nz_4-analytic_magnus_False-lamb_2-magnus_order_1-omega_0.00-polarization_left-t1_1.00-t2_0.50-td_2_kx-3.14_3.14_ky-3.14_3.14_mesh100_data_set7"

# Load data
print("Loading data...")
e1 = np.load(os.path.join(dir1, "eigenvalues.npy"))
vec1 = np.load(os.path.join(dir1, "eigenfunctions.npy"))

e2 = np.load(os.path.join(dir2, "eigenvalues.npy"))
vec2 = np.load(os.path.join(dir2, "eigenfunctions.npy"))

print(f"Shapes: e1={e1.shape}, vec1={vec1.shape}")

# Find max diff
diff = np.abs(e1 - e2)
max_diff = np.max(diff)
print(f"Max Eigenvalue Difference: {max_diff}")

if max_diff > 1e-4:
    # Find index of max diff
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    i, j, b = idx
    print(f"Max Diff at index ({i}, {j}), Band {b}")
    print(f"Eigenvalues Set 3: {e1[i, j]}")
    print(f"Eigenvalues Set 4: {e2[i, j]}")
    
    # Check overlaps for Band b
    psi1 = vec1[i, j, b]
    psi2 = vec2[i, j, b]
    overlap = np.abs(np.vdot(psi1, psi2))
    print(f"Overlap of Band {b} eigenvectors: {overlap:.6f}")
    
    # Check if psi1 matches ANY band in Set 4
    all_overlaps = []
    for k in range(e2.shape[2]):
        ov = np.abs(np.vdot(psi1, vec2[i, j, k]))
        all_overlaps.append(ov)
    print(f"Overlap of Set 3 Band {b} with all bands in Set 4: {np.round(all_overlaps, 4)}")
    
else:
    print("Differences are negligible.")
