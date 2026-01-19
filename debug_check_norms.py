import numpy as np
import os

dir_path = "results/2D_Eigen_results/AltermagnetHamiltonian/2D_A0_0.00-J_1.00-Nz_4-analytic_magnus_False-lamb_2-magnus_order_1-omega_0.00-polarization_left-t1_1.00-t2_0.50-td_2_kx-3.14_3.14_ky-3.14_3.14_mesh100_data_set6"
target_idx = (0, 24)

print(f"Checking dataset: {dir_path}")
vecs = np.load(os.path.join(dir_path, "eigenfunctions.npy"))
vals = np.load(os.path.join(dir_path, "eigenvalues.npy"))

print(f"Shape: {vecs.shape}")
psi = vecs[target_idx[0], target_idx[1]]
print(f"Eigenvectors at {target_idx}:")
for b in range(4):
    norm = np.linalg.norm(psi[b])
    print(f"  Band {b} (E={vals[target_idx][b]:.4f}): Norm = {norm:.6f}")
    print(f"  Vector: {psi[b]}")
