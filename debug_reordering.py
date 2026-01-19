import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from Library.Eigenvector import Eigenvectors
from Library.Hamiltonian.Altermagnet_Hamiltonian import AltermagnetHamiltonian
from Library.eigenvalue_calc_lib import eigenvalues_and_vectors_eigenvalue_ordering

# Setup
dir_path = "results/2D_Eigen_results/AltermagnetHamiltonian/2D_A0_0.00-J_1.00-Nz_4-analytic_magnus_False-lamb_2-magnus_order_1-omega_0.00-polarization_left-t1_1.00-t2_0.50-td_2_kx-3.14_3.14_ky-3.14_3.14_mesh100_data_set6"
target_idx = (0, 24)
delta_k = 1e-5

# Load Stored Data (Previous)
print("Loading Stored Data (Previous)...")
vecs_stored = np.load(os.path.join(dir_path, "eigenfunctions.npy"))
vals_stored = np.load(os.path.join(dir_path, "eigenvalues.npy"))
psi_old = vecs_stored[target_idx[0], target_idx[1]] # Shape (4,4)
val_old = vals_stored[target_idx[0], target_idx[1]]
kx = -1.6184
ky = -3.1416

print(f"Old Vals: {val_old}")

# Calculate New Data (k+dk)
print("Calculating New Data (k+dk)...")
H = AltermagnetHamiltonian(t1=1.0, t2=0.5, td=2.0, lamb=2.0, J=1.0, Nz=4.0)
vals_new, vecs_new = eigenvalues_and_vectors_eigenvalue_ordering(H, kx + delta_k, ky)
# vecs_new is list of arrays. Convert to array for easier handling?
# The Eigenvector class handles list.
print(f"New Vals (Energy Ordered): {vals_new}")

# Manual Debug of Permutation Logic
print("\n--- MATRIX OF DOT PRODUCTS ---")
print("Row i = Old[i], Col j = New[j]")
dim = 4
score_identity = 0
score_swap = 0 # Assuming swap 0<->1 for discussion
matrix = np.zeros((dim, dim))

for i in range(dim):
    for j in range(dim):
        v_old = psi_old[i]
        v_new = vecs_new[j]
        dot = np.abs(np.vdot(v_old, v_new))
        matrix[i, j] = dot
        # print(f"Old[{i}] (E={val_old[i]:.2f}) . New[{j}] (E={vals_new[j]:.2f}) = {dot:.4f}")

print("      New0   New1   New2   New3")
for i in range(dim):
    row_str = f"Old{i} | "
    for j in range(dim):
        row_str += f"{matrix[i,j]:.4f} "
    print(row_str)

# Calculate Scores manually
# Identity: 0->0, 1->1, 2->2, 3->3
score_matrix_trace = 0
for i in range(dim):
    score_matrix_trace += np.abs(1 - matrix[i, i])
print(f"\nScore Identity (Diag): {score_matrix_trace:.4f}")

# Find Best Permutation
from itertools import permutations
best_p = None
min_sc = 100
for p in permutations(range(dim)):
    sc = 0
    for i in range(dim):
        sc += np.abs(1 - matrix[i, p[i]])
    if sc < min_sc:
        min_sc = sc
        best_p = p

print(f"Best Permutation Found: {best_p} with score {min_sc:.4f}")

if best_p == (0, 1, 2, 3):
    print("Result: Identity (No Swap). This explains Overlap=0 if bands were swapped.")
else:
    print(f"Result: Swap occurring. New indices map to Old indices as: {best_p}")
