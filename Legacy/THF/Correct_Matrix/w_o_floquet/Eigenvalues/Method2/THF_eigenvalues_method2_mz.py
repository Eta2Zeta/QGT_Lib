import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cmath

# Constants
nu_star = -4.303  # Adjust as needed
nu_star_prime = 1.622  # Adjust as needed
nu_star_prime_prime = nu_star_prime / 10  # Adjust as needed
lambda_val = 0.03  # Adjust as needed
M = 3.697  # Adjust as needed
gamma = -24.75
z_limit = 6  # Set the z-axis limit

# Define parameters
mesh_spacing = 200
k_range = 5
k_max = k_range * np.pi  # Adjust based on the realistic Brillouin zone
kx = np.linspace(-k_max, k_max, mesh_spacing)
ky = np.linspace(-k_max, k_max, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)
k = np.sqrt(kx**2 + ky**2)
theta = np.arctan2(ky, kx)

# Define the coefficients based on kx, ky, and other constants
def get_coefficients(k, theta):
    C = np.exp(-k**2 * lambda_val**2 / 2) * gamma
    D = np.exp(-k**2 * lambda_val**2 / 2) * nu_star_prime * k
    A = nu_star * k

    return A, C, D

# Define Q based on the coefficients
def compute_Q(A, C, D, M):
    return 9 * A**2 * M - 18 * C**2 * M - 36 * C * D * M - 18 * D**2 * M + 2 * M**3 + cmath.sqrt(
        4 * (-3 * A**2 - 3 * C**2 - 6 * C * D - 3 * D**2 - M**2)**3 + (9 * A**2 * M - 18 * C**2 * M - 36 * C * D * M - 18 * D**2 * M + 2 * M**3)**2)

# Define P based on the coefficients
def compute_P(A, C, D, M):
    return 9 * A**2 * M - 18 * C**2 * M + 36 * C * D * M - 18 * D**2 * M + 2 * M**3 + cmath.sqrt(
        4 * (-3 * A**2 - 3 * C**2 + 6 * C * D - 3 * D**2 - M**2)**3 + (9 * A**2 * M - 18 * C**2 * M + 36 * C * D * M - 18 * D**2 * M + 2 * M**3)**2)

# Analytical solutions for eigenvalues
def analytical_eigenvalue_1(A, C, D, M, Q):
    term1 = M / 3
    term2 = (2**(1/3) * (-3 * A**2 - 3 * C**2 - 6 * C * D - 3 * D**2 - M**2)) / (3 * Q**(1/3))
    term3 = Q**(1/3) / (3 * 2**(1/3))
    return term1 - term2 + term3

def analytical_eigenvalue_2(A, C, D, M, Q):
    term1 = M / 3
    term2 = ((1 + 1j * np.sqrt(3)) * (-3 * A**2 - 3 * C**2 - 6 * C * D - 3 * D**2 - M**2)) / (3 * 2**(2/3) * Q**(1/3))
    term3 = ((1 - 1j * np.sqrt(3)) * Q**(1/3)) / (6 * 2**(1/3))
    return term1 + term2 - term3

def analytical_eigenvalue_3(A, C, D, M, Q):
    term1 = M / 3
    term2 = ((1 - 1j * np.sqrt(3)) * (-3 * A**2 - 3 * C**2 - 6 * C * D - 3 * D**2 - M**2)) / (3 * 2**(2/3) * Q**(1/3))
    term3 = ((1 + 1j * np.sqrt(3)) * Q**(1/3)) / (6 * 2**(1/3))
    return term1 + term2 - term3

def analytical_eigenvalue_4(A, C, D, M, P):
    term1 = M / 3
    term2 = (2**(1/3) * (-3 * A**2 - 3 * C**2 + 6 * C * D - 3 * D**2 - M**2)) / (3 * P**(1/3))
    term3 = P**(1/3) / (3 * 2**(1/3))
    return term1 - term2 + term3

def analytical_eigenvalue_5(A, C, D, M, P):
    term1 = M / 3
    term2 = ((1 + 1j * np.sqrt(3)) * (-3 * A**2 - 3 * C**2 + 6 * C * D - 3 * D**2 - M**2)) / (3 * 2**(2/3) * P**(1/3))
    term3 = ((1 - 1j * np.sqrt(3)) * P**(1/3)) / (6 * 2**(1/3))
    return term1 + term2 - term3

def analytical_eigenvalue_6(A, C, D, M, P):
    term1 = M / 3
    term2 = ((1 - 1j * np.sqrt(3)) * (-3 * A**2 - 3 * C**2 + 6 * C * D - 3 * D**2 - M**2)) / (3 * 2**(2/3) * P**(1/3))
    term3 = ((1 + 1j * np.sqrt(3)) * P**(1/3)) / (6 * 2**(1/3))
    return term1 + term2 - term3

# Initialize arrays to store eigenvalues
eigenvalues_1 = np.zeros((mesh_spacing, mesh_spacing), dtype=complex)
eigenvalues_2 = np.zeros((mesh_spacing, mesh_spacing), dtype=complex)
eigenvalues_3 = np.zeros((mesh_spacing, mesh_spacing), dtype=complex)
eigenvalues_4 = np.zeros((mesh_spacing, mesh_spacing), dtype=complex)
eigenvalues_5 = np.zeros((mesh_spacing, mesh_spacing), dtype=complex)
eigenvalues_6 = np.zeros((mesh_spacing, mesh_spacing), dtype=complex)

for i in range(mesh_spacing):
    for j in range(mesh_spacing):
        k_ij = k[i, j]
        theta_ij = theta[i, j]
        A, C, D = get_coefficients(k_ij, theta_ij)
        Q = compute_Q(A, C, D, M)
        P = compute_P(A, C, D, M)
        
        eigenvalues_1[i, j] = analytical_eigenvalue_1(A, C, D, M, Q)
        eigenvalues_2[i, j] = analytical_eigenvalue_2(A, C, D, M, Q)
        eigenvalues_3[i, j] = analytical_eigenvalue_3(A, C, D, M, Q)
        eigenvalues_4[i, j] = analytical_eigenvalue_4(A, C, D, M, P)
        eigenvalues_5[i, j] = analytical_eigenvalue_5(A, C, D, M, P)
        eigenvalues_6[i, j] = analytical_eigenvalue_6(A, C, D, M, P)

# Apply z-limit to the real part of the eigenvalues
eigenvalues_1_real = np.clip(np.real(eigenvalues_1), -z_limit, z_limit)
eigenvalues_2_real = np.clip(np.real(eigenvalues_2), -z_limit, z_limit)
eigenvalues_3_real = np.clip(np.real(eigenvalues_3), -z_limit, z_limit)
eigenvalues_4_real = np.clip(np.real(eigenvalues_4), -z_limit, z_limit)
eigenvalues_5_real = np.clip(np.real(eigenvalues_5), -z_limit, z_limit)
eigenvalues_6_real = np.clip(np.real(eigenvalues_6), -z_limit, z_limit)

# Plot the eigenvalues in the same 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
stride_size = 3

# Plot all six sets of eigenvalues in the same plot
# ax.plot_surface(kx, ky, eigenvalues_1_real, cmap='viridis', alpha=0.5, rstride=stride_size, cstride=stride_size)
# ax.plot_surface(kx, ky, eigenvalues_2_real, cmap='plasma', alpha=0.5, rstride=stride_size, cstride=stride_size)
ax.plot_surface(kx, ky, eigenvalues_3_real, cmap='inferno', alpha=0.5, rstride=stride_size, cstride=stride_size)
# ax.plot_surface(kx, ky, eigenvalues_4_real, cmap='magma', alpha=0.5, rstride=stride_size, cstride=stride_size)
# ax.plot_surface(kx, ky, eigenvalues_5_real, cmap='cividis', alpha=0.5, rstride=stride_size, cstride=stride_size)
ax.plot_surface(kx, ky, eigenvalues_6_real, cmap='coolwarm', alpha=0.5, rstride=stride_size, cstride=stride_size)

ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.set_zlabel('Eigenvalues')
ax.set_zlim(-z_limit, z_limit)
plt.title('Analytical Eigenvalues of the Cubic Equation')
plt.tight_layout()
plt.show()
