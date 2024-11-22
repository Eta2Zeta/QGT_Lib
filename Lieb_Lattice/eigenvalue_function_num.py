import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_spiral_indices(n):
    indices = []
    left, right = 0, n - 1
    top, bottom = 0, n - 1
    spiral_indices_matrix = np.empty((n, n), dtype=object)


    while left <= right and top <= bottom:
        for i in range(left, right + 1):
            indices.append((top, i))
        top += 1

        for i in range(top, bottom + 1):
            indices.append((i, right))
        right -= 1

        for i in range(right, left - 1, -1):
            indices.append((bottom, i))
        bottom -= 1

        for i in range(bottom, top - 1, -1):
            indices.append((i, left))
        left += 1
    
    i_index = 0
    for i in range(n): 
        for j in range (n):
            spiral_indices_matrix[i,j] = indices[i_index]
            i_index += 1

    return spiral_indices_matrix



# Brillouin zone vectors
a = 1.0  # Lattice constant

# Define parameters
mesh_spacing = 100
k_max = 2* (np.pi / a)  # Maximum k value for the first Brillouin zone
k_dirac_cone = k_max / 2
k_range = 0.1
ky_range = 0.1

# Create kx and ky arrays
kx = np.linspace((1 - k_range) * k_dirac_cone, (1 + k_range) * k_dirac_cone, mesh_spacing)
ky = np.linspace((1 - ky_range) * k_dirac_cone, (1 + ky_range) * k_dirac_cone, mesh_spacing)
kx, ky = np.meshgrid(kx, ky)

spiral_indices = get_spiral_indices(mesh_spacing)

for row in spiral_indices:
    print(row)

class Eigenvector:
    def __init__(self):
        self.previous_eigenvectors = None
        self.phase_factor = None

    def set_eigenvectors(self, new_eigenvectors):
        phase_factor_array = np.array([0,0,0])
        if self.previous_eigenvectors is not None:
            for i in range(len(new_eigenvectors)):
                dot_product = np.vdot(self.previous_eigenvectors[i], new_eigenvectors[i])
                phase_diff = np.angle(dot_product)
                phase_factor = np.exp(-1j * phase_diff)
                if (i == 1) & (phase_factor < 0):
                    pass
                phase_factor_array[i] = phase_factor
                new_eigenvectors[i] = new_eigenvectors[i] * phase_factor

        self.previous_eigenvectors = new_eigenvectors
        self.phase_factor = phase_factor_array
        return new_eigenvectors
    
    def get_phase_factors(self):
        return self.phase_factor
    
# Hamiltonian for the Lieb lattice
def H(kx, ky, a):
    return 2 * np.array([
        [0, np.cos(ky * a/2), np.cos(kx * a/2)],
        [np.cos(ky * a/2), 0, 0],
        [np.cos(kx * a/2), 0, 0]
    ])

# Function to calculate eigenvalues and eigenvectors
def eigenvalues_and_vectors(kx, ky, eigenvector, a):
    H_k = H(kx, ky, a)
    eigenvalues, eigenvectors = np.linalg.eig(H_k)
    eigenvectors = np.transpose(eigenvectors)

    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[idx, :]
    
    # Set the new eigenvectors with phase correction
    eigenvectors = eigenvector.set_eigenvectors(eigenvectors)
    
    return eigenvalues, eigenvectors

# Initialize arrays to store eigenfunctions and eigenvalues
eigenfunctions = np.zeros((mesh_spacing, mesh_spacing, 3, 3), dtype=complex)
eigenvalues = np.zeros((mesh_spacing, mesh_spacing, 3), dtype=float)
phase_factors_array = np.zeros((mesh_spacing, mesh_spacing, 3), dtype=float)

eigenvector = Eigenvector()

for i in range(mesh_spacing):
    for j in range(mesh_spacing):
        k,l = spiral_indices[i,j]
        
        vals, vecs = eigenvalues_and_vectors(kx[k, l], ky[k, l], eigenvector, a)
        phase_factors = eigenvector.get_phase_factors()
        
        eigenfunctions[k, l] = vecs
        eigenvalues[k, l] = vals
        phase_factors_array[k, l] = phase_factors

# Verify the eigenvalue-eigenfunction relationship
def verify_eigenpairs(kx, ky, a, eigenvalues, eigenfunctions):
    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            H_k = H(kx[i, j], ky[i, j], a)
            for band in range(3):
                vec = eigenfunctions[i, j, band, :]
                val = eigenvalues[i, j, band]
                lhs = np.dot(H_k, vec)
                rhs = val * vec
                allclose = np.allclose(lhs, rhs)
                if not allclose:
                    print(f"Mismatch at (kx, ky) = ({kx[i, j]}, {ky[i, j]}) for band {band + 1}")
                    print(f"H_k * eigenvector = {lhs}")
                    print(f"eigenvalue * eigenvector = {rhs}")
                    raise ValueError("Eigenpair verification failed")

# Verify eigenpairs
verify_eigenpairs(kx, ky, a, eigenvalues, eigenfunctions)

# Plot the eigenvalues and eigenfunctions
fig = plt.figure(figsize=(24, 8))

X, Y = kx, ky

for band in range(3):
    # Plot eigenvalues
    ax_eigenvalue = fig.add_subplot(3, 5, band * 5 + 1, projection='3d')
    Z_eigenvalue = eigenvalues[:, :, band]
    ax_eigenvalue.plot_surface(X, Y, Z_eigenvalue, cmap='viridis')
    ax_eigenvalue.set_title(f'Eigenvalue {band + 1}')
    ax_eigenvalue.set_xlabel('kx')
    ax_eigenvalue.set_ylabel('ky')
    ax_eigenvalue.set_zlabel('Eigenvalue')

    ax_phasefactor = fig.add_subplot(3, 5, band * 5 + 2, projection='3d')
    Z_phasefactor = phase_factors_array[:, :, band]
    ax_phasefactor.plot_surface(X, Y, Z_phasefactor, cmap='viridis')
    ax_phasefactor.set_title(f'Eigenvalue {band + 1}')
    ax_phasefactor.set_xlabel('kx')
    ax_phasefactor.set_ylabel('ky')
    ax_phasefactor.set_zlabel('Eigenvalue')



    # Plot corresponding eigenfunction components
    for component in range(3):
        ax_eigenfunction = fig.add_subplot(3, 5, band * 5 + component + 3, projection='3d')
        Z_eigenfunction = eigenfunctions[:, :, band, component]
        ax_eigenfunction.plot_surface(X, Y, Z_eigenfunction, cmap='viridis')
        ax_eigenfunction.set_title(f'Eigenfunction Component {component + 1}, Band {band + 1}')
        ax_eigenfunction.set_xlabel('kx')
        ax_eigenfunction.set_ylabel('ky')
        # ax_eigenfunction.set_zlabel('Magnitude')

plt.tight_layout()
plt.show()
