import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec
from qutip import Qobj
from mpl_toolkits.mplot3d import Axes3D

# Parameters
A0 = 1.0           # Driving amplitude
omega = 2.0 * np.pi  # Driving frequency
m = 1.0           # Mass
alpha = 0.5       # Coupling constant

# Define the analytical H1 and H-1 for a single kx, ky
def analytical_H1_single(kx, ky):
    return np.array([
        [A0 / (2 * m) * (kx - 1j * ky), 0],
        [-1j * alpha * A0, A0 / (2 * m) * (kx - 1j * ky)]
    ])

def analytical_Hm1_single(kx, ky):
    return np.array([
        [A0 / (2 * m) * (kx + 1j * ky), 1j * alpha * A0],
        [0, A0 / (2 * m) * (kx + 1j * ky)]
    ])

# Define the time-dependent Hamiltonian for numerical computation
def H(t, kx, ky):
    kx_t = kx + A0 * np.cos(omega * t)
    ky_t = ky + A0 * np.sin(omega * t)
    k_squared = kx_t**2 + ky_t**2
    H_matrix = np.array([
        [k_squared / (2 * m), alpha * (ky_t + 1j * kx_t)],
        [alpha * (ky_t - 1j * kx_t), k_squared / (2 * m)]
    ])
    return H_matrix

# Compute numerical Fourier components
def compute_numerical_Hn(n, kx, ky):
    integral = quad_vec(lambda t: H(t, kx, ky) * np.exp(-1j * n * omega * t), 0, 2 * np.pi / omega, epsrel=1e-8)
    return integral[0]

# Generate a grid of kx and ky values
kx_vals = np.linspace(-2, 2, 30)
ky_vals = np.linspace(-2, 2, 30)
kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals)

# Compute results for each matrix component
def plot_2x4(H_analytical_fn, H_numerical_fn, n, title):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    components = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for idx, (row, col) in enumerate(components):
        analytical = np.zeros_like(kx_grid, dtype=np.complex_)
        numerical = np.zeros_like(kx_grid, dtype=np.complex_)
        
        for i in range(kx_grid.shape[0]):
            for j in range(kx_grid.shape[1]):
                kx, ky = kx_grid[i, j], ky_grid[i, j]
                analytical[i, j] = H_analytical_fn(kx, ky)[row, col]
                numerical[i, j] = H_numerical_fn(n, kx, ky)[row, col]
        
        # Analytical plot
        ax = axes[0, idx]
        contour = ax.contourf(kx_grid, ky_grid, analytical.real, levels=100, cmap="viridis")
        fig.colorbar(contour, ax=ax, shrink=0.8)
        ax.set_title(f"Analytical Re(H[{row},{col}])")
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
        
        # Numerical plot
        ax = axes[1, idx]
        contour = ax.contourf(kx_grid, ky_grid, numerical.real, levels=100, cmap="viridis")
        fig.colorbar(contour, ax=ax, shrink=0.8)
        ax.set_title(f"Numerical Re(H[{row},{col}])")
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def commutator(H1, H2):
    return np.dot(H1, H2) - np.dot(H2, H1)

def plot_commutator_components_3d(A0, alpha, compute_numerical_Hn, kx_vals, ky_vals, omega, threshold=1e-12):
    """
    Plot the components of the commutator [H1, H-1] for analytical and numerical results as 3D surface plots.
    """
    # Generate a grid of kx and ky values
    kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals)

    # Define the analytical commutator
    def analytical_commutator(kx, ky):
        return np.array([
            [-A0**2 * alpha**2, 0],
            [0, A0**2 * alpha**2]
        ])

    # Initialize arrays for analytical and numerical results
    analytical_results = np.zeros(kx_grid.shape + (2, 2), dtype=np.complex_)
    numerical_results = np.zeros(kx_grid.shape + (2, 2), dtype=np.complex_)

    # Compute results
    for i in range(kx_grid.shape[0]):
        for j in range(kx_grid.shape[1]):
            kx, ky = kx_grid[i, j], ky_grid[i, j]
            H1_numerical = compute_numerical_Hn(1, kx, ky)
            Hm1_numerical = compute_numerical_Hn(-1, kx, ky)
            numerical_results[i, j] = commutator(H1_numerical, Hm1_numerical)
            analytical_results[i, j] = analytical_commutator(kx, ky)

    # Apply threshold to numerical results to filter noise
    numerical_results = np.where(np.abs(numerical_results) < threshold, 0, numerical_results)

    # Set up 3D plots for the 4 components with same figsize
    fig = plt.figure(figsize=(16, 7))
    components = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, (row, col) in enumerate(components):
        # Analytical 3D plot
        ax = fig.add_subplot(2, 4, idx * 2 + 1, projection='3d')
        ax.plot_surface(
            kx_grid, ky_grid, analytical_results[:, :, row, col].real,
            cmap="viridis", edgecolor="k", alpha=0.8
        )
        ax.set_title(f"Analytical Re([H1, H-1][{row},{col}])")
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
        ax.set_zlabel("Value")

        # Numerical 3D plot
        ax = fig.add_subplot(2, 4, idx * 2 + 2, projection='3d')
        ax.plot_surface(
            kx_grid, ky_grid, numerical_results[:, :, row, col].real,
            cmap="plasma", edgecolor="k", alpha=0.8
        )
        ax.set_title(f"Numerical Re([H1, H-1][{row},{col}])")
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
        ax.set_zlabel("Value")

    fig.suptitle("[H1, H-1] Components (Analytical vs Numerical)", fontsize=16)
    plt.tight_layout()
    plt.show()


# # Plot H1
# plot_2x4(analytical_H1_single, compute_numerical_Hn, 1, "H1 Components (Analytical vs Numerical)")

# # Plot H-1
# plot_2x4(analytical_Hm1_single, compute_numerical_Hn, -1, "H-1 Components (Analytical vs Numerical)")


plot_commutator_components_3d(A0, alpha, compute_numerical_Hn, kx_vals, ky_vals, omega)
