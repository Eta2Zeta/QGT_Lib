import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Optional: for progress tracking

def compute_point(kx, ky, Hamiltonian_Obj, dim):
    """
    Compute eigenvalues and eigenfunctions for a single (kx, ky) point.
    """
    H_k = Hamiltonian_Obj.compute_static(kx, ky)
    eigenvalues, eigenvectors = np.linalg.eigh(H_k)
    return eigenvalues, eigenvectors

def compute_chunk(chunk_indices, kx_flat, ky_flat, Hamiltonian_Obj, dim):
    """
    Compute eigenvalues and eigenfunctions for a chunk of the grid.
    """
    results = []
    for idx in chunk_indices:
        kx, ky = kx_flat[idx], ky_flat[idx]
        eigenvalues, eigenfunctions = compute_point(kx, ky, Hamiltonian_Obj, dim)
        results.append((idx, eigenvalues, eigenfunctions))
    return results

def parallel_compute(kx, ky, Hamiltonian_Obj, dim, num_workers=None):
    """
    Parallel computation of eigenvalues and eigenfunctions over the kx-ky grid.

    Parameters:
    - kx, ky: 2D arrays of the k-space grid.
    - Hamiltonian_Obj: Hamiltonian object to compute eigenvalues and eigenfunctions.
    - dim: Dimension of the Hamiltonian.
    - num_workers: Number of parallel workers. Defaults to the number of CPU cores.
    """
    # Flatten the kx and ky grids
    kx_flat = kx.flatten()
    ky_flat = ky.flatten()
    total_points = len(kx_flat)

    # Split the indices into chunks for parallel processing
    chunk_size = total_points // (num_workers or os.cpu_count())
    chunks = [range(i, min(i + chunk_size, total_points)) for i in range(0, total_points, chunk_size)]

    # Parallel computation
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_chunk, chunk, kx_flat, ky_flat, Hamiltonian_Obj, dim) for chunk in chunks]
        for future in tqdm(futures, desc="Computing eigenvalues/eigenfunctions"):
            results.extend(future.result())

    # Initialize arrays to store results
    eigenvalues = np.full((kx.shape[0], kx.shape[1], dim), np.nan, dtype=float)
    eigenfunctions = np.full((kx.shape[0], kx.shape[1], dim, dim), np.nan, dtype=complex)

    # Populate results
    for idx, vals, funcs in results:
        i, j = np.unravel_index(idx, kx.shape)
        eigenvalues[i, j, :] = vals
        eigenfunctions[i, j, :, :] = funcs

    return eigenvalues, eigenfunctions

# Example usage:
kx = np.linspace(-np.pi, np.pi, 100)
ky = np.linspace(-np.pi, np.pi, 100)
kx, ky = np.meshgrid(kx, ky)

Hamiltonian_Obj = TwoOrbitalUnspinfulHamiltonian(zeta=0.5, A0=0, mu=2)
dim = Hamiltonian_Obj.dim

eigenvalues, eigenfunctions = parallel_compute(kx, ky, Hamiltonian_Obj, dim)
