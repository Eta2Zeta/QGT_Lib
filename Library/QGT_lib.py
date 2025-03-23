from .eigenvalue_calc_lib import *    
from .utilities import sign_check

# & Calculation with analytical eigenfunctions
# Projection operator
def projection_operator(psi):
    return np.outer(psi, np.conj(psi))

# Numerical derivative w.r.t. kx
def dpsi_dx(psi, kx, ky, delta_k):
    psi_front = psi(kx + delta_k, ky)
    psi_back = psi(kx - delta_k, ky, prev_psi = psi_front)
    return (psi_front - psi_back) / (2 * delta_k)

# Numerical derivative w.r.t. ky
def dpsi_dy(psi, kx, ky, delta_k):
    psi_front = psi(kx, ky + delta_k)
    psi_back = psi(kx, ky - delta_k, prev_psi = psi_front)
    return (psi_front - psi_back) / (2 * delta_k)


# Calculate the quantum geometric tensor components
def quantum_geometric_tensor(psi, I, kx, ky, delta_k):
    dpsi_dx_val = dpsi_dx(psi, kx, ky, delta_k)
    dpsi_dy_val = dpsi_dy(psi, kx, ky, delta_k)
    psi_val = psi(kx, ky)
    P = projection_operator(psi_val)
    
    g_xx = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dx_val).real
    g_xy_real = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).real
    g_xy_imag = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).imag
    g_yy = np.vdot(dpsi_dy_val, (I - P) @ dpsi_dy_val).real
    
    return g_xx, g_xy_real, g_xy_imag, g_yy

# & sanity checks
# Calculate the quantum geometric tensor components
def quantum_geometric_tensor_term1(psi, I, kx, ky, delta_k):
    dpsi_dx_val = dpsi_dx(psi, kx, ky, delta_k)
    dpsi_dy_val = dpsi_dy(psi, kx, ky, delta_k)
    dpsi_dx_val, dpsi_dy_val = sign_check(dpsi_dx_val, dpsi_dy_val)
    g_xx = np.vdot(dpsi_dx_val, I @ dpsi_dx_val).real
    g_xy_real = np.vdot(dpsi_dx_val, I @ dpsi_dy_val).real
    g_xy_imag = np.vdot(dpsi_dx_val, I @ dpsi_dy_val).imag
    g_yy = np.vdot(dpsi_dy_val, I @ dpsi_dy_val).real
    
    return g_xx, g_xy_real, g_xy_imag, g_yy


# Calculate the quantum geometric tensor components
def quantum_geometric_tensor_term2(psi, kx, ky, delta_k):
    dpsi_dx_val = dpsi_dx(psi, kx, ky, delta_k)
    dpsi_dy_val = dpsi_dy(psi, kx, ky, delta_k)
    dpsi_dx_val, dpsi_dy_val = sign_check(dpsi_dx_val, dpsi_dy_val)
    psi_val = psi(kx, ky)

    P = projection_operator(psi_val)
    
    g_xx = np.vdot(dpsi_dx_val, P @ dpsi_dx_val).real
    g_xy_real = np.vdot(dpsi_dx_val, P @ dpsi_dy_val).real
    g_xy_imag = np.vdot(dpsi_dx_val, P @ dpsi_dy_val).imag
    g_yy = np.vdot(dpsi_dy_val, P @ dpsi_dy_val).real
    
    return g_xx, g_xy_real, g_xy_imag, g_yy


# & Calculation with numerical eigenfunctions
# Numerical derivative w.r.t. kx
def dpsi_dx_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index):
    eigenvector_plus = Eigenvectors(len(eigenfunction))
    eigenvector_minus = Eigenvectors(len(eigenfunction))
    eigenvector_plus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky)
    eigenvector_minus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky)

    # Calculate for kx + delta_k
    eigenvalues_plus, psi_plus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx + delta_k, ky)
    psi_plus_ordered = eigenvector_plus.set_eigenvectors_eigenvalue_preordered(psi_plus, eigenvalues_plus, kx + delta_k, ky)

    # Calculate for kx - delta_k
    eigenvalues_minus, psi_minus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx - delta_k, ky)
    psi_minus_ordered = eigenvector_minus.set_eigenvectors_eigenvalue_preordered(psi_minus, eigenvalues_minus, kx - delta_k, ky)

    # Return the derivative for the specified band
    return (psi_plus_ordered[band_index] - psi_minus_ordered[band_index]) / (2 * delta_k)

# Numerical derivative w.r.t. ky
def dpsi_dy_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index):
    eigenvector_plus = Eigenvectors(len(eigenfunction))
    eigenvector_minus = Eigenvectors(len(eigenfunction))
    eigenvector_plus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky)
    eigenvector_minus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky)

    # Calculate for ky + delta_k
    eigenvalues_plus, psi_plus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky + delta_k)
    psi_plus_ordered = eigenvector_plus.set_eigenvectors_eigenvalue_preordered(psi_plus, eigenvalues_plus, kx, ky + delta_k)

    # Calculate for ky - delta_k
    eigenvalues_minus, psi_minus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky - delta_k)
    psi_minus_ordered = eigenvector_minus.set_eigenvectors_eigenvalue_preordered(psi_minus, eigenvalues_minus, kx, ky - delta_k)

    # Return the derivative for the specified band
    return (psi_plus_ordered[band_index] - psi_minus_ordered[band_index]) / (2 * delta_k)

# Quantum geometric tensor components calculation using numerically obtained eigenfunctions
def quantum_geometric_tensor_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index):
    dpsi_dx_val = dpsi_dx_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index)
    dpsi_dy_val = dpsi_dy_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index)
    psi_val = eigenfunction[band_index]

    dim = Hamiltonian.dim
    I = np.eye(dim)
    P = projection_operator(psi_val)
    
    g_xx = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dx_val).real
    g_xy_real = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).real
    g_xy_imag = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).imag
    g_yy = np.vdot(dpsi_dy_val, (I - P) @ dpsi_dy_val).real
    
    return g_xx, g_xy_real, g_xy_imag, g_yy


def QGT_grid(
    kx, ky, eigenvalues, eigenfunctions, quantum_geometric_tensor_func, 
    hamiltonian, delta_k, band_index, z_cutoff=None
):
    """
    Calculate the quantum geometric tensor (QGT) components for a kx-ky grid with a progress bar.

    Parameters:
    - kx, ky: 2D arrays defining the k-space grid.
    - eigenvalues: 2D array of eigenvalues corresponding to the k-space grid.
    - eigenfunctions: 2D array of eigenfunctions corresponding to the k-space grid.
    - quantum_geometric_tensor_func: Function to calculate QGT components.
    - hamiltonian: The Hamiltonian function for the system.
    - delta_k: Small step for numerical differentiation.
    - band_index: Band index for which QGT is calculated.
    - z_cutoff: Optional maximum value for clipping the QGT components.

    Returns:
    - g_xx_array: 2D array of g_xx components.
    - g_xy_real_array: 2D array of real parts of g_xy components.
    - g_xy_imag_array: 2D array of imaginary parts of g_xy components.
    - g_yy_array: 2D array of g_yy components.
    - trace_array: 2D array of trace components (g_xx + g_yy).
    """
    # Initialize arrays to store tensor components
    g_xx_array = np.zeros(kx.shape)
    g_xy_real_array = np.zeros(kx.shape)
    g_xy_imag_array = np.zeros(kx.shape)
    g_yy_array = np.zeros(kx.shape)
    trace_array = np.zeros(kx.shape)

    total_points = kx.shape[0] * kx.shape[1]  # Total number of k-points

    # Create a progress bar for the nested loop
    with tqdm(total=total_points, desc="Computing QGT grid", unit="point") as pbar:
        for i in range(kx.shape[0]):
            for j in range(kx.shape[1]):
                eigenfunction = eigenfunctions[i, j]
                eigenvalue = eigenvalues[i, j]

                g_xx, g_xy_real, g_xy_imag, g_yy = quantum_geometric_tensor_func(
                    hamiltonian, kx[i, j], ky[i, j], delta_k, eigenvalue, eigenfunction, band_index
                )

                # Store computed QGT components
                g_xx_array[i, j] = g_xx
                g_xy_real_array[i, j] = g_xy_real
                g_xy_imag_array[i, j] = g_xy_imag
                g_yy_array[i, j] = g_yy
                trace_array[i, j] = g_xx + g_yy

                # Update progress bar
                pbar.update(1)

    # Apply the cutoff if specified
    if z_cutoff is not None:
        g_xx_array = np.clip(g_xx_array, None, z_cutoff)
        g_xy_real_array = np.clip(g_xy_real_array, None, z_cutoff)
        g_xy_imag_array = np.clip(g_xy_imag_array, None, z_cutoff)
        g_yy_array = np.clip(g_yy_array, None, z_cutoff)
        trace_array = np.clip(trace_array, None, z_cutoff)

    return g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array



def QGT_line(Hamiltonian, line_kx, line_ky, delta_k, band_index):
    """
    Calculate the Quantum Geometric Tensor (QGT) along a line in the kx-ky plane.

    Parameters:
    - Hamiltonian: Function to compute the Hamiltonian matrix.
    - k_line: 1D array of k-values along the line.
    - k_angle: The angle of the line in radians.
    - delta_k: Small step for numerical differentiation.
    - dim: The dimension of the system.
    - band_index: Band index for which to calculate the QGT.

    Returns:
    - g_xx_values: Array of g_xx components along the line.
    - g_xy_real_values: Array of real parts of g_xy components along the line.
    - g_xy_imag_values: Array of imaginary parts of g_xy components along the line.
    - g_yy_values: Array of g_yy components along the line.
    - trace_values: Array of trace components (g_xx + g_yy) along the line.
    """
    # Step 1: Get eigenvalues and eigenfunctions along the line
    eigenvalues, eigenfunctions, _ = line_eigenvalues_eigenfunctions(Hamiltonian, line_kx, line_ky, band_index)

    # Ensure eigenvalues is at least 2D (e.g., [points, bands])
    eigenvalues = np.asarray(eigenvalues)

    if eigenvalues.ndim == 1:
        # If eigenvalues is 1D (e.g., just one band at each k-point)
        eigenvalues_band = eigenvalues
    elif eigenvalues.ndim >= 2:
        # General case: eigenvalues is 2D or more, extract the specified band
        eigenvalues_band = eigenvalues[..., band_index]
    else:
        raise ValueError("Invalid eigenvalues shape.")
    

    # Step 2: Initialize arrays to store QGT components
    g_xx_values = []
    g_xy_real_values = []
    g_xy_imag_values = []
    g_yy_values = []
    trace_values = []

    # Step 3: Calculate QGT components at each point along the line
    for i, (kx, ky) in enumerate(zip(line_kx, line_ky)):
        eigenvalue = eigenvalues[i]
        eigenfunction = eigenfunctions[i]

        g_xx, g_xy_real, g_xy_imag, g_yy = quantum_geometric_tensor_num(
            Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index
        )

        g_xx_values.append(g_xx)
        g_xy_real_values.append(g_xy_real)
        g_xy_imag_values.append(g_xy_imag)
        g_yy_values.append(g_yy)
        trace_values.append(g_xx + g_yy)

    # Convert results to numpy arrays
    g_xx_values = np.array(g_xx_values)
    g_xy_real_values = np.array(g_xy_real_values)
    g_xy_imag_values = np.array(g_xy_imag_values)
    g_yy_values = np.array(g_yy_values)
    trace_values = np.array(trace_values)

    return eigenvalues_band, g_xx_values, g_xy_real_values, g_xy_imag_values, g_yy_values, trace_values
