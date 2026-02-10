from .eigenvalue_calc_lib import *    
from .utilities import sign_check

# & Calculation with semi-analytical eigenfunctions
# & 
# & I think the idea is to get the expressions for the eigenfunctions 
# & and then use them to calculate the quantum geometric tensor
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


def quantum_geometric_tensor_semi_num(hamiltonian, band_index, kx, ky, delta_k):
    """
    Semi-analytic QGT for the chiral (holomorphic) pseudo-eigenvectors.

    Parameters
    ----------
    hamiltonian : object
        Must provide:
          - .dim (int): Hilbert-space dimension (2n)
          - .pseudo_eigenvector(band_index) -> callable psi(kx, ky, prev_psi=None)
    band_index : int
        0 -> psiA (A-chiral), 1 -> psiB (B-chiral)
    kx, ky : float
        Momentum coordinates
    delta_k : float
        Central-difference step for numerical derivatives

    Returns
    -------
    g_xx, g_xy_real, g_xy_imag, g_yy : floats
        QGT components for the chosen pseudo-eigenvector at (kx, ky)
    """
    # get the callable pseudo-eigenvector psi(kx, ky, prev_psi=None)
    if not hasattr(hamiltonian, 'pseudo_eigenvector'):
        raise AttributeError("hamiltonian must implement .pseudo_eigenvector(band_index)")

    psi = hamiltonian.pseudo_eigenvector(band_index)

    # identity in the full 2n-dimensional space
    I = np.eye(hamiltonian.dim, dtype=complex)

    # finite differences (uses prev_psi internally to smooth gauge)
    dpsi_dx_val = dpsi_dx(psi, kx, ky, delta_k)
    dpsi_dy_val = dpsi_dy(psi, kx, ky, delta_k)

    # projector onto the chosen state
    psi_val = psi(kx, ky)
    P = projection_operator(psi_val)

    # QGT components: g_{μν} = Re <∂μψ | (1 - |ψ><ψ|) | ∂νψ>,  Im-part gives Berry curvature/2
    g_xx      = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dx_val).real
    g_xy_real = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).real
    g_xy_imag = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).imag
    g_yy      = np.vdot(dpsi_dy_val, (I - P) @ dpsi_dy_val).real

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
def dpsi_dx_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=0):
    eigenvector_plus = Eigenvectors(len(eigenfunction))
    eigenvector_minus = Eigenvectors(len(eigenfunction))
    eigenvector_plus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky, kz=kz)
    eigenvector_minus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky, kz=kz)

    # Calculate for kx + delta_k
    eigenvalues_plus, psi_plus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx + delta_k, ky, kz=kz)
    psi_plus_ordered = eigenvector_plus.set_eigenvectors_eigenvalue_preordered(psi_plus, eigenvalues_plus, kx + delta_k, ky, kz=kz)

    # Calculate for kx - delta_k
    eigenvalues_minus, psi_minus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx - delta_k, ky, kz=kz)
    psi_minus_ordered = eigenvector_minus.set_eigenvectors_eigenvalue_preordered(psi_minus, eigenvalues_minus, kx - delta_k, ky, kz=kz)

    # Return the derivative for the specified band
    return (psi_plus_ordered[band_index] - psi_minus_ordered[band_index]) / (2 * delta_k)

# Numerical derivative w.r.t. ky
def dpsi_dy_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=0):
    eigenvector_plus = Eigenvectors(len(eigenfunction))
    eigenvector_minus = Eigenvectors(len(eigenfunction))
    eigenvector_plus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky, kz=kz)
    eigenvector_minus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky, kz=kz)

    # Calculate for ky + delta_k
    eigenvalues_plus, psi_plus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky + delta_k, kz=kz)
    psi_plus_ordered = eigenvector_plus.set_eigenvectors_eigenvalue_preordered(psi_plus, eigenvalues_plus, kx, ky + delta_k, kz=kz)

    # Calculate for ky - delta_k
    eigenvalues_minus, psi_minus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky - delta_k, kz=kz)
    psi_minus_ordered = eigenvector_minus.set_eigenvectors_eigenvalue_preordered(psi_minus, eigenvalues_minus, kx, ky - delta_k, kz=kz)

    # Return the derivative for the specified band
    return (psi_plus_ordered[band_index] - psi_minus_ordered[band_index]) / (2 * delta_k)

# Numerical derivative w.r.t. kx with Eigenvector Ordering
def dpsi_dx_num_eigenvector_ordered(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=0):
    eigenvector_plus = Eigenvectors(len(eigenfunction))
    eigenvector_minus = Eigenvectors(len(eigenfunction))
    eigenvector_plus.set_eigenvectors_eigenvector_ordered(eigenfunction, eigenvalue, kx, ky, kz=kz)
    eigenvector_minus.set_eigenvectors_eigenvector_ordered(eigenfunction, eigenvalue, kx, ky, kz=kz)

    # Calculate for kx + delta_k
    eigenvalues_plus, psi_plus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx + delta_k, ky, kz=kz)
    eigenvalues_plus_ordered, psi_plus_ordered = eigenvector_plus.set_eigenvectors_eigenvector_ordered(psi_plus, eigenvalues_plus, kx + delta_k, ky, kz=kz)

    # Calculate for kx - delta_k
    eigenvalues_minus, psi_minus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx - delta_k, ky, kz=kz)
    eigenvalues_minus_ordered, psi_minus_ordered = eigenvector_minus.set_eigenvectors_eigenvector_ordered(psi_minus, eigenvalues_minus, kx - delta_k, ky, kz=kz)

    # Return the derivative for the specified band
    return (psi_plus_ordered[band_index] - psi_minus_ordered[band_index]) / (2 * delta_k)

# Numerical derivative w.r.t. ky with Eigenvector Ordering
def dpsi_dy_num_eigenvector_ordered(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=0):
    eigenvector_plus = Eigenvectors(len(eigenfunction))
    eigenvector_minus = Eigenvectors(len(eigenfunction))
    eigenvector_plus.set_eigenvectors_eigenvector_ordered(eigenfunction, eigenvalue, kx, ky, kz=kz)
    eigenvector_minus.set_eigenvectors_eigenvector_ordered(eigenfunction, eigenvalue, kx, ky, kz=kz)

    # Calculate for ky + delta_k
    eigenvalues_plus, psi_plus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky + delta_k, kz=kz)
    eigenvalues_plus_ordered, psi_plus_ordered = eigenvector_plus.set_eigenvectors_eigenvector_ordered(psi_plus, eigenvalues_plus, kx, ky + delta_k, kz=kz)

    # Calculate for ky - delta_k
    eigenvalues_minus, psi_minus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky - delta_k, kz=kz)
    eigenvalues_minus_ordered, psi_minus_ordered = eigenvector_minus.set_eigenvectors_eigenvector_ordered(psi_minus, eigenvalues_minus, kx, ky - delta_k, kz=kz)

    # Return the derivative for the specified band
    return (psi_plus_ordered[band_index] - psi_minus_ordered[band_index]) / (2 * delta_k)


# Numerical derivative w.r.t. kz
def dpsi_dz_num(Hamiltonian, kx, ky, kz, delta_k, eigenvalue, eigenfunction, band_index):
    eigenvector_plus = Eigenvectors(len(eigenfunction))
    eigenvector_minus = Eigenvectors(len(eigenfunction))
    eigenvector_plus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky, kz=kz)
    eigenvector_minus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky, kz=kz)

    # Calculate for kz + delta_k
    eigenvalues_plus, psi_plus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky, kz=kz + delta_k)
    psi_plus_ordered = eigenvector_plus.set_eigenvectors_eigenvalue_preordered(psi_plus, eigenvalues_plus, kx, ky, kz=kz + delta_k)

    # Calculate for kz - delta_k
    eigenvalues_minus, psi_minus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky, kz=kz - delta_k)
    psi_minus_ordered = eigenvector_minus.set_eigenvectors_eigenvalue_preordered(psi_minus, eigenvalues_minus, kx, ky, kz=kz - delta_k)

    # Return the derivative for the specified band
    return (psi_plus_ordered[band_index] - psi_minus_ordered[band_index]) / (2 * delta_k)


# Numerical derivative w.r.t. kz with Eigenvector Ordering
def dpsi_dz_num_eigenvector_ordered(Hamiltonian, kx, ky, kz, delta_k, eigenvalue, eigenfunction, band_index):
    eigenvector_plus = Eigenvectors(len(eigenfunction))
    eigenvector_minus = Eigenvectors(len(eigenfunction))
    eigenvector_plus.set_eigenvectors_eigenvector_ordered(eigenfunction, eigenvalue, kx, ky, kz=kz)
    eigenvector_minus.set_eigenvectors_eigenvector_ordered(eigenfunction, eigenvalue, kx, ky, kz=kz)

    # Calculate for kz + delta_k
    eigenvalues_plus, psi_plus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky, kz=kz + delta_k)
    eigenvalues_plus_ordered, psi_plus_ordered = eigenvector_plus.set_eigenvectors_eigenvector_ordered(psi_plus, eigenvalues_plus, kx, ky, kz=kz + delta_k)

    # Calculate for kz - delta_k
    eigenvalues_minus, psi_minus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky, kz=kz - delta_k)
    eigenvalues_minus_ordered, psi_minus_ordered = eigenvector_minus.set_eigenvectors_eigenvector_ordered(psi_minus, eigenvalues_minus, kx, ky, kz=kz - delta_k)

    # Return the derivative for the specified band
    return (psi_plus_ordered[band_index] - psi_minus_ordered[band_index]) / (2 * delta_k)

# Quantum geometric tensor components calculation using numerically obtained eigenfunctions
def quantum_geometric_tensor_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=0):
    dpsi_dx_val = dpsi_dx_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=kz)
    dpsi_dy_val = dpsi_dy_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=kz)
    psi_val = eigenfunction[band_index]

    dim = Hamiltonian.dim
    I = np.eye(dim)
    P = projection_operator(psi_val)
    
    g_xx = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dx_val).real
    g_xy_real = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).real
    g_xy_imag = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).imag
    g_yy = np.vdot(dpsi_dy_val, (I - P) @ dpsi_dy_val).real
    
    return g_xx, g_xy_real, g_xy_imag, g_yy

# Quantum geometric tensor components calculation using numerically obtained eigenfunctions with Eigenvector Ordering
def quantum_geometric_tensor_num_eigenvector_ordered(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=0):
    dpsi_dx_val = dpsi_dx_num_eigenvector_ordered(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=kz)
    dpsi_dy_val = dpsi_dy_num_eigenvector_ordered(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=kz)
    psi_val = eigenfunction[band_index]

    dim = Hamiltonian.dim
    I = np.eye(dim)
    P = projection_operator(psi_val)
    
    g_xx = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dx_val).real
    g_xy_real = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).real
    g_xy_imag = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).imag
    g_yy = np.vdot(dpsi_dy_val, (I - P) @ dpsi_dy_val).real
    
    return g_xx, g_xy_real, g_xy_imag, g_yy


# 3D Quantum Geometric Tensor Num
def quantum_geometric_tensor_3d_num(Hamiltonian, kx, ky, kz, delta_k, eigenvalue, eigenfunction, band_index):
    dpsi_dx_val = dpsi_dx_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=kz)
    dpsi_dy_val = dpsi_dy_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=kz)
    dpsi_dz_val = dpsi_dz_num(Hamiltonian, kx, ky, kz, delta_k, eigenvalue, eigenfunction, band_index)
    psi_val = eigenfunction[band_index]

    dim = Hamiltonian.dim
    I = np.eye(dim)
    P = projection_operator(psi_val)

    # XY components
    g_xx = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dx_val).real
    g_xy_real = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).real
    g_xy_imag = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).imag
    g_yy = np.vdot(dpsi_dy_val, (I - P) @ dpsi_dy_val).real

    # XZ components
    g_xz_real = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dz_val).real
    g_xz_imag = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dz_val).imag
    g_zz = np.vdot(dpsi_dz_val, (I - P) @ dpsi_dz_val).real

    # YZ components
    g_yz_real = np.vdot(dpsi_dy_val, (I - P) @ dpsi_dz_val).real
    g_yz_imag = np.vdot(dpsi_dy_val, (I - P) @ dpsi_dz_val).imag

    return g_xx, g_yy, g_zz, g_xy_real, g_xy_imag, g_xz_real, g_xz_imag, g_yz_real, g_yz_imag


# 3D Quantum Geometric Tensor Num Eigenvector Ordered
def quantum_geometric_tensor_3d_num_eigenvector_ordered(Hamiltonian, kx, ky, kz, delta_k, eigenvalue, eigenfunction, band_index):
    dpsi_dx_val = dpsi_dx_num_eigenvector_ordered(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=kz)
    dpsi_dy_val = dpsi_dy_num_eigenvector_ordered(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=kz)
    dpsi_dz_val = dpsi_dz_num_eigenvector_ordered(Hamiltonian, kx, ky, kz, delta_k, eigenvalue, eigenfunction, band_index)
    psi_val = eigenfunction[band_index]

    dim = Hamiltonian.dim
    I = np.eye(dim)
    P = projection_operator(psi_val)

    # XY components
    g_xx = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dx_val).real
    g_xy_real = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).real
    g_xy_imag = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).imag
    g_yy = np.vdot(dpsi_dy_val, (I - P) @ dpsi_dy_val).real

    # XZ components
    g_xz_real = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dz_val).real
    g_xz_imag = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dz_val).imag
    g_zz = np.vdot(dpsi_dz_val, (I - P) @ dpsi_dz_val).real

    # YZ components
    g_yz_real = np.vdot(dpsi_dy_val, (I - P) @ dpsi_dz_val).real
    g_yz_imag = np.vdot(dpsi_dy_val, (I - P) @ dpsi_dz_val).imag

    return g_xx, g_yy, g_zz, g_xy_real, g_xy_imag, g_xz_real, g_xz_imag, g_yz_real, g_yz_imag

def quantum_geometric_tensor_analytic(Hamiltonian, kx, ky, band=-1):
    """
    Returns the analytical components of the quantum geometric tensor (QGT)
    for a given Hamiltonian at momentum (kx, ky).

    If g_xy_imag is not implemented, this function tries to compute it
    from the Berry curvature:
        g_xy_imag = -1/2 * Omega_full

    Parameters:
        Hamiltonian: Hamiltonian object with optional analytical QGT methods
        kx, ky: momentum components
        band: +1 or -1 (Berry curvature sign depends on band)

    Returns:
        g_xx, g_xy_real, g_xy_imag, g_yy, trace
    """

    # --- Standard metric components ---
    g_xx = Hamiltonian.g_xx(kx, ky) if hasattr(Hamiltonian, 'g_xx') else None
    g_xy_real = Hamiltonian.g_xy_real(kx, ky) if hasattr(Hamiltonian, 'g_xy_real') else None
    g_yy = Hamiltonian.g_yy(kx, ky) if hasattr(Hamiltonian, 'g_yy') else None
    trace = Hamiltonian.trace(kx, ky) if hasattr(Hamiltonian, 'trace') else None

    # --- Imaginary part (Berry curvature contribution) ---
    if hasattr(Hamiltonian, 'g_xy_imag'):
        # Direct access if implemented
        g_xy_imag = Hamiltonian.g_xy_imag(kx, ky)

    elif hasattr(Hamiltonian, 'berry_curvature_full'):
        # Compute using full projected curvature
        # g_xy_imag = -(1/2) * Omega_full
        Omega = Hamiltonian.berry_curvature_full(kx, ky, band=band)
        g_xy_imag = -0.5 * Omega

    elif hasattr(Hamiltonian, 'berry_curvature_full_radial'):
        # If only radial curvature exists
        k = (kx**2 + ky**2)**0.5
        Omega = Hamiltonian.berry_curvature_full_radial(k, band=band)
        g_xy_imag = -0.5 * Omega

    else:
        # Cannot infer
        g_xy_imag = None

    return g_xx, g_xy_real, g_xy_imag, g_yy, trace


def QGT_grid_num(
    ki, kj, eigenvalues, eigenfunctions, quantum_geometric_tensor_func, 
    hamiltonian, delta_k, band_index, z_cutoff=None,
    progress_label=None, kk=0
):
    """
    Calculate the quantum geometric tensor (QGT) components for a ki-kj grid with a fixed kk.

    Parameters:
    - ki, kj: 2D arrays defining the k-space grid (meshgrids).
    - eigenvalues: 2D array of eigenvalues corresponding to the grid.
    - eigenfunctions: 2D array of eigenfunctions corresponding to the grid.
    - quantum_geometric_tensor_func: Function to calculate QGT components (must return 9 components).
    - hamiltonian: The Hamiltonian function for the system.
    - delta_k: Small step for numerical differentiation.
    - band_index: Band index for which QGT is calculated.
    - z_cutoff: Optional maximum value for clipping the QGT components.
    - kk: Fixed value for the third momentum component.

    Returns:
    - g_xx_array, g_yy_array, g_zz_array
    - g_xy_real_array, g_xy_imag_array
    - g_xz_real_array, g_xz_imag_array
    - g_yz_real_array, g_yz_imag_array
    """
    # Initialize arrays to store tensor components
    g_xx_array = np.zeros(ki.shape)
    g_yy_array = np.zeros(ki.shape)
    g_zz_array = np.zeros(ki.shape)

    g_xy_real_array = np.zeros(ki.shape)
    g_xy_imag_array = np.zeros(ki.shape)

    g_xz_real_array = np.zeros(ki.shape)
    g_xz_imag_array = np.zeros(ki.shape)

    g_yz_real_array = np.zeros(ki.shape)
    g_yz_imag_array = np.zeros(ki.shape)

    total_points = ki.shape[0] * ki.shape[1]

    desc = f"QGT grid [{progress_label}]" if progress_label else "Computing QGT grid"
    # leave=False avoids leaving dozens of bars when using many processes
    with tqdm(total=total_points, desc=desc, unit="kpt", leave=False) as pbar:
        for i in range(ki.shape[0]):
            for j in range(ki.shape[1]):
                eigenfunction = eigenfunctions[i, j]
                eigenvalue = eigenvalues[i, j]
                
                # Call the 3D QGT function with fixed kk
                # The signature expected here is:
                # func(Hamiltonian, kx, ky, kz, delta_k, eigenvalue, eigenfunction, band_index)
                g_xx, g_yy, g_zz, g_xy_real, g_xy_imag, g_xz_real, g_xz_imag, g_yz_real, g_yz_imag = quantum_geometric_tensor_func(
                    hamiltonian, ki[i, j], kj[i, j], kk, delta_k, eigenvalue, eigenfunction, band_index
                )
                
                g_xx_array[i, j] = g_xx
                g_yy_array[i, j] = g_yy
                g_zz_array[i, j] = g_zz
                
                g_xy_real_array[i, j] = g_xy_real
                g_xy_imag_array[i, j] = g_xy_imag
                
                g_xz_real_array[i, j] = g_xz_real
                g_xz_imag_array[i, j] = g_xz_imag
                
                g_yz_real_array[i, j] = g_yz_real
                g_yz_imag_array[i, j] = g_yz_imag

                pbar.update(1)

    # Clipping if z_cutoff is provided
    if z_cutoff is not None:
        arrays = [g_xx_array, g_yy_array, g_zz_array, 
                  g_xy_real_array, g_xy_imag_array, 
                  g_xz_real_array, g_xz_imag_array, 
                  g_yz_real_array, g_yz_imag_array]
                  
        for idx, arr in enumerate(arrays):
            arrays[idx] = np.clip(arr, -z_cutoff, z_cutoff)
            
        return tuple(arrays)

    return (g_xx_array, g_yy_array, g_zz_array, 
            g_xy_real_array, g_xy_imag_array, 
            g_xz_real_array, g_xz_imag_array, 
            g_yz_real_array, g_yz_imag_array) 


def QGT_grid_3d_num(
    kx_vals, ky_vals, kz_vals, eigenvalues_3d, eigenfunctions_3d, quantum_geometric_tensor_func, 
    hamiltonian, delta_k, band_index, z_cutoff=None
):
    """
    Calculate the 3D quantum geometric tensor (QGT) components for a kx-ky-kz grid.
    
    Parameters:
    - kx_vals, ky_vals, kz_vals: 1D arrays defining the grid.
    - eigenvalues_3d: 3D array of eigenvalues [nkx, nky, nkz].
    - eigenfunctions_3d: 3D array of eigenfunctions [nkx, nky, nkz].
    - quantum_geometric_tensor_func: Function to calculate 3D QGT components.
    - hamiltonian: The Hamiltonian function.
    - delta_k: Small step for numerical differentiation.
    - band_index: Band index for which QGT is calculated.
    - z_cutoff: Optional maximum value for clipping.
    
    Returns:
    - 9 arrays of shape [nkx, nky, nkz] for metric components.
    """
    nkx = len(kx_vals)
    nky = len(ky_vals)
    nkz = len(kz_vals)
    
    # Initialize arrays
    g_xx_arr = np.zeros((nkx, nky, nkz))
    g_yy_arr = np.zeros((nkx, nky, nkz))
    g_zz_arr = np.zeros((nkx, nky, nkz))
    
    g_xy_real_arr = np.zeros((nkx, nky, nkz))
    g_xy_imag_arr = np.zeros((nkx, nky, nkz))
    
    g_xz_real_arr = np.zeros((nkx, nky, nkz))
    g_xz_imag_arr = np.zeros((nkx, nky, nkz))
    
    g_yz_real_arr = np.zeros((nkx, nky, nkz))
    g_yz_imag_arr = np.zeros((nkx, nky, nkz))
    
    total_points = nkx * nky * nkz
    
    with tqdm(total=total_points, desc="Computing 3D QGT Grid", unit="kpt") as pbar:
        for i, kx in enumerate(kx_vals):
            for j, ky in enumerate(ky_vals):
                for k, kz in enumerate(kz_vals):
                    eigenfunction = eigenfunctions_3d[i, j, k]
                    eigenvalue = eigenvalues_3d[i, j, k]
                    
                    g_xx, g_yy, g_zz, g_xy_real, g_xy_imag, g_xz_real, g_xz_imag, g_yz_real, g_yz_imag = quantum_geometric_tensor_func(
                        hamiltonian, kx, ky, kz, delta_k, eigenvalue, eigenfunction, band_index
                    )
                    
                    g_xx_arr[i, j, k] = g_xx
                    g_yy_arr[i, j, k] = g_yy
                    g_zz_arr[i, j, k] = g_zz
                    
                    g_xy_real_arr[i, j, k] = g_xy_real
                    g_xy_imag_arr[i, j, k] = g_xy_imag
                    
                    g_xz_real_arr[i, j, k] = g_xz_real
                    g_xz_imag_arr[i, j, k] = g_xz_imag
                    
                    g_yz_real_arr[i, j, k] = g_yz_real
                    g_yz_imag_arr[i, j, k] = g_yz_imag
                    
                    pbar.update(1)
                    
    # Clipping
    if z_cutoff is not None:
         # Helper to clip in place or return clipped
         arrays = [g_xx_arr, g_yy_arr, g_zz_arr, g_xy_real_arr, g_xy_imag_arr, g_xz_real_arr, g_xz_imag_arr, g_yz_real_arr, g_yz_imag_arr]
         for idx, arr in enumerate(arrays):
             arrays[idx] = np.clip(arr, -z_cutoff, z_cutoff)
         return tuple(arrays)
         
    return (g_xx_arr, g_yy_arr, g_zz_arr, 
            g_xy_real_arr, g_xy_imag_arr, 
            g_xz_real_arr, g_xz_imag_arr, 
            g_yz_real_arr, g_yz_imag_arr)


def QGT_grid_semi_num(
    kx, ky,
    quantum_geometric_tensor_func,
    hamiltonian,
    delta_k,
    band_index,
    z_cutoff=None
):
    """
    Calculate the semi-analytic quantum geometric tensor (QGT) components
    on a kx-ky grid using pseudo-eigenvectors from the Hamiltonian.

    Parameters
    ----------
    kx, ky : 2D np.ndarray
        Grids of kx, ky values (same shape).
    quantum_geometric_tensor_func : callable
        Function with signature:
            (hamiltonian, band_index, kx, ky, delta_k) -> (g_xx, g_xy_real, g_xy_imag, g_yy)
        e.g. your quantum_geometric_tensor_semi_num(...) defined earlier.
    hamiltonian : object
        Must provide:
          - .dim (int)
          - .pseudo_eigenvector(band_index) -> callable psi(kx, ky, prev_psi=None)
    delta_k : float
        Central-difference step for k-derivatives.
    band_index : int
        0 -> psiA, 1 -> psiB (by your convention).
    z_cutoff : float or None
        If provided, clip outputs above this value (upper bound only, like QGT_grid_num).

    Returns
    -------
    g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array : 2D np.ndarray
        Arrays of the same shape as kx/ky with QGT components (trace = g_xx + g_yy).
    """
    # Allocate outputs
    g_xx_array      = np.zeros(kx.shape)
    g_xy_real_array = np.zeros(kx.shape)
    g_xy_imag_array = np.zeros(kx.shape)
    g_yy_array      = np.zeros(kx.shape)
    trace_array     = np.zeros(kx.shape)

    total_points = kx.shape[0] * kx.shape[1]

    with tqdm(total=total_points, desc="Computing Semi-Num QGT Grid", unit="point") as pbar:
        for i in range(kx.shape[0]):
            for j in range(kx.shape[1]):
                g_xx, g_xy_real, g_xy_imag, g_yy = quantum_geometric_tensor_func(
                    hamiltonian, band_index, kx[i, j], ky[i, j], delta_k
                )

                g_xx_array[i, j]      = g_xx
                g_xy_real_array[i, j] = g_xy_real
                g_xy_imag_array[i, j] = g_xy_imag
                g_yy_array[i, j]      = g_yy
                trace_array[i, j]     = g_xx + g_yy

                pbar.update(1)

    # Optional clipping (upper bound only, matching QGT_grid_num behavior)
    if z_cutoff is not None:
        g_xx_array      = np.clip(g_xx_array,      None, z_cutoff)
        g_xy_real_array = np.clip(g_xy_real_array, None, z_cutoff)
        g_xy_imag_array = np.clip(g_xy_imag_array, None, z_cutoff)
        g_yy_array      = np.clip(g_yy_array,      None, z_cutoff)
        trace_array     = np.clip(trace_array,     None, z_cutoff)

    return g_xx_array, g_xy_real_array, g_xy_imag_array, g_yy_array, trace_array


def QGT_grid_analytic(
    kx, ky,
    quantum_geometric_tensor_func,
    hamiltonian,
    z_cutoff=None
):
    """
    Calculate the analytical quantum geometric tensor (QGT) components on a kx-ky grid.

    Parameters:
    - kx, ky: 2D arrays defining the k-space grid.
    - quantum_geometric_tensor_func: Analytical function returning QGT components.
    - hamiltonian: The Hamiltonian object.
    - z_cutoff: Optional upper bound to clip the QGT components.

    Returns:
    - g_xx_array: 2D array of g_xx components.
    - g_xy_real_array: 2D array of real parts of g_xy components.
    - g_xy_imag_array: 2D array of imaginary parts of g_xy components.
    - g_yy_array: 2D array of g_yy components.
    - trace_array: 2D array of g_xx + g_yy.
    """
    g_xx_array = np.zeros(kx.shape)
    g_xy_real_array = np.zeros(kx.shape)
    g_xy_imag_array = np.zeros(kx.shape)
    g_yy_array = np.zeros(kx.shape)
    trace_array = np.zeros(kx.shape)

    total_points = kx.shape[0] * kx.shape[1]

    with tqdm(total=total_points, desc="Computing Analytical QGT Grid", unit="point") as pbar:
        for i in range(kx.shape[0]):
            for j in range(kx.shape[1]):
                g_xx, g_xy_real, g_xy_imag, g_yy, trace = quantum_geometric_tensor_func(
                    hamiltonian, kx[i, j], ky[i, j]
                )

                g_xx_array[i, j] = g_xx if g_xx is not None else 0.0
                g_xy_real_array[i, j] = g_xy_real if g_xy_real is not None else 0.0
                g_xy_imag_array[i, j] = g_xy_imag if g_xy_imag is not None else 0.0
                g_yy_array[i, j] = g_yy if g_yy is not None else 0.0
                trace_array[i, j] = trace if trace is not None else 0.0

                pbar.update(1)

    if z_cutoff is not None:
        g_xx_array = np.clip(g_xx_array, -z_cutoff, z_cutoff)
        g_xy_real_array = np.clip(g_xy_real_array, -z_cutoff, z_cutoff)
        g_xy_imag_array = np.clip(g_xy_imag_array, -z_cutoff, z_cutoff)
        g_yy_array = np.clip(g_yy_array, -z_cutoff, z_cutoff)
        trace_array = np.clip(trace_array, -z_cutoff, z_cutoff)

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
    eigenvalues, eigenfunctions, _, perturbations, magnus_operator_norm = line_eigenvalues_eigenfunctions(Hamiltonian, line_kx, line_ky, band_index)

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

    return eigenvalues, perturbations, g_xx_values, g_xy_real_values, g_xy_imag_values, g_yy_values, trace_values, magnus_operator_norm


def compute_QGT_projector(eigenvectors, band_idx, dk_x, dk_y):
    """
    Computes the full Quantum Geometric Tensor (Metric + Curvature) using the
    gauge-invariant Projector method.
    
    Parameters:
    - eigenvectors: complex array (Nx, Ny, num_bands, dim_hilbert)
      Note: Calc_Eigenvalues produces (Nx, Ny, dim, dim) where
            eigenvectors[i, j, m, :] is the m-th eigenvector components.
    - band_idx: index of the band to compute (e.g., 0 for ground state)
    - dk_x, dk_y: grid spacing in kx and ky directions
    
    Returns:
    - g_xx, g_xy, g_yy: Components of the Quantum Metric (Real part of QGT)
    - berry_curvature: The Berry Curvature (Imaginary part of QGT, Omega_xy)
    """
    Nx, Ny, num_bands, dim_hilbert = eigenvectors.shape
    
    # 1. Extract the specific band we want
    # shape: (Nx, Ny, dim_hilbert)
    # The eigenvectors array is [Nx, Ny, band, component]
    psi = eigenvectors[:, :, band_idx, :]
    
    # 2. Construct the Projector P = |u><u| at every k-point
    # We want an array of matrices (Nx, Ny, dim_hilbert, dim_hilbert)
    # P[k] = outer(psi[k], conj(psi[k]))
    
    # Efficient broadcasting way to do outer product:
    # psi[:,:,:,None] is (Nx, Ny, dim, 1)
    # conj(psi[:,:,None,:]) is (Nx, Ny, 1, dim)
    # The product is (Nx, Ny, dim, dim)
    P = psi[:, :, :, None] * np.conj(psi[:, :, None, :])
    
    # 3. Calculate Derivatives of the Projector (dP/dk)
    # np.gradient uses central differences, which is stable for smooth P
    # axis 0 is kx, axis 1 is ky
    dP_dx = np.gradient(P, dk_x, axis=0)
    dP_dy = np.gradient(P, dk_y, axis=1)
    
    # 4. Compute QGT Components using Trace Formulas
    
    # --- Quantum Metric g_mu_nu = 0.5 * Tr( dP_mu * dP_nu ) ---
    # We use Einstein summation for the trace: "ab,ba -> scalar"
    
    # g_xx
    prod_xx = np.matmul(dP_dx, dP_dx) # Matrix product (dP/dx)(dP/dx)
    g_xx = 0.5 * np.trace(prod_xx, axis1=2, axis2=3).real
    
    # g_yy
    prod_yy = np.matmul(dP_dy, dP_dy)
    g_yy = 0.5 * np.trace(prod_yy, axis1=2, axis2=3).real
    
    # g_xy (Symmetric part)
    prod_xy = np.matmul(dP_dx, dP_dy)
    prod_yx = np.matmul(dP_dy, dP_dx)
    # Note: g_xy = 0.5 * Tr( dP_x dP_y + dP_y dP_x )? 
    # Actually for metric usually defined as Re(Q_xy).
    # Since dP is hermitian, Tr(dP_x dP_y) is complex.
    # The real part is the metric, imaginary part is curvature related.
    trace_xy = np.trace(prod_xy, axis1=2, axis2=3)
    g_xy = 0.5 * (trace_xy + np.conj(trace_xy)).real 
    
    # --- Berry Curvature Omega_xy = i * Tr( P * [dP_x, dP_y] ) ---
    
    # Commutator [dP_x, dP_y]
    comm = prod_xy - prod_yx
    
    # Multiply by P: P * [dP_x, dP_y]
    P_comm = np.matmul(P, comm)
    
    # Trace and multiply by i
    # The projector formula Omega = i Tr(P [dP_x, dP_y]) matches standard definitions.
    berry_curvature = 1j * np.trace(P_comm, axis1=2, axis2=3)
    
    # Return real part of curvature (it should be real physically, imag part is numerical noise)
    return g_xx, g_xy, g_yy, berry_curvature.real
