from .eigenvalue_calc_lib import *    
from .utilities import sign_check
from .dimension_lib import map_k_by_order
from .calculus_lib import *
from .QGT_calc_functions_2comp import *
from .QGT_calc_functions_3comp import *


def QGT_grid_num(
    ki, kj, eigenvalues, eigenfunctions, quantum_geometric_tensor_func,
    hamiltonian, delta_k, band_index,
    progress_label=None, kk=0, order="xyz", show_progress=True
):
    """
    Calculate the quantum geometric tensor (QGT) components for a ki-kj grid with a fixed kk,
    with flexible axis ordering controlled by `order`.

    Mapping rule:
      - order[0] gets ki[i,j]
      - order[1] gets kj[i,j]
      - order[2] gets kk

    So:
      order='xyz' -> (kx,ky,kz)=(ki,kj,kk)
      order='yzx' -> (kx,ky,kz)=(kk,ki,kj)
      order='xzy' -> (kx,ky,kz)=(ki,kk,kj)
      etc.
    """
    # Initialize arrays to store tensor components
    shape = ki.shape
    g_xx_array = np.zeros(shape) 
    g_yy_array = np.zeros(shape)
    g_zz_array = np.zeros(shape)

    g_xy_real_array = np.zeros(shape)
    g_xy_imag_array = np.zeros(shape)

    g_xz_real_array = np.zeros(shape)
    g_xz_imag_array = np.zeros(shape)

    g_yz_real_array = np.zeros(shape)
    g_yz_imag_array = np.zeros(shape)

    total_points = shape[0] * shape[1]
    desc = f"QGT grid [{progress_label}]" if progress_label else "Computing QGT grid"

    with tqdm(
        total=total_points,
        desc=desc,
        unit="kpt",
        leave=False,
        disable=not show_progress,
    ) as pbar:
        for i in range(shape[0]):
            for j in range(shape[1]):
                eigenfunction = eigenfunctions[i, j]
                eigenvalue = eigenvalues[i, j]

                # Map (ki,kj,kk) -> (kx,ky,kz) per order
                kx, ky, kz = map_k_by_order(ki[i, j], kj[i, j], kk, order)

                # Call the 3D QGT function:
                g_xx, g_yy, g_zz, g_xy_real, g_xy_imag, g_xz_real, g_xz_imag, g_yz_real, g_yz_imag = \
                    quantum_geometric_tensor_func(
                        hamiltonian, kx, ky, kz, delta_k, eigenvalue, eigenfunction, band_index
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

    return (g_xx_array, g_yy_array, g_zz_array,
            g_xy_real_array, g_xy_imag_array,
            g_xz_real_array, g_xz_imag_array,
            g_yz_real_array, g_yz_imag_array)

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
    ki, kj,
    quantum_geometric_tensor_func,
    hamiltonian,
    kk=0.0,
    z_cutoff=None,
    eigenvalues=None,
    band_index=None,
    order="xyz"
):
    """
    Calculate the analytical quantum geometric tensor (QGT) components on a ki-kj grid.

    Parameters:
    - ki, kj: 2D arrays defining the k-space grid (e.g. kx, ky).
    - quantum_geometric_tensor_func: Analytical function returning QGT components.
    - hamiltonian: The Hamiltonian object.
    - kk: Momentum component in the out-of-plane direction (default 0.0).
    - z_cutoff: Optional upper bound to clip the QGT components.
    - eigenvalues: 2D array of eigenvalues to be used for components.
    - order: string defining the axis mapping, e.g. 'xyz' -> (kx,ky,kz)=(ki,kj,kk).

    # Returns:
    # - 9-tuple of 2D arrays: g_xx, g_yy, g_zz, g_xy_real, g_xy_imag, g_xz_real, g_xz_imag, g_yz_real, g_yz_imag
    """
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

    with tqdm(total=total_points, desc="Computing Analytical QGT Grid", unit="point") as pbar:
        for i in range(ki.shape[0]):
            for j in range(ki.shape[1]):
                energy = eigenvalues[i, j] if eigenvalues is not None else None
                
                # Map (ki,kj,kk) -> (kx,ky,kz) per order mapping
                kx, ky, kz = map_k_by_order(ki[i, j], kj[i, j], kk, order)

                g_xx, g_yy, g_zz, g_xy_real, g_xy_imag, g_xz_real, g_xz_imag, g_yz_real, g_yz_imag = quantum_geometric_tensor_func(
                    hamiltonian, kx, ky, kz=kz, band=band_index, energy=energy
                )

                g_xx_array[i, j] = g_xx if g_xx is not None else 0.0
                g_yy_array[i, j] = g_yy if g_yy is not None else 0.0
                g_zz_array[i, j] = g_zz if g_zz is not None else 0.0
                g_xy_real_array[i, j] = g_xy_real if g_xy_real is not None else 0.0
                g_xy_imag_array[i, j] = g_xy_imag if g_xy_imag is not None else 0.0
                g_xz_real_array[i, j] = g_xz_real if g_xz_real is not None else 0.0
                g_xz_imag_array[i, j] = g_xz_imag if g_xz_imag is not None else 0.0
                g_yz_real_array[i, j] = g_yz_real if g_yz_real is not None else 0.0
                g_yz_imag_array[i, j] = g_yz_imag if g_yz_imag is not None else 0.0

                pbar.update(1)

    if z_cutoff is not None:
        g_xx_array = np.clip(g_xx_array, -z_cutoff, z_cutoff)
        g_yy_array = np.clip(g_yy_array, -z_cutoff, z_cutoff)
        g_zz_array = np.clip(g_zz_array, -z_cutoff, z_cutoff)
        g_xy_real_array = np.clip(g_xy_real_array, -z_cutoff, z_cutoff)
        g_xy_imag_array = np.clip(g_xy_imag_array, -z_cutoff, z_cutoff)
        g_xz_real_array = np.clip(g_xz_real_array, -z_cutoff, z_cutoff)
        g_xz_imag_array = np.clip(g_xz_imag_array, -z_cutoff, z_cutoff)
        g_yz_real_array = np.clip(g_yz_real_array, -z_cutoff, z_cutoff)
        g_yz_imag_array = np.clip(g_yz_imag_array, -z_cutoff, z_cutoff)

    return g_xx_array, g_yy_array, g_zz_array, g_xy_real_array, g_xy_imag_array, g_xz_real_array, g_xz_imag_array, g_yz_real_array, g_yz_imag_array


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
