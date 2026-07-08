from .eigenvalue_calc_lib import *    
from .utilities import sign_check
from .dimension_lib import map_k_by_order
from .calculus_lib import *

# Projection operator
def projection_operator(psi):
    return np.outer(psi, np.conj(psi))


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




def quantum_geometric_tensor_analytic(Hamiltonian, kx, ky, kz=0.0, band=-1, energy=None):
    """
    Returns the analytical components of the quantum geometric tensor (QGT)
    for a given Hamiltonian at momentum (kx, ky).

    Parameters:
        Hamiltonian: Hamiltonian object with optional analytical QGT methods
        kx, ky: momentum components
        band: +1 or -1 (Berry curvature sign depends on band)
        energy: band energy value E(kx,ky) required by some imaginary components.

    Returns:
        g_xx, g_yy, g_zz, g_xy_real, g_xy_imag, g_xz_real, g_xz_imag, g_yz_real, g_yz_imag, trace
    """

    # --- Standard metric components ---
    g_xx = Hamiltonian.g_xx(kx, ky, kz) if hasattr(Hamiltonian, 'g_xx') else None
    g_yy = Hamiltonian.g_yy(kx, ky, kz) if hasattr(Hamiltonian, 'g_yy') else None
    g_zz = Hamiltonian.g_zz(kx, ky, kz) if hasattr(Hamiltonian, 'g_zz') else None

    # --- Cross metric components (real parts) ---
    g_xy_real = Hamiltonian.g_xy_real(kx, ky, kz) if hasattr(Hamiltonian, 'g_xy_real') else None
    g_xz_real = Hamiltonian.g_xz_real(kx, ky, kz) if hasattr(Hamiltonian, 'g_xz_real') else None
    g_yz_real = Hamiltonian.g_yz_real(kx, ky, kz) if hasattr(Hamiltonian, 'g_yz_real') else None

    # --- Trace ---
    trace = Hamiltonian.trace(kx, ky, kz) if hasattr(Hamiltonian, 'trace') else None

    # --- Imaginary part (Berry curvature contribution) ---
    g_xy_imag = Hamiltonian.g_xy_imag(kx, ky, kz, band=band) if hasattr(Hamiltonian, 'g_xy_imag') else None
    g_xz_imag = Hamiltonian.g_xz_imag(kx, ky, kz, band=band, energy=energy) if hasattr(Hamiltonian, 'g_xz_imag') else None
    g_yz_imag = Hamiltonian.g_yz_imag(kx, ky, kz, band=band, energy=energy) if hasattr(Hamiltonian, 'g_yz_imag') else None

    return g_xx, g_yy, g_zz, g_xy_real, g_xy_imag, g_xz_real, g_xz_imag, g_yz_real, g_yz_imag
