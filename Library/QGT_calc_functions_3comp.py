from .eigenvalue_calc_lib import *    
from .utilities import sign_check
from .dimension_lib import map_k_by_order
from .calculus_lib import *

# Projection operator
def projection_operator(psi):
    return np.outer(psi, np.conj(psi))


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

def quantum_geometric_tensor_3d_num_phase_corrected(Hamiltonian, kx, ky, kz, delta_k, eigenvalue, eigenfunction, band_index):
    dpsi_dx_val = dpsi_dx_num_phase_corrected(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=kz)
    dpsi_dy_val = dpsi_dy_num_phase_corrected(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=kz)
    dpsi_dz_val = dpsi_dz_num_phase_corrected(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=kz)
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

def quantum_geometric_tensor_3d_analytic(Hamiltonian, kx, ky, kz, *, trace_from_sum=True, **kwargs):
    """
    Analytic 3D QGT accessor with passthrough kwargs.

    Change vs your version:
      - If a component method does NOT exist (or errors), returns None for that component
        instead of raising.
      - trace:
          * if Hamiltonian.trace exists and works -> use it
          * else if trace_from_sum and all diagonals exist -> sum them
          * else -> None
    """

    def _call(name):
        fn = getattr(Hamiltonian, name, None)
        if not callable(fn):
            return None
        try:
            return fn(kx, ky, kz, **kwargs)
        except TypeError:
            # method might not accept kwargs
            try:
                return fn(kx, ky, kz)
            except Exception:
                return None
        except Exception:
            return None

    g_xx = _call("g_xx")
    g_yy = _call("g_yy")
    g_zz = _call("g_zz")

    g_xy_real = _call("g_xy_real")
    g_xy_imag = _call("g_xy_imag")

    g_xz_real = _call("g_xz_real")
    g_xz_imag = _call("g_xz_imag")

    g_yz_real = _call("g_yz_real")
    g_yz_imag = _call("g_yz_imag")

    trace = _call("trace")
    if trace is None and trace_from_sum:
        if (g_xx is not None) and (g_yy is not None) and (g_zz is not None):
            trace = g_xx + g_yy + g_zz

    return (
        g_xx, g_yy, g_zz,
        g_xy_real, g_xy_imag,
        g_xz_real, g_xz_imag,
        g_yz_real, g_yz_imag,
        trace
    )