from .eigenvalue_calc_lib import *    
from .utilities import sign_check
from .dimension_lib import map_k_by_order


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

# Numerical derivative w.r.t. kx with Phase Corrected
def dpsi_dx_num_phase_corrected(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=0):
    eigenvector_plus = Eigenvectors(len(eigenfunction))
    eigenvector_minus = Eigenvectors(len(eigenfunction))
    eigenvector_plus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky, kz=kz)
    eigenvector_minus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky, kz=kz)

    # Calculate for kx + delta_k
    eigenvalues_plus, psi_plus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx + delta_k, ky, kz=kz)
    eigenvalues_plus_ordered, psi_plus_ordered = eigenvector_plus.set_eigenvectors_phase_corrected(psi_plus, eigenvalues_plus, kx + delta_k, ky, kz=kz)

    # Calculate for kx - delta_k
    eigenvalues_minus, psi_minus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx - delta_k, ky, kz=kz)
    eigenvalues_minus_ordered, psi_minus_ordered = eigenvector_minus.set_eigenvectors_phase_corrected(psi_minus, eigenvalues_minus, kx - delta_k, ky, kz=kz)

    # Return the derivative for the specified band
    return (psi_plus_ordered[band_index] - psi_minus_ordered[band_index]) / (2 * delta_k)

def dpsi_dy_num_phase_corrected(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=0):
    eigenvector_plus = Eigenvectors(len(eigenfunction))
    eigenvector_minus = Eigenvectors(len(eigenfunction))
    eigenvector_plus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky, kz=kz)
    eigenvector_minus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky, kz=kz)

    # Calculate for ky + delta_k
    eigenvalues_plus, psi_plus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky + delta_k, kz=kz)
    eigenvalues_plus_ordered, psi_plus_ordered = eigenvector_plus.set_eigenvectors_phase_corrected(psi_plus, eigenvalues_plus, kx, ky + delta_k, kz=kz)

    # Calculate for ky - delta_k
    eigenvalues_minus, psi_minus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky - delta_k, kz=kz)
    eigenvalues_minus_ordered, psi_minus_ordered = eigenvector_minus.set_eigenvectors_phase_corrected(psi_minus, eigenvalues_minus, kx, ky - delta_k, kz=kz)

    # Return the derivative for the specified band
    return (psi_plus_ordered[band_index] - psi_minus_ordered[band_index]) / (2 * delta_k)

def dpsi_dz_num_phase_corrected(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=0):
    eigenvector_plus = Eigenvectors(len(eigenfunction))
    eigenvector_minus = Eigenvectors(len(eigenfunction))
    eigenvector_plus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky, kz=kz)
    eigenvector_minus.set_eigenvectors_eigenvalue_preordered(eigenfunction, eigenvalue, kx, ky, kz=kz)

    # Calculate for kz + delta_k
    eigenvalues_plus, psi_plus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky, kz=kz + delta_k)
    eigenvalues_plus_ordered, psi_plus_ordered = eigenvector_plus.set_eigenvectors_phase_corrected(psi_plus, eigenvalues_plus, kx, ky, kz=kz + delta_k)

    # Calculate for kz - delta_k
    eigenvalues_minus, psi_minus = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx, ky, kz=kz - delta_k)
    eigenvalues_minus_ordered, psi_minus_ordered = eigenvector_minus.set_eigenvectors_phase_corrected(psi_minus, eigenvalues_minus, kx, ky, kz=kz - delta_k)

    # Return the derivative for the specified band
    return (psi_plus_ordered[band_index] - psi_minus_ordered[band_index]) / (2 * delta_k)