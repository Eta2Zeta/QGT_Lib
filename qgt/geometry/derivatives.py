from qgt.spectra.eigenvectors import Eigenvectors
from qgt.spectra.diagonalization import eigenvalues_and_vectors_eigenvalue_ordering


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _shifted(axis, kx, ky, kz, step):
    k = [kx, ky, kz]
    k[_AXIS_INDEX[axis]] += step
    return k


def dpsi_from_function(axis, psi, kx, ky, delta_k):
    """Central difference along kx or ky of an eigenvector function ``psi(kx, ky, prev_psi=None)``."""
    kx_plus, ky_plus, _ = _shifted(axis, kx, ky, 0, delta_k)
    kx_minus, ky_minus, _ = _shifted(axis, kx, ky, 0, -delta_k)
    psi_front = psi(kx_plus, ky_plus)
    psi_back = psi(kx_minus, ky_minus, prev_psi=psi_front)
    return (psi_front - psi_back) / (2 * delta_k)


def _dpsi_central(axis, Hamiltonian, kx, ky, kz, delta_k, eigenvalue, eigenfunction, band_index,
                  set_reference, align):
    """Central difference of eigenvector ``band_index`` along ``axis``.

    The eigenvectors at k +- delta_k are aligned to those at k with
    ``align(tracker, vectors, values, kx, ky, kz)``.
    """
    aligned = []
    for step in (delta_k, -delta_k):
        tracker = Eigenvectors(len(eigenfunction))
        set_reference(tracker, eigenfunction, eigenvalue, kx, ky, kz)
        kx_s, ky_s, kz_s = _shifted(axis, kx, ky, kz, step)
        values, vectors = eigenvalues_and_vectors_eigenvalue_ordering(Hamiltonian, kx_s, ky_s, kz=kz_s)
        aligned.append(align(tracker, vectors, values, kx_s, ky_s, kz_s))
    return (aligned[0][band_index] - aligned[1][band_index]) / (2 * delta_k)


def _eigenvalue_preordered(tracker, vectors, values, kx, ky, kz):
    return tracker.set_eigenvectors_eigenvalue_preordered(vectors, values, kx, ky, kz=kz)


def _eigenvector_ordered(tracker, vectors, values, kx, ky, kz):
    return tracker.set_eigenvectors_eigenvector_ordered(vectors, values, kx, ky, kz=kz)[1]


def _phase_corrected(tracker, vectors, values, kx, ky, kz):
    return tracker.set_eigenvectors_phase_corrected(vectors, values, kx, ky, kz=kz)[1]


def dpsi_num(axis, Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=0):
    """Numerical d psi / d k_axis, with eigenvectors matched by eigenvalue order."""
    return _dpsi_central(axis, Hamiltonian, kx, ky, kz, delta_k, eigenvalue, eigenfunction, band_index,
                         _eigenvalue_preordered, _eigenvalue_preordered)


def dpsi_num_eigenvector_ordered(axis, Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=0):
    """Numerical d psi / d k_axis, with eigenvectors matched by overlap with those at k."""
    return _dpsi_central(axis, Hamiltonian, kx, ky, kz, delta_k, eigenvalue, eigenfunction, band_index,
                         _eigenvector_ordered, _eigenvector_ordered)


def dpsi_num_phase_corrected(axis, Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, band_index, kz=0):
    """Numerical d psi / d k_axis, with eigenvalue order and a phase fixed against those at k."""
    return _dpsi_central(axis, Hamiltonian, kx, ky, kz, delta_k, eigenvalue, eigenfunction, band_index,
                         _eigenvalue_preordered, _phase_corrected)
