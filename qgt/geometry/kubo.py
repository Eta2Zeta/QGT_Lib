"""The quantum geometric tensor from the Kubo (sum-over-states) formula.

For band ``n`` of a Hermitian ``H(k)`` with eigenstates ``|m>`` and energies ``E_m``,

    Q_ij = sum_{m != n} <n|d_i H|m> <m|d_j H|n> / (E_n - E_m)^2,

which equals ``<d_i n|(1 - |n><n|)|d_j n>``: ``g_ij = Re Q_ij`` is the quantum
metric and ``Omega_ij = -2 Im Q_ij`` the Berry curvature. For a group ``S`` of
bands the sum runs over ``n`` in ``S`` and ``m`` outside it; this is the QGT of
the projector onto ``S``, which stays finite where the bands of ``S`` touch
each other and diverges only where ``S`` touches a band outside it. At an
exact touching (energies equal to ``DEGENERATE``) the result is NaN.

Only ``H`` and its first derivatives enter, so no eigenvector phase or order
has to be fixed. ``dH/dk`` is a central difference of ``H``, which is smooth,
and a driven model uses its Floquet effective Hamiltonian.

Momenta are arrays of shape ``(..., 3)`` holding ``(kx, ky, kz)``; any leading
shape works (a point, a line, a plane, a volume, a sphere, scattered points).
"""

import numpy as np

from qgt.hamiltonians.evaluate import hamiltonian_at
from qgt.kspace.grids import map_k_by_order
from qgt.topology import berry_curvature_components_from_qgt

# Points evaluated together; bounds the memory of the (N, 3, 3, d, d) products.
CHUNK = 20000
# Energies closer than this (relative to the largest |E|, at least 1) count as degenerate.
DEGENERATE = 1e-12


def hamiltonian_gradient(model, k, delta_k=1e-5):
    """``dH/dk_i`` of shape ``(..., 3, d, d)`` by central differences with step ``delta_k``."""
    k = np.asarray(k, dtype=float)
    shifts = delta_k * np.eye(3)
    plus = hamiltonian_at(model, k[..., None, :] + shifts)
    minus = hamiltonian_at(model, k[..., None, :] - shifts)
    return (plus - minus) / (2 * delta_k)


def _match_stored_bands(U, eigenvectors):
    """For each stored eigenvector, the index of the eigenstate in ``U`` it overlaps most.

    ``U[..., :, a]`` are the eigenstates at k; ``eigenvectors[..., b, :]`` the
    stored ones, in the layout of the eigen runs.
    """
    overlap = np.abs(np.conj(eigenvectors) @ U) ** 2
    return np.argmax(overlap, axis=-1)


def _qgt_chunk(model, k, bands, delta_k, eigenvectors):
    E, U = np.linalg.eigh(hamiltonian_at(model, k))
    dH = hamiltonian_gradient(model, k, delta_k)
    Ud = np.conj(np.swapaxes(U, -1, -2))
    V = Ud[..., None, :, :] @ dH @ U[..., None, :, :]  # <n|d_i H|m>, shape (N, 3, d, d)

    d = E.shape[-1]
    if eigenvectors is not None:
        order = _match_stored_bands(U, eigenvectors)
    else:
        order = np.broadcast_to(np.arange(d), E.shape)

    gap = E[..., :, None] - E[..., None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse_gap2 = 1.0 / gap ** 2
    scale = np.maximum(1.0, np.max(np.abs(E), axis=-1))[..., None, None]
    degenerate = np.abs(gap) <= DEGENERATE * scale
    # T[..., i, j, n, m] = <n|d_i H|m> <m|d_j H|n>
    T = V[..., :, None, :, :] * np.swapaxes(V, -1, -2)[..., None, :, :, :]

    def group_qgt(group):
        inside = np.zeros(E.shape, dtype=bool)
        np.put_along_axis(inside, order[..., group], True, axis=-1)
        pairs = inside[..., :, None] & ~inside[..., None, :]
        weight = np.where(pairs & ~degenerate, inverse_gap2, 0.0)
        Q = np.einsum("...ijnm,...nm->...ij", T, weight)
        # Where the group touches a band outside it, its QGT is undefined.
        Q[np.any(pairs & degenerate, axis=(-2, -1))] = complex(np.nan, np.nan)
        return Q

    if bands is None:
        Q = np.stack([group_qgt([b]) for b in range(d)], axis=-3)
    else:
        Q = group_qgt(list(np.atleast_1d(bands)))
    if eigenvectors is not None:
        # A stored spectrum marks failed points with NaN; keep them NaN.
        failed = np.isnan(eigenvectors).any(axis=(-2, -1))
        Q[failed] = complex(np.nan, np.nan)
    return Q


def quantum_geometric_tensor(model, k, bands=None, *, delta_k=1e-5, eigenvectors=None):
    """Complex QGT ``Q_ij`` of a band or a group of bands at momenta ``k`` (shape ``(..., 3)``).

    ``bands`` is a band index, a sequence of indices treated as one group, or
    ``None`` for every band separately. The result has shape ``(..., 3, 3)``,
    or ``(..., d, 3, 3)`` for ``bands=None``.

    Bands are numbered by energy, unless ``eigenvectors`` of shape ``(..., d, d)``
    holds a stored spectrum at the same momenta (``eigenvectors[..., b, :]`` is
    band ``b``, the layout of the eigen runs): band ``b`` is then the eigenstate
    that overlaps most with the stored vector, so bands tracked through
    crossings keep their identity.
    """
    k = np.asarray(k, dtype=float)
    lead = k.shape[:-1]
    flat_k = k.reshape(-1, 3)
    flat_v = None if eigenvectors is None else np.asarray(eigenvectors).reshape(
        (-1,) + np.shape(eigenvectors)[-2:])
    parts = [
        _qgt_chunk(model, flat_k[s:s + CHUNK], bands, delta_k,
                   None if flat_v is None else flat_v[s:s + CHUNK])
        for s in range(0, len(flat_k), CHUNK)
    ]
    Q = np.concatenate(parts, axis=0) if parts else np.zeros((0,) + (3, 3), dtype=complex)
    return Q.reshape(lead + Q.shape[1:])


def berry_curvature(Q):
    """Berry curvature vector ``(Omega_x, Omega_y, Omega_z) = (Omega_yz, Omega_zx, Omega_xy)``, shape ``(..., 3)``."""
    return np.stack(berry_curvature_components_from_qgt(Q[..., 0, 1].imag, Q[..., 0, 2].imag, Q[..., 1, 2].imag), axis=-1)


def qgt_components(Q):
    """The nine arrays the workflows save, in their order.

    ``g_xx, g_yy, g_zz, g_xy_real, g_xy_imag, g_xz_real, g_xz_imag, g_yz_real, g_yz_imag``
    """
    return (Q[..., 0, 0].real, Q[..., 1, 1].real, Q[..., 2, 2].real,
            Q[..., 0, 1].real, Q[..., 0, 1].imag,
            Q[..., 0, 2].real, Q[..., 0, 2].imag,
            Q[..., 1, 2].real, Q[..., 1, 2].imag)


def plane_momenta(ki, kj, kk=0, order="xyz"):
    """Momenta of shape ``ki.shape + (3,)`` for a plane given in a coordinate order (see ``map_k_by_order``)."""
    kx, ky, kz = np.broadcast_arrays(*map_k_by_order(np.asarray(ki), np.asarray(kj), kk, order))
    return np.stack([kx, ky, kz], axis=-1).astype(float)
