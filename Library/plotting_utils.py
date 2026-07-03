import os
import pickle
import numpy as np

def load_qgt(folder_name):
    """Load QGT entries (np object array) and meta dict from a sweep folder."""
    base = os.path.join(os.getcwd(), "results", "2D_QGT_omega_sweep", folder_name)
    qgt_path  = os.path.join(base, "QGT_2D.npy")
    meta_path = os.path.join(base, "meta_info.pkl")
    if not os.path.exists(qgt_path):
        raise FileNotFoundError(f"QGT data not found in '{base}'.")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    entries = np.load(qgt_path, allow_pickle=True)
    return entries, meta

def filter_entries_by_omega(entries, omega_min=None, omega_max=None):
    """Return a list of entries whose float(entry['omega']) lies in [omega_min, omega_max]."""
    def _in_range(w):
        if (omega_min is not None) and (w < omega_min): return False
        if (omega_max is not None) and (w > omega_max): return False
        return True
    filtered = [e for e in entries if _in_range(float(e["omega"]))]
    if len(filtered) == 0:
        raise ValueError("No omega slices fall within the requested range.")
    return filtered


def get_bvecs_from_meta(meta):
    """
    Extract reciprocal vectors b1/b2 from an ND QGT meta dict.

    Falls back to the graphene-style convention used by several legacy
    Hamiltonians when the saved template does not expose b1/b2.
    """
    htemp = meta.get("Hamiltonian_Template") if meta else None
    if htemp is not None and hasattr(htemp, "b1") and hasattr(htemp, "b2"):
        return np.asarray(htemp.b1, dtype=float), np.asarray(htemp.b2, dtype=float)

    a = getattr(htemp, "a", 1.0) if htemp is not None else 1.0
    b1 = (2 * np.pi / (3 * a)) * np.array([1.0, np.sqrt(3.0)], dtype=float)
    b2 = (2 * np.pi / (3 * a)) * np.array([1.0, -np.sqrt(3.0)], dtype=float)
    return b1, b2


def bz_wigner_seitz_from_bvecs(b1, b2, search_radius=2, tol=1e-9):
    """
    Return the closed 2D first Brillouin-zone polygon from reciprocal vectors.

    The first BZ is the Wigner-Seitz cell of the reciprocal lattice around
    Gamma, so this does not assume the polygon shape in advance.
    """
    b1 = np.asarray(b1, dtype=float)
    b2 = np.asarray(b2, dtype=float)

    Gs = []
    for m in range(-search_radius, search_radius + 1):
        for n in range(-search_radius, search_radius + 1):
            if m == 0 and n == 0:
                continue
            Gs.append(m * b1 + n * b2)
    Gs = np.asarray(Gs, dtype=float)
    Gnorm2 = np.sum(Gs * Gs, axis=1)

    vertices = []
    for i, G1 in enumerate(Gs):
        for G2 in Gs[i + 1:]:
            A = np.stack([G1, G2], axis=0)
            if abs(np.linalg.det(A)) < tol:
                continue

            rhs = 0.5 * np.array([np.dot(G1, G1), np.dot(G2, G2)])
            k = np.linalg.solve(A, rhs)

            if np.all(Gs @ k <= 0.5 * Gnorm2 + tol):
                vertices.append(k)

    if not vertices:
        raise ValueError("Could not construct Brillouin-zone polygon from b1/b2.")

    unique = []
    for v in vertices:
        if not any(np.linalg.norm(v - u) < 1e-7 for u in unique):
            unique.append(v)
    verts = np.asarray(unique)

    order = np.argsort(np.arctan2(verts[:, 1], verts[:, 0]))
    verts = verts[order]
    verts = np.vstack([verts, verts[0]])
    return verts[:, 0], verts[:, 1]
