import os
import pickle
import numpy as np


def overlay_hamiltonian_symmetry_path(
    axes,
    kx,
    ky,
    hamiltonian,
    *,
    kk=0.0,
    sym_kz_threshold=0.02,
    color="white",
    line_width=1.5,
    point_size=30,
    label_fontsize=10,
    show_legend=False,
):
    """Overlay ``hamiltonian.get_sym_path()`` on one or more kx-ky axes.

    Two-dimensional symmetry points are drawn directly. For three-dimensional
    paths, only nodes and complete path segments within the plotted ``kz=kk``
    slice are drawn.

    Returns
    -------
    bool
        Whether any symmetry path segment or point was drawn.
    """
    if hamiltonian is None:
        return False
    if not 0 <= sym_kz_threshold <= 1:
        raise ValueError("sym_kz_threshold must be between 0 and 1.")

    get_sym_path = getattr(hamiltonian, "get_sym_path", None)
    if get_sym_path is None:
        print(
            f"{hamiltonian.__class__.__name__} does not define get_sym_path(); "
            "skipping the symmetry overlay."
        )
        return False

    try:
        sym_points, path_names = get_sym_path()
    except NotImplementedError:
        print(
            f"{hamiltonian.__class__.__name__} does not define a symmetry path; "
            "skipping the symmetry overlay."
        )
        return False

    path_names = list(path_names)
    if not path_names:
        return False

    if hasattr(axes, "plot"):
        plot_axes = (axes,)
    else:
        plot_axes = tuple(np.asarray(axes, dtype=object).reshape(-1))
    if not plot_axes:
        raise ValueError("axes must contain at least one Matplotlib axis.")

    kx = np.asarray(kx)
    ky = np.asarray(ky)
    kx_range = float(np.nanmax(kx) - np.nanmin(kx))
    ky_range = float(np.nanmax(ky) - np.nanmin(ky))
    kz_tolerance = sym_kz_threshold * max(kx_range, ky_range)
    fixed_kz = 0.0 if kk is None else float(kk)

    visible_points = {}
    for name in dict.fromkeys(path_names):
        if name not in sym_points:
            raise KeyError(f"Symmetry path references undefined point {name!r}.")

        point = np.asarray(sym_points[name], dtype=float).reshape(-1)
        if point.size not in (2, 3):
            raise ValueError(
                f"Symmetry point {name!r} must contain two or three coordinates."
            )
        if point.size == 3 and abs(point[2] - fixed_kz) > kz_tolerance:
            visible_points[name] = None
        else:
            visible_points[name] = point[:2]

    segments = []
    for start_name, end_name in zip(path_names[:-1], path_names[1:]):
        start_xy = visible_points[start_name]
        end_xy = visible_points[end_name]
        if start_xy is not None and end_xy is not None:
            segments.append((start_xy, end_xy))

    labeled_points = [
        (name, visible_points[name])
        for name in dict.fromkeys(path_names)
        if visible_points[name] is not None
    ]
    if not segments and not labeled_points:
        return False

    for ax in plot_axes:
        for segment_index, (start_xy, end_xy) in enumerate(segments):
            ax.plot(
                [start_xy[0], end_xy[0]],
                [start_xy[1], end_xy[1]],
                color=color,
                linewidth=line_width,
                linestyle="--",
                alpha=0.95,
                label=(
                    "Symmetry Path"
                    if show_legend and segment_index == 0
                    else None
                ),
            )

        for name, xy in labeled_points:
            ax.scatter(
                xy[0],
                xy[1],
                color=color,
                edgecolor="black",
                linewidth=0.4,
                s=point_size,
                zorder=5,
            )
            display_name = r"$\Gamma$" if name in {"G", "Gamma", "Γ"} else name
            ax.annotate(
                display_name,
                (xy[0], xy[1]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                color=color,
                fontsize=label_fontsize,
                fontweight="bold",
            )

        if show_legend and segments:
            ax.legend(loc="upper right")

    return True


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
