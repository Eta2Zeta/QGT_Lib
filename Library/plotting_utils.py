import os
import pickle
import numpy as np

from Library.dimension_lib import cylindrical_order_axes, is_cylindrical_order


def _fixed_axis_from_order(order):
    if is_cylindrical_order(order):
        return cylindrical_order_axes(order)[3]
    return order[2]


def symmetry_points_on_fixed_slice(
    data,
    order,
    plane_axes,
    *,
    tolerance_fraction=0.01,
):
    """Return saved symmetry nodes lying on a fixed-coordinate slice."""
    tolerance_fraction = float(tolerance_fraction)
    if not 0.0 <= tolerance_fraction <= 1.0:
        raise ValueError("symmetry_slice_tolerance must be between 0 and 1")
    if "path_points" not in data.files or "path_labels" not in data.files:
        return [], np.empty((0, 2), dtype=float), 0.0

    path_points = np.asarray(data["path_points"], dtype=float)
    if path_points.ndim != 2 or path_points.shape[1] not in (2, 3):
        raise ValueError("path_points must have shape (number_of_nodes, 2 or 3)")
    if path_points.shape[1] == 2:
        path_points = np.column_stack(
            (path_points, np.zeros(path_points.shape[0], dtype=float))
        )

    path_labels = [str(label) for label in np.asarray(data["path_labels"])]
    if len(path_labels) != path_points.shape[0]:
        raise ValueError("path_labels and path_points must have the same length")

    fixed_axis = _fixed_axis_from_order(order)
    fixed_axis_index = "xyz".index(fixed_axis)
    plane_indices = ["xyz".index(axis) for axis in plane_axes]
    fixed_coordinate = (
        float(np.asarray(data["kk"]).item()) if "kk" in data.files else 0.0
    )

    fixed_values = path_points[:, fixed_axis_index]
    reciprocal_boundary = float(np.max(np.abs(fixed_values)))
    reciprocal_period = 2.0 * reciprocal_boundary
    if reciprocal_period > 0.0:
        tolerance = tolerance_fraction * reciprocal_period
        separation = np.mod(
            np.abs(fixed_values - fixed_coordinate),
            reciprocal_period,
        )
        separation = np.minimum(separation, reciprocal_period - separation)
    else:
        tolerance = max(1e-12, tolerance_fraction * 2.0 * np.pi)
        separation = np.abs(fixed_values - fixed_coordinate)

    visible_labels = []
    visible_points = []
    seen_labels = set()
    for label, point, distance in zip(path_labels, path_points, separation):
        if distance > tolerance or label in seen_labels:
            continue
        seen_labels.add(label)
        visible_labels.append(label)
        visible_points.append(point[plane_indices])

    if not visible_points:
        return [], np.empty((0, 2), dtype=float), tolerance
    return visible_labels, np.asarray(visible_points, dtype=float), tolerance


def draw_symmetry_point_overlay(
    axes,
    labels,
    points,
    *,
    marker_color="#ffd84d",
):
    """Draw labeled symmetry nodes without changing the map limits."""
    if not labels:
        return [], []

    plot_axes = list(np.asarray(axes, dtype=object).reshape(-1))
    scatter_artists = []
    annotation_artists = []
    for axis in plot_axes:
        original_xlim = axis.get_xlim()
        original_ylim = axis.get_ylim()
        xmin, xmax = sorted(original_xlim)
        ymin, ymax = sorted(original_ylim)
        visible = (
            (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
        )
        visible_points = points[visible]
        visible_labels = [label for label, keep in zip(labels, visible) if keep]
        if not visible_labels:
            continue

        scatter = axis.scatter(
            visible_points[:, 0],
            visible_points[:, 1],
            s=52,
            facecolor=marker_color,
            edgecolor="#161616",
            linewidth=0.8,
            zorder=8,
        )
        scatter_artists.append(scatter)
        for label, point in zip(visible_labels, visible_points):
            display_label = r"$\Gamma$" if label in {"G", "Gamma", "Γ"} else label
            annotation_artists.append(
                axis.annotate(
                    display_label,
                    (point[0], point[1]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    color="#111111",
                    fontsize=11,
                    fontweight="bold",
                    zorder=9,
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.72,
                    },
                )
            )
        axis.set_xlim(original_xlim)
        axis.set_ylim(original_ylim)

    return scatter_artists, annotation_artists


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
