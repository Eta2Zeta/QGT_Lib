"""Coordinate-order validation, grid construction, and Cartesian conversion."""

from itertools import permutations

import numpy as np


CARTESIAN_ORDERS = frozenset("".join(order) for order in permutations("xyz"))
CYLINDRICAL_ORDERS = frozenset({"xpz", "ypz", "xpy", "zpy", "ypx", "zpx"})
SUPPORTED_ORDERS = CARTESIAN_ORDERS | CYLINDRICAL_ORDERS

_BASIS = {
    "x": np.array([1, 0, 0], dtype=int),
    "y": np.array([0, 1, 0], dtype=int),
    "z": np.array([0, 0, 1], dtype=int),
}


def normalize_coordinate_order(order):
    """Return a validated lowercase Cartesian or cylindrical order."""
    if not isinstance(order, str):
        raise TypeError("order must be a string such as 'xyz' or 'xpz'")

    normalized = order.lower()
    if normalized not in SUPPORTED_ORDERS:
        allowed = ", ".join(sorted(SUPPORTED_ORDERS))
        raise ValueError(
            f"Unsupported coordinate order {order!r}. Supported orders: {allowed}."
        )
    return normalized


def is_cylindrical_order(order):
    """Return whether ``order`` describes a radial-angular-fixed-axis grid."""
    return normalize_coordinate_order(order) in CYLINDRICAL_ORDERS


def coordinate_order_info(order):
    """Describe the input coordinates and physical convention for ``order``."""
    order = normalize_coordinate_order(order)
    if order in CARTESIAN_ORDERS:
        return {
            "coordinate_system": "cartesian",
            "coordinate_labels": [f"k{order[0]}", f"k{order[1]}", f"k{order[2]}"],
            "radial_reference_axis": None,
            "rotation_axis": None,
            "angular_orientation": None,
            "phi_periodic": False,
        }

    reference_axis = order[0]
    fixed_axis = order[2]
    return {
        "coordinate_system": "cylindrical",
        "coordinate_labels": ["r", "phi", f"k{fixed_axis}"],
        "radial_reference_axis": reference_axis,
        "rotation_axis": fixed_axis,
        "angular_orientation": "right_handed",
        "phi_periodic": True,
    }


def map_k_by_order(ki_ij, kj_ij, kk, order):
    """Map ordered input coordinates to physical Cartesian ``(kx, ky, kz)``.

    Cartesian orders are permutations of ``xyz``. The first order character
    receives ``ki_ij``, the second receives ``kj_ij``, and the third receives
    the fixed value ``kk``.

    Cylindrical orders have the form ``apb``. ``ki_ij`` is the radius,
    ``kj_ij`` is phi in radians, ``a`` is the positive radial direction at
    phi=0, and ``b`` is the fixed axis receiving ``kk``. Positive phi follows
    the right-hand rule about the positive fixed axis.

    Examples
    --------
    ``xyz``: ``(kx, ky, kz) = (ki, kj, kk)``

    ``xpz``: ``(kx, ky, kz) = (r*cos(phi), r*sin(phi), kk)``

    ``ypz``: ``(kx, ky, kz) = (-r*sin(phi), r*cos(phi), kk)``
    """
    order = normalize_coordinate_order(order)

    if order in CARTESIAN_ORDERS:
        slots = {order[0]: ki_ij, order[1]: kj_ij, order[2]: kk}
        return slots["x"], slots["y"], slots["z"]

    radius = np.asarray(ki_ij)
    phi = np.asarray(kj_ij)
    reference_axis = order[0]
    fixed_axis = order[2]

    tangent = np.cross(_BASIS[fixed_axis], _BASIS[reference_axis])
    tangent_index = int(np.flatnonzero(tangent)[0])
    tangent_axis = "xyz"[tangent_index]
    tangent_sign = int(tangent[tangent_index])

    zero = np.zeros(np.broadcast_shapes(radius.shape, phi.shape), dtype=float)
    slots = {"x": zero.copy(), "y": zero.copy(), "z": zero.copy()}
    slots[reference_axis] = radius * np.cos(phi)
    slots[tangent_axis] = tangent_sign * radius * np.sin(phi)
    slots[fixed_axis] = zero + kk
    return slots["x"], slots["y"], slots["z"]


def create_2d_coordinate_grid(
    k_max,
    mesh_spacing,
    order="xyz",
    include_endpoints=True,
):
    """Create the two varying coordinate arrays for a 2D calculation.

    Cartesian inputs span ``[-k_max, k_max]``. Cylindrical inputs use a radius
    spanning ``[0, k_max]`` and a periodic angular grid spanning ``[0, 2*pi)``.
    The angular endpoint is deliberately omitted because phi=0 and phi=2*pi
    are the same physical line.

    Returns
    -------
    ki, kj : ndarray
        Meshgrid arrays consumed by :func:`map_k_by_order`.
    grid_info : dict
        JSON-safe coordinate ranges, spacings, labels, and conventions.
    """
    order = normalize_coordinate_order(order)
    k_max = float(k_max)
    mesh_spacing = int(mesh_spacing)

    if not np.isfinite(k_max) or k_max <= 0:
        raise ValueError("k_max must be a finite positive number")
    if mesh_spacing < 2:
        raise ValueError("mesh_spacing must be at least 2")

    info = coordinate_order_info(order)
    if order in CARTESIAN_ORDERS:
        if include_endpoints:
            ki_vals = np.linspace(-k_max, k_max, mesh_spacing)
            kj_vals = np.linspace(-k_max, k_max, mesh_spacing)
            sampling = "endpoints"
        else:
            ki_vals = np.linspace(-k_max, k_max, mesh_spacing + 2)[1:-1]
            kj_vals = np.linspace(-k_max, k_max, mesh_spacing + 2)[1:-1]
            sampling = "interior"
    else:
        if include_endpoints:
            radial_vals = np.linspace(0.0, k_max, mesh_spacing)
            phi_vals = np.linspace(0.0, 2.0 * np.pi, mesh_spacing, endpoint=False)
            sampling = "radial_endpoints_periodic_phi"
        else:
            radial_step = k_max / mesh_spacing
            phi_step = 2.0 * np.pi / mesh_spacing
            radial_vals = (np.arange(mesh_spacing) + 0.5) * radial_step
            phi_vals = (np.arange(mesh_spacing) + 0.5) * phi_step
            sampling = "cell_centers_periodic_phi"
        ki_vals, kj_vals = radial_vals, phi_vals

    ki, kj = np.meshgrid(ki_vals, kj_vals)
    grid_info = {
        **info,
        "order": order,
        "sampling": sampling,
        "ki_range": [float(ki_vals[0]), float(ki_vals[-1])],
        "kj_range": [float(kj_vals[0]), float(kj_vals[-1])],
        "dki": float(ki_vals[1] - ki_vals[0]),
        "dkj": float(kj_vals[1] - kj_vals[0]),
        "phi_domain": [0.0, float(2.0 * np.pi)] if info["phi_periodic"] else None,
        "phi_endpoint_included": False if info["phi_periodic"] else None,
    }
    return ki, kj, grid_info
