import numpy as np

from .dimension_lib import (
    CARTESIAN_ORDERS,
    cylindrical_order_axes,
    normalize_coordinate_order,
)


def berry_curvature_components_from_qgt(
    g_xy_imag,
    g_xz_imag,
    g_yz_imag,
):
    """Return Cartesian Berry-curvature components from imaginary QGT parts.

    The library convention is ``Omega_ij = -2 Im(Q_ij)``. Therefore the
    pseudovector components are ``Omega_x = Omega_yz``,
    ``Omega_y = Omega_zx = -Omega_xz``, and ``Omega_z = Omega_xy``.
    """
    g_xy_imag = np.asarray(g_xy_imag, dtype=float)
    g_xz_imag = np.asarray(g_xz_imag, dtype=float)
    g_yz_imag = np.asarray(g_yz_imag, dtype=float)
    if not (
        g_xy_imag.shape == g_xz_imag.shape == g_yz_imag.shape
    ):
        raise ValueError("All imaginary QGT component arrays must have the same shape")

    omega_x = -2.0 * g_yz_imag
    omega_y = 2.0 * g_xz_imag
    omega_z = -2.0 * g_xy_imag
    return omega_x, omega_y, omega_z


def integrate_berry_flux_2d(
    omega_x,
    omega_y,
    omega_z,
    ki_grid,
    kj_grid,
    *,
    order="xyz",
    phi_periodic=None,
):
    """Integrate Berry flux on a Cartesian or cylindrical coordinate grid."""
    order = normalize_coordinate_order(order)
    ki_grid = np.asarray(ki_grid, dtype=float)
    kj_grid = np.asarray(kj_grid, dtype=float)
    omega_by_axis = {
        "x": np.asarray(omega_x, dtype=float),
        "y": np.asarray(omega_y, dtype=float),
        "z": np.asarray(omega_z, dtype=float),
    }
    if ki_grid.ndim != 2 or kj_grid.ndim != 2 or ki_grid.shape != kj_grid.shape:
        raise ValueError("ki_grid and kj_grid must be matching two-dimensional arrays")
    if any(component.shape != ki_grid.shape for component in omega_by_axis.values()):
        raise ValueError("Berry components and coordinate grids must have matching shapes")

    ki_values = ki_grid[0, :]
    kj_values = kj_grid[:, 0]
    if not np.allclose(ki_grid, ki_values[np.newaxis, :]):
        raise ValueError("ki_grid must vary only along axis 1")
    if not np.allclose(kj_grid, kj_values[:, np.newaxis]):
        raise ValueError("kj_grid must vary only along axis 0")

    if order in CARTESIAN_ORDERS:
        basis = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        normal = np.cross(basis[order[0]], basis[order[1]])
        normal_position = int(np.flatnonzero(normal)[0])
        normal_axis = "xyz"[normal_position]
        normal_sign = float(normal[normal_position])
        integrand = normal_sign * omega_by_axis[normal_axis]
        phi_periodic = False
    else:
        _, _, _, fixed_axis = cylindrical_order_axes(order)
        integrand = ki_grid * omega_by_axis[fixed_axis]
        if phi_periodic is None:
            phi_periodic = True

    trapezoid = getattr(np, "trapezoid", np.trapz)
    integral_along_ki = trapezoid(integrand, x=ki_values, axis=1)
    if phi_periodic:
        kj_integral_values = np.concatenate(
            (kj_values, [kj_values[0] + 2.0 * np.pi])
        )
        integral_along_ki = np.concatenate(
            (integral_along_ki, [integral_along_ki[0]])
        )
    else:
        kj_integral_values = kj_values
    return float(trapezoid(integral_along_ki, x=kj_integral_values, axis=0))


def winding_number_on_closed_curve(
    field_real,
    field_imag,
    *,
    relative_zero_tolerance=1e-10,
):
    """Calculate the winding of a two-component field on a closed curve.

    ``field_real + 1j*field_imag`` is sampled in traversal order without a
    duplicated endpoint. ``nan`` is returned when the sampled field vanishes.
    """
    field_real = np.asarray(field_real, dtype=float)
    field_imag = np.asarray(field_imag, dtype=float)
    if field_real.ndim != 1 or field_imag.ndim != 1:
        raise ValueError("Closed-curve field components must be one-dimensional")
    if field_real.shape != field_imag.shape:
        raise ValueError("Closed-curve field components must have matching shapes")
    if field_real.size < 3:
        raise ValueError("At least three points are required on a closed curve")

    field = field_real + 1j * field_imag
    if not np.all(np.isfinite(field)):
        return np.nan

    magnitude = np.abs(field)
    scale = float(np.max(magnitude))
    if scale == 0.0 or np.min(magnitude) <= relative_zero_tolerance * scale:
        return np.nan

    phase_steps = np.angle(np.roll(field, -1) * np.conjugate(field))
    return float(np.sum(phase_steps) / (2.0 * np.pi))


def winding_numbers_vs_radius(
    radius_grid,
    phi_grid,
    field_real,
    field_imag,
    *,
    relative_zero_tolerance=1e-10,
):
    """Calculate a closed-curve winding number for every polar-grid radius.

    The expected grid convention is the one produced by
    ``create_2d_coordinate_grid``: phi varies along axis 0 and radius along
    axis 1. The radius-zero entry is undefined and is returned as ``nan``.
    """
    radius_grid = np.asarray(radius_grid, dtype=float)
    phi_grid = np.asarray(phi_grid, dtype=float)
    field_real = np.asarray(field_real, dtype=float)
    field_imag = np.asarray(field_imag, dtype=float)

    if radius_grid.ndim != 2 or phi_grid.ndim != 2:
        raise ValueError("radius_grid and phi_grid must be two-dimensional")
    if not (
        radius_grid.shape
        == phi_grid.shape
        == field_real.shape
        == field_imag.shape
    ):
        raise ValueError("Polar grids and field arrays must have matching shapes")

    radius_values = radius_grid[0, :]
    phi_values = phi_grid[:, 0]
    if not np.allclose(radius_grid, radius_values[np.newaxis, :]):
        raise ValueError("radius_grid must vary only along axis 1")
    if not np.allclose(phi_grid, phi_values[:, np.newaxis]):
        raise ValueError("phi_grid must vary only along axis 0")

    winding = np.full(radius_values.shape, np.nan, dtype=float)
    for radius_index, radius in enumerate(radius_values):
        if radius <= 0.0:
            continue
        winding[radius_index] = winding_number_on_closed_curve(
            field_real[:, radius_index],
            field_imag[:, radius_index],
            relative_zero_tolerance=relative_zero_tolerance,
        )

    return radius_values.copy(), winding


def first_bz_hex_mask(kx_grid, ky_grid, b1, b2, center=(0.0, 0.0)):
    """
    Boolean mask selecting points inside the 1st BZ hexagon centered at 'center'.
    Uses |k·G| <= |G|^2/2 for G in {b1, b2, b1+b2}.
    
    kx_grid, ky_grid: 2D arrays defining your k-grid (same shape as g_xy_imag_array).
    b1, b2: reciprocal primitive vectors (shape (2,)).
    center: (kx0, ky0) — center of the hexagon; default is Γ at (0,0).
    """
    kx0, ky0 = center
    # shift so hexagon is centered at 'center'
    kx = kx_grid - kx0
    ky = ky_grid - ky0

    # stack k points
    K = np.stack([kx, ky], axis=-1)  # shape (Nx, Ny, 2)

    Gs = np.array([b1, b2, b1 + b2])  # shape (3, 2)
    # Dot products K·G for each G (broadcast over grid)
    KP = np.tensordot(K, Gs.T, axes=([2], [0]))  # shape (Nx, Ny, 3)

    # thresholds |K·G| <= |G|^2/2 for each G
    Gnorm2 = np.sum(Gs**2, axis=1)  # (3,)
    thresh = 0.5 * Gnorm2           # (3,)

    # need both ±G: equivalent to |K·G| <= |G|^2/2
    inside_each = np.abs(KP) <= thresh  # (Nx, Ny, 3)
    mask = np.all(inside_each, axis=2)  # (Nx, Ny)

    return mask

def compute_chern_number(g_xy_imag_array, delta_kx, delta_ky, kx_grid, ky_grid, b1, b2, center=(0.0, 0.0)):
    """
    Compute the Chern number by integrating the imaginary part of g_xy over the Brillouin zone.

    Parameters:
    - g_xy_imag_array: 2D array of Im(g_xy) values on the kx-ky grid.
    - delta_kx: Grid spacing in the kx direction.
    - delta_ky: Grid spacing in the ky direction.

    Returns:
    - Chern number (float)
    """
    mask = first_bz_hex_mask(kx_grid, ky_grid, b1, b2, center=center)
    Berry_Curvature = -2*g_xy_imag_array
    Berry_Curvature_masked = np.where(mask, Berry_Curvature, 0.0)

    # Use the trapezoidal rule along both axes
    integral = np.trapz(
        np.trapz(Berry_Curvature_masked, dx=delta_ky, axis=1),
        dx=delta_kx, axis=0
    )
    # Normalize by 2π
    chern_number = integral / (2 * np.pi)
    return chern_number
