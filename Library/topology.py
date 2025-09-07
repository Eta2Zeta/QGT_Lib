import numpy as np
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


