import numpy as np
def compute_chern_number(g_xy_imag_array, delta_kx, delta_ky):
    """
    Compute the Chern number by integrating the imaginary part of g_xy over the Brillouin zone.

    Parameters:
    - g_xy_imag_array: 2D array of Im(g_xy) values on the kx-ky grid.
    - delta_kx: Grid spacing in the kx direction.
    - delta_ky: Grid spacing in the ky direction.

    Returns:
    - Chern number (float)
    """
    Berry_Curvature = -2*g_xy_imag_array
    # Use the trapezoidal rule along both axes
    integral = np.trapz(np.trapz(Berry_Curvature, dx=delta_ky, axis=1), dx=delta_kx, axis=0)
    # Normalize by 2π
    chern_number = integral / (2 * np.pi)
    return chern_number


