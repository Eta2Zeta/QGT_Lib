from tqdm import tqdm  # Import tqdm for progress bar

def range_of_omega(spacing='log', omega_min=5e0, omega_max=1e2, num_points=10):
    """
    Calculate QGT for a range of omega values and save the results to a file.
    The output file name is dynamically set based on the spacing type.

    Parameters:
    - spacing: Type of spacing for omega values ('log' or 'linear').
    - omega: Minimum value of omega.
    - omega: Maximum value of omega.
    - num_points: Number of omega values.

    Returns:
    - None: Saves the results to a file.
    """
    # Give some amplitude to the light
    Hamiltonian_Obj.A0 = 100

    # Generate omega values based on the specified spacing
    if spacing == 'log':
        omega_values = np.logspace(np.log10(omega_max), np.log10(omega_min), num_points)
        file_name = "g_results_log.npy"
    elif spacing == 'linear':
        omega_values = np.linspace(omega_max, omega_min, num_points)
        file_name = "g_results_linear.npy"
    else:
        raise ValueError("Invalid spacing. Choose 'log' or 'linear'.")

    # Initialize a list to store results for each G
    g_results = []

    # Use tqdm to create a progress bar for the loop
    for omega in tqdm(omega_values, desc="Processing omega values", unit="omega"):
        # Create the Hamiltonian for the current G
        Hamiltonian_Obj.omega = omega
        
        # Calculate QGT along the line
        g_xx, g_xy_real, g_xy_imag, g_yy, trace = QGT_line(
            Hamiltonian_Obj, line_kx, line_ky, delta_k, band_index=band
        )
        
        # Store the results as a dictionary for this G
        g_results.append({
            'omega': omega,
            'g_xx': g_xx,
            'g_xy_real': g_xy_real,
            'g_xy_imag': g_xy_imag,
            'g_yy': g_yy,
            'trace': trace,
        })

    # Save the results to an .npy file with the dynamically set name
    np.save(file_name, g_results)
    print(f"Results saved to {file_name}")


range_of_omega(spacing="linear")

exit()
