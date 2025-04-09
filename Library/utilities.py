import numpy as np
import re
import os

def replace_zeros_with_nan(Z):
    """Replace zero values in the array with NaN."""
    return np.where(Z == 0, np.nan, Z)


# Sign checker
def sign_check(vec1, vec2): 
    if np.dot(vec1, vec2) < 0: 
        return vec1, -vec2
    else: 
        return vec1, vec2


def setup_results_directory(hamiltonian, kx_range, ky_range, mesh_spacing):
    """
    Creates a unique results directory for storing computed data, structured by Hamiltonian parameters and k-space info.
    If a directory with the same name exists and contains all necessary files, it is reused.
    Otherwise, a new directory with an incremented number is created.

    Parameters:
        hamiltonian (object): The Hamiltonian object.
        kx_range (tuple): Tuple of (min_kx, max_kx).
        ky_range (tuple): Tuple of (min_ky, max_ky).
        mesh_spacing (int): The number of k-space points.

    Returns:
        dict: Dictionary containing file paths.
        bool: Whether to use existing data (if all files are found).
        str: Path to the results directory used.
    """
    # Ensure the main "results" directory exists
    results_dir = os.path.join(os.getcwd(), "results", "2D_Eigen_results")
    os.makedirs(results_dir, exist_ok=True)

    # Base subdirectory name
    base_subdir_name = f"2D_{hamiltonian.get_filename()}_kx{kx_range[0]:.2f}_{kx_range[1]:.2f}_ky{ky_range[0]:.2f}_{ky_range[1]:.2f}_mesh{mesh_spacing}"
    base_subdir_name = re.sub(r'[^\w.-]', '_', base_subdir_name)  # Sanitize filename

    # Check for existing directories with the same base name
    existing_dirs = [d for d in os.listdir(results_dir) if d.startswith(base_subdir_name)]
    
    for existing_dir in sorted(existing_dirs):
        existing_path = os.path.join(results_dir, existing_dir)
        
        # Define expected file paths within this directory
        file_paths = {
            "eigenvalues": os.path.join(existing_path, "eigenvalues.npy"),
            "eigenfunctions": os.path.join(existing_path, "eigenfunctions.npy"),
            "phasefactors": os.path.join(existing_path, "phasefactors.npy"),
            "neighbor_phase_array": os.path.join(existing_path, "neighbor_phase_array.npy"),
            "magnus_first": os.path.join(existing_path, "magnus_first.npy"),
            "magnus_second": os.path.join(existing_path, "magnus_second.npy"),
            "meta_info": os.path.join(existing_path, "meta_info.pkl"),  # Use pickle for metadata
        }

        # Check if all required files exist
        if all(os.path.exists(path) for path in file_paths.values()):
            print(f"Using existing results directory: {existing_path}")
            return file_paths, True, existing_path  # Use existing directory

    # If no suitable directory was found, create a new one with an incremented number
    next_number = 1
    while os.path.exists(os.path.join(results_dir, f"{base_subdir_name}_{next_number}")):
        next_number += 1

    # Define new results directory
    results_subdir = os.path.join(results_dir, f"{base_subdir_name}_{next_number}")
    os.makedirs(results_subdir, exist_ok=True)

    # Define all file paths in the new directory
    file_paths = {
        "eigenvalues": os.path.join(results_subdir, "eigenvalues.npy"),
        "eigenfunctions": os.path.join(results_subdir, "eigenfunctions.npy"),
        "phasefactors": os.path.join(results_subdir, "phasefactors.npy"),
        "neighbor_phase_array": os.path.join(results_subdir, "neighbor_phase_array.npy"),
        "magnus_first": os.path.join(results_subdir, "magnus_first.npy"),
        "magnus_second": os.path.join(results_subdir, "magnus_second.npy"),
        "meta_info": os.path.join(results_subdir, "meta_info.pkl"),  # Use pickle for metadata
    }

    print(f"Created new results directory: {results_subdir}")
    return file_paths, False, results_subdir  # New directory, so use_existing=False


def setup_results_directory_1d(hamiltonian, k_angle, kx_shift, ky_shift, num_points, k_max):
    """
    Creates a unique results directory for storing computed data in 1D calculations.
    If a directory with the same name exists and contains all necessary files, it is reused.
    Otherwise, a new directory with an incremented number is created.

    Parameters:
        hamiltonian (object): The Hamiltonian object.
        k_angle (float): Angle of the k-line in degrees.
        kx_shift (float): Shift in kx.
        ky_shift (float): Shift in ky.
        num_points (int): Number of points along the line.

    Returns:
        dict: Dictionary containing file paths.
        bool: Whether to use existing data (if all files are found).
        str: Path to the results directory used.
    """
    # Ensure the main "results" directory exists
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Get Hamiltonian parameter string
    hamiltonian_params = hamiltonian.get_filename()  # Uses the method to format parameter string

    # Base subdirectory name with Hamiltonian parameters and 1D settings
    base_subdir_name = f"1D_{hamiltonian_params}_angle{k_angle:.1f}_kxshift{kx_shift:.2f}_kyshift{ky_shift:.2f}_points{num_points}_kmax{k_max:.2f}"
    base_subdir_name = re.sub(r'[^\w.-]', '_', base_subdir_name)  # Sanitize filename


    # Check for existing directories with the same base name
    existing_dirs = [d for d in os.listdir(results_dir) if d.startswith(base_subdir_name)]
    
    for existing_dir in sorted(existing_dirs):
        existing_path = os.path.join(results_dir, existing_dir)
        
        # Define expected file paths within this directory
        file_paths = {
            "eigenvalues": os.path.join(existing_path, "eigenvalues.npy"),
            "eigenfunctions": os.path.join(existing_path, "eigenfunctions.npy"),
            "meta_info": os.path.join(existing_path, "meta_info.pkl"),  # Use pickle for metadata
        }

        # Check if all required files exist
        if all(os.path.exists(path) for path in file_paths.values()):
            print(f"Using existing results directory: {existing_path}")
            return file_paths, True, existing_path  # Use existing directory

    # If no suitable directory was found, create a new one with an incremented number
    next_number = 1
    while os.path.exists(os.path.join(results_dir, f"{base_subdir_name}_{next_number}")):
        next_number += 1

    # Define new results directory
    results_subdir = os.path.join(results_dir, f"{base_subdir_name}_{next_number}")
    os.makedirs(results_subdir, exist_ok=True)

    # Define all file paths in the new directory
    file_paths = {
        "eigenvalues": os.path.join(results_subdir, "eigenvalues.npy"),
        "eigenfunctions": os.path.join(results_subdir, "eigenfunctions.npy"),
        "meta_info": os.path.join(results_subdir, "meta_info.pkl"),  # Use pickle for metadata
    }

    print(f"Created new results directory: {results_subdir}")
    return file_paths, False, results_subdir  # New directory, so use_existing=False


def setup_QGT_results_directory(hamiltonian, kx_range, ky_range, mesh_spacing):
    """
    Creates a results directory for storing QGT computed data, structured by Hamiltonian parameters and k-space info.
    If a directory with the same name exists and contains all necessary files, it is reused.
    Otherwise, a new directory with an incremented number is created.

    Parameters:
        hamiltonian (object): The Hamiltonian object.
        kx_range (tuple): Tuple of (min_kx, max_kx).
        ky_range (tuple): Tuple of (min_ky, max_ky).
        mesh_spacing (int): The number of k-space points.

    Returns:
        dict: Dictionary containing file paths.
        bool: Whether to use existing data (if all files are found).
        str: Path to the results directory used.
    """
    # Ensure the main "QGT_results" directory exists
    results_dir = os.path.join(os.getcwd(), "results", "2D_QGT_results")
    os.makedirs(results_dir, exist_ok=True)

    # Base subdirectory name
    base_subdir_name = f"QGT_{hamiltonian.get_filename(parameter = '2D')}_kx{kx_range[0]:.2f}_{kx_range[1]:.2f}_ky{ky_range[0]:.2f}_{ky_range[1]:.2f}_mesh{mesh_spacing}"
    base_subdir_name = re.sub(r'[^\w.-]', '_', base_subdir_name)  # Sanitize filename

    # Check for existing directories with the same base name
    existing_dirs = [d for d in os.listdir(results_dir) if d.startswith(base_subdir_name)]

    for existing_dir in sorted(existing_dirs):
        existing_path = os.path.join(results_dir, existing_dir)

        # Define expected file paths within this directory
        file_paths = {
            "g_xx": os.path.join(existing_path, "g_xx.npy"),
            "g_xy_real": os.path.join(existing_path, "g_xy_real.npy"),
            "g_xy_imag": os.path.join(existing_path, "g_xy_imag.npy"),
            "g_yy": os.path.join(existing_path, "g_yy.npy"),
            "trace": os.path.join(existing_path, "trace.npy"),
            "meta_info": os.path.join(existing_path, "meta_info.pkl"),  # Metadata
        }

         # Check if all required files exist
        if all(os.path.exists(path) for path in file_paths.values()):
            print(f"Using existing QGT results directory: {existing_path}")
            return file_paths, True, existing_path  # Use existing directory

    # If no suitable directory was found, create a new one with an incremented number
    next_number = 1
    while os.path.exists(os.path.join(results_dir, f"{base_subdir_name}_{next_number}")):
        next_number += 1

    # Define new results directory
    results_subdir = os.path.join(results_dir, f"{base_subdir_name}_{next_number}")
    os.makedirs(results_subdir, exist_ok=True)

    # Define all file paths in the new directory
    file_paths = {
        "g_xx": os.path.join(results_subdir, "g_xx.npy"),
        "g_xy_real": os.path.join(results_subdir, "g_xy_real.npy"),
        "g_xy_imag": os.path.join(results_subdir, "g_xy_imag.npy"),
        "g_yy": os.path.join(results_subdir, "g_yy.npy"),
        "trace": os.path.join(results_subdir, "trace.npy"),
        "meta_info": os.path.join(results_subdir, "meta_info.pkl"),  # Metadata
    }

    print(f"Created new QGT results directory: {results_subdir}")
    return file_paths, False, results_subdir  # New directory, so use_existing=False


def setup_QGT_results_directory_1D(
    hamiltonian,
    k_angle,
    kx_shift,
    ky_shift,
    num_k_points,
    num_omega_points,
    k_max,
    omega_min,
    omega_max,
    spacing,
    force_new=False,  # <-- New parameter to force directory creation
):
    """
    Creates a results directory for storing 1D QGT computed data, structured by Hamiltonian parameters and k-space info.
    If a directory with the same name exists and contains all necessary files, it is reused.
    Otherwise, a new directory with an incremented number is created.

    Parameters:
        hamiltonian (object): The Hamiltonian object.
        k_angle (float): Angle of the k-line in degrees.
        kx_shift (float): Shift in kx.
        ky_shift (float): Shift in ky.
        num_points (int): Number of points along the line.
        k_max (float): Maximum k-value.

    Returns:
        dict: Dictionary containing the file path.
        bool: Whether to use existing data (if all files are found).
        str: Path to the results directory used.
    """
    Hamiltonian_name = hamiltonian.name
    # Ensure the main "QGT_results_1D" directory exists
    results_dir = os.path.join(os.getcwd(), "results", "1D_QGT_results", Hamiltonian_name)
    os.makedirs(results_dir, exist_ok=True)

    # Get Hamiltonian parameter string
    hamiltonian_params = hamiltonian.get_filename(parameter = "1D")  # Uses the method to format parameter string

    # Base subdirectory name with Hamiltonian parameters and 1D QGT settings
    base_subdir_name = (
    f"{hamiltonian_params}_angle{k_angle:.1f}_kxshift{kx_shift:.2f}_"
    f"kyshift{ky_shift:.2f}_points{num_k_points}_kmax{k_max:.2f}_"
    f"omega{omega_min:.2e}_{omega_max:.2e}_spacing_{str(spacing)}_points{num_omega_points}"
    )


    base_subdir_name = re.sub(r'[^\w.-]', '_', base_subdir_name)  # Sanitize filename

    # Check for existing directories with the same base name
    existing_dirs = [d for d in os.listdir(results_dir) if d.startswith(base_subdir_name)]
    if not force_new:
        for existing_dir in sorted(existing_dirs):
            existing_path = os.path.join(results_dir, existing_dir)

            # Define expected file path within this directory
            file_path = os.path.join(existing_path, "QGT_1D.npy")
            meta_info_path = os.path.join(existing_path, "meta_info.pkl")

            # Check if the required files exist
            if os.path.exists(file_path) and os.path.exists(meta_info_path):
                print(f"Using existing QGT results directory: {existing_path}")
                return {"QGT_1D": file_path, "meta_info": meta_info_path}, True, existing_path  # Use existing directory

    # If no suitable directory was found, create a new one with an incremented number
    next_number = 1
    while os.path.exists(os.path.join(results_dir, f"{base_subdir_name}_{next_number}")):
        next_number += 1

    # Define new results directory
    results_subdir = os.path.join(results_dir, f"{base_subdir_name}_{next_number}")
    os.makedirs(results_subdir, exist_ok=True)

    # Define the file path for the QGT result
    file_paths = {
        "QGT_1D": os.path.join(results_subdir, "QGT_1D.npy"),
        "meta_info": os.path.join(results_subdir, "meta_info.pkl"),  # Metadata
    }

    print(f"Created new QGT results directory: {results_subdir}")
    return file_paths, False, results_subdir  # New directory, so use_existing=False




def setup_QGT_results_directory_2D_omega_range(
    hamiltonian,
    kx_range,
    ky_range,
    mesh_spacing,
    omega_min,
    omega_max,
    num_omega_points,
    spacing,
    force_new=False,
):
    """
    Creates a results directory for storing 2D QGT omega sweep data, structured by Hamiltonian parameters.

    Parameters:
        hamiltonian (object): The Hamiltonian object.
        kx_range (tuple): Tuple of (min_kx, max_kx).
        ky_range (tuple): Tuple of (min_ky, max_ky).
        mesh_spacing (int): Number of k-points along each axis.
        omega_min (float): Minimum omega.
        omega_max (float): Maximum omega.
        num_omega_points (int): Number of omega values to sweep.
        spacing (str): 'log' or 'linear'.
        force_new (bool): If True, always creates a new results directory.

    Returns:
        dict: Dictionary of file paths.
        bool: Whether to use existing data.
        str: Path to the results directory.
    """
    Hamiltonian_name = hamiltonian.name

    # Main results directory for 2D omega range sweeps
    results_dir = os.path.join(os.getcwd(), "results", "2D_QGT_omega_sweep", Hamiltonian_name)
    os.makedirs(results_dir, exist_ok=True)

    # Parameter string
    hamiltonian_params = hamiltonian.get_filename(parameter="2D")

    # Base subdir name
    base_subdir_name = (
        f"{hamiltonian_params}_kx{kx_range[0]:.2f}_{kx_range[1]:.2f}_"
        f"ky{ky_range[0]:.2f}_{ky_range[1]:.2f}_mesh{mesh_spacing}_"
        f"omega{omega_min:.2e}_{omega_max:.2e}_spacing_{spacing}_points{num_omega_points}"
    )

    base_subdir_name = re.sub(r'[^\w.-]', '_', base_subdir_name)  # Sanitize

    # Check for existing subdirectories
    existing_dirs = [d for d in os.listdir(results_dir) if d.startswith(base_subdir_name)]
    if not force_new:
        for existing_dir in sorted(existing_dirs):
            existing_path = os.path.join(results_dir, existing_dir)

            # Define expected files
            file_paths = {
                "QGT_2D": os.path.join(existing_path, "QGT_2D.npy"),
                "meta_info": os.path.join(existing_path, "meta_info.pkl"),
            }

            if all(os.path.exists(path) for path in file_paths.values()):
                print(f"Using existing QGT 2D omega sweep directory: {existing_path}")
                return file_paths, True, existing_path

    # Create new results directory with increment
    next_number = 1
    while os.path.exists(os.path.join(results_dir, f"{base_subdir_name}_{next_number}")):
        next_number += 1

    results_subdir = os.path.join(results_dir, f"{base_subdir_name}_{next_number}")
    os.makedirs(results_subdir, exist_ok=True)

    file_paths = {
        "QGT_2D": os.path.join(results_subdir, "QGT_2D.npy"),
        "meta_info": os.path.join(results_subdir, "meta_info.pkl"),
    }

    print(f"Created new QGT 2D omega sweep directory: {results_subdir}")
    return file_paths, False, results_subdir