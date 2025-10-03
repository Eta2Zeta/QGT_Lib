import numpy as np
import re
import os
from typing import Iterable, Callable, Optional, Tuple


def replace_zeros_with_nan(Z):
    """Replace zero values in the array with NaN."""
    return np.where(Z == 0, np.nan, Z)


# Sign checker
def sign_check(vec1, vec2): 
    if np.dot(vec1, vec2) < 0: 
        return vec1, -vec2
    else: 
        return vec1, vec2


def in_range(w, omega_min, omega_max):
    if (omega_min is not None) and (w < omega_min): return False
    if (omega_max is not None) and (w > omega_max): return False
    return True




def pick_or_create_result_dir(
    base_root: str,
    base_name: str,
    *,
    required_files: Optional[Iterable[str]] = None,
    validator: Optional[Callable[[str], bool]] = None,
    force_new: bool = False,
    suffix_template: str = "_data_set{n}",
    start_index: int = 1,
) -> Tuple[str, bool]:
    """
    Reuse ONLY if:
      - validator(dir) returns True, OR
      - all required_files exist in the dir.
    If neither validator nor required_files is provided, we NEVER reuse.
    If no candidate passes (or force_new=True), create a new numbered dir.

    Returns: (dir_path, used_existing)
    """
    os.makedirs(base_root, exist_ok=True)

    candidates = [d for d in os.listdir(base_root)
                  if d == base_name or d.startswith(base_name + "_")]
    candidates.sort()

    if not force_new and (validator is not None or required_files is not None):
        for d in candidates:
            dir_path = os.path.join(base_root, d)
            passed = False
            if validator is not None:
                passed = bool(validator(dir_path))
            if not passed and required_files is not None:
                passed = all(os.path.exists(os.path.join(dir_path, f)) for f in required_files)
            if passed:
                return dir_path, True  # reuse only when checks pass

    # Create new numbered directory
    n = start_index
    while True:
        name = f"{base_name}{suffix_template.format(n=n)}"
        dir_path = os.path.join(base_root, name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            return dir_path, False
        n += 1



# def setup_results_directory(hamiltonian, kx_range, ky_range, mesh_spacing, force_new=False):
#     """
#     Creates a unique results directory for storing computed data, structured by Hamiltonian parameters and k-space info.
#     If a directory with the same name exists and contains all necessary files, it is reused unless force_new=True.
#     Otherwise, a new directory with an incremented number is created.

#     Parameters:
#         hamiltonian (object): The Hamiltonian object.
#         kx_range (tuple): Tuple of (min_kx, max_kx).
#         ky_range (tuple): Tuple of (min_ky, max_ky).
#         mesh_spacing (int): The number of k-space points.
#         force_new (bool): If True, force creation of a new directory even if one already exists.

#     Returns:
#         dict: Dictionary containing file paths.
#         bool: Whether to use existing data (if all files are found and force_new=False).
#         str: Path to the results directory used.
#     """
#     # Group by Hamiltonian name (fallback to generic if missing)
#     Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")

#     # Main results directory grouped by Hamiltonian
#     results_dir = os.path.join(os.getcwd(), "results", "2D_Eigen_results", Hamiltonian_name)
#     os.makedirs(results_dir, exist_ok=True)

#     base_subdir_name = (
#         f"2D_{hamiltonian.get_filename()}_"
#         f"kx{kx_range[0]:.2f}_{kx_range[1]:.2f}_"
#         f"ky{ky_range[0]:.2f}_{ky_range[1]:.2f}_mesh{mesh_spacing}"
#     )
#     base_subdir_name = re.sub(r'[^\w.-]', '_', base_subdir_name)

#     existing_dirs = [d for d in os.listdir(results_dir) if d.startswith(base_subdir_name)]

#     if not force_new:
#         for existing_dir in sorted(existing_dirs):
#             existing_path = os.path.join(results_dir, existing_dir)

#             file_paths = {
#                 "eigenvalues": os.path.join(existing_path, "eigenvalues.npy"),
#                 "eigenfunctions": os.path.join(existing_path, "eigenfunctions.npy"),
#                 "phasefactors": os.path.join(existing_path, "phasefactors.npy"),
#                 "neighbor_phase_array": os.path.join(existing_path, "neighbor_phase_array.npy"),
#                 "magnus_first": os.path.join(existing_path, "magnus_first.npy"),
#                 "magnus_second": os.path.join(existing_path, "magnus_second.npy"),
#                 "meta_info": os.path.join(existing_path, "meta_info.pkl"),
#             }

#             if all(os.path.exists(path) for path in file_paths.values()):
#                 print(f"Using existing results directory: {existing_path}")
#                 return file_paths, True, existing_path

#     next_number = 1
#     while os.path.exists(os.path.join(results_dir, f"{base_subdir_name}_{next_number}")):
#         next_number += 1

#     results_subdir = os.path.join(results_dir, f"{base_subdir_name}_{next_number}")
#     os.makedirs(results_subdir, exist_ok=True)

#     file_paths = {
#         "eigenvalues": os.path.join(results_subdir, "eigenvalues.npy"),
#         "eigenfunctions": os.path.join(results_subdir, "eigenfunctions.npy"),
#         "phasefactors": os.path.join(results_subdir, "phasefactors.npy"),
#         "neighbor_phase_array": os.path.join(results_subdir, "neighbor_phase_array.npy"),
#         "magnus_first": os.path.join(results_subdir, "magnus_first.npy"),
#         "magnus_second": os.path.join(results_subdir, "magnus_second.npy"),
#         "meta_info": os.path.join(results_subdir, "meta_info.pkl"),
#     }

#     print(f"Created new results directory: {results_subdir}")
#     return file_paths, False, results_subdir

def setup_results_directory(hamiltonian, kx_range, ky_range, mesh_spacing, force_new=False):
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "2D_Eigen_results", Hamiltonian_name)

    base_name = re.sub(
        r'[^\w.-]', '_',
        f"2D_{hamiltonian.get_filename()}_"
        f"kx{kx_range[0]:.2f}_{kx_range[1]:.2f}_"
        f"ky{ky_range[0]:.2f}_{ky_range[1]:.2f}_mesh{mesh_spacing}"
    )

    required_files = [
        "eigenvalues.npy",
        "eigenfunctions.npy",
        "phasefactors.npy",
        "neighbor_phase_array.npy",
        "magnus_first.npy",
        "magnus_second.npy",
        "meta_info.pkl",
    ]

    dir_path, used = pick_or_create_result_dir(
        base_root, base_name,
        required_files=required_files,
        validator=None,
        force_new=force_new,
        suffix_template="_data_set{n}",
        start_index=1
    )

    file_paths = {k: os.path.join(dir_path, fname) for k, fname in {
        "eigenvalues": "eigenvalues.npy",
        "eigenfunctions": "eigenfunctions.npy",
        "phasefactors": "phasefactors.npy",
        "neighbor_phase_array": "neighbor_phase_array.npy",
        "magnus_first": "magnus_first.npy",
        "magnus_second": "magnus_second.npy",
        "meta_info": "meta_info.pkl",
    }.items()}

    print(("Using existing results directory: " if used else "Created new results directory: ") + dir_path)
    return file_paths, used, dir_path


# def setup_results_directory_1d(hamiltonian, k_angle, kx_shift, ky_shift, num_points, k_max):
#     """
#     Creates a unique results directory for storing computed data in 1D calculations.
#     If a directory with the same name exists and contains all necessary files, it is reused.
#     Otherwise, a new directory with an incremented number is created.

#     Parameters:
#         hamiltonian (object): The Hamiltonian object.
#         k_angle (float): Angle of the k-line in degrees.
#         kx_shift (float): Shift in kx.
#         ky_shift (float): Shift in ky.
#         num_points (int): Number of points along the line.

#     Returns:
#         dict: Dictionary containing file paths.
#         bool: Whether to use existing data (if all files are found).
#         str: Path to the results directory used.
#     """
#     # Ensure the main "results" directory exists
#     results_dir = os.path.join(os.getcwd(), "results")
#     os.makedirs(results_dir, exist_ok=True)

#     # Get Hamiltonian parameter string
#     hamiltonian_params = hamiltonian.get_filename()  # Uses the method to format parameter string

#     # Base subdirectory name with Hamiltonian parameters and 1D settings
#     base_subdir_name = f"1D_{hamiltonian_params}_angle{k_angle:.1f}_kxshift{kx_shift:.2f}_kyshift{ky_shift:.2f}_points{num_points}_kmax{k_max:.2f}"
#     base_subdir_name = re.sub(r'[^\w.-]', '_', base_subdir_name)  # Sanitize filename


#     # Check for existing directories with the same base name
#     existing_dirs = [d for d in os.listdir(results_dir) if d.startswith(base_subdir_name)]
    
#     for existing_dir in sorted(existing_dirs):
#         existing_path = os.path.join(results_dir, existing_dir)
        
#         # Define expected file paths within this directory
#         file_paths = {
#             "eigenvalues": os.path.join(existing_path, "eigenvalues.npy"),
#             "eigenfunctions": os.path.join(existing_path, "eigenfunctions.npy"),
#             "meta_info": os.path.join(existing_path, "meta_info.pkl"),  # Use pickle for metadata
#         }

#         # Check if all required files exist
#         if all(os.path.exists(path) for path in file_paths.values()):
#             print(f"Using existing results directory: {existing_path}")
#             return file_paths, True, existing_path  # Use existing directory

#     # If no suitable directory was found, create a new one with an incremented number
#     next_number = 1
#     while os.path.exists(os.path.join(results_dir, f"{base_subdir_name}_{next_number}")):
#         next_number += 1

#     # Define new results directory
#     results_subdir = os.path.join(results_dir, f"{base_subdir_name}_{next_number}")
#     os.makedirs(results_subdir, exist_ok=True)

#     # Define all file paths in the new directory
#     file_paths = {
#         "eigenvalues": os.path.join(results_subdir, "eigenvalues.npy"),
#         "eigenfunctions": os.path.join(results_subdir, "eigenfunctions.npy"),
#         "meta_info": os.path.join(results_subdir, "meta_info.pkl"),  # Use pickle for metadata
#     }

#     print(f"Created new results directory: {results_subdir}")
#     return file_paths, False, results_subdir  # New directory, so use_existing=False

def setup_results_directory_1d(hamiltonian, k_angle, kx_shift, ky_shift, num_points, k_max, *, force_new=False):
    base_root = os.path.join(os.getcwd(), "results")
    base_name = re.sub(
        r'[^\w.-]', '_',
        f"1D_{hamiltonian.get_filename()}_angle{k_angle:.1f}_"
        f"kxshift{kx_shift:.2f}_kyshift{ky_shift:.2f}_points{num_points}_kmax{k_max:.2f}"
    )

    required_files = ["eigenvalues.npy", "eigenfunctions.npy", "meta_info.pkl"]

    dir_path, used = pick_or_create_result_dir(
        base_root, base_name,
        required_files=required_files,
        validator=None,
        force_new=force_new,
        suffix_template="_data_set{n}",
        start_index=1
    )

    file_paths = {
        "eigenvalues": os.path.join(dir_path, "eigenvalues.npy"),
        "eigenfunctions": os.path.join(dir_path, "eigenfunctions.npy"),
        "meta_info": os.path.join(dir_path, "meta_info.pkl"),
    }

    print(("Using existing results directory: " if used else "Created new results directory: ") + dir_path)
    return file_paths, used, dir_path



# def setup_QGT_results_directory(hamiltonian, kx_range, ky_range, mesh_spacing, force_new=False):
#     """
#     Creates a results directory for storing QGT computed data, structured by Hamiltonian parameters and k-space info.
#     If a directory with the same name exists and contains all necessary files, it is reused unless force_new=True.
#     Otherwise, a new directory with an incremented number is created.

#     Parameters:
#         hamiltonian (object): The Hamiltonian object.
#         kx_range (tuple): Tuple of (min_kx, max_kx).
#         ky_range (tuple): Tuple of (min_ky, max_ky).
#         mesh_spacing (int): The number of k-space points.
#         force_new (bool): If True, force creation of a new directory even if one already exists.

#     Returns:
#         dict: Dictionary containing file paths.
#         bool: Whether to use existing data (if all files are found and force_new=False).
#         str: Path to the results directory used.
#     """
#     # Get Hamiltonian name for top-level grouping
#     Hamiltonian_name = hamiltonian.name if hasattr(hamiltonian, "name") else "Hamiltonian"

#     # Main result directory grouped by Hamiltonian name
#     results_dir = os.path.join(os.getcwd(), "results", "2D_QGT_results", Hamiltonian_name)
#     os.makedirs(results_dir, exist_ok=True)

#     # Parameter string and subdir naming
#     base_subdir_name = (
#         f"QGT_{hamiltonian.get_filename(parameter='2D')}_"
#         f"kx{kx_range[0]:.2f}_{kx_range[1]:.2f}_"
#         f"ky{ky_range[0]:.2f}_{ky_range[1]:.2f}_mesh{mesh_spacing}"
#     )
#     base_subdir_name = re.sub(r'[^\w.-]', '_', base_subdir_name)

#     existing_dirs = [d for d in os.listdir(results_dir) if d.startswith(base_subdir_name)]

#     if not force_new:
#         for existing_dir in sorted(existing_dirs):
#             existing_path = os.path.join(results_dir, existing_dir)

#             file_paths = {
#                 "g_xx": os.path.join(existing_path, "g_xx.npy"),
#                 "g_xy_real": os.path.join(existing_path, "g_xy_real.npy"),
#                 "g_xy_imag": os.path.join(existing_path, "g_xy_imag.npy"),
#                 "g_yy": os.path.join(existing_path, "g_yy.npy"),
#                 "trace": os.path.join(existing_path, "trace.npy"),
#                 "meta_info": os.path.join(existing_path, "meta_info.pkl"),
#             }

#             if all(os.path.exists(path) for path in file_paths.values()):
#                 print(f"Using existing QGT results directory: {existing_path}")
#                 return file_paths, True, existing_path

#     next_number = 1
#     while os.path.exists(os.path.join(results_dir, f"{base_subdir_name}_{next_number}")):
#         next_number += 1

#     results_subdir = os.path.join(results_dir, f"{base_subdir_name}_{next_number}")
#     os.makedirs(results_subdir, exist_ok=True)

#     file_paths = {
#         "g_xx": os.path.join(results_subdir, "g_xx.npy"),
#         "g_xy_real": os.path.join(results_subdir, "g_xy_real.npy"),
#         "g_xy_imag": os.path.join(results_subdir, "g_xy_imag.npy"),
#         "g_yy": os.path.join(results_subdir, "g_yy.npy"),
#         "trace": os.path.join(results_subdir, "trace.npy"),
#         "meta_info": os.path.join(results_subdir, "meta_info.pkl"),
#     }

#     print(f"Created new QGT results directory: {results_subdir}")
#     return file_paths, False, results_subdir

def setup_QGT_results_directory(hamiltonian, kx_range, ky_range, mesh_spacing, force_new=False):
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "2D_QGT_results", Hamiltonian_name)

    base_name = re.sub(
        r'[^\w.-]', '_',
        f"QGT_{hamiltonian.get_filename(parameter='2D')}_"
        f"kx{kx_range[0]:.2f}_{kx_range[1]:.2f}_"
        f"ky{ky_range[0]:.2f}_{ky_range[1]:.2f}_mesh{mesh_spacing}"
    )

    required_files = ["g_xx.npy","g_xy_real.npy","g_xy_imag.npy","g_yy.npy","trace.npy","meta_info.pkl"]

    dir_path, used = pick_or_create_result_dir(
        base_root, base_name,
        required_files=required_files,
        validator=None,
        force_new=force_new,
        suffix_template="_data_set{n}",
        start_index=1
    )

    file_paths = {k: os.path.join(dir_path, fname) for k, fname in {
        "g_xx": "g_xx.npy",
        "g_xy_real": "g_xy_real.npy",
        "g_xy_imag": "g_xy_imag.npy",
        "g_yy": "g_yy.npy",
        "trace": "trace.npy",
        "meta_info": "meta_info.pkl",
    }.items()}

    print(("Using existing QGT results directory: " if used else "Created new QGT results directory: ") + dir_path)
    return file_paths, used, dir_path


# def setup_QGT_results_directory_1D(
#     hamiltonian,
#     k_angle,
#     kx_shift,
#     ky_shift,
#     num_k_points,
#     num_omega_points,
#     k_max,
#     omega_min,
#     omega_max,
#     spacing,
#     force_new=False,  # <-- New parameter to force directory creation
# ):
#     """
#     Creates a results directory for storing 1D QGT computed data, structured by Hamiltonian parameters and k-space info.
#     If a directory with the same name exists and contains all necessary files, it is reused.
#     Otherwise, a new directory with an incremented number is created.

#     Parameters:
#         hamiltonian (object): The Hamiltonian object.
#         k_angle (float): Angle of the k-line in degrees.
#         kx_shift (float): Shift in kx.
#         ky_shift (float): Shift in ky.
#         num_points (int): Number of points along the line.
#         k_max (float): Maximum k-value.

#     Returns:
#         dict: Dictionary containing the file path.
#         bool: Whether to use existing data (if all files are found).
#         str: Path to the results directory used.
#     """
#     Hamiltonian_name = hamiltonian.name
#     # Ensure the main "QGT_results_1D" directory exists
#     results_dir = os.path.join(os.getcwd(), "results", "1D_QGT_results", Hamiltonian_name)
#     os.makedirs(results_dir, exist_ok=True)

#     # Get Hamiltonian parameter string
#     hamiltonian_params = hamiltonian.get_filename(parameter = "1D")  # Uses the method to format parameter string

#     # Base subdirectory name with Hamiltonian parameters and 1D QGT settings
#     base_subdir_name = (
#     f"{hamiltonian_params}_angle{k_angle:.1f}_kxshift{kx_shift:.2f}_"
#     f"kyshift{ky_shift:.2f}_points{num_k_points}_kmax{k_max:.2f}_"
#     f"omega{omega_min:.2e}_{omega_max:.2e}_spacing_{str(spacing)}_points{num_omega_points}"
#     )


#     base_subdir_name = re.sub(r'[^\w.-]', '_', base_subdir_name)  # Sanitize filename

#     # Check for existing directories with the same base name
#     existing_dirs = [d for d in os.listdir(results_dir) if d.startswith(base_subdir_name)]
#     if not force_new:
#         for existing_dir in sorted(existing_dirs):
#             existing_path = os.path.join(results_dir, existing_dir)

#             # Define expected file path within this directory
#             file_path = os.path.join(existing_path, "QGT_1D.npy")
#             meta_info_path = os.path.join(existing_path, "meta_info.pkl")

#             # Check if the required files exist
#             if os.path.exists(file_path) and os.path.exists(meta_info_path):
#                 print(f"Using existing QGT results directory: {existing_path}")
#                 return {"QGT_1D": file_path, "meta_info": meta_info_path}, True, existing_path  # Use existing directory

#     # If no suitable directory was found, create a new one with an incremented number
#     next_number = 1
#     while os.path.exists(os.path.join(results_dir, f"{base_subdir_name}_{next_number}")):
#         next_number += 1

#     # Define new results directory
#     results_subdir = os.path.join(results_dir, f"{base_subdir_name}_{next_number}")
#     os.makedirs(results_subdir, exist_ok=True)

#     # Define the file path for the QGT result
#     file_paths = {
#         "QGT_1D": os.path.join(results_subdir, "QGT_1D.npy"),
#         "meta_info": os.path.join(results_subdir, "meta_info.pkl"),  # Metadata
#     }

#     print(f"Created new QGT results directory: {results_subdir}")
#     return file_paths, False, results_subdir  # New directory, so use_existing=False


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
    force_new=False,
):
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "1D_QGT_results", Hamiltonian_name)

    base_name = re.sub(
        r'[^\w.-]', '_',
        f"{hamiltonian.get_filename(parameter='1D')}_angle{k_angle:.1f}_kxshift{kx_shift:.2f}_"
        f"kyshift{ky_shift:.2f}_points{num_k_points}_kmax{k_max:.2f}_"
        f"omega{omega_min:.2e}_{omega_max:.2e}_spacing_{str(spacing)}_points{num_omega_points}"
    )

    required_files = ["QGT_1D.npy", "meta_info.pkl"]

    dir_path, used = pick_or_create_result_dir(
        base_root, base_name,
        required_files=required_files,
        validator=None,
        force_new=force_new,
        suffix_template="_data_set{n}",
        start_index=1
    )

    file_paths = {
        "QGT_1D": os.path.join(dir_path, "QGT_1D.npy"),
        "meta_info": os.path.join(dir_path, "meta_info.pkl"),
    }

    print(("Using existing QGT results directory: " if used else "Created new QGT results directory: ") + dir_path)
    return file_paths, used, dir_path


def setup_QGT_results_directory_1D_single_param(
    hamiltonian,
    *,
    param_name: str,
    vmin: float,
    vmax: float,
    spacing: str,
    num_param_points: int,
    num_k_points: int,
    angle_deg: float,
    kx_shift: float,
    ky_shift: float,
    k_max: float,
    force_new: bool = False,
) -> Tuple[dict, bool, str]:
    """
    Create/reuse a results directory for a 1D sweep of ONE Hamiltonian parameter.
    Returns (file_paths_dict, used_existing, out_dir).
    """
    # Top-level group by Hamiltonian name
    Hname = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "1D_QGT_results", _sanitize(Hname))
    os.makedirs(base_root, exist_ok=True)

    # Hamiltonian signature for filename
    if hasattr(hamiltonian, "get_filename"):
        hsig = hamiltonian.get_filename(parameter="1D")
    else:
        hsig = "H"

    base_name = _sanitize(
        f"{hsig}_angle{angle_deg:.1f}_kx{kx_shift:.2f}_ky{ky_shift:.2f}_"
        f"kmax{k_max:.2f}_param_{param_name}_{vmin:.6g}_{vmax:.6g}_"
        f"spacing_{spacing}_N{int(num_param_points)}_kN{int(num_k_points)}"
    )

    required_files = ["QGT_1D.npy", "meta_info.pkl"]

    # Use your generic picker/creator
    from Library.utilities import pick_or_create_result_dir  # adjust import to your layout
    out_dir, used_existing = pick_or_create_result_dir(
        base_root,
        base_name,
        required_files=required_files,
        validator=None,         # rely on required_files existence
        force_new=force_new,
        suffix_template="_data_set{n}",
        start_index=1,
    )

    file_paths = {
        "QGT_1D":   os.path.join(out_dir, "QGT_1D.npy"),
        "meta_info":os.path.join(out_dir, "meta_info.pkl"),
    }
    return file_paths, used_existing, out_dir


# def setup_QGT_results_directory_2D_omega_range(
#     hamiltonian,
#     kx_range,
#     ky_range,
#     mesh_spacing,
#     omega_min,
#     omega_max,
#     num_omega_points,
#     spacing,
#     force_new=False,
# ):
#     """
#     Creates a results directory for storing 2D QGT omega sweep data, structured by Hamiltonian parameters.

#     Parameters:
#         hamiltonian (object): The Hamiltonian object.
#         kx_range (tuple): Tuple of (min_kx, max_kx).
#         ky_range (tuple): Tuple of (min_ky, max_ky).
#         mesh_spacing (int): Number of k-points along each axis.
#         omega_min (float): Minimum omega.
#         omega_max (float): Maximum omega.
#         num_omega_points (int): Number of omega values to sweep.
#         spacing (str): 'log' or 'linear'.
#         force_new (bool): If True, always creates a new results directory.

#     Returns:
#         dict: Dictionary of file paths.
#         bool: Whether to use existing data.
#         str: Path to the results directory.
#     """
#     Hamiltonian_name = hamiltonian.name

#     # Main results directory for 2D omega range sweeps
#     results_dir = os.path.join(os.getcwd(), "results", "2D_QGT_omega_sweep", Hamiltonian_name)
#     os.makedirs(results_dir, exist_ok=True)

#     # Parameter string
#     hamiltonian_params = hamiltonian.get_filename(parameter="2D")

#     # Base subdir name
#     base_subdir_name = (
#         f"{hamiltonian_params}_kx{kx_range[0]:.2f}_{kx_range[1]:.2f}_"
#         f"ky{ky_range[0]:.2f}_{ky_range[1]:.2f}_mesh{mesh_spacing}_"
#         f"omega{omega_min:.2e}_{omega_max:.2e}_spacing_{spacing}_points{num_omega_points}"
#     )

#     base_subdir_name = re.sub(r'[^\w.-]', '_', base_subdir_name)  # Sanitize

#     # Check for existing subdirectories
#     existing_dirs = [d for d in os.listdir(results_dir) if d.startswith(base_subdir_name)]
#     if not force_new:
#         for existing_dir in sorted(existing_dirs):
#             existing_path = os.path.join(results_dir, existing_dir)

#             # Define expected files
#             file_paths = {
#                 "QGT_2D": os.path.join(existing_path, "QGT_2D.npy"),
#                 "meta_info": os.path.join(existing_path, "meta_info.pkl"),
#             }

#             if all(os.path.exists(path) for path in file_paths.values()):
#                 print(f"Using existing QGT 2D omega sweep directory: {existing_path}")
#                 return file_paths, True, existing_path

#     # Create new results directory with increment
#     next_number = 1
#     while os.path.exists(os.path.join(results_dir, f"{base_subdir_name}_{next_number}")):
#         next_number += 1

#     results_subdir = os.path.join(results_dir, f"{base_subdir_name}_{next_number}")
#     os.makedirs(results_subdir, exist_ok=True)

#     file_paths = {
#         "QGT_2D": os.path.join(results_subdir, "QGT_2D.npy"),
#         "meta_info": os.path.join(results_subdir, "meta_info.pkl"),
#     }

#     print(f"Created new QGT 2D omega sweep directory: {results_subdir}")
#     return file_paths, False, results_subdir

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
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "2D_QGT_omega_sweep", Hamiltonian_name)

    base_name = re.sub(
        r'[^\w.-]', '_',
        f"{hamiltonian.get_filename(parameter='2D')}_"
        f"kx{kx_range[0]:.2f}_{kx_range[1]:.2f}_"
        f"ky{ky_range[0]:.2f}_{ky_range[1]:.2f}_mesh{mesh_spacing}_"
        f"omega{omega_min:.2e}_{omega_max:.2e}_spacing_{spacing}_points{num_omega_points}"
    )

    required_files = ["QGT_2D.npy", "meta_info.pkl"]

    dir_path, used = pick_or_create_result_dir(
        base_root, base_name,
        required_files=required_files,
        validator=None,
        force_new=force_new,
        suffix_template="_data_set{n}",
        start_index=1
    )

    file_paths = {
        "QGT_2D": os.path.join(dir_path, "QGT_2D.npy"),
        "meta_info": os.path.join(dir_path, "meta_info.pkl"),
    }

    print(("Using existing QGT 2D omega sweep directory: " if used else "Created new QGT 2D omega sweep directory: ") + dir_path)
    return file_paths, used, dir_path


def _sanitize(name: str) -> str:
    """Keep [A-Za-z0-9_ . -], replace everything else with underscore."""
    return re.sub(r'[^\w.\-]', '_', str(name))


def _point_dir_name_from_values(param_values: dict, decimals=2):
    """
    Build point dir name like: "t1_-1.00-t2_0.33-psi_1.57".
    param_values: dict {name: value}
    """
    # stable order by key
    items = sorted(param_values.items(), key=lambda kv: str(kv[0]))
    parts = []
    for k, v in items:
        if isinstance(v, float):
            parts.append(f"{k}_{v:.{decimals}f}")
        else:
            parts.append(f"{k}_{v}")
    return "-".join(parts)

# ---------- public API ----------

def _normalize_param_ranges(param_ranges):
    """
    Return a stable, sorted list of (name, vmin, vmax) as floats/strings.
    Accepts dict {name: (vmin, vmax)} or iterable [(name, vmin, vmax), ...].
    """
    if isinstance(param_ranges, dict):
        items = [(str(k), float(v[0]), float(v[1])) for k, v in param_ranges.items()]
    else:
        items = []
        for tup in param_ranges:
            # allow (name, (vmin, vmax)) or (name, vmin, vmax)
            if len(tup) == 2 and isinstance(tup[1], (tuple, list)) and len(tup[1]) == 2:
                n, (a, b) = tup
            elif len(tup) == 3:
                n, a, b = tup
            else:
                raise ValueError("param_ranges items must be (name,(min,max)) or (name,min,max)")
            items.append((str(n), float(a), float(b)))
    # stable order by name
    return sorted(items, key=lambda x: x[0])


def _normalize_spacing(parameter_spacing, names):
    """
    Return a dict {name: count} given:
      - int -> same count for all names
      - dict -> per-name counts (missing names default to 1)
      - None -> default all 1
    """
    if parameter_spacing is None:
        return {n: 1 for n in names}
    if isinstance(parameter_spacing, int):
        return {n: int(parameter_spacing) for n in names}
    # dict
    out = {}
    for n in names:
        c = parameter_spacing.get(n, 1)
        out[n] = int(c)
    return out


def _range_dir_name_with_spacing(param_ranges, parameter_spacing, decimals=2):
    """
    Build name like:
      "M_-2.00_2.00_N32-psi_-3.14_3.14_N32"
    so that changing spacing creates a different directory.
    """
    rng_list = _normalize_param_ranges(param_ranges)
    names = [n for (n, _, _) in rng_list]
    counts = _normalize_spacing(parameter_spacing, names)

    parts = [
        f"{n}_{vmin:.{decimals}f}_{vmax:.{decimals}f}_N{counts[n]}"
        for (n, vmin, vmax) in rng_list
    ]
    return "-".join(parts)



# def setup_phase_diagram_results_general(
#     hamiltonian_template,
#     param_ranges,
#     parameter_spacing=None,   # <-- new argument
#     decimals=2,
#     force_new_range=False
# ):
#     """
#     Create (or reuse) the calc-range directory under:
#       results/phase_diagram/<HamiltonianName>/<range_dir>

#     <range_dir> now encodes BOTH the parameter ranges AND the spacing, e.g.:
#       M_-2.00_2.00_N32-psi_-3.14_3.14_N32_data_set1

#     Args:
#         hamiltonian_template: Hamiltonian instance (used for name only).
#         param_ranges: [(name, vmin, vmax), ...] OR {name: (vmin, vmax)}.
#         parameter_spacing: int or {name: count}. If None, defaults to 1 per param.
#         decimals: float formatting for range labels.
#         force_new_range: if True, always create a new numbered dir.

#     Returns:
#         (range_root_dir, used_existing: bool)
#     """
#     Hname = getattr(hamiltonian_template, "name", "Hamiltonian")
#     base_root = os.path.join(os.getcwd(), "results", "phase_diagram", _sanitize(Hname))
#     os.makedirs(base_root, exist_ok=True)

#     base = _sanitize(_range_dir_name_with_spacing(param_ranges, parameter_spacing, decimals=decimals))

#     if not force_new_range:
#         # reuse exact or numbered matches
#         for d in sorted(os.listdir(base_root)):
#             if d == base or d.startswith(base + "_"):
#                 existing = os.path.join(base_root, d)
#                 print(f"Using existing phase-diagram range directory: {existing}")
#                 return existing, True

#     # create new with increment
#     idx = 1
#     while os.path.exists(os.path.join(base_root, f"{base}_data_set{idx}")):
#         idx += 1
#     range_root = os.path.join(base_root, f"{base}_data_set{idx}")

#     os.makedirs(range_root, exist_ok=True)
#     print(f"Created new phase-diagram range directory: {range_root}")
#     return range_root, False


def setup_phase_diagram_results_general(
    hamiltonian_template,
    param_ranges,
    parameter_spacing=None,
    decimals=2,
    force_new_range=False
):
    Hname = getattr(hamiltonian_template, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "phase_diagram", re.sub(r'[^\w.-]','_',Hname))

    base_name = re.sub(
        r'[^\w.-]', '_',
        _range_dir_name_with_spacing(param_ranges, parameter_spacing, decimals=decimals)
    )

    # Only reuse when the bundle is present; otherwise create new
    dir_path, used = pick_or_create_result_dir(
        base_root, base_name,
        required_files=["qgt_nd_bundle.npz"],  # the artifact that proves completeness
        validator=None,
        force_new=force_new_range,
        suffix_template="_data_set{n}",
        start_index=1
    )

    print(("Using existing phase-diagram range directory: " if used else "Created new phase-diagram range directory: ") + dir_path)
    return dir_path, used


# def setup_phase_point_directory_general(range_root_dir, param_values: dict, decimals=2, force_new_point=False):
#     """
#     Create (or reuse) a subdirectory for one parameter point under <range_root_dir>.
#     Name is constructed from param_values dict (general, not tied to psi/M).

#     Args:
#         range_root_dir: directory returned by setup_phase_diagram_results_general.
#         param_values: dict {param_name: value} (e.g. {"t1":-1.0, "t2":0.33, "psi":1.57, "M":0.0})
#         decimals: float formatting of values.
#         force_new_point: if True, do not reuse existing even if complete.

#     Returns:
#         (file_paths_dict, used_existing: bool, point_dir: str)
#     """
#     point_name = _sanitize(_point_dir_name_from_values(param_values, decimals=decimals))
#     point_dir = os.path.join(range_root_dir, point_name)

#     if not force_new_point and os.path.isdir(point_dir):
#         fps = _phase_point_file_paths(point_dir)
#         if all(os.path.exists(p) for p in fps.values()):
#             print(f"Using existing phase-point directory: {point_dir}")
#             return fps, True, point_dir

#     os.makedirs(point_dir, exist_ok=True)
#     fps = _phase_point_file_paths(point_dir)
#     print(f"Created phase-point directory: {point_dir}")
#     return fps, False, point_dir


def setup_phase_point_directory_general(range_root_dir, param_values: dict, decimals=2, force_new_point=False):
    point_name = _sanitize(_point_dir_name_from_values(param_values, decimals=decimals))

    required_files = [
        "eigenvalues.npy",
        "eigenfunctions.npy",
        "g_xx.npy",
        "g_xy_real.npy",
        "g_xy_imag.npy",
        "g_yy.npy",
        "trace.npy",
        "chern.npy",
        "meta_info.pkl",
    ]

    dir_path, used = pick_or_create_result_dir(
        base_root=range_root_dir,
        base_name=point_name,
        required_files=required_files,
        validator=None,
        force_new=force_new_point,
        suffix_template="",     # points aren’t numbered; exact name per values
        start_index=1
    )

    # Build paths
    fps = {k: os.path.join(dir_path, fname) for k, fname in {
        "eigenvalues": "eigenvalues.npy",
        "eigenfunctions": "eigenfunctions.npy",
        "g_xx": "g_xx.npy",
        "g_xy_real": "g_xy_real.npy",
        "g_xy_imag": "g_xy_imag.npy",
        "g_yy": "g_yy.npy",
        "trace": "trace.npy",
        "chern": "chern.npy",
        "meta_info": "meta_info.pkl",
    }.items()}

    print(("Using existing phase-point directory: " if used else "Created phase-point directory: ") + dir_path)
    return fps, used, dir_path


def setup_qgt_nd_results_dir(
    hamiltonian_template,
    param_ranges,
    parameter_spacing,
    kx_range,
    ky_range,
    mesh_spacing,
    decimals=3,
    force_new=False
):
    """
    Create (or reuse) the root dir to hold a single N-D QGT npz bundle.
    Reuse only when 'qgt_nd_bundle.npz' is present; otherwise make a new numbered dir.
    Returns: (root_dir, used_existing)
    """
    Hname = getattr(hamiltonian_template, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "QGT_ND", _sanitize(Hname))

    # ---- optional Hamiltonian-specific prefix via get_filename ----
    try:
        h_prefix = str(hamiltonian_template.get_filename(parameter="ND"))
    except Exception:
        # fall back to something stable if not available
        h_prefix = Hname
    h_prefix = _sanitize(h_prefix)

    # ---- normalize parameter ranges ----
    if isinstance(param_ranges, dict):
        items = sorted(param_ranges.items(), key=lambda kv: kv[0])  # [(name,(min,max)),...]
    else:
        items = sorted([(n, (a, b)) for (n, a, b) in param_ranges], key=lambda x: x[0])

    range_parts = [
        f"{name}_{float(vmin):.{decimals}f}_{float(vmax):.{decimals}f}"
        for name, (vmin, vmax) in items
    ]

    # ---- spacing label: accept int or (n, 'linear'|'log') per-param ----
    spacing_parts = []
    if isinstance(parameter_spacing, int):
        for name, _ in items:
            spacing_parts.append(f"{name}_{int(parameter_spacing)}_linear")
    elif isinstance(parameter_spacing, dict):
        for name, _ in items:
            spec = parameter_spacing.get(name, 1)
            if isinstance(spec, int):
                n, scale = int(spec), "linear"
            elif isinstance(spec, (tuple, list)) and len(spec) >= 2:
                n, scale = int(spec[0]), str(spec[1]).lower()
                if scale not in ("linear", "log"):
                    raise ValueError(f"Unsupported scale '{scale}' for {name} (use 'linear' or 'log').")
            else:
                raise ValueError(f"Unsupported spacing spec for {name}: {spec}")
            spacing_parts.append(f"{name}_{n}_{scale}")
    else:
        raise ValueError("parameter_spacing must be int or dict")

    label_ranges  = "RANGES["  + "-".join(range_parts)   + "]"
    label_spacing = "SPACING[" + "-".join(spacing_parts) + "]"
    klabel = f"kx{kx_range[0]:.2f}_{kx_range[1]:.2f}__ky{ky_range[0]:.2f}_{ky_range[1]:.2f}__mesh{mesh_spacing}"

    # Final base name uses the Hamiltonian filename prefix + explicit sweep metadata
    base_name = _sanitize(f"{h_prefix}-{label_ranges}-{label_spacing}-{klabel}")

    # ---- reuse/create via the shared helper ----
    required_files = ["qgt_nd_bundle.npz"]
    dir_path, used = pick_or_create_result_dir(
        base_root=base_root,
        base_name=base_name,
        required_files=required_files,
        validator=None,
        force_new=force_new,
        suffix_template="_data_set{n}",
        start_index=1
    )

    print(("Using existing QGT N-D sweep directory: " if used else "Created QGT N-D sweep directory: ") + dir_path)
    return dir_path, used
