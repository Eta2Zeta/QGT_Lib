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


def setup_results_directory(hamiltonian, kx_range, ky_range, mesh_spacing, force_new=False):
    """
    Creates a unique results directory for storing computed data, structured by Hamiltonian parameters and k-space info.
    If a directory with the same name exists and contains all necessary files, it is reused unless force_new=True.
    Otherwise, a new directory with an incremented number is created.

    Parameters:
        hamiltonian (object): The Hamiltonian object.
        kx_range (tuple): Tuple of (min_kx, max_kx).
        ky_range (tuple): Tuple of (min_ky, max_ky).
        mesh_spacing (int): The number of k-space points.
        force_new (bool): If True, force creation of a new directory even if one already exists.

    Returns:
        dict: Dictionary containing file paths.
        bool: Whether to use existing data (if all files are found and force_new=False).
        str: Path to the results directory used.
    """
    # Group by Hamiltonian name (fallback to generic if missing)
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")

    # Main results directory grouped by Hamiltonian
    results_dir = os.path.join(os.getcwd(), "results", "2D_Eigen_results", Hamiltonian_name)
    os.makedirs(results_dir, exist_ok=True)

    base_subdir_name = (
        f"2D_{hamiltonian.get_filename()}_"
        f"kx{kx_range[0]:.2f}_{kx_range[1]:.2f}_"
        f"ky{ky_range[0]:.2f}_{ky_range[1]:.2f}_mesh{mesh_spacing}"
    )
    base_subdir_name = re.sub(r'[^\w.-]', '_', base_subdir_name)

    existing_dirs = [d for d in os.listdir(results_dir) if d.startswith(base_subdir_name)]

    if not force_new:
        for existing_dir in sorted(existing_dirs):
            existing_path = os.path.join(results_dir, existing_dir)

            file_paths = {
                "eigenvalues": os.path.join(existing_path, "eigenvalues.npy"),
                "eigenfunctions": os.path.join(existing_path, "eigenfunctions.npy"),
                "phasefactors": os.path.join(existing_path, "phasefactors.npy"),
                "neighbor_phase_array": os.path.join(existing_path, "neighbor_phase_array.npy"),
                "magnus_first": os.path.join(existing_path, "magnus_first.npy"),
                "magnus_second": os.path.join(existing_path, "magnus_second.npy"),
                "meta_info": os.path.join(existing_path, "meta_info.pkl"),
            }

            if all(os.path.exists(path) for path in file_paths.values()):
                print(f"Using existing results directory: {existing_path}")
                return file_paths, True, existing_path

    next_number = 1
    while os.path.exists(os.path.join(results_dir, f"{base_subdir_name}_{next_number}")):
        next_number += 1

    results_subdir = os.path.join(results_dir, f"{base_subdir_name}_{next_number}")
    os.makedirs(results_subdir, exist_ok=True)

    file_paths = {
        "eigenvalues": os.path.join(results_subdir, "eigenvalues.npy"),
        "eigenfunctions": os.path.join(results_subdir, "eigenfunctions.npy"),
        "phasefactors": os.path.join(results_subdir, "phasefactors.npy"),
        "neighbor_phase_array": os.path.join(results_subdir, "neighbor_phase_array.npy"),
        "magnus_first": os.path.join(results_subdir, "magnus_first.npy"),
        "magnus_second": os.path.join(results_subdir, "magnus_second.npy"),
        "meta_info": os.path.join(results_subdir, "meta_info.pkl"),
    }

    print(f"Created new results directory: {results_subdir}")
    return file_paths, False, results_subdir


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


def setup_QGT_results_directory(hamiltonian, kx_range, ky_range, mesh_spacing, force_new=False):
    """
    Creates a results directory for storing QGT computed data, structured by Hamiltonian parameters and k-space info.
    If a directory with the same name exists and contains all necessary files, it is reused unless force_new=True.
    Otherwise, a new directory with an incremented number is created.

    Parameters:
        hamiltonian (object): The Hamiltonian object.
        kx_range (tuple): Tuple of (min_kx, max_kx).
        ky_range (tuple): Tuple of (min_ky, max_ky).
        mesh_spacing (int): The number of k-space points.
        force_new (bool): If True, force creation of a new directory even if one already exists.

    Returns:
        dict: Dictionary containing file paths.
        bool: Whether to use existing data (if all files are found and force_new=False).
        str: Path to the results directory used.
    """
    # Get Hamiltonian name for top-level grouping
    Hamiltonian_name = hamiltonian.name if hasattr(hamiltonian, "name") else "Hamiltonian"

    # Main result directory grouped by Hamiltonian name
    results_dir = os.path.join(os.getcwd(), "results", "2D_QGT_results", Hamiltonian_name)
    os.makedirs(results_dir, exist_ok=True)

    # Parameter string and subdir naming
    base_subdir_name = (
        f"QGT_{hamiltonian.get_filename(parameter='2D')}_"
        f"kx{kx_range[0]:.2f}_{kx_range[1]:.2f}_"
        f"ky{ky_range[0]:.2f}_{ky_range[1]:.2f}_mesh{mesh_spacing}"
    )
    base_subdir_name = re.sub(r'[^\w.-]', '_', base_subdir_name)

    existing_dirs = [d for d in os.listdir(results_dir) if d.startswith(base_subdir_name)]

    if not force_new:
        for existing_dir in sorted(existing_dirs):
            existing_path = os.path.join(results_dir, existing_dir)

            file_paths = {
                "g_xx": os.path.join(existing_path, "g_xx.npy"),
                "g_xy_real": os.path.join(existing_path, "g_xy_real.npy"),
                "g_xy_imag": os.path.join(existing_path, "g_xy_imag.npy"),
                "g_yy": os.path.join(existing_path, "g_yy.npy"),
                "trace": os.path.join(existing_path, "trace.npy"),
                "meta_info": os.path.join(existing_path, "meta_info.pkl"),
            }

            if all(os.path.exists(path) for path in file_paths.values()):
                print(f"Using existing QGT results directory: {existing_path}")
                return file_paths, True, existing_path

    next_number = 1
    while os.path.exists(os.path.join(results_dir, f"{base_subdir_name}_{next_number}")):
        next_number += 1

    results_subdir = os.path.join(results_dir, f"{base_subdir_name}_{next_number}")
    os.makedirs(results_subdir, exist_ok=True)

    file_paths = {
        "g_xx": os.path.join(results_subdir, "g_xx.npy"),
        "g_xy_real": os.path.join(results_subdir, "g_xy_real.npy"),
        "g_xy_imag": os.path.join(results_subdir, "g_xy_imag.npy"),
        "g_yy": os.path.join(results_subdir, "g_yy.npy"),
        "trace": os.path.join(results_subdir, "trace.npy"),
        "meta_info": os.path.join(results_subdir, "meta_info.pkl"),
    }

    print(f"Created new QGT results directory: {results_subdir}")
    return file_paths, False, results_subdir

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


def _sanitize(name: str) -> str:
    """Keep [A-Za-z0-9_ . -], replace everything else with underscore."""
    return re.sub(r'[^\w.\-]', '_', str(name))

def _range_dir_name(param_ranges, decimals=2):
    """
    Build calc-range dir name like: "t1_-2.00_2.00-t2_0.00_1.00-psi_-3.14_3.14".
    param_ranges: iterable of (name, vmin, vmax) or dict {name: (vmin, vmax)}.
    """
    if isinstance(param_ranges, dict):
        items = list(param_ranges.items())
    else:
        # assume iterable of (name, vmin, vmax)
        items = [(k, (vmin, vmax)) if not isinstance(vmin, tuple) else (k, vmin)
                 for (k, vmin, vmax) in [(n, a, b) for (n, a, b) in param_ranges]]

    # normalize to [(name, (vmin, vmax)), ...]
    norm = []
    for it in items:
        name, rng = it[0], it[1]
        vmin, vmax = rng
        norm.append((str(name), float(vmin), float(vmax)))

    # stable order by param name
    norm.sort(key=lambda x: x[0])

    parts = [
        f"{n}_{vmin:.{decimals}f}_{vmax:.{decimals}f}"
        for (n, vmin, vmax) in norm
    ]
    return "-".join(parts)

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

def _phase_point_file_paths(root_dir):
    """Standard files for one phase point."""
    return {
        "eigenvalues":    os.path.join(root_dir, "eigenvalues.npy"),
        "eigenfunctions": os.path.join(root_dir, "eigenfunctions.npy"),
        "g_xx":           os.path.join(root_dir, "g_xx.npy"),
        "g_xy_real":      os.path.join(root_dir, "g_xy_real.npy"),
        "g_xy_imag":      os.path.join(root_dir, "g_xy_imag.npy"),
        "g_yy":           os.path.join(root_dir, "g_yy.npy"),
        "trace":          os.path.join(root_dir, "trace.npy"),
        "chern":          os.path.join(root_dir, "chern.npy"),
        "meta_info":      os.path.join(root_dir, "meta_info.pkl"),
    }

# ---------- public API ----------

def setup_phase_diagram_results_general(hamiltonian_template, param_ranges, decimals=2, force_new_range=False):
    """
    Create (or reuse) the calc-range directory under:
      results/phase_diagram/<HamiltonianName>/<range_dir>
    where <range_dir> is built from a general list/dict of parameter ranges.

    Args:
        hamiltonian_template: your Hamiltonian instance (used only for name).
        param_ranges: iterable [(name, vmin, vmax), ...] OR dict {name: (vmin, vmax)}.
        decimals: float formatting for range labels.
        force_new_range: if True, always create a new numbered dir.

    Returns:
        (range_root_dir, used_existing: bool)
    """
    Hname = getattr(hamiltonian_template, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "phase_diagram", _sanitize(Hname))
    os.makedirs(base_root, exist_ok=True)

    base = _sanitize(_range_dir_name(param_ranges, decimals=decimals))

    if not force_new_range:
        # exact or numbered reuses
        for d in sorted(os.listdir(base_root)):
            if d == base or d.startswith(base + "_"):
                existing = os.path.join(base_root, d)
                print(f"Using existing phase-diagram range directory: {existing}")
                return existing, True

    # create new with increment
    idx = 1
    while os.path.exists(os.path.join(base_root, f"{base}_data_set{idx}")):
        idx += 1
    range_root = os.path.join(base_root, f"{base}_data_set{idx}")
    
    os.makedirs(range_root, exist_ok=True)
    print(f"Created new phase-diagram range directory: {range_root}")
    return range_root, False

def setup_phase_point_directory_general(range_root_dir, param_values: dict, decimals=2, force_new_point=False):
    """
    Create (or reuse) a subdirectory for one parameter point under <range_root_dir>.
    Name is constructed from param_values dict (general, not tied to psi/M).

    Args:
        range_root_dir: directory returned by setup_phase_diagram_results_general.
        param_values: dict {param_name: value} (e.g. {"t1":-1.0, "t2":0.33, "psi":1.57, "M":0.0})
        decimals: float formatting of values.
        force_new_point: if True, do not reuse existing even if complete.

    Returns:
        (file_paths_dict, used_existing: bool, point_dir: str)
    """
    point_name = _sanitize(_point_dir_name_from_values(param_values, decimals=decimals))
    point_dir = os.path.join(range_root_dir, point_name)

    if not force_new_point and os.path.isdir(point_dir):
        fps = _phase_point_file_paths(point_dir)
        if all(os.path.exists(p) for p in fps.values()):
            print(f"Using existing phase-point directory: {point_dir}")
            return fps, True, point_dir

    os.makedirs(point_dir, exist_ok=True)
    fps = _phase_point_file_paths(point_dir)
    print(f"Created phase-point directory: {point_dir}")
    return fps, False, point_dir
