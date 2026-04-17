import os, re, json, math, pickle
from typing import Optional, Iterable, Callable, Tuple, Dict, Any
import numpy as np

def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _load_pkl(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def pick_or_create_dataset_dir(
    base_root: str,
    *,
    meta_target: Dict[str, Any],
    required_files: Optional[Iterable[str]] = None,
    meta_matcher: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
    force_new: bool = False,
    prefix: str = "data_set_",
    start_index: int = 1,
) -> Tuple[str, bool]:
    """
    Dataset dirs are named like data_set_1, data_set_2, ...
    Reuse if meta.json matches meta_target (via meta_matcher) AND required_files exist (if provided).
    Otherwise create a new data_set_N.

    Returns: (dir_path, used_existing)
    """
    os.makedirs(base_root, exist_ok=True)

    # Collect existing dataset dirs
    existing = []
    for d in os.listdir(base_root):
        if d.startswith(prefix):
            m = re.match(rf"^{re.escape(prefix)}(\d+)$", d)
            if m:
                existing.append((int(m.group(1)), d))
    existing.sort()

    if not force_new and meta_matcher is not None:
        for _, dname in existing:
            dpath = os.path.join(base_root, dname)
            meta_path = os.path.join(dpath, "meta_info.pkl")
            meta = _load_pkl(meta_path)
            if meta is None:
                continue

            if not meta_matcher(meta, meta_target):
                continue

            if required_files is not None:
                ok = all(os.path.exists(os.path.join(dpath, f)) for f in required_files)
                if not ok:
                    continue

            return dpath, True

    # Create new dataset dir
    n = start_index
    if existing:
        n = max(n, existing[-1][0] + 1)

    while True:
        dname = f"{prefix}{n}"
        dpath = os.path.join(base_root, dname)
        if not os.path.exists(dpath):
            os.makedirs(dpath, exist_ok=True)
            return dpath, False
        n += 1

def meta_matcher_all_fields(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    sig_digits: int = 7,
) -> bool:
    """
    Require a and b to be identical in structure and values, except floats which
    are compared with ~sig_digits significant-digit tolerance (recursively).
    
    - Dicts: same key set, values match recursively.
    - Lists/Tuples: same length, elements match recursively (tuple/list treated as same sequence type).
    - Numbers: float/int compared numerically with tolerance; bool stays strict.
    - Everything else: compared with ==.
    """

    # tolerance ~ 0.5 * 10^(-sig_digits) relative (e.g. 7 digits -> ~5e-8)
    rel_tol = 0.5 * 10 ** (-sig_digits)

    def is_number(x: Any) -> bool:
        # bool is a subclass of int; keep it strict, not numeric.
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def float_close(x: float, y: float) -> bool:
        # Handle NaNs/infs explicitly
        if math.isnan(x) or math.isnan(y):
            return math.isnan(x) and math.isnan(y)
        if math.isinf(x) or math.isinf(y):
            return x == y
        # Significant-digit-ish tolerance:
        # abs(x-y) <= rel_tol * max(1, |x|, |y|)
        scale = max(1.0, abs(x), abs(y))
        return abs(x - y) <= rel_tol * scale

    def eq(x: Any, y: Any) -> bool:
        if x is y:
            return True

        # Dicts: same keys, compare each value
        if isinstance(x, dict) and isinstance(y, dict):
            if x.keys() != y.keys():
                return False
            return all(eq(x[k], y[k]) for k in x.keys())

        # Sequences: list/tuple treated equivalently
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            if len(x) != len(y):
                return False
            return all(eq(xi, yi) for xi, yi in zip(x, y))

        # Numpy Arrays
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if x.shape != y.shape:
                return False
            if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
                return bool(np.allclose(x, y, rtol=rel_tol, atol=rel_tol, equal_nan=True))
            else:
                return bool(np.array_equal(x, y))

        # Numeric tolerance
        if is_number(x) and is_number(y):
            return float_close(float(x), float(y))

        # Custom Objects: compare __dict__ if both have it
        if hasattr(x, '__dict__') and hasattr(y, '__dict__'):
            if type(x) is not type(y):
                return False
            x_dict = x.__dict__
            y_dict = y.__dict__
            return eq(x_dict, y_dict)

        # Everything else exact
        try:
            return bool(x == y)
        except Exception:
            return False

    return eq(a, b)


def setup_3D_Eigen_results_directory(
    hamiltonian,
    kx_range, ky_range, kz_range,
    mesh_shape,
    include_endpoints=True,
    force_new=False,
    kvals_mode: str = "endpoints",  # or "centered"
):
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "3D_Eigen_results", Hamiltonian_name)

    required_files = ["eigenvalues_3d.npy", "eigenvectors_3d.npy", "meta.json", "meta_info.pkl"]

    # Get Hamiltonian parameters natively as a dictionary
    if hasattr(hamiltonian, "get_parameters_dict"):
        ham_params = hamiltonian.get_parameters_dict(parameter="3D")
    else:
        ham_params = {}

    meta_target = {
        "hamiltonian_name": Hamiltonian_name,
        "hamiltonian_params": ham_params,
        "include_endpoints": bool(include_endpoints),
        "k_range": float(max(abs(kx_range[0]), abs(kx_range[1]))),  # or store all 3 ranges below
        "kvals_mode": str(kvals_mode),
        "kx_range": [float(kx_range[0]), float(kx_range[1])],
        "ky_range": [float(ky_range[0]), float(ky_range[1])],
        "kz_range": [float(kz_range[0]), float(kz_range[1])],
    }

    dir_path, used = pick_or_create_dataset_dir(
        base_root,
        meta_target=meta_target,
        required_files=required_files,
        meta_matcher=meta_matcher_all_fields,  # or meta_matcher_exact
        force_new=force_new,
        prefix="data_set_",
        start_index=1,
    )

    file_paths = {
        "eigenvalues": os.path.join(dir_path, "eigenvalues_3d.npy"),
        "eigenfunctions": os.path.join(dir_path, "eigenvectors_3d.npy"),
        "meta_json": os.path.join(dir_path, "meta.json"),
        "meta_pkl": os.path.join(dir_path, "meta_info.pkl"),
    }


    print(("Using existing 3D Eigen results directory: " if used else "Created new 3D Eigen results directory: ") + dir_path)
    return file_paths, used, dir_path, meta_target


def setup_3D_QGT_results_directory(
    hamiltonian,
    kx_range, ky_range, kz_range,
    mesh_shape,
    include_endpoints=True,
    force_new=False,
    kvals_mode: str = "endpoints",
    *,
    # NEW: include these in meta-matching so ALL-bands runs don't collide with single-band runs
    method_name: str = "numerical",
    band_index="ALL",          # int or "ALL"
    n_bands=None,              # required if band_index == "ALL"
):
    """
    Creates (or reuses) a results directory for 3D QGT computations.

    Supports two modes:
      - band_index is an int: single-band results saved (still as .npy arrays).
      - band_index == "ALL": stacked results saved with shape (n_bands, nx, ny, nz).

    Returns:
      file_paths: dict of output file paths
      used_existing: bool
      dir_path: str
      meta_target: dict used for matching/saving
    """
    nx, ny, nz = map(int, mesh_shape)

    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "3D_QGT_results", Hamiltonian_name)

    # Get Hamiltonian parameters natively as a dictionary
    if hasattr(hamiltonian, "get_parameters_dict"):
        ham_params = hamiltonian.get_parameters_dict(parameter="3D")
    else:
        ham_params = {}

    # Normalize band_index
    band_key = band_index
    if isinstance(band_key, str):
        band_key = band_key.upper()
    is_all = (band_key == "ALL")

    if is_all:
        if n_bands is None:
            raise ValueError("setup_3D_QGT_results_directory: n_bands must be provided when band_index='ALL'")
        n_bands = int(n_bands)
    else:
        # Single band: store as int
        band_index = int(band_index)

    # Match target (what defines this dataset)
    meta_target = {
        "hamiltonian_name": str(Hamiltonian_name),
        "hamiltonian_params": ham_params,
        "mesh_shape": [nx, ny, nz],
        "include_endpoints": bool(include_endpoints),
        "kvals_mode": str(kvals_mode),
        "kx_range": [float(kx_range[0]), float(kx_range[1])],
        "ky_range": [float(ky_range[0]), float(ky_range[1])],
        "kz_range": [float(kz_range[0]), float(kz_range[1])],

        # NEW: make meta matching robust across methods and band modes
        "method_name": str(method_name),
        "band_index": ("ALL" if is_all else int(band_index)),
        "n_bands": (int(n_bands) if is_all else None),
    }

    # Required files for reuse
    required_files = [
        "g_xx.npy", "g_yy.npy", "g_zz.npy",
        "g_xy_real.npy", "g_xy_imag.npy",
        "g_xz_real.npy", "g_xz_imag.npy",
        "g_yz_real.npy", "g_yz_imag.npy",
        "trace.npy",          # NEW
        "meta.json",          # for matching
        "meta_info.pkl",      # for loading Hamiltonian_Obj later if you want
    ]

    # If your meta_matcher treats None fields strictly, it might fail matches between
    # (single band) and (ALL bands). That's GOOD — we *want* them separated.
    dir_path, used = pick_or_create_dataset_dir(
        base_root,
        meta_target=meta_target,
        required_files=required_files,
        meta_matcher=meta_matcher_all_fields,
        force_new=force_new,
        prefix="data_set_",
        start_index=1,
    )

    file_paths = {
        "g_xx": os.path.join(dir_path, "g_xx.npy"),
        "g_yy": os.path.join(dir_path, "g_yy.npy"),
        "g_zz": os.path.join(dir_path, "g_zz.npy"),
        "g_xy_real": os.path.join(dir_path, "g_xy_real.npy"),
        "g_xy_imag": os.path.join(dir_path, "g_xy_imag.npy"),
        "g_xz_real": os.path.join(dir_path, "g_xz_real.npy"),
        "g_xz_imag": os.path.join(dir_path, "g_xz_imag.npy"),
        "g_yz_real": os.path.join(dir_path, "g_yz_real.npy"),
        "g_yz_imag": os.path.join(dir_path, "g_yz_imag.npy"),
        "trace": os.path.join(dir_path, "trace.npy"),          # NEW
        "meta_json": os.path.join(dir_path, "meta.json"),
        "meta_pkl": os.path.join(dir_path, "meta_info.pkl"),
    }

    print(("Using existing 3D QGT results directory: " if used else "Created new 3D QGT results directory: ") + dir_path)
    return file_paths, used, dir_path, meta_target

def setup_sym_points_results_directory(
    hamiltonian,
    path_points,
    path_labels,
    num_points_per_segment,
    force_new=False
):
    """
    Create or reuse a results directory for symmetry-path band structure calculations.
    Works for both 2D and 3D paths — the dimensionality is inferred from path_points.
    """
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "Sym_Points_results", Hamiltonian_name)

    required_files = ["eigenvalues.npy", "meta.json", "meta_info.pkl"]

    path_points_list = [np.array(p).tolist() for p in path_points]

    if hasattr(hamiltonian, "get_parameters_dict"):
        ham_params = hamiltonian.get_parameters_dict(parameter="sym")
    else:
        ham_params = {}

    meta_target = {
        "hamiltonian_name": Hamiltonian_name,
        "hamiltonian_params": ham_params,
        "path_labels": path_labels,
        "num_points_per_segment": num_points_per_segment,
        "path_points": path_points_list,
    }

    dir_path, used = pick_or_create_dataset_dir(
        base_root,
        meta_target=meta_target,
        required_files=required_files,
        meta_matcher=meta_matcher_all_fields,
        force_new=force_new,
        prefix="data_set_",
        start_index=1,
    )

    file_paths = {
        "eigenvalues": os.path.join(dir_path, "eigenvalues.npy"),
        "meta_json":   os.path.join(dir_path, "meta.json"),
        "meta_pkl":    os.path.join(dir_path, "meta_info.pkl"),
    }

    print(("Using existing Sym Points results directory: " if used else "Created new Sym Points results directory: ") + dir_path)
    return file_paths, used, dir_path, meta_target


def setup_1D_angles_results_directory(
    hamiltonian,
    k_max,
    num_angles,
    num_points_per_line,
    force_new=False
):
    """
    Create or reuse a results directory for 1D angled line band structure calculations.
    lives under 1D_Angles_results.
    """
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "1D_Angles_results", Hamiltonian_name)

    required_files = ["eigenvalues.npy", "meta.json", "meta_info.pkl"]

    if hasattr(hamiltonian, "get_parameters_dict"):
        ham_params = hamiltonian.get_parameters_dict(parameter="1D_Angles")
    else:
        ham_params = {}

    meta_target = {
        "hamiltonian_name": Hamiltonian_name,
        "hamiltonian_params": ham_params,
        "k_max": k_max,
        "num_angles": num_angles,
        "num_points_per_line": num_points_per_line,
    }

    dir_path, used = pick_or_create_dataset_dir(
        base_root,
        meta_target=meta_target,
        required_files=required_files,
        meta_matcher=meta_matcher_all_fields,
        force_new=force_new,
        prefix="data_set_",
        start_index=1,
    )

    file_paths = {
        "eigenvalues": os.path.join(dir_path, "eigenvalues.npy"),
        "meta_json":   os.path.join(dir_path, "meta.json"),
        "meta_pkl":    os.path.join(dir_path, "meta_info.pkl"),
    }

    print(("Using existing 1D Angles results directory: " if used else "Created new 1D Angles results directory: ") + dir_path)
    return file_paths, used, dir_path, meta_target
