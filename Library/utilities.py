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
        "meta_info": "meta_info.pkl",
    }.items()}

    print(("Using existing results directory: " if used else "Created new results directory: ") + dir_path)
    return file_paths, used, dir_path


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



def setup_QGT_results_directory(hamiltonian, kx_range, ky_range, mesh_spacing, force_new=False, method_name=None):
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "2D_QGT_results", Hamiltonian_name)

    # Sanitize method name if provided
    method_str = f"_{method_name}" if method_name else ""

    base_name = re.sub(
        r'[^\w.-]', '_',
        f"QGT{method_str}_{hamiltonian.get_filename(parameter='2D')}_"
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


def setup_QGT_results_directory_2D_omega_range(
    hamiltonian,
    kx_range,
    ky_range,
    mesh_spacing,
    omega_min,
    omega_max,
    num_omega_points,
    spacing,
    band,                 # <— NEW: required
    force_new=False,
):
    Hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "2D_QGT_omega_sweep", Hamiltonian_name)

    base_name = re.sub(
        r'[^\w.-]', '_',
        f"{hamiltonian.get_filename(parameter='2D')}_"
        f"kx{kx_range[0]:.2f}_{kx_range[1]:.2f}_"
        f"ky{ky_range[0]:.2f}_{ky_range[1]:.2f}_mesh{mesh_spacing}_"
        f"omega{omega_min:.2e}_{omega_max:.2e}_spacing_{spacing}_points{num_omega_points}_"
        f"band{int(band)}"   # <— include band in dir name
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
    parameter_spacing,   # int OR {name: {"count":N,"scale":"linear|log|inv-linear|inv-log"}}
    kx_range,
    ky_range,
    mesh_spacing,
    decimals=3,
    force_new=False
):
    """
    Create (or reuse) the root dir to hold a single N-D QGT npz bundle.
    Reuse only when 'qgt_nd_bundle.npz' is present; otherwise make a new numbered dir.

    parameter_spacing: unified spec (matches builder)
      - int -> same count for all params, linear
      - dict -> per-parameter spec; each value may be:
          * int                      -> count, linear
          * {"count": N}             -> linear
          * {"count": N, "scale": S} -> S in {"linear","log","inv-linear","inv-log"}

    Returns: (root_dir, used_existing)
    """
    # -------- helpers --------
    def _sanitize(name: str) -> str:
        return re.sub(r"[^\w.\-]", "_", str(name))

    def _norm_ranges(ranges):
        if isinstance(ranges, dict):
            items = sorted(ranges.items(), key=lambda kv: kv[0])
            return [(k, float(v[0]), float(v[1])) for k, v in items]
        # iterable of (name, min, max)
        items = sorted([(n, float(a), float(b)) for (n, a, b) in ranges], key=lambda x: x[0])
        return items

    def _parse_spacing_spec(spec):
        """
        Returns (count:int, scale_token:str)
          scale_token ∈ {"linear","log","inv-linear","inv-log"}
        """
        # default
        count = None
        scale = "linear"

        if isinstance(spec, int):
            count = int(spec)
        elif isinstance(spec, dict):
            # accept a few key aliases
            count = int(spec.get("count", spec.get("n", spec.get("points", 1))))
            scale = str(spec.get("scale", spec.get("spacing", "linear"))).lower().strip()
        else:
            raise ValueError(f"Unrecognized spacing spec: {spec!r}")

        # normalize scale aliases
        alias = {
            "lin": "linear",
            "log10": "log",
            "inv": "inv-linear",  # if user wrote just "inv"
            "inverse": "inv-linear",
            "inverse-linear": "inv-linear",
            "inverse-log": "inv-log",
            "invlog": "inv-log",
            "invlin": "inv-linear",
        }
        scale = alias.get(scale, scale)
        if scale not in {"linear", "log", "inv-linear", "inv-log"}:
            raise ValueError(f"Unsupported scale '{scale}' (use 'linear','log','inv-linear','inv-log').")

        return count, scale

    # -------- base paths / names --------
    Hname = getattr(hamiltonian_template, "name", "Hamiltonian")
    base_root = os.path.join(os.getcwd(), "results", "QGT_ND", _sanitize(Hname))

    try:
        h_prefix = str(hamiltonian_template.get_filename(parameter="ND"))
    except Exception:
        h_prefix = Hname
    h_prefix = _sanitize(h_prefix)

    # -------- normalize ranges --------
    range_items = _norm_ranges(param_ranges)  # [(name, vmin, vmax), ...]
    range_parts = [f"{n}_{vmin:.{decimals}f}_{vmax:.{decimals}f}" for (n, vmin, vmax) in range_items]

    # -------- normalize spacing --------
    if isinstance(parameter_spacing, int):
        spacing_map = {n: {"count": int(parameter_spacing), "scale": "linear"} for (n, _, _) in range_items}
    elif isinstance(parameter_spacing, dict):
        spacing_map = {}
        for (n, _, _) in range_items:
            spec = parameter_spacing.get(n, 1)  # default linear
            cnt, scl = _parse_spacing_spec(spec)
            spacing_map[n] = {"count": cnt, "scale": scl}
    else:
        raise ValueError("parameter_spacing must be int or dict (per-parameter spec).")

    # For the label, compress "inv-linear" -> "invlinear", "inv-log" -> "invlog" for filesystem neatness
    def _label_scale(s):
        return s.replace("-", "")

    spacing_parts = [
        f"{n}_{spacing_map[n]['count']}_{_label_scale(spacing_map[n]['scale'])}"
        for (n, _, _) in range_items
    ]

    label_ranges  = "RANGES["  + "-".join(range_parts)   + "]"
    label_spacing = "SPACING[" + "-".join(spacing_parts) + "]"
    klabel = f"kx{kx_range[0]:.2f}_{kx_range[1]:.2f}__ky{ky_range[0]:.2f}_{ky_range[1]:.2f}__mesh{mesh_spacing}"

    base_name = _sanitize(f"{h_prefix}-{label_ranges}-{label_spacing}-{klabel}")

    # -------- reuse/create --------
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
