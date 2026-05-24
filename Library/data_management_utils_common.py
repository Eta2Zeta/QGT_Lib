import os, re, json, math, pickle
from typing import Optional, Iterable, Callable, Tuple, Dict, Any
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def dump_metadata(meta_dict: dict, file_path: str):
    with open(file_path, "w") as f:
        json.dump(meta_dict, f, indent=4, cls=NumpyEncoder)

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

def pick_or_create_result_dir_simple(
    base_root: str,
    base_name: str = "dataset_",
    required_params: Optional[dict] = None,
    force_new: bool = False,
    required_files: Optional[list] = None,
) -> Tuple[str, bool]:
    """
    Creates or picks a directory named '{base_name}1', '{base_name}2', ...
    
    Logic:
    1. Scan for directories matching pattern f"{base_name}N".
    2. Ideally sorted by N.
    3. If not force_new and required_params provided:
       - Open 'parameters.json' in each candidate.
       - If param dict matches required_params (using float tolerance), REUSE that dir.
       
    4. If no match found (or force_new), find first available N.
       - Create default empty dir.
       - Caller is responsible for writing parameters.json immediately after.
       
    Returns: (dir_path, used_existing)
    """
    os.makedirs(base_root, exist_ok=True)
    
    # 1. Gather candidates
    candidates = []
    if os.path.exists(base_root):
        for d in os.listdir(base_root):
            if d.startswith(base_name):
                try:
                    # Extract number suffix
                    suffix = d[len(base_name):]
                    if suffix.isdigit():
                        n = int(suffix)
                        candidates.append((n, d))
                except ValueError:
                    pass
    
    # Sort by number
    candidates.sort(key=lambda x: x[0])
    
    # 2. Check for reuse
    if not force_new and required_params is not None:
        for n, d_name in candidates:
            d_path = os.path.join(base_root, d_name)
            json_path = os.path.join(d_path, "meta.json")
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        existing_params = json.load(f)
                    
                    match = True
                    for k, v in required_params.items():
                        if k not in existing_params:
                            match = False
                            break
                        if not meta_matcher_all_fields(existing_params[k], v):
                            match = False
                            break
                    
                    if match:
                        if required_files:
                            files_exist = all(os.path.exists(os.path.join(d_path, f)) for f in required_files)
                            if not files_exist:
                                continue
                        return d_path, True
                        
                except Exception as e:
                    print(f"Warning: Failed to read {json_path}: {e}")
                    continue

    # 3. Create new dir
    n = 1
    if candidates:
        n = candidates[-1][0] + 1
        
    while True:
        dname = f"{base_name}{n}"
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

def get_mapped_axis(order: str, axis: str) -> str:
    """
    Maps an axis ('x', 'y', 'z') to an index-based character ('i', 'j', 'k')
    based on the given `order`.
    """
    ijk = ["i", "j", "k"]
    if axis in order:
        idx = order.index(axis)
        if idx < len(ijk):
            return ijk[idx]
    return axis

def read_meta_axis(meta_info: dict, prefix: str, axis: str, suffix: str = ""):
    """
    Reads a parameter from meta_info mapping the axis to i, j, k.
    Example: read_meta_axis(meta_info, 'k', 'x', '_range') 
    checks order, e.g., 'xyz' -> 'x' is 'i', returns meta_info['ki_range']
    """
    order = meta_info.get("order", "xyz")
    mapped_char = get_mapped_axis(order, axis)
    key = f"{prefix}{mapped_char}{suffix}"
    
    # Fallback if mapped key doesn't exist but the exact requested one does
    if key not in meta_info and f"{prefix}{axis}{suffix}" in meta_info:
        return meta_info[f"{prefix}{axis}{suffix}"]
        
    return meta_info[key]

def write_meta_axis(meta_info: dict, prefix: str, axis: str, value: Any, suffix: str = ""):
    """
    Writes a parameter to meta_info mapping the axis to i, j, k.
    Example: write_meta_axis(meta_info, 'k', 'x', kx_array, '_range')
    checks order, outputs to meta_info['ki_range'].
    """
    order = meta_info.get("order", "xyz")
    mapped_char = get_mapped_axis(order, axis)
    key = f"{prefix}{mapped_char}{suffix}"
    meta_info[key] = value

