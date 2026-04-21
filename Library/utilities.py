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







# =============================================================================
# NEW: Simplified Dataset Storage with JSON Metadata
# =============================================================================

import json

def pick_or_create_result_dir_simple(
    base_root: str,
    base_name: str = "dataset",
    required_params: Optional[dict] = None,
    force_new: bool = False,
) -> Tuple[str, bool]:
    """
    Creates or picks a directory named '{base_name}1', '{base_name}2', ...
    
    Logic:
    1. Scan for directories matching pattern f"{base_name}N".
    2. Ideally sorted by N.
    3. If not force_new and required_params provided:
       - Open 'parameters.json' in each candidate.
       - If param dict matches required_params, REUSE that dir.
       
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
            json_path = os.path.join(d_path, "parameters.json")
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        existing_params = json.load(f)
                    
                    # Compare dictionaries (subset match logic)
                    match = True
                    for k, v in required_params.items():
                        if k not in existing_params:
                            match = False
                            break
                        # Simple equality check
                        if existing_params[k] != v:
                            match = False
                            break
                    
                    if match:
                        return d_path, True
                        
                except Exception as e:
                    print(f"Warning: Failed to read {json_path}: {e}")
                    continue

    # 3. Create new
    if candidates:
        last_n = candidates[-1][0]
        new_n = last_n + 1
    else:
        new_n = 1
        
    new_dir_name = f"{base_name}{new_n}"
    new_dir_path = os.path.join(base_root, new_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)
    
    # Write parameters.json immediately
    if required_params:
        with open(os.path.join(new_dir_path, "parameters.json"), 'w') as f:
            json.dump(required_params, f, indent=4, default=str)
            
    return new_dir_path, False




def centered_kvals(k_range: float, N: int) -> np.ndarray:
    """
    Return N cell-centered k points spanning [-k_range, +k_range], symmetric about 0.

    Uses the "edges -> centers" construction:
      edges   = linspace(-k_range, +k_range, N+1)
      centers = 0.5*(edges[:-1] + edges[1:])
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    edges = np.linspace(-k_range, k_range, N + 1, endpoint=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers

def generate_3d_sym_lines(num_points_per_segment=100, space_group=58, a=1.0):
    """
    Legacy routine, symmetry points should be moved to the Hamiltonian Objects. 

    Generates k-points along a high-symmetry path for the requested space group.

    Parameters
    ----------
    num_points_per_segment : int
        Number of k-points *per segment* (exclusive of the start point of
        each segment, inclusive of the end point).
    space_group : int
        Crystal space group number.
        - 58  : Orthorhombic (Pnnm).  Path: Z-G-S-R-T-Y-S-X-U-R-Z.
                Points use units of π (a = b = c = 1 implied).
        - 194 : Hexagonal (P6₃/mmc).  Path: G-M-K-G-A-L-H-A.
                In-plane coordinates use lattice constant `a`;
                out-of-plane coordinate uses π (c-axis half-reciprocal-lattice).
    a : float
        Lattice constant for space group 194 (default 1.0).
        Ignored for space group 58.

    Returns
    -------
    all_k_points : ndarray, shape (N, 3)
        k-points along the path.
    all_k_dist : ndarray, shape (N,)
        Cumulative distance along the path.
    node_indices : list of int
        Indices into the arrays where the path reaches a high-symmetry point.
    path_labels : list of str
        Name of each high-symmetry node in sequence (length = len(node_indices)).
    path_points : ndarray, shape (M, 3)
        Cartesian k-coordinates of the nodes in sequence.
    """

    # ------------------------------------------------------------------
    # Space group 58 – Orthorhombic (Pnnm)
    # ------------------------------------------------------------------
    if space_group == 58:
        points = {
            "G": np.array([0,      0,      0     ]),
            "X": np.array([np.pi,  0,      0     ]),
            "S": np.array([np.pi,  np.pi,  0     ]),
            "Y": np.array([0,      np.pi,  0     ]),
            "Z": np.array([0,      0,      np.pi ]),
            "U": np.array([np.pi,  0,      np.pi ]),
            "R": np.array([np.pi,  np.pi,  np.pi ]),
            "T": np.array([0,      np.pi,  np.pi ]),
        }
        path_labels = ["Z", "G", "S", "R", "T", "Y", "S", "X", "U", "R", "Z"]

    # ------------------------------------------------------------------
    # Space group 194 – Hexagonal (P6₃/mmc)
    # ------------------------------------------------------------------
    elif space_group == 194:
        #   M-point : (0,       2π/√3,    0)   ← (0, 2/√3) × π
        #   K-point : (2π/3,    2π/√3,    0)   ← (2/3, 2/√3) × π
        #   A       : (0,       0,         π)   ← Γ shifted by kz = π
        #   L       : (0,       2π/√3,    π)   ← M shifted by kz = π
        #   H       : (2π/3,    2π/√3,    π)   ← K shifted by kz = π
        kM_y = 2 * np.pi / np.sqrt(3)   # = (2/√3) × π
        kK_x = 2 * np.pi / 3            # = (2/3)  × π
        kz_A = np.pi

        kB_x = np.pi
        kB_y = np.pi / np.sqrt(3)

        points = {
            "G": np.array([0,     0,      0    ]),
            "M": np.array([0,     kM_y,   0    ]),
            "K": np.array([kK_x,  kM_y,   0    ]),
            "A": np.array([0,     0,      kz_A ]),
            "L": np.array([0,     kM_y,   kz_A ]),
            "H": np.array([kK_x,  kM_y,   kz_A ]),
            "U": np.array([0,     kM_y,   kz_A / 2]), # Between L and M
            "D": np.array([0,     0,      kz_A / 2]), # Between A and G
            "P": np.array([kK_x,  kM_y,   kz_A / 2]), # Between H and K
            "B": np.array([kB_x,  kB_y,   kz_A ]),
            "C": np.array([kB_x,  kB_y,   kz_A / 2]),
            "E": np.array([kB_x,  kB_y,   0    ]),
        }
        # Standard hexagonal path:  G -> M -> K -> G -> A -> L -> H -> A
        # path_labels = ["G", "M", "K", "G", "A", "L", "H", "A"]
        
        # Old requested path: L -> H -> A -> L -> M -> H -> K -> M -> G -> K -> A -> G -> L
        # path_labels = ["L", "H", "A", "L", "M", "H", "K", "M", "G", "K", "A", "G", "L"]
        
        # Old requested path: LAHLUHPUDPKUMGKMADG
        # path_labels = list("LAHLUHPUDPKUMGKMADG")
        
        # Newest requested path
        path_labels = list("ALHABHPLUADUPDCPBCAPKDGMKGEKUMDECK")

    else:
        raise ValueError(
            f"space_group={space_group} is not supported. "
            "Currently supported: 58 (orthorhombic), 194 (hexagonal)."
        )

    # ------------------------------------------------------------------
    # Build path (common logic)
    # ------------------------------------------------------------------
    path_points = [points[label] for label in path_labels]

    all_k_points = []
    all_k_dist   = []
    node_indices  = []
    cum_dist      = 0.0

    # First node
    all_k_points.append(path_points[0])
    all_k_dist.append(cum_dist)
    node_indices.append(0)

    for i in range(len(path_points) - 1):
        start = path_points[i]
        end   = path_points[i + 1]

        dist  = np.linalg.norm(end - start)
        pts   = np.linspace(start, end, num_points_per_segment + 1)[1:]
        dists = np.linspace(0, dist,  num_points_per_segment + 1)[1:]

        for p, d in zip(pts, dists):
            all_k_points.append(p)
            all_k_dist.append(cum_dist + d)

        cum_dist += dist
        node_indices.append(len(all_k_points) - 1)

    return (
        np.array(all_k_points),
        np.array(all_k_dist),
        node_indices,
        path_labels,
        np.array(path_points),
    )


def generate_1d_lines_at_angles(k_max, num_angles, num_points_per_line):
    """
    Generate 1D straight-line k-paths passing through the origin at various angles.
    
    The lines span from -k_max to +k_max. The angle theta varies from 0 to pi
    (since [-k_max, k_max] along [0, pi) covers the full 2D plane).
    
    Parameters
    ----------
    k_max : float
        Maximum magnitude of k along the line.
    num_angles : int
        Number of distinct angles to sample between 0 and pi (exclusive of pi).
    num_points_per_line : int
        Number of k-points requested along a single line.
        
    Returns
    -------
    k_path : ndarray, shape (num_angles * num_points_per_line, 2)
        Flattened array of all k-coordinates for all angles.
    k_vals : ndarray, shape (num_points_per_line,)
        The 1D signed distance along the line (-k_max to k_max).
    angles : ndarray, shape (num_angles,)
        The angles sampled (in radians).
    """
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    k_vals = np.linspace(-k_max, k_max, num_points_per_line)
    
    k_path_list = []
    
    for theta in angles:
        kx = k_vals * np.cos(theta)
        ky = k_vals * np.sin(theta)
        # Stack into (num_points_per_line, 2)
        line_k_points = np.column_stack((kx, ky))
        k_path_list.append(line_k_points)
        
    # Flatten across the angles dimension: shape is (num_angles * num_points_per_line, 2)
    k_path = np.vstack(k_path_list)
    
    return k_path, k_vals, angles
