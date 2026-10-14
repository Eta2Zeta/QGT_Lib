import numpy as np


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




def generate_2d_sym_lines(Hamiltonian_Obj, num_points_per_segment=100):
    """
    Generate interpolated k-points along the high-symmetry path of a 2D Hamiltonian.

    The symmetry points and path order are obtained from
    ``Hamiltonian_Obj.get_sym_path()``, so the geometry is always
    self-consistent with the model's own Brillouin zone definition.

    Parameters
    ----------
    Hamiltonian_Obj : hamiltonian
        A Hamiltonian instance that implements ``get_sym_path()``.
        The method must return ``(sym_points, path_labels)`` where
        ``sym_points`` is a dict mapping label -> 1-D array of length 2
        and ``path_labels`` is an ordered list of labels (may repeat).
    num_points_per_segment : int
        Number of k-points per segment, exclusive of the start point and
        inclusive of the end point (same convention as generate_3d_sym_lines).

    Returns
    -------
    all_k_points : ndarray, shape (N, 2)
        Interpolated k-points along the full path.
    all_k_dist : ndarray, shape (N,)
        Cumulative arc-length distance along the path (useful as x-axis
        when plotting band structures).
    node_indices : list of int
        Indices into ``all_k_points`` / ``all_k_dist`` where the path
        lands on a high-symmetry node.  Length equals ``len(path_labels)``.
    path_labels : list of str
        Ordered label sequence as returned by ``get_sym_path()``
        (may include repeated entries, e.g. ``["G","M","X","G"]``).
    path_points : ndarray, shape (M, 2)
        Cartesian k-coordinates of the nodes in sequence.

    Example
    -------
    >>> H = SquareLatticeHamiltonian()
    >>> k_pts, k_dist, node_idx, labels, nodes = generate_2d_sym_lines(H)
    """
    sym_points, path_labels = Hamiltonian_Obj.get_sym_path()

    # Resolve labels → coordinate arrays (shape (2,) each)
    path_points = [np.asarray(sym_points[label], dtype=float)
                   for label in path_labels]

    # ------------------------------------------------------------------
    # Build path (same logic as generate_3d_sym_lines)
    # ------------------------------------------------------------------
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
        np.array(all_k_points),   # (N, 2)
        np.array(all_k_dist),     # (N,)
        node_indices,             # list of int, length = len(path_labels)
        path_labels,              # list of str
        np.array(path_points),    # (M, 2)
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
