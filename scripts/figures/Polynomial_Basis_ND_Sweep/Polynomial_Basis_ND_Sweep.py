"""Standalone real-time polar viewer for a two-component polynomial basis."""

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons, Slider
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qgt.io.paths import legacy_dir


OUTPUT_DIR = Path(legacy_dir("Plot_Specific", "Polynomial_Basis_ND_Sweep"))

# Coefficient defaults. Parameters without a range remain fixed at these values.
COEFFICIENT_DEFAULTS = {
    "a": 1.0,
    "b": 0.0,
    "c": 0.0,
    "d": 0.0,
    "e": 0.0,
    "f": 0.0,
    "g": 0.0,
    "h": 0.0,
    "i": 0.0,
}

# Every basis pair can also be multiplied by (r^2)^n, where n is a
# nonnegative integer. These controls therefore generate only even powers of r.
RADIAL_POWER_DEFAULTS = {
    "a": 0,
    "b": 0,
    "c": 0,
    "d": 0,
    "e": 0,
    "f": 0,
    "g": 0,
    "h": 0,
    "i": 0,
}
RADIAL_POWER_RANGE = (0, 6)
BERRY_LIM_PERCENTILE = 60.0
SIGN_FILTER_COMPONENTS = True

# Interactive coefficient controls.
COEFFICIENT_STEP = 0.1
param_ranges = {
    "a": (-4.0, 4.0),
    "b": (-4.0, 4.0),
    "c": (-4.0, 4.0),
    "d": (-4.0, 4.0),
    "e": (-4.0, 4.0),
    "f": (-4.0, 4.0),
    "g": (-4.0, 4.0),
    "h": (-4.0, 4.0),
    "i": (-4.0, 4.0),
}

# Polar sampling setup. The two earlier modes use Z_VALUE; the constrained
# D6h/E2g mode is evaluated strictly on z=0 and ignores it.
R_RANGE = (0.0, 2.0)
N_R = 120
N_PHI = 240
Z_VALUE = 1.0

D6H_E2G_MODE = "D6h / E2g"
D3D_EG_MODE = "D3d / Eg"
D6H_E2G_SYMMETRY_MODE = "D6h/E2g symmetry constrained"
BASIS_MODES = (D6H_E2G_MODE, D3D_EG_MODE, D6H_E2G_SYMMETRY_MODE)
DEFAULT_BASIS_MODE = D6H_E2G_MODE
D6H_BASIS_NAMES = tuple("abcde")
D3D_BASIS_NAMES = tuple(COEFFICIENT_DEFAULTS)

# Finite, real-time truncation of each invariant scalar power series. Increase
# this value to expose more independently controlled terms in both sums.
SYMMETRY_TERMS_PER_FAMILY = 3
SYMMETRY_A_TERMS = tuple(
    f"a{index}" for index in range(SYMMETRY_TERMS_PER_FAMILY)
)
SYMMETRY_B_TERMS = tuple(
    f"b{index}" for index in range(SYMMETRY_TERMS_PER_FAMILY)
)
SYMMETRY_TERM_NAMES = SYMMETRY_A_TERMS + SYMMETRY_B_TERMS
SYMMETRY_COEFFICIENT_DEFAULTS = {
    name: 1.0 if name == "a0" else 0.0 for name in SYMMETRY_TERM_NAMES
}
SYMMETRY_COEFFICIENT_RANGES = {
    name: (-4.0, 4.0) for name in SYMMETRY_TERM_NAMES
}
SYMMETRY_N_DEFAULTS = {
    name: 1 if name.endswith("1") else 0 for name in SYMMETRY_TERM_NAMES
}
SYMMETRY_Q_DEFAULTS = {
    name: 1 if name.endswith("2") else 0 for name in SYMMETRY_TERM_NAMES
}
SYMMETRY_N_RANGE = (0, 6)
SYMMETRY_Q_RANGE = (0, 4)


D3D_BASIS_LABELS = {
    "a": {"x": "C_2 = x^2-y^2", "y": "S_2 = 2xy"},
    "b": {"x": "zC_1 = xz", "y": "zS_1 = yz"},
    "c": {
        "x": "C_4 = (x^2-y^2)^2-4x^2y^2",
        "y": "S_4 = 4xy(x^2-y^2)",
    },
    "d": {"x": "z^2C_2", "y": "z^2S_2"},
    "e": {"x": "z^3C_1", "y": "z^3S_1"},
    "f": {
        "x": "zC_5",
        "y": "zS_5",
    },
    "g": {"x": "z^2C_4", "y": "z^2S_4"},
    "h": {"x": "z^4C_2", "y": "z^4S_2"},
    "i": {"x": "z^5C_1", "y": "z^5S_1"},
}

D6H_BASIS_LABELS = {
    "a": {"x": "C_2 = x^2-y^2", "y": "S_2 = 2xy"},
    "b": {
        "x": "C_4 = (x^2-y^2)^2-4x^2y^2",
        "y": "S_4 = 4xy(x^2-y^2)",
    },
    "c": {"x": "z^2C_2", "y": "z^2S_2"},
    "d": {"x": "z^2C_4", "y": "z^2S_4"},
    "e": {"x": "z^4C_2", "y": "z^4S_2"},
}

SYMMETRY_CONSTRAINED_BASIS_LABELS = {
    "h2": {"x": "C_2 = x^2-y^2", "y": "S_2 = 2xy"},
    "h4": {
        "x": "C_4 = (x^2-y^2)^2-4x^2y^2",
        "y": "-S_4 = -4xy(x^2-y^2)",
    },
    "invariant": {
        "r_squared": "x^2+y^2",
        "C_6": "x^6-15x^4y^2+15x^2y^4-y^6",
    },
}

# Keep the original D3d names available to callers of this standalone script.
BASIS_LABELS = D3D_BASIS_LABELS

D3D_EQUATION_LINES = (
    r"$C_m=r^m\cos(m\phi),\qquad S_m=r^m\sin(m\phi),\qquad "
    r"r^2=x^2+y^2$",
    r"$\mathbf{P}=(P_x,P_y),\qquad n_a,n_b,\ldots,n_i"
    r"\in\mathbb{Z}_{\geq 0}$",
    r"$\mathbf{P}=a(r^2)^{n_a}(C_2,S_2)"
    r"+b(r^2)^{n_b}z(C_1,S_1)"
    r"+c(r^2)^{n_c}(C_4,S_4)$",
    r"$\quad+d(r^2)^{n_d}z^2(C_2,S_2)"
    r"+e(r^2)^{n_e}z^3(C_1,S_1)"
    r"+f(r^2)^{n_f}z(C_5,S_5)$",
    r"$\quad+g(r^2)^{n_g}z^2(C_4,S_4)"
    r"+h(r^2)^{n_h}z^4(C_2,S_2)"
    r"+i(r^2)^{n_i}z^5(C_1,S_1)$",
)

D6H_EQUATION_LINES = (
    r"$C_m=r^m\cos(m\phi),\qquad S_m=r^m\sin(m\phi),\qquad "
    r"r^2=x^2+y^2$",
    r"$\mathbf{P}=(P_x,P_y),\qquad n_a,n_b,\ldots,n_e"
    r"\in\mathbb{Z}_{\geq 0}$",
    r"$\mathbf{P}=a(r^2)^{n_a}(C_2,S_2)"
    r"+b(r^2)^{n_b}(C_4,S_4)"
    r"+c(r^2)^{n_c}z^2(C_2,S_2)$",
    r"$\quad+d(r^2)^{n_d}z^2(C_4,S_4)"
    r"+e(r^2)^{n_e}z^4(C_2,S_2)$",
)

SYMMETRY_CONSTRAINED_EQUATION_LINES = (
    r"$C_m=r^m\cos(m\phi),\quad S_m=r^m\sin(m\phi),\quad "
    r"C_6=r^6\cos(6\phi)$",
    r"$\mathbf{h}_2=(C_2,S_2),\qquad\mathbf{h}_4=(C_4,-S_4),\qquad z=0$",
    rf"$\mathbf{{P}}=\sum_{{i=0}}^{{{SYMMETRY_TERMS_PER_FAMILY - 1}}}"
    r"a_i(r^2)^{n_i}(C_6)^{q_i}\mathbf{h}_2"
    rf"+\sum_{{i=0}}^{{{SYMMETRY_TERMS_PER_FAMILY - 1}}}"
    r"b_i(r^2)^{n_i}(C_6)^{q_i}\mathbf{h}_4$",
    r"$a_i,b_i\in\mathbb{R},\qquad n_i,q_i\in\mathbb{Z}_{\geq0}$",
)

# Preserve the original public name as the D3d equation.
EQUATION_LINES = D3D_EQUATION_LINES

BASIS_LABELS_BY_MODE = {
    D6H_E2G_MODE: D6H_BASIS_LABELS,
    D3D_EG_MODE: D3D_BASIS_LABELS,
    D6H_E2G_SYMMETRY_MODE: SYMMETRY_CONSTRAINED_BASIS_LABELS,
}
EQUATION_LINES_BY_MODE = {
    D6H_E2G_MODE: D6H_EQUATION_LINES,
    D3D_EG_MODE: D3D_EQUATION_LINES,
    D6H_E2G_SYMMETRY_MODE: SYMMETRY_CONSTRAINED_EQUATION_LINES,
}
BASIS_NAMES_BY_MODE = {
    D6H_E2G_MODE: D6H_BASIS_NAMES,
    D3D_EG_MODE: D3D_BASIS_NAMES,
    D6H_E2G_SYMMETRY_MODE: SYMMETRY_TERM_NAMES,
}

HARMONIC_TABLE_ROWS = (
    (r"$1$", r"$x$", r"$y$"),
    (r"$2$", r"$x^2-y^2$", r"$2xy$"),
    (
        r"$4$",
        r"$(x^2-y^2)^2-4x^2y^2$",
        r"$4xy(x^2-y^2)$",
    ),
    (
        r"$5$",
        r"$x(x^4-10x^2y^2+5y^4)$",
        r"$y(5x^4-10x^2y^2+y^4)$",
    ),
    (
        r"$6$",
        r"$x^6-15x^4y^2+15x^2y^4-y^6$",
        r"$2xy(3x^4-10x^2y^2+3y^4)$",
    ),
)

HARMONIC_TABLE_ROWS_BY_MODE = {
    D6H_E2G_MODE: HARMONIC_TABLE_ROWS,
    D3D_EG_MODE: HARMONIC_TABLE_ROWS,
    D6H_E2G_SYMMETRY_MODE: (
        HARMONIC_TABLE_ROWS[1],
        HARMONIC_TABLE_ROWS[2],
        HARMONIC_TABLE_ROWS[4],
    ),
}


def make_polar_grid(r_range=R_RANGE, n_r=N_R, n_phi=N_PHI):
    """Return cell-centered polar samples and Cartesian cell boundaries."""
    r_min, r_max = map(float, r_range)
    n_r = int(n_r)
    n_phi = int(n_phi)
    if r_min < 0.0 or r_max <= r_min:
        raise ValueError("r_range must be increasing and nonnegative")
    if n_r < 2 or n_phi < 4:
        raise ValueError("n_r must be at least 2 and n_phi at least 4")

    r_edges = np.linspace(r_min, r_max, n_r + 1)
    phi_edges = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])
    phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    radius, angle = np.meshgrid(r, phi)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    radius_edges, angle_edges = np.meshgrid(r_edges, phi_edges)
    x_edges = radius_edges * np.cos(angle_edges)
    y_edges = radius_edges * np.sin(angle_edges)
    return {
        "r": r,
        "phi": phi,
        "r_edges": r_edges,
        "phi_edges": phi_edges,
        "radius": radius,
        "angle": angle,
        "x": x,
        "y": y,
        "x_edges": x_edges,
        "y_edges": y_edges,
    }


def cartesian_harmonics(x, y):
    """Return the harmonic-normalized Cartesian polynomials used here."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c1, s1 = x, y
    c2 = c1 * c1 - s1 * s1
    s2 = 2.0 * c1 * s1
    c4 = c2 * c2 - s2 * s2
    s4 = 2.0 * c2 * s2
    c5 = c4 * c1 - s4 * s1
    s5 = s4 * c1 + c4 * s1
    c6 = c5 * c1 - s5 * s1
    s6 = s5 * c1 + c5 * s1
    return {
        1: {"C": c1, "S": s1},
        2: {"C": c2, "S": s2},
        4: {"C": c4, "S": s4},
        5: {"C": c5, "S": s5},
        6: {"C": c6, "S": s6},
    }


def polynomial_basis_terms(x, y, z):
    """Evaluate the harmonic-normalized D3d E_g basis pairs."""
    harmonics = cartesian_harmonics(x, y)
    z = float(z)

    return {
        "a": {"x": harmonics[2]["C"], "y": harmonics[2]["S"]},
        "b": {"x": z * harmonics[1]["C"], "y": z * harmonics[1]["S"]},
        "c": {"x": harmonics[4]["C"], "y": harmonics[4]["S"]},
        "d": {"x": z**2 * harmonics[2]["C"], "y": z**2 * harmonics[2]["S"]},
        "e": {"x": z**3 * harmonics[1]["C"], "y": z**3 * harmonics[1]["S"]},
        "f": {"x": z * harmonics[5]["C"], "y": z * harmonics[5]["S"]},
        "g": {"x": z**2 * harmonics[4]["C"], "y": z**2 * harmonics[4]["S"]},
        "h": {"x": z**4 * harmonics[2]["C"], "y": z**4 * harmonics[2]["S"]},
        "i": {"x": z**5 * harmonics[1]["C"], "y": z**5 * harmonics[1]["S"]},
    }


def d6h_e2g_basis_terms(x, y, z):
    """Evaluate the five harmonic-normalized D6h E2g basis pairs."""
    harmonics = cartesian_harmonics(x, y)
    z = float(z)

    c2 = harmonics[2]["C"]
    s2 = harmonics[2]["S"]
    c4 = harmonics[4]["C"]
    s4 = harmonics[4]["S"]
    return {
        "a": {"x": c2, "y": s2},
        "b": {"x": c4, "y": s4},
        "c": {"x": z**2 * c2, "y": z**2 * s2},
        "d": {"x": z**2 * c4, "y": z**2 * s4},
        "e": {"x": z**4 * c2, "y": z**4 * s2},
    }


def _resolved_symmetry_integer_powers(values, defaults, label):
    unknown = set(values) - set(defaults)
    if unknown:
        raise KeyError(f"Unknown {label} parameters: {sorted(unknown)}")

    resolved = dict(defaults)
    for name, value in values.items():
        rounded = int(round(float(value)))
        if not np.isclose(float(value), rounded) or rounded < 0:
            raise ValueError(f"{label} for {name!r} must be a nonnegative integer")
        resolved[name] = rounded
    return resolved


def evaluate_d6h_e2g_symmetry_constrained(
    x,
    y,
    coefficients,
    *,
    n_powers=SYMMETRY_N_DEFAULTS,
    q_powers=SYMMETRY_Q_DEFAULTS,
):
    """Evaluate a finite truncation of the symmetry-constrained scalar series."""
    unknown = set(coefficients) - set(SYMMETRY_COEFFICIENT_DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown constrained coefficients: {sorted(unknown)}")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    harmonics = cartesian_harmonics(x, y)
    r_squared = x * x + y * y
    c6 = harmonics[6]["C"]
    n_values = _resolved_symmetry_integer_powers(
        n_powers,
        SYMMETRY_N_DEFAULTS,
        "radial exponent",
    )
    q_values = _resolved_symmetry_integer_powers(
        q_powers,
        SYMMETRY_Q_DEFAULTS,
        "C6 exponent",
    )

    h2 = (harmonics[2]["C"], harmonics[2]["S"])
    h4 = (harmonics[4]["C"], -harmonics[4]["S"])
    x_component = np.zeros_like(x, dtype=float)
    y_component = np.zeros_like(y, dtype=float)

    for name in SYMMETRY_TERM_NAMES:
        coefficient = float(
            coefficients.get(name, SYMMETRY_COEFFICIENT_DEFAULTS[name])
        )
        invariant = (r_squared ** n_values[name]) * (c6 ** q_values[name])
        covariant = h2 if name.startswith("a") else h4
        x_component = x_component + coefficient * invariant * covariant[0]
        y_component = y_component + coefficient * invariant * covariant[1]
    return x_component, y_component


def _resolved_radial_powers(radial_powers):
    powers = dict(RADIAL_POWER_DEFAULTS)
    for name, value in radial_powers.items():
        if name not in powers:
            raise KeyError(f"Unknown radial-power parameter: {name!r}")
        rounded = int(round(float(value)))
        if not np.isclose(float(value), rounded) or rounded < 0:
            raise ValueError(
                f"Radial power for {name!r} must be a nonnegative integer"
            )
        powers[name] = rounded
    return powers


def _evaluate_basis_terms(x, y, coefficients, radial_powers, terms, basis_names):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r_squared = x * x + y * y
    powers = _resolved_radial_powers(radial_powers)
    x_component = np.zeros_like(x, dtype=float)
    y_component = np.zeros_like(y, dtype=float)

    for name in basis_names:
        coefficient = float(coefficients.get(name, COEFFICIENT_DEFAULTS[name]))
        radial_factor = r_squared ** powers[name]
        x_component = x_component + coefficient * radial_factor * terms[name]["x"]
        y_component = y_component + coefficient * radial_factor * terms[name]["y"]
    return x_component, y_component


def evaluate_polynomial(
    x,
    y,
    z,
    coefficients,
    *,
    radial_powers=RADIAL_POWER_DEFAULTS,
):
    """Evaluate the original D3d E_g polynomial components."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    terms = polynomial_basis_terms(x, y, z)
    return _evaluate_basis_terms(
        x,
        y,
        coefficients,
        radial_powers,
        terms,
        D3D_BASIS_NAMES,
    )


def evaluate_d6h_e2g_polynomial(
    x,
    y,
    z,
    coefficients,
    *,
    radial_powers=RADIAL_POWER_DEFAULTS,
):
    """Evaluate the D6h E2g polynomial components."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    terms = d6h_e2g_basis_terms(x, y, z)
    return _evaluate_basis_terms(
        x,
        y,
        coefficients,
        radial_powers,
        terms,
        D6H_BASIS_NAMES,
    )


def evaluate_basis_polynomial(
    x,
    y,
    z,
    coefficients,
    *,
    basis_mode=DEFAULT_BASIS_MODE,
    radial_powers=RADIAL_POWER_DEFAULTS,
    symmetry_n_powers=SYMMETRY_N_DEFAULTS,
    symmetry_q_powers=SYMMETRY_Q_DEFAULTS,
):
    """Evaluate the selected point-group basis without changing either model."""
    if basis_mode == D6H_E2G_MODE:
        return evaluate_d6h_e2g_polynomial(
            x,
            y,
            z,
            coefficients,
            radial_powers=radial_powers,
        )
    if basis_mode == D3D_EG_MODE:
        return evaluate_polynomial(
            x,
            y,
            z,
            coefficients,
            radial_powers=radial_powers,
        )
    if basis_mode == D6H_E2G_SYMMETRY_MODE:
        return evaluate_d6h_e2g_symmetry_constrained(
            x,
            y,
            coefficients,
            n_powers=symmetry_n_powers,
            q_powers=symmetry_q_powers,
        )
    raise ValueError(
        f"Unknown basis mode {basis_mode!r}; expected one of {BASIS_MODES}"
    )


def save_configuration(grid, output_dir=OUTPUT_DIR, *, z_value=Z_VALUE):
    """Save the basis and interactive-control setup without precomputed fields."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "parameters.json"

    metadata = {
        "evaluation": "real_time",
        "component_order": ["x", "y"],
        "harmonic_notation": {
            "C_m": "r^m cos(m phi)",
            "S_m": "r^m sin(m phi)",
            "C_6": "r^6 cos(6 phi)",
            "r_squared": "x^2+y^2",
        },
        "default_basis_mode": DEFAULT_BASIS_MODE,
        "basis_modes": {
            D6H_E2G_MODE: {
                "point_group": "D6h",
                "irrep": "E2g",
                "active_coefficients": list(D6H_BASIS_NAMES),
                "basis_pairs": D6H_BASIS_LABELS,
            },
            D3D_EG_MODE: {
                "point_group": "D3d",
                "irrep": "Eg",
                "active_coefficients": list(D3D_BASIS_NAMES),
                "basis_pairs": D3D_BASIS_LABELS,
            },
            D6H_E2G_SYMMETRY_MODE: {
                "point_group": "D6h",
                "irrep": "E2g",
                "z_plane": 0.0,
                "aligned_covariants": SYMMETRY_CONSTRAINED_BASIS_LABELS,
                "terms_per_family": SYMMETRY_TERMS_PER_FAMILY,
                "term_names": list(SYMMETRY_TERM_NAMES),
                "coefficient_defaults": SYMMETRY_COEFFICIENT_DEFAULTS,
                "coefficient_ranges": {
                    name: list(values)
                    for name, values in SYMMETRY_COEFFICIENT_RANGES.items()
                },
                "n_defaults": SYMMETRY_N_DEFAULTS,
                "n_range": list(SYMMETRY_N_RANGE),
                "q_defaults": SYMMETRY_Q_DEFAULTS,
                "q_range": list(SYMMETRY_Q_RANGE),
            },
        },
        # Retained as the original D3d basis metadata.
        "basis_pairs": BASIS_LABELS,
        "coefficient_defaults": COEFFICIENT_DEFAULTS,
        "coefficient_step": COEFFICIENT_STEP,
        "param_ranges": {name: list(values) for name, values in param_ranges.items()},
        "radial_factor": "(r^2)^n = (x^2+y^2)^n",
        "radial_power_defaults": RADIAL_POWER_DEFAULTS,
        "radial_power_range": list(RADIAL_POWER_RANGE),
        "berry_lim_percentile": BERRY_LIM_PERCENTILE,
        "sign_filter_components": SIGN_FILTER_COMPONENTS,
        "polar_grid": {
            "r_range": [float(grid["r_edges"][0]), float(grid["r_edges"][-1])],
            "n_r": int(len(grid["r"])),
            "n_phi": int(len(grid["phi"])),
            "z_value": float(z_value),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def _symmetric_limit(values, percentile=None):
    """Return a symmetric color limit, optionally clipping by percentile."""
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return 1.0
    absolute_values = np.abs(finite)
    absolute_maximum = float(np.max(absolute_values))
    if percentile is None:
        limit = absolute_maximum
    else:
        percentile = float(percentile)
        if not 0.0 < percentile <= 100.0:
            raise ValueError("berry_lim_percentile must be in (0, 100]")
        limit = float(np.percentile(absolute_values, percentile))
        if not np.isfinite(limit) or limit <= 0.0:
            limit = absolute_maximum
    return limit if limit > 0.0 else 1.0


def plot_polynomial_basis(
    grid,
    *,
    z_value=Z_VALUE,
    ranges=param_ranges,
    initial_basis_mode=DEFAULT_BASIS_MODE,
    initial_coefficients=COEFFICIENT_DEFAULTS,
    initial_radial_powers=RADIAL_POWER_DEFAULTS,
    radial_power_range=RADIAL_POWER_RANGE,
    symmetry_ranges=SYMMETRY_COEFFICIENT_RANGES,
    initial_symmetry_coefficients=SYMMETRY_COEFFICIENT_DEFAULTS,
    initial_symmetry_n_powers=SYMMETRY_N_DEFAULTS,
    initial_symmetry_q_powers=SYMMETRY_Q_DEFAULTS,
    symmetry_n_range=SYMMETRY_N_RANGE,
    symmetry_q_range=SYMMETRY_Q_RANGE,
    berry_lim_percentile=BERRY_LIM_PERCENTILE,
    sign_filter_components=SIGN_FILTER_COMPONENTS,
    output_png=None,
    show=True,
):
    """Evaluate both components as coefficients and radial powers change."""
    if initial_basis_mode not in BASIS_MODES:
        raise ValueError(
            f"Unknown basis mode {initial_basis_mode!r}; expected one of {BASIS_MODES}"
        )
    names = list(ranges)
    unknown = set(names) - set(initial_coefficients)
    if unknown:
        raise KeyError(f"Unknown polynomial coefficients: {sorted(unknown)}")

    coefficients = {
        name: float(value) for name, value in initial_coefficients.items()
    }
    for name in names:
        minimum, maximum = map(float, ranges[name])
        if maximum <= minimum:
            raise ValueError(f"Range for {name!r} must be increasing")
        coefficients[name] = float(np.clip(coefficients[name], minimum, maximum))

    power_minimum, power_maximum = map(int, radial_power_range)
    if power_minimum < 0 or power_maximum <= power_minimum:
        raise ValueError("radial_power_range must be increasing and nonnegative")
    radial_powers = dict(RADIAL_POWER_DEFAULTS)
    for name, value in initial_radial_powers.items():
        if name not in radial_powers:
            raise KeyError(f"Unknown radial-power parameter: {name!r}")
        rounded = int(round(float(value)))
        if not np.isclose(float(value), rounded):
            raise ValueError(f"Initial radial power for {name!r} must be an integer")
        radial_powers[name] = int(
            np.clip(rounded, power_minimum, power_maximum)
        )

    symmetry_coefficients = dict(SYMMETRY_COEFFICIENT_DEFAULTS)
    unknown = set(initial_symmetry_coefficients) - set(symmetry_coefficients)
    if unknown:
        raise KeyError(f"Unknown constrained coefficients: {sorted(unknown)}")
    symmetry_coefficients.update(
        {name: float(value) for name, value in initial_symmetry_coefficients.items()}
    )
    for name in SYMMETRY_TERM_NAMES:
        minimum, maximum = map(float, symmetry_ranges[name])
        if maximum <= minimum:
            raise ValueError(f"Range for constrained coefficient {name!r} must increase")
        symmetry_coefficients[name] = float(
            np.clip(symmetry_coefficients[name], minimum, maximum)
        )

    symmetry_n_minimum, symmetry_n_maximum = map(int, symmetry_n_range)
    symmetry_q_minimum, symmetry_q_maximum = map(int, symmetry_q_range)
    if symmetry_n_minimum < 0 or symmetry_n_maximum <= symmetry_n_minimum:
        raise ValueError("symmetry_n_range must be increasing and nonnegative")
    if symmetry_q_minimum < 0 or symmetry_q_maximum <= symmetry_q_minimum:
        raise ValueError("symmetry_q_range must be increasing and nonnegative")
    symmetry_n_powers = _resolved_symmetry_integer_powers(
        initial_symmetry_n_powers,
        SYMMETRY_N_DEFAULTS,
        "radial exponent",
    )
    symmetry_q_powers = _resolved_symmetry_integer_powers(
        initial_symmetry_q_powers,
        SYMMETRY_Q_DEFAULTS,
        "C6 exponent",
    )
    for name in SYMMETRY_TERM_NAMES:
        symmetry_n_powers[name] = int(
            np.clip(
                symmetry_n_powers[name],
                symmetry_n_minimum,
                symmetry_n_maximum,
            )
        )
        symmetry_q_powers[name] = int(
            np.clip(
                symmetry_q_powers[name],
                symmetry_q_minimum,
                symmetry_q_maximum,
            )
        )

    display_options = {
        "basis_mode": initial_basis_mode,
        "sign_filter": bool(sign_filter_components),
    }

    def active_basis_names():
        return BASIS_NAMES_BY_MODE[display_options["basis_mode"]]

    def symmetry_mode_active():
        return display_options["basis_mode"] == D6H_E2G_SYMMETRY_MODE

    def evaluate_current_mode():
        if symmetry_mode_active():
            return evaluate_d6h_e2g_symmetry_constrained(
                grid["x"],
                grid["y"],
                symmetry_coefficients,
                n_powers=symmetry_n_powers,
                q_powers=symmetry_q_powers,
            )
        return evaluate_basis_polynomial(
            grid["x"],
            grid["y"],
            z_value,
            coefficients,
            basis_mode=display_options["basis_mode"],
            radial_powers=radial_powers,
        )

    def component_display_values(x_component, y_component):
        if display_options["sign_filter"]:
            return np.sign(x_component), np.sign(y_component)
        return x_component, y_component

    figure, plot_axes = plt.subplots(1, 3, figsize=(19, 11.5))
    figure.subplots_adjust(
        left=0.055,
        right=0.965,
        top=0.69,
        bottom=0.27,
        wspace=0.34,
    )
    equation_text = figure.text(
        0.30,
        0.935,
        "\n".join(EQUATION_LINES_BY_MODE[display_options["basis_mode"]]),
        ha="center",
        va="top",
        fontsize=16.5,
        linespacing=1.55,
    )
    harmonic_table_axis = figure.add_axes([0.57, 0.755, 0.39, 0.17])
    harmonic_table_axis.axis("off")
    harmonic_table_axis.set_title(
        "Explicit Cartesian harmonics",
        fontsize=10,
        pad=6,
    )
    harmonic_table = harmonic_table_axis.table(
        cellText=HARMONIC_TABLE_ROWS,
        colLabels=(r"$m$", r"$C_m$", r"$S_m$"),
        colWidths=(0.08, 0.45, 0.47),
        cellLoc="left",
        colLoc="center",
        loc="center",
    )
    harmonic_table.auto_set_font_size(False)
    harmonic_table.set_fontsize(8.5)
    harmonic_table.scale(1.0, 1.35)
    for (row, _column), cell in harmonic_table.get_celld().items():
        cell.set_edgecolor("#9aa1ad")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#e4e8ef")
            cell.set_text_props(weight="bold")

    def update_harmonic_table(mode):
        rows = HARMONIC_TABLE_ROWS_BY_MODE[mode]
        for row_index in range(1, len(HARMONIC_TABLE_ROWS) + 1):
            visible = row_index <= len(rows)
            for column_index in range(3):
                cell = harmonic_table[(row_index, column_index)]
                cell.set_visible(visible)
                if visible:
                    cell.get_text().set_text(rows[row_index - 1][column_index])

    x_frame, y_frame = evaluate_current_mode()
    x_display_frame, y_display_frame = component_display_values(x_frame, y_frame)
    magnitude_frame = np.hypot(x_frame, y_frame)
    x_limit = (
        1.0
        if display_options["sign_filter"]
        else _symmetric_limit(x_frame, berry_lim_percentile)
    )
    y_limit = (
        1.0
        if display_options["sign_filter"]
        else _symmetric_limit(y_frame, berry_lim_percentile)
    )
    magnitude_limit = _symmetric_limit(magnitude_frame, berry_lim_percentile)

    x_mesh = plot_axes[0].pcolormesh(
        grid["x_edges"],
        grid["y_edges"],
        x_display_frame,
        cmap="RdBu_r",
        vmin=-x_limit,
        vmax=x_limit,
        shading="flat",
    )
    y_mesh = plot_axes[1].pcolormesh(
        grid["x_edges"],
        grid["y_edges"],
        y_display_frame,
        cmap="RdBu_r",
        vmin=-y_limit,
        vmax=y_limit,
        shading="flat",
    )
    magnitude_mesh = plot_axes[2].pcolormesh(
        grid["x_edges"],
        grid["y_edges"],
        magnitude_frame,
        cmap="viridis",
        vmin=0.0,
        vmax=magnitude_limit,
        shading="flat",
    )

    figure.colorbar(x_mesh, ax=plot_axes[0], fraction=0.046, pad=0.035)
    figure.colorbar(y_mesh, ax=plot_axes[1], fraction=0.046, pad=0.035)
    figure.colorbar(magnitude_mesh, ax=plot_axes[2], fraction=0.046, pad=0.035)
    plot_axes[0].set_title(r"$x$ component $P_x$")
    plot_axes[1].set_title(r"$y$ component $P_y$")
    plot_axes[2].set_title(r"Magnitude $|\mathbf{P}|$ and direction")

    stride_phi = max(1, len(grid["phi"]) // 24)
    stride_r = max(1, len(grid["r"]) // 14)
    sample = (slice(None, None, stride_phi), slice(None, None, stride_r))

    def normalized_vectors(x_component, y_component):
        magnitude = np.hypot(x_component, y_component)
        safe = np.where(magnitude > 0.0, magnitude, 1.0)
        return x_component / safe, y_component / safe

    direction_x, direction_y = normalized_vectors(x_frame, y_frame)
    quiver = plot_axes[2].quiver(
        grid["x"][sample],
        grid["y"][sample],
        direction_x[sample],
        direction_y[sample],
        color="white",
        alpha=0.82,
        pivot="mid",
        scale=28,
        width=0.004,
    )

    r_max = float(grid["r_edges"][-1])
    for axis in plot_axes:
        axis.set_aspect("equal")
        axis.set_xlim(-r_max, r_max)
        axis.set_ylim(-r_max, r_max)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_facecolor("#d9dde5")

    def update_component_titles():
        if display_options["sign_filter"]:
            plot_axes[0].set_title(r"Sign of $x$ component: $\mathrm{sgn}(P_x)$")
            plot_axes[1].set_title(r"Sign of $y$ component: $\mathrm{sgn}(P_y)$")
        else:
            plot_axes[0].set_title(r"$x$ component $P_x$")
            plot_axes[1].set_title(r"$y$ component $P_y$")

    def update_title():
        mode = display_options["basis_mode"]
        active_names = active_basis_names()
        if symmetry_mode_active():
            coefficient_names = ",".join(active_names)
            coefficient_values = ",".join(
                f"{symmetry_coefficients[name]:.3g}" for name in active_names
            )
            powers_text = "  ".join(
                rf"${name[0]}_{{{name[1:]}}}$:"
                rf"$(n,q)=({symmetry_n_powers[name]},{symmetry_q_powers[name]})$"
                for name in active_names
            )
            displayed_z = 0.0
        else:
            coefficient_names = ",".join(active_names)
            coefficient_values = ",".join(
                f"{coefficients[name]:.3g}" for name in active_names
            )
            power_labels = ",".join(rf"n_{{{name}}}" for name in active_names)
            power_values = ",".join(str(radial_powers[name]) for name in active_names)
            powers_text = rf"Radial powers $({power_labels})=$ " + f"({power_values})"
            displayed_z = float(z_value)
        component_mode = "sign components" if display_options["sign_filter"] else "values"
        figure.suptitle(
            f"Real-time polar polynomial basis | {mode} | z={displayed_z:.4g}"
            + f" | ({coefficient_names})=({coefficient_values})"
            + f" | {component_mode}",
            y=0.985,
            fontsize=14,
        )
        power_value_text.set_text(powers_text)
        power_value_text.set_fontsize(8.5 if symmetry_mode_active() else 10)
        equation_text.set_text("\n".join(EQUATION_LINES_BY_MODE[mode]))
        harmonic_table_axis.set_title(
            f"Explicit Cartesian harmonics | {mode}",
            fontsize=10,
            pad=6,
        )
        update_harmonic_table(mode)
        update_component_titles()

    def update_plot():
        x_component, y_component = evaluate_current_mode()
        magnitude = np.hypot(x_component, y_component)
        x_display, y_display = component_display_values(x_component, y_component)
        x_mesh.set_array(x_display.ravel())
        y_mesh.set_array(y_display.ravel())
        magnitude_mesh.set_array(magnitude.ravel())
        if display_options["sign_filter"]:
            x_limit = 1.0
            y_limit = 1.0
        else:
            x_limit = _symmetric_limit(x_component, berry_lim_percentile)
            y_limit = _symmetric_limit(y_component, berry_lim_percentile)
        x_mesh.set_clim(-x_limit, x_limit)
        y_mesh.set_clim(-y_limit, y_limit)
        magnitude_mesh.set_clim(
            0.0,
            _symmetric_limit(magnitude, berry_lim_percentile),
        )
        direction_x, direction_y = normalized_vectors(x_component, y_component)
        quiver.set_UVC(direction_x[sample], direction_y[sample])
        update_title()
        figure.canvas.draw_idle()

    power_value_text = figure.text(
        0.5,
        0.245,
        "",
        ha="center",
        va="center",
        fontsize=10,
    )
    sign_filter_axis = figure.add_axes([0.455, 0.207, 0.10, 0.034])
    sign_filter_axis.set_frame_on(False)
    sign_filter_toggle = CheckButtons(
        sign_filter_axis,
        ["Sign filter"],
        [display_options["sign_filter"]],
    )

    def on_sign_filter_change(_label):
        display_options["sign_filter"] = bool(sign_filter_toggle.get_status()[0])
        update_plot()

    sign_filter_toggle.on_clicked(on_sign_filter_change)

    basis_mode_axis = figure.add_axes([0.39, 0.692, 0.22, 0.072])
    basis_mode_axis.set_title("Basis mode", fontsize=9, pad=2)
    basis_mode_toggle = RadioButtons(
        basis_mode_axis,
        BASIS_MODES,
        active=BASIS_MODES.index(display_options["basis_mode"]),
    )
    for label in basis_mode_toggle.labels:
        label.set_fontsize(8.5)

    coefficient_sliders = {}
    radial_power_sliders = {}
    symmetry_coefficient_sliders = {}
    symmetry_n_sliders = {}
    symmetry_q_sliders = {}
    controls = [("coefficient", name) for name in names]
    controls.extend(("power", name) for name in RADIAL_POWER_DEFAULTS)
    slider_left = (0.065, 0.305, 0.545, 0.785)
    slider_width = 0.15
    slider_height = 0.017
    first_slider_y = 0.195
    slider_gap = 0.035

    def add_slider(control_index, label, minimum, maximum, initial, *, valstep=None):
        row = control_index // 4
        column = control_index % 4
        slider_axis = figure.add_axes(
            [
                slider_left[column],
                first_slider_y - row * slider_gap,
                slider_width,
                slider_height,
            ]
        )
        return Slider(
            slider_axis,
            label,
            valmin=minimum,
            valmax=maximum,
            valinit=initial,
            valstep=valstep,
            valfmt="%d" if valstep == 1 else "%.4g",
        )

    for control_index, (control_type, name) in enumerate(controls):
        if control_type == "coefficient":
            minimum, maximum = map(float, ranges[name])
            slider = add_slider(
                control_index,
                name,
                minimum,
                maximum,
                coefficients[name],
                valstep=COEFFICIENT_STEP,
            )

            def on_coefficient_change(value, parameter_name=name):
                coefficients[parameter_name] = float(value)
                update_plot()

            slider.on_changed(on_coefficient_change)
            coefficient_sliders[name] = slider
            continue

        exponent_label = rf"$n_{name}$"
        slider = add_slider(
            control_index,
            exponent_label,
            power_minimum,
            power_maximum,
            radial_powers[name],
            valstep=1,
        )

        def on_power_change(value, basis_name=name):
            radial_powers[basis_name] = int(round(value))
            update_plot()

        slider.on_changed(on_power_change)
        radial_power_sliders[name] = slider

    for name in SYMMETRY_TERM_NAMES:
        family = name[0]
        index = name[1:]
        latex_name = rf"{family}_{{{index}}}"
        minimum, maximum = map(float, symmetry_ranges[name])

        coefficient_slider = add_slider(
            0,
            rf"${latex_name}$",
            minimum,
            maximum,
            symmetry_coefficients[name],
            valstep=COEFFICIENT_STEP,
        )

        def on_symmetry_coefficient_change(value, parameter_name=name):
            symmetry_coefficients[parameter_name] = float(value)
            update_plot()

        coefficient_slider.on_changed(on_symmetry_coefficient_change)
        symmetry_coefficient_sliders[name] = coefficient_slider

        n_slider = add_slider(
            0,
            rf"$n_{{{latex_name}}}$",
            symmetry_n_minimum,
            symmetry_n_maximum,
            symmetry_n_powers[name],
            valstep=1,
        )

        def on_symmetry_n_change(value, parameter_name=name):
            symmetry_n_powers[parameter_name] = int(round(value))
            update_plot()

        n_slider.on_changed(on_symmetry_n_change)
        symmetry_n_sliders[name] = n_slider

        q_slider = add_slider(
            0,
            rf"$q_{{{latex_name}}}$",
            symmetry_q_minimum,
            symmetry_q_maximum,
            symmetry_q_powers[name],
            valstep=1,
        )

        def on_symmetry_q_change(value, parameter_name=name):
            symmetry_q_powers[parameter_name] = int(round(value))
            update_plot()

        q_slider.on_changed(on_symmetry_q_change)
        symmetry_q_sliders[name] = q_slider

    def update_control_layout():
        all_slider_groups = (
            coefficient_sliders,
            radial_power_sliders,
            symmetry_coefficient_sliders,
            symmetry_n_sliders,
            symmetry_q_sliders,
        )
        for slider_group in all_slider_groups:
            for slider in slider_group.values():
                slider.ax.set_visible(False)

        if symmetry_mode_active():
            constrained_left = (0.08, 0.40, 0.72)
            constrained_width = 0.18
            constrained_gap = 0.029
            for row, name in enumerate(SYMMETRY_TERM_NAMES):
                term_sliders = (
                    symmetry_coefficient_sliders[name],
                    symmetry_n_sliders[name],
                    symmetry_q_sliders[name],
                )
                for column, slider in enumerate(term_sliders):
                    slider.ax.set_position(
                        [
                            constrained_left[column],
                            first_slider_y - row * constrained_gap,
                            constrained_width,
                            slider_height,
                        ]
                    )
                    slider.ax.set_visible(True)
            return

        active_names = set(active_basis_names())
        active_controls = []
        for name, slider in coefficient_sliders.items():
            if name in active_names:
                active_controls.append(slider)
        for name, slider in radial_power_sliders.items():
            if name in active_names:
                active_controls.append(slider)

        for control_index, slider in enumerate(active_controls):
            row = control_index // 4
            column = control_index % 4
            slider.ax.set_position(
                [
                    slider_left[column],
                    first_slider_y - row * slider_gap,
                    slider_width,
                    slider_height,
                ]
            )
            slider.ax.set_visible(True)

    def on_basis_mode_change(label):
        display_options["basis_mode"] = label
        update_control_layout()
        update_plot()

    basis_mode_toggle.on_clicked(on_basis_mode_change)
    update_control_layout()
    update_title()
    figure.polynomial_viewer = {
        "sliders": list(coefficient_sliders.values())
        + list(radial_power_sliders.values())
        + list(symmetry_coefficient_sliders.values())
        + list(symmetry_n_sliders.values())
        + list(symmetry_q_sliders.values()),
        "coefficient_sliders": coefficient_sliders,
        "radial_power_sliders": radial_power_sliders,
        "symmetry_coefficient_sliders": symmetry_coefficient_sliders,
        "symmetry_n_sliders": symmetry_n_sliders,
        "symmetry_q_sliders": symmetry_q_sliders,
        "basis_mode_toggle": basis_mode_toggle,
        "sign_filter_toggle": sign_filter_toggle,
        "display_options": display_options,
        "coefficients": coefficients,
        "radial_powers": radial_powers,
        "symmetry_coefficients": symmetry_coefficients,
        "symmetry_n_powers": symmetry_n_powers,
        "symmetry_q_powers": symmetry_q_powers,
        "meshes": [x_mesh, y_mesh, magnitude_mesh],
        "quiver": quiver,
        "axes": plot_axes,
        "equation_text": equation_text,
        "harmonic_table": harmonic_table,
        "harmonic_table_axis": harmonic_table_axis,
        "power_value_text": power_value_text,
    }
    if output_png is not None:
        figure.savefig(output_png, dpi=180)
    if show:
        plt.show()
    return figure


def main(show=True):
    grid = make_polar_grid()
    metadata_path = save_configuration(grid)
    figure_path = OUTPUT_DIR / "polynomial_basis_realtime.png"
    plot_polynomial_basis(grid, output_png=figure_path, show=show)
    print(
        "\nReal-time polynomial basis viewer ready. Configuration and preview saved at:"
        f"\n > {metadata_path}"
        f"\n > {figure_path}"
    )


if __name__ == "__main__":
    main()
