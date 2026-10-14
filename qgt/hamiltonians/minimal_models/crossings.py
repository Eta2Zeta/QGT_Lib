"""Exact band crossings of the minimal altermagnet models.

Every minimal model has the form

    H(k) = eps0 + tx tau_x + tz tau_z + tau_y (lambda . sigma) + tau_z (J . sigma),

whose bands are E1, E2 = eps0 - sqrt(X +/- 2Y) and E3, E4 = eps0 + sqrt(X -/+ 2Y), with

    X = tx^2 + tz^2 + |lambda|^2 + |J|^2,    Y = sqrt(tz^2 |J|^2 + |lambda x J|^2).

Because X^2 - 4Y^2 = tx^4 + 2 tx^2 (A + |J|^2) + (A - |J|^2)^2 + 4 (lambda . J)^2,
with A = tz^2 + |lambda|^2, is a sum of squares, the bands touch exactly where

    bands 2-3:              tx = 0,  lambda . J = 0,  tz^2 + |lambda|^2 = |J|^2
    bands 1-2 and 3-4:      tz = 0,  lambda x J = 0          (they always touch together)

for J != 0. For J = 0, bands 1-2 and 3-4 are degenerate everywhere and bands 2-3
touch where tx = tz = 0 and lambda = 0. Research Note 4-0 derives the same conditions.

find_band_crossings() writes these conditions exactly with sympy from the model's
own form factors. Along each axis the form factors are polynomials in the cosine
and sine of one half angle, theta = (k_x/2, k_y/2, k_z/2), or (k_x/2, sqrt(3) k_y/2,
k_z/2) in the hexagonal setting. The conditions are split into components:

- each condition is factored (over Q(sqrt 3) for hexagonal models); a factor in one
  angle is solved exactly, which fixes that momentum;
- a factor whose form in z_i = exp(i theta_i) contains a polynomial in one monomial
  z^e vanishes on the lines (or planes) e . theta = arg(root); these are solved
  exactly as linear relations, such as k_y = k_x or k_y = sqrt(3) k_x;
- whatever is left mixes several angles; it is kept as exact equations and sampled
  by Newton projection onto them.

Every sample is checked by diagonalizing the model, and each Crossing records the
largest gap found.
"""

import math
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from itertools import product
from types import SimpleNamespace

import numpy as np
import sympy as sp
from scipy.spatial import cKDTree

KX, KY, KZ = K = sp.symbols("k_x k_y k_z", real=True)
C = sp.symbols("C_x C_y C_z", real=True)  # cos(theta_i)
S = sp.symbols("S_x S_y S_z", real=True)  # sin(theta_i)
Z = sp.symbols("z_x z_y z_z")             # exp(i theta_i)
R = sp.Symbol("R", positive=True)         # sqrt(3) in hexagonal form factors, kept symbolic for factoring
AXES = ("x", "y", "z")
PAIRS = ("2-3", "1-2,3-4")
PAIR_NAMES = {"2-3": "bands 2-3", "1-2,3-4": "bands 1-2 and 3-4"}
_HEXAGONAL = ("trigonal", "hexagonal")
_TEST_ANGLES = ((0.3711, 1.9137, -2.4311), (2.7183, -0.8309, 0.5923), (-1.2917, 2.3305, 1.1731))


# Symbols standing for exact constants: R = sqrt(3), and cos/sin of a plane's half angle
# (see _plane_point). Factoring treats them as variables; evaluation substitutes them.
_VALUES = {R: sp.sqrt(3)}
_ANGLES = {}  # cos symbol -> its exact angle


def _real(expr):
    """Replace the constant symbols by their exact values."""
    return expr.xreplace(_VALUES) if isinstance(expr, sp.Basic) else expr


def _reduced(expr):
    """Replace R by sqrt(3) only (so sqrt(3)^2 = 3 is applied); other constant symbols stay."""
    return expr.xreplace({R: sp.sqrt(3)}) if isinstance(expr, sp.Basic) else expr


def _constants(expr):
    return sorted((x for x in expr.free_symbols if x in _VALUES), key=str)


def _plane_point(kz):
    """(cos, sin) of the half angle k_z/2 for a fixed plane: exact numbers when they are
    algebraic in closed form, else symbols registered in _VALUES."""
    theta = kz / 2
    c, s = sp.cos(theta), sp.sin(theta)
    if not (c.has(sp.cos, sp.sin) or s.has(sp.cos, sp.sin)):
        return c, s
    symbols = sp.Symbol(f"cos({sp.sstr(theta)})", real=True), sp.Symbol(f"sin({sp.sstr(theta)})", real=True)
    _VALUES.update(zip(symbols, (c, s)))
    _ANGLES[symbols[0]] = theta
    return symbols


# --------------------------------------------------------------------------- exact form factors

def _exact(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return sp.Integer(int(value))
    if isinstance(value, (float, np.floating)):
        return sp.Rational(repr(float(value)))
    return value


class _SymbolicModel:
    """A view of a model with exact parameters and sympy trig terms (sqrt(3) coefficients as R)."""

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        attribute = getattr(type(self._model), name, None)
        if callable(attribute):
            return attribute.__get__(self)
        return _exact(getattr(self._model, name))

    def _trig_terms(self, kx, ky, kz):
        cos, sin = sp.cos, sp.sin
        sx, sx_2 = sin(kx), sin(kx / 2)
        csq3ky_2, ssq3ky_2 = cos(sp.sqrt(3) * ky / 2), sin(sp.sqrt(3) * ky / 2)
        return SimpleNamespace(
            sqrt3=R, cx=cos(kx), cy=cos(ky), cz=cos(kz), sx=sx, sy=sin(ky), sz=sin(kz),
            cx_2=cos(kx / 2), cy_2=cos(ky / 2), cz_2=cos(kz / 2),
            sx_2=sx_2, sy_2=sin(ky / 2), sz_2=sin(kz / 2),
            csq3ky_2=csq3ky_2, ssq3ky_2=ssq3ky_2,
            fx=sx + sx_2 * csq3ky_2, fy=R * cos(kx / 2) * ssq3ky_2, fz=sin(kz),
        )


def _terms(model):
    return type(model)._model_terms(_SymbolicModel(model), KX, KY, KZ)


def _conditions(model):
    _, tx, tz, lx, ly, lz = _terms(model)
    lam = sp.Matrix([lx, ly, lz])
    J = sp.Matrix([_exact(model.Jx), _exact(model.Jy), _exact(model.Jz)])
    if all(component == 0 for component in J):
        return {"2-3": [tx, tz, lx, ly, lz], "1-2,3-4": []}
    return {"2-3": [tx, lam.dot(J), tz**2 + lam.dot(lam) - J.dot(J)],
            "1-2,3-4": [tz, *lam.cross(J)]}


def symbolic_terms(model):
    """Exact (eps0, tx, tz, lambda_x, lambda_y, lambda_z) as sympy expressions in k_x, k_y, k_z."""
    return tuple(_real(sp.sympify(term)) for term in _terms(model))


def crossing_conditions(model):
    """Exact crossing conditions: {pair: [expressions in k that must all vanish]}.

    An empty list means the pair is degenerate everywhere.
    """
    return {pair: [_real(e) for e in equations] for pair, equations in _conditions(model).items()}


def _hexagonal(model):
    return model.crystal_system in _HEXAGONAL


def _scales(hexagonal):
    """a_i in theta_i = a_i k_i / 2."""
    return (1, sp.sqrt(3), 1) if hexagonal else (1, 1, 1)


def _to_polynomial(expr, hexagonal):
    """Rewrite an expression in k as a polynomial in C_i = cos(theta_i), S_i = sin(theta_i) (and R)."""
    theta = sp.symbols("theta_x theta_y theta_z", real=True)
    a = _scales(hexagonal)
    poly = sp.sympify(expr).subs({K[i]: 2 * theta[i] / a[i] for i in range(3)})
    poly = sp.expand(sp.expand_trig(sp.expand(poly)))
    poly = poly.xreplace({sp.cos(theta[i]): C[i] for i in range(3)} | {sp.sin(theta[i]): S[i] for i in range(3)})
    if poly.has(sp.sin, sp.cos) or poly.free_symbols - set(C) - set(S) - {R}:
        raise ValueError(f"Form factor is not a polynomial in the half-angle variables: {expr}")
    return sp.expand(poly)


# --------------------------------------------------------------------------- algebra helpers

def _axes_of(poly):
    return [i for i in range(3) if poly.has(C[i]) or poly.has(S[i])]


def _gens(poly):
    return [g for i in _axes_of(poly) for g in (C[i], S[i]) if poly.has(g)]


def _surds(expr):
    return sorted({a for a in expr.atoms(sp.Pow) if a.base.is_Rational and a.exp.is_Rational
                   and not a.exp.is_Integer}, key=sp.default_sort_key)


def _scale(poly):
    gens = _gens(poly)
    coefficients = sp.Poly(poly, *gens).coeffs() if gens else [poly]
    return max(abs(complex(sp.N(_real(c), 30))) for c in coefficients)


@lru_cache(maxsize=4096)
def _is_zero(poly):
    """True if poly vanishes on the torus (evaluated at generic angles to 50 digits)."""
    poly = sp.expand(_real(poly))
    if poly == 0:
        return True
    if not _axes_of(poly):
        return abs(complex(sp.N(poly, 50))) < 1e-30
    scale = _scale(poly)
    for angles in _TEST_ANGLES:
        values = {C[i]: sp.cos(sp.Float(a, 50)) for i, a in enumerate(angles)}
        values |= {S[i]: sp.sin(sp.Float(a, 50)) for i, a in enumerate(angles)}
        if abs(complex(sp.N(poly.xreplace(values), 50))) > 1e-35 * scale:
            return False
    return True


def _factor_over_q(poly, gens):
    """Distinct factors of poly over Q, with R and each surd treated as further variables; or None."""
    surds = _surds(poly)
    aux = sp.symbols(f"r0:{len(surds)}")
    replaced = poly.xreplace(dict(zip(surds, aux)))
    variables = [*gens, *aux, *_constants(replaced)]
    try:
        if sp.Poly(replaced, *variables).domain not in (sp.ZZ, sp.QQ):
            return None
        _, factors = sp.factor_list(replaced, *variables)
    except sp.PolynomialError:
        return None
    back = dict(zip(aux, surds))
    return [sp.expand(f.xreplace(back)) for f, _ in factors]


@lru_cache(maxsize=4096)
def _factors(poly):
    """Distinct non-constant factors of poly in C, S.

    Hexagonal coefficients are tried both with sqrt(3)^2 = 3 applied and with sqrt(3)
    kept as the symbol R: each finds factorizations over Q(sqrt 3) that the other misses.
    """
    gens = _gens(poly)
    if not gens:
        return (poly,)
    for form in dict.fromkeys([sp.expand(_reduced(poly)), poly]):
        found = [f for f in _factor_over_q(form, gens) or [] if _axes_of(f)]
        if len(found) > 1:
            return tuple(found)
    return (poly,)


def _divides(a, b):
    """True if b is in the ideal of a and the circle relations C_i^2 + S_i^2 = 1, so that
    a = 0 implies b = 0 (surds enter as symbols r with r^2 = n)."""
    if not set(_gens(a)) <= set(_gens(b)):
        return False
    a, b = sp.expand(_reduced(a)), sp.expand(_reduced(b))
    surds = sorted(set(_surds(a)) | set(_surds(b)), key=sp.default_sort_key)
    aux = sp.symbols(f"r0:{len(surds)}")
    a, b = (e.xreplace(dict(zip(surds, aux))) for e in (a, b))
    axes = sorted(set(_axes_of(a)) | set(_axes_of(b)))
    relations = [C[i]**2 + S[i]**2 - 1 for i in axes] + [r**2 - s**2 for r, s in zip(aux, surds)]
    constants = sorted(set(_constants(a)) | set(_constants(b)), key=str)
    try:
        basis = sp.groebner([a, *relations], *[g for i in axes for g in (C[i], S[i])], *aux, *constants,
                            order="grevlex")
        return basis.reduce(b)[1] == 0
    except (sp.PolynomialError, sp.CoercionFailed, NotImplementedError):
        return False


def _drop_redundant(system):
    """Drop equations implied by another equation (a multiple of it)."""
    kept = []
    for eq in sorted(system, key=lambda e: (len(sp.Add.make_args(e)), sp.default_sort_key(e))):
        if not any(_divides(other, eq) for other in kept):
            kept.append(eq)
    return kept


def _rationalized(expr):
    """expr times its conjugates under the sign of each surd: a polynomial with rational coefficients."""
    expr = sp.expand(_real(expr))
    for surd in _surds(expr):
        expr = sp.expand(expr * expr.xreplace({surd: -surd}))
    return expr


def _real_roots(expr, var):
    """Distinct exact real roots of a polynomial in var. Surd coefficients are allowed; with
    other coefficients (a plane's cos and sin) only quadratics, in var or in var^2, are solved."""
    poly = sp.Poly(_rationalized(expr), var)
    if poly.domain in (sp.ZZ, sp.QQ):
        return list(dict.fromkeys(poly.real_roots()))
    poly = sp.Poly(sp.expand(_real(expr)), var)
    if poly.degree() <= 2:
        candidates = list(sp.roots(poly))
    elif poly.degree() == 4 and all(not c for (n,), c in poly.terms() if n % 2):
        u = sp.Dummy("u")
        squares = sp.roots(sp.Poly(sum(c * u**(n // 2) for (n,), c in poly.terms()), u))
        candidates = [sign * sp.sqrt(r) for r in squares if complex(sp.N(r, 50)).real >= 0 for sign in (1, -1)]
    else:
        return None
    return [r for r in candidates if abs(complex(sp.N(r, 50)).imag) < 1e-40]


def _circle_roots(factor, i):
    """Exact (cos theta_i, sin theta_i) where a one-angle factor vanishes; None if not solvable."""
    factor = sp.expand(_real(factor))
    c, s = C[i], S[i]
    try:
        if factor.has(c) and factor.has(s):
            values, var = _real_roots(sp.resultant(factor, c**2 + s**2 - 1, s), c), c
        else:
            var = s if factor.has(s) else c
            values = _real_roots(factor, var)
    except (sp.PolynomialError, NotImplementedError):
        return None
    if values is None:
        return None
    roots, seen = [], set()
    for value in values:
        if abs(float(value)) > 1 + 1e-12:
            continue
        other = sp.sqrt(1 - value**2)
        for sign in (1, -1):
            cos_value, sin_value = (value, sign * other) if var == c else (sign * other, value)
            key = (round(float(cos_value), 10), round(float(sin_value), 10))
            if key not in seen and _is_zero(factor.xreplace({c: cos_value, s: sin_value})):
                seen.add(key)
                roots.append((cos_value, sin_value))
    return roots


# --------------------------------------------------------------------------- lines from monomial factors

def _laurent(poly):
    """poly with C_i = (z_i + 1/z_i)/2, S_i = (z_i - 1/z_i)/2i, times a monomial: a polynomial in z."""
    axes = _axes_of(poly)
    gens = [g for i in axes for g in (C[i], S[i])]
    P = sp.Poly(poly, *gens)
    degree = [max(m[2 * n] + m[2 * n + 1] for m in P.monoms()) for n in range(len(axes))]
    total = 0
    for monomial, coefficient in P.terms():
        term = coefficient
        for n, i in enumerate(axes):
            c_power, s_power = monomial[2 * n], monomial[2 * n + 1]
            term *= ((Z[i]**2 + 1)**c_power * (Z[i]**2 - 1)**s_power * Z[i]**(degree[n] - c_power - s_power)
                     / (2**(c_power + s_power) * sp.I**s_power))
        total += term
    total = sp.expand(total)
    if total.has(sp.I):
        rotated = sp.expand(total / sp.I)
        if not rotated.has(sp.I):
            return rotated
    return total


def _unit_circle_roots(q, w):
    """Exact roots of q(w) with |w| = 1, as (cos, sin) of their arguments."""
    q = sp.expand(_real(q))
    rational = sp.Poly(_rationalized(q), w)
    if rational.domain in (sp.ZZ, sp.QQ):
        candidates = sp.roots(rational)
        if sum(candidates.values()) != rational.degree():
            candidates = dict.fromkeys(rational.all_roots())
    elif sp.Poly(q, w).degree() == 1:
        candidates = sp.roots(sp.Poly(q, w))
    else:
        return []
    roots, seen = [], set()
    for root in candidates:
        value = complex(sp.N(root, 50))
        key = (round(value.real, 10), round(value.imag, 10))
        if abs(abs(value) - 1) > 1e-20 or key in seen or abs(complex(sp.N(q.subs(w, root), 50))) > 1e-30:
            continue
        seen.add(key)
        roots.append((sp.radsimp(sp.re(root)), sp.radsimp(sp.im(root))))
    return roots


@lru_cache(maxsize=4096)
def _monomial_lines(poly):
    """Linear relations on which poly vanishes, from factors of poly(z) that are polynomials
    in one monomial z^e (e with at least two nonzero entries).

    Returns (relations, complete): each relation is (e, (cos c, sin c)) for e . theta = c
    (mod 2 pi); complete is True when these relations make up the whole zero set.
    """
    axes = _axes_of(poly)
    if len(axes) < 2:
        return (), False
    zs = [Z[i] for i in axes]
    factors = _factor_over_q(_laurent(sp.expand(_reduced(poly))), zs)
    if factors is None:
        return (), False
    relations, complete = [], True
    w = sp.Symbol("w")
    for factor in factors:
        if not any(factor.has(z) for z in zs):
            continue
        terms = sp.Poly(factor, *zs).terms()
        first = terms[0][0]
        deltas = [tuple(m[n] - first[n] for n in range(len(zs))) for m, _ in terms]
        nonzero = [d for d in deltas if any(d)]
        g = reduce(math.gcd, (abs(v) for v in nonzero[0]))
        e = tuple(v // g for v in nonzero[0])
        multiples = []
        for d in deltas:
            parallel = all((ev and v % ev == 0) or (not ev and not v) for v, ev in zip(d, e))
            ratios = {v // ev for v, ev in zip(d, e) if ev} if parallel else set()
            if len(ratios) != 1:
                break
            multiples.append(ratios.pop())
        if len(multiples) != len(deltas) or sum(1 for v in e if v) < 2:
            complete = False
            continue
        low = min(multiples)
        q = sum(coefficient * w**(n - low) for (_, coefficient), n in zip(terms, multiples))
        full = [0, 0, 0]
        for n, i in enumerate(axes):
            full[i] = e[n]
        relations += [(tuple(full), root) for root in _unit_circle_roots(q, w)]
    return tuple(relations), complete


def _relation_link(relation):
    """(dep, coeffs, constant) with theta_dep = sum n_r theta_r + c for e . theta = c, or None."""
    e, (cos_c, sin_c) = relation
    units = [i for i in range(3) if abs(e[i]) == 1]
    if not units:
        return None
    dep = units[-1]
    sigma = e[dep]
    coeffs = tuple((r, -sigma * e[r]) for r in range(3) if r != dep and e[r])
    return dep, coeffs, (cos_c, sigma * sin_c)


# --------------------------------------------------------------------------- linear relations (links)

def _times(a, b, n=1):
    """(cos, sin) of angle_a + n angle_b, exactly."""
    w = (a[0] + sp.I * a[1]) * ((b[0] + sp.I * b[1]) if n >= 0 else (b[0] - sp.I * b[1]))**abs(n)
    real, imaginary = sp.expand(w).as_real_imag()
    return sp.radsimp(real), sp.radsimp(imaginary)


@lru_cache(maxsize=None)
def _phase_polys(coeffs, constant):
    """cos and sin of (sum n_r theta_r + c) as polynomials in C_r, S_r."""
    w = constant[0] + sp.I * constant[1]
    for r, n in coeffs:
        w *= (C[r] + sp.I * S[r])**n if n > 0 else (C[r] - sp.I * S[r])**(-n)
    real, imaginary = sp.expand(w).as_real_imag()
    return sp.expand(real), sp.expand(imaginary)


def _add_link(links, dep, coeffs, constant):
    """links plus theta_dep = sum n_r theta_r + c, with earlier links expressed through it."""
    updated = {}
    for d, (cs, const) in links.items():
        cs = dict(cs)
        if dep in cs:
            n = cs.pop(dep)
            for r, m in coeffs:
                cs[r] = cs.get(r, 0) + n * m
            const = _times(const, constant, n)
        updated[d] = (tuple(sorted((r, m) for r, m in cs.items() if m)), const)
    updated[dep] = (coeffs, constant)
    return updated


def _substitute(poly, fixed, links):
    first = {}
    for dep, (coeffs, constant) in links.items():
        first[C[dep]], first[S[dep]] = _phase_polys(coeffs, constant)
    second = {g: v for i, (c, s) in fixed.items() for g, v in ((C[i], c), (S[i], s))}
    return sp.expand(poly.xreplace(first).xreplace(second))


# --------------------------------------------------------------------------- splitting into components

def _split(equations, fixed, links, excluded, done, branches):
    """Depth-first split of a polynomial system into branches (fixed, links, remaining, excluded).

    excluded: relations already reported as their own branches, left out when sampling.
    done: equations whose monomial lines were already split off.
    """
    system = []
    for eq in equations:
        eq = _substitute(eq, fixed, links)
        if _is_zero(eq):
            continue
        if not _axes_of(eq):
            return  # a nonzero constant: no solutions
        system.append(eq)
    if not system:
        branches.append((fixed, links, [], excluded))
        return
    system = _drop_redundant(system) if len(system) > 1 else system
    for index, eq in enumerate(system):
        factors = _factors(eq)
        if len(factors) > 1:
            rest = system[:index] + system[index + 1:]
            for factor in factors:
                _split(rest + [factor], fixed, links, excluded, done, branches)
            return
    for index, eq in enumerate(system):
        axes = _axes_of(eq)
        if len(axes) == 1:
            roots = _circle_roots(eq, axes[0])
            if roots is None:
                continue
            rest = system[:index] + system[index + 1:]
            for root in roots:
                _split(rest, {**fixed, axes[0]: root}, links, excluded, done, branches)
            return
    for index, eq in enumerate(system):
        if eq in done:
            continue
        relations, complete = _monomial_lines(eq)
        new = [link for link in map(_relation_link, relations) if link is not None]
        complete = complete and len(new) == len(relations)
        if not new:
            continue
        rest = system[:index] + system[index + 1:]
        for dep, coeffs, constant in new:
            _split(rest, fixed, _add_link(links, dep, coeffs, constant), excluded, done, branches)
        if not complete:
            _split(rest + [eq], fixed, links, excluded + new, done | {eq}, branches)
        return
    branches.append((fixed, links, system, excluded))


# --------------------------------------------------------------------------- Brillouin zone

def _hex_vectors():
    return np.array([0.0, 4 * np.pi / np.sqrt(3)]), np.array([2 * np.pi, 2 * np.pi / np.sqrt(3)])


def _wrap_period(values, period):
    """Map values into (-period/2, period/2]."""
    return values - period * np.ceil((values - period / 2) / period - 1e-12)


def wrap_to_bz(k, hexagonal):
    """Map momenta (N, 3) into the first Brillouin zone: the box (-pi, pi]^3, or the
    hexagonal prism (M at (0, 2 pi/sqrt 3)) for trigonal and hexagonal models."""
    k = np.array(k, dtype=float).reshape(-1, 3)
    k[:, 2] = _wrap_period(k[:, 2], 2 * np.pi)
    if not hexagonal:
        k[:, :2] = _wrap_period(k[:, :2], 2 * np.pi)
        return k
    b1, b2 = _hex_vectors()
    shifts = np.array([m * b1 + n * b2 for m in range(-3, 4) for n in range(-3, 4)])
    candidates = k[:, None, :2] - shifts[None]                       # (N, 49, 2)
    norms = np.linalg.norm(candidates, axis=-1)
    best = norms <= norms.min(axis=1, keepdims=True) + 1e-9
    kx = np.where(best, candidates[..., 0], -np.inf)                 # ties: largest k_x, then k_y
    best &= candidates[..., 0] >= kx.max(axis=1, keepdims=True) - 1e-9
    ky = np.where(best, candidates[..., 1], -np.inf)
    k[:, :2] = candidates[np.arange(len(k)), ky.argmax(axis=1)]
    return k


def _theta_of_k(k, hexagonal):
    a = np.array([1.0, np.sqrt(3) if hexagonal else 1.0, 1.0])
    return np.asarray(k, dtype=float) * a / 2


def _k_of_theta(theta, hexagonal):
    a = np.array([1.0, np.sqrt(3) if hexagonal else 1.0, 1.0])
    return 2 * np.asarray(theta, dtype=float) / a


def _lattice_shifts(hexagonal):
    """Shifts of theta that leave the Brillouin-zone point unchanged (modulo 2 pi)."""
    if hexagonal:
        return np.array([[n * np.pi, n * np.pi, m * np.pi] for n in (0, 1) for m in (0, 1)])
    return np.pi * np.array(list(product((0, 1), repeat=3)), dtype=float)


def _angle(pair):
    return math.atan2(float(_real(pair[1])), float(_real(pair[0])))


def _link_mismatch(theta, dep, coeffs, constant):
    """|theta_dep - sum n_r theta_r - c| modulo 2 pi, for theta (N, 3)."""
    value = theta[:, dep] - sum(n * theta[:, r] for r, n in coeffs) - _angle(constant)
    return np.abs((value + np.pi) % (2 * np.pi) - np.pi)


def _unique_rows(points, tol):
    """points without near-duplicates (max-norm distance below tol)."""
    if len(points) < 2:
        return points
    tree = cKDTree(points)
    removed = np.zeros(len(points), bool)
    for i in range(len(points)):
        if not removed[i]:
            for j in tree.query_ball_point(points[i], tol, p=np.inf):
                removed[j] = removed[j] or j > i
    return points[~removed]


# --------------------------------------------------------------------------- numerical sampling

def _angle_functions(polys, free):
    """Vectorized residuals and Jacobians of polys over the free angles, each poly scaled to unit size."""
    angle = sp.symbols(f"a0:{len(free)}", real=True)
    trig = {C[i]: sp.cos(angle[n]) for n, i in enumerate(free)} | {S[i]: sp.sin(angle[n]) for n, i in enumerate(free)}
    exprs = [sp.N(_real(p / _scale(p)).xreplace(trig), 17) for p in polys]
    values = sp.lambdify([angle], exprs, "numpy", cse=True)
    jacobian = sp.lambdify([angle], [[sp.diff(e, a) for a in angle] for e in exprs], "numpy", cse=True)

    def column(value, n):
        return np.broadcast_to(np.asarray(value, dtype=float), (n,))

    def residual(x):
        return np.stack([column(v, len(x)) for v in values(x.T)], axis=-1)

    def derivative(x):
        return np.stack([np.stack([column(v, len(x)) for v in row], axis=-1) for row in jacobian(x.T)], axis=1)

    return residual, derivative


def _newton(residual, derivative, x, iterations=200):
    """Project points onto {residual = 0} by Gauss-Newton steps (minimum norm, with backtracking).

    Returns the final points and a mask of those that converged. Many iterations are
    allowed because zeros of high order converge only linearly.
    """
    x = np.array(x, dtype=float)
    r = residual(x)
    f = np.sum(r * r, axis=1)
    active = np.ones(len(x), bool)
    for iteration in range(iterations):
        index = np.flatnonzero(active)
        if len(index) == 0:
            break
        step = np.einsum("nij,nj->ni", np.linalg.pinv(derivative(x[index]), rcond=1e-9), r[index])
        norm = np.linalg.norm(step, axis=1)
        alpha = np.minimum(1.0, 0.5 / np.maximum(norm, 1e-300))
        pending = np.ones(len(index), bool)
        for _ in range(10):
            p = np.flatnonzero(pending)
            trial = x[index[p]] - alpha[p, None] * step[p]
            r_trial = residual(trial)
            f_trial = np.sum(r_trial * r_trial, axis=1)
            better = f_trial < f[index[p]]
            accepted = index[p[better]]
            x[accepted], r[accepted], f[accepted] = trial[better], r_trial[better], f_trial[better]
            pending[p[better]] = False
            alpha[p[~better]] *= 0.5
            if not pending.any():
                break
        stop = pending | (alpha * norm < 1e-15) | ((iteration > 30) & (f[index] > 1e-6))
        active[index[stop]] = False
    return x, np.all(np.abs(r) < 1e-11, axis=1)


def _local_dimension(points, residual, derivative, h=1e-3, probes=12):
    """Dimension of the zero set near the points: project small random displacements back
    onto it and count the directions in which the projections spread."""
    rng = np.random.default_rng(1)
    chosen = points[rng.choice(len(points), min(len(points), 24), replace=False)]
    d = points.shape[1]
    directions = rng.normal(size=(len(chosen), probes, d))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    start = (chosen[:, None, :] + h * directions).reshape(-1, d)
    projected, converged = _newton(residual, derivative, start)
    shift = (projected.reshape(len(chosen), probes, d) - chosen[:, None, :] + np.pi) % (2 * np.pi) - np.pi
    shift[~converged.reshape(len(chosen), probes)] = 0.0
    singular = np.linalg.svd(shift, compute_uv=False)
    ranks = np.sum(singular > 0.1 * h, axis=1)
    return int(np.median(ranks))


def _sample_equations(polys, free, resolution):
    """Points (angles of the free axes, each in [-pi, pi)) on {polys = 0}, and the set's dimension."""
    d, m = len(free), len(polys)
    residual, derivative = _angle_functions(polys, free)
    grid = np.linspace(-np.pi, np.pi, resolution, endpoint=False)
    vertices = np.stack(np.meshgrid(*([grid] * d), indexing="ij"), axis=-1)
    F = residual(vertices.reshape(-1, d)).reshape(vertices.shape[:-1] + (m,))
    low, high = F.copy(), F.copy()
    for offset in product((0, 1), repeat=d):
        if any(offset):
            rolled = np.roll(F, shift=[-o for o in offset], axis=tuple(range(d)))
            low, high = np.minimum(low, rolled), np.maximum(high, rolled)
    cells = np.all((low <= 0) & (high >= 0), axis=-1)
    norm = np.abs(F).max(axis=-1)
    minima = norm < 0.05  # also seed at small local minima, for zeros where nothing changes sign
    for axis in range(d):
        for shift in (1, -1):
            minima &= norm <= np.roll(norm, shift, axis=axis)
    seeds = np.concatenate([vertices[cells] + np.pi / resolution, vertices[minima]])
    if len(seeds) == 0:
        return np.empty((0, d)), 0
    points, converged = _newton(residual, derivative, seeds)
    points = _unique_rows((points[converged] + np.pi) % (2 * np.pi) - np.pi, 1e-7)
    if len(points) == 0:
        return np.empty((0, d)), 0
    dimension = _local_dimension(points, residual, derivative)
    if dimension == 0:  # zeros of high order are located only to about eps^(1/order)
        points = _unique_rows(points, 1e-5)
    return points, dimension


# --------------------------------------------------------------------------- components

@dataclass
class Crossing:
    """One component of the crossings of a band pair.

    pair: "2-3", or "1-2,3-4" for bands 1-2 and 3-4, which touch together.
    dimension: 0 points, 1 lines or curves, 2 planes or surfaces, 3 the whole zone.
    fixed: {axis: exact momentum} for directions pinned to a value.
    relations: {axis: exact expression in the other momenta}, e.g. {"y": sqrt(3)*k_x}.
    equations: remaining exact conditions (sympy expressions in k that vanish).
    exact: True when the component is given by fixed values and relations only.
    samples: (N, 3) momenta on the component, inside the first Brillouin zone.
    gaps: gap of the pair at each sample, from diagonalizing the model.
    """
    pair: str
    dimension: int
    fixed: dict
    relations: dict
    equations: list
    exact: bool
    samples: np.ndarray = field(repr=False)
    gaps: np.ndarray = field(repr=False)
    hexagonal: bool = field(default=False, repr=False)
    residual: object = field(default=None, repr=False, compare=False)

    @property
    def max_gap(self):
        return float(np.max(self.gaps))

    def contains(self, k, tol=1e-6):
        """Mask of the momenta (N, 3) that lie on this component, up to reciprocal lattice vectors."""
        theta = _theta_of_k(np.asarray(k, dtype=float).reshape(-1, 3), self.hexagonal)
        best = np.full(len(theta), np.inf)
        for shift in _lattice_shifts(self.hexagonal):
            best = np.minimum(best, self.residual(theta + shift))
        return best < tol

    def describe(self, width=160):
        """One line. Values with CRootOf are shown numerically; long equations are cut at width."""
        def show(value):
            if value.has(sp.CRootOf):
                return f"{sp.sstr(sp.N(value, 8))} (approx.)"
            text = sp.sstr(value)
            return text if len(text) <= width else text[:width] + " ..."

        parts = [f"k_{axis} = {show(value)}" for axis, value in {**self.fixed, **self.relations}.items()]
        if self.exact:
            kind = ("point", "line", "plane", "everywhere")[self.dimension]
        else:
            kind = ("points", "curve", "surface")[self.dimension]
        if self.dimension == 0 and not self.exact:
            parts = ["(k_x, k_y, k_z) = " + ", ".join(
                "(" + ", ".join(f"{v:.6f}" for v in point) + ")" for point in self.samples)]
        else:
            parts += [f"{show(eq)} = 0" for eq in self.equations]
        how = "exact" if self.exact else f"exact equations, {len(self.samples)} samples"
        return f"{kind}: {'; '.join(parts) or 'all k'}  [{how}; max gap {self.max_gap:.1e}]"

    def to_dict(self):
        return {"pair": self.pair, "dimension": self.dimension, "exact": self.exact,
                "fixed": {a: sp.sstr(v) for a, v in self.fixed.items()},
                "relations": {a: sp.sstr(v) for a, v in self.relations.items()},
                "equations": [sp.sstr(eq) for eq in self.equations],
                "description": self.describe(width=10**6), "max_gap": self.max_gap,
                "samples": np.round(self.samples, 12).tolist()}


def _wrap_exact(value, period):
    """An exact value shifted by a multiple of period into (-period/2, period/2]."""
    n = math.ceil(float(sp.N((value - period / 2) / period, 30)) - 1e-12)
    return sp.simplify(value - n * period) if n else value


def _exact_momenta(fixed, links, hexagonal):
    """Exact fixed momenta and relations of a branch, reduced into the first zone where possible."""
    a = _scales(hexagonal)
    theta = {i: _ANGLES[c] if c in _ANGLES else sp.atan2(s, c) for i, (c, s) in fixed.items()}
    relations = {}
    for dep, (coeffs, constant) in links.items():
        value = sum(n * theta.get(r, a[r] * K[r] / 2) for r, n in coeffs) + sp.atan2(constant[1], constant[0])
        if all(r in theta for r, _ in coeffs):
            theta[dep] = value
        else:
            relations[dep] = sp.expand(2 * value / a[dep])
    momenta = {i: sp.simplify(2 * t / a[i]) for i, t in theta.items()}
    for i in momenta:
        if i == 2 or not hexagonal:
            momenta[i] = _wrap_exact(momenta[i], 2 * sp.pi)
    for j in relations:
        if j == 2 or not hexagonal:
            constant = relations[j].subs({k: 0 for k in K})
            relations[j] = sp.expand(relations[j] - constant + _wrap_exact(constant, 2 * sp.pi))
    if hexagonal and 0 in momenta and 1 in momenta:
        b1, b2 = (0, 4 * sp.pi / sp.sqrt(3)), (2 * sp.pi, 2 * sp.pi / sp.sqrt(3))
        point = np.array([[float(momenta[0]), float(momenta[1]), 0.0]])
        shift = point[0, :2] - wrap_to_bz(point, True)[0, :2]
        basis = np.array([[float(b1[0]), float(b2[0])], [float(b1[1]), float(b2[1])]])
        m, n = np.round(np.linalg.solve(basis, shift)).astype(int)
        momenta[0] = sp.simplify(momenta[0] - m * b1[0] - n * b2[0])
        momenta[1] = sp.simplify(momenta[1] - m * b1[1] - n * b2[1])
    elif hexagonal and 0 in momenta and 1 not in momenta and 1 not in relations:
        momenta[0] = _wrap_exact(momenta[0], 2 * sp.pi)
    return ({AXES[i]: momenta[i] for i in sorted(momenta)}, {AXES[j]: relations[j] for j in sorted(relations)})


def _equation_in_k(poly, hexagonal):
    """A remaining polynomial condition written in the momenta.

    Along an axis where every term has even degree in (C_i, S_i), the condition is
    written in cos(k_i) and sin(k_i) (at most linear in sin); otherwise in half angles.
    """
    a = _scales(hexagonal)
    gens = [g for i in range(3) for g in (C[i], S[i])]
    terms = sp.Poly(sp.expand(_real(poly)), *gens).terms()
    even = [all((m[2 * i] + m[2 * i + 1]) % 2 == 0 for m, _ in terms) for i in range(3)]
    expr = 0
    for monomial, coefficient in terms:
        term = coefficient
        for i in range(3):
            c_power, s_power = monomial[2 * i], monomial[2 * i + 1]
            if even[i]:
                full = a[i] * K[i]
                cos2, sin2 = (1 + sp.cos(full)) / 2, (1 - sp.cos(full)) / 2
                if c_power % 2:
                    term *= sp.sin(full) / 2 * cos2 ** (c_power // 2) * sin2 ** (s_power // 2)
                else:
                    term *= cos2 ** (c_power // 2) * sin2 ** (s_power // 2)
            else:
                term *= sp.cos(a[i] * K[i] / 2) ** c_power * sp.sin(a[i] * K[i] / 2) ** s_power
        expr += term
    expr = sp.expand(expr)
    if sp.count_ops(expr) < 300 and not expr.has(sp.CRootOf):
        expr = sp.factor(expr)
    _, expr = sp.factor_terms(expr).as_content_primitive()
    return -expr if expr.could_extract_minus_sign() else expr


def _component_residual(fixed, links, remaining):
    """Largest scaled residual of a branch's defining conditions, as a function of theta (N, 3)."""
    fixed_angles = {i: _angle(root) for i, root in fixed.items()}
    equations = None
    if remaining:
        theta = sp.symbols("t0:3", real=True)
        trig = {C[i]: sp.cos(theta[i]) for i in range(3)} | {S[i]: sp.sin(theta[i]) for i in range(3)}
        equations = sp.lambdify([theta], [sp.N(_real(p / _scale(p)).xreplace(trig), 17) for p in remaining],
                                "numpy", cse=True)

    def residual(x):
        x = np.asarray(x, dtype=float)
        worst = np.zeros(len(x))
        for i, angle in fixed_angles.items():
            worst = np.maximum(worst, np.abs((x[:, i] - angle + np.pi) % (2 * np.pi) - np.pi))
        for dep, (coeffs, constant) in links.items():
            worst = np.maximum(worst, _link_mismatch(x, dep, coeffs, constant))
        if equations is not None:
            for value in equations(x.T):
                worst = np.maximum(worst, np.abs(np.broadcast_to(np.asarray(value, dtype=float), (len(x),))))
        return worst

    return residual


def band_gaps(model, k, pair):
    """Gap of a band pair ("2-3" or "1-2,3-4") at momenta k (N, 3), by diagonalization."""
    energies = np.array([np.linalg.eigvalsh(model.compute_static(*point)) for point in k]).reshape(-1, 4)
    if pair == "2-3":
        return energies[:, 2] - energies[:, 1]
    return np.maximum(energies[:, 1] - energies[:, 0], energies[:, 3] - energies[:, 2])


def _branch_key(pair, fixed, links, remaining, hexagonal):
    """Branches with equal keys give the same set in the Brillouin zone."""
    def reduced(i, angle):
        period = 2 * np.pi if (i == 2 or not hexagonal) else 4 * np.pi
        return round((2 * angle) % period, 9) % round(period, 9)

    fixed_key = tuple(sorted((i, reduced(i, _angle(root))) for i, root in fixed.items()))
    links_key = tuple(sorted((dep, coeffs, reduced(dep, _angle(constant))) for dep, (coeffs, constant) in links.items()))
    equations = tuple(sorted(sp.sstr(sp.expand(eq / sp.Poly(eq, *_gens(eq)).LC())) for eq in remaining))
    return pair, fixed_key, links_key, equations


def exact_momentum(value):
    """An exact sympy number for a momentum: floats are matched to rational multiples of pi."""
    if isinstance(value, (float, np.floating)):
        return sp.nsimplify(float(value), [sp.pi], tolerance=1e-12)
    return sp.sympify(value)


def find_band_crossings(model, pairs=PAIRS, *, kz=None, resolution=64, line_samples=64, plane_samples=24):
    """All band crossings of a minimal model, as a list of Crossing components.

    pairs: any of "2-3" and "1-2,3-4" (bands 1-2 and 3-4 always touch together).
    kz: if given, only the crossings in the plane k_z = kz (a number, or an exact
        expression such as "2*pi/100"; floats are matched to multiples of pi).
    resolution: seed grid per angle (over two zone periods) for components sampled
        from their equations; components with three free directions use 40.
    line_samples, plane_samples: samples per zone period along exact lines and planes.
    """
    hexagonal = _hexagonal(model)
    conditions = _conditions(model)
    plane = {} if kz is None else {2: _plane_point(exact_momentum(kz))}
    found = []
    for pair in pairs:
        branches = []
        _split([_to_polynomial(e, hexagonal) for e in conditions[pair]], plane, {}, [], frozenset(), branches)
        seen = set()
        for fixed, links, remaining, excluded in branches:
            key = _branch_key(pair, fixed, links, remaining, hexagonal)
            if key not in seen:
                seen.add(key)
                found += _component(model, pair, fixed, links, remaining, excluded, hexagonal,
                                    resolution, line_samples, plane_samples)
    return _deduplicate(found)


def _component(model, pair, fixed, links, remaining, excluded, hexagonal, resolution, line_samples, plane_samples):
    free = [i for i in range(3) if i not in fixed and i not in links]
    if remaining:
        angles, dimension = _sample_equations(remaining, free, resolution if len(free) < 3 else 40)
    else:
        count = (1, 2 * line_samples, 2 * plane_samples, 16)[len(free)]
        grid = np.linspace(-np.pi, np.pi, count, endpoint=False)
        angles = (np.stack(np.meshgrid(*([grid] * len(free)), indexing="ij"), axis=-1).reshape(-1, len(free))
                  if free else np.zeros((1, 0)))
        dimension = len(free)
    theta = np.zeros((len(angles), 3))
    theta[:, free] = angles
    for i, root in fixed.items():
        theta[:, i] = _angle(root)
    for dep, (coeffs, constant) in links.items():
        theta[:, dep] = sum(n * theta[:, r] for r, n in coeffs) + _angle(constant)
    if remaining:
        for dep, coeffs, constant in excluded:
            theta = theta[_link_mismatch(theta, dep, coeffs, constant) > 1e-7]
        if len(theta) == 0:
            return []
    k = _unique_rows(wrap_to_bz(_k_of_theta(theta, hexagonal), hexagonal), 1e-7)
    momenta, relations = _exact_momenta(fixed, links, hexagonal)
    return [Crossing(pair, dimension, momenta, relations, [_equation_in_k(p, hexagonal) for p in remaining],
                     not remaining, k, band_gaps(model, k, pair), hexagonal,
                     _component_residual(fixed, links, remaining))]


def _deduplicate(found):
    """Drop components that lie inside another component of the same pair; sampled point
    sets also lose the points that lie on another component."""
    order = sorted(found, key=lambda c: (-c.dimension, not c.exact, len(c.equations), len(c.relations), len(c.fixed)))
    kept = []
    for candidate in order:
        others = [o for o in kept if o.pair == candidate.pair and o.dimension >= candidate.dimension]
        loose = 1e-5 if candidate.dimension == 0 and not candidate.exact else 1e-6
        covered = np.zeros(len(candidate.samples), bool)
        for other in others:
            covered |= other.contains(candidate.samples, tol=loose)
        if covered.all():
            continue
        if candidate.dimension == 0 and not candidate.exact and covered.any():
            candidate.samples, candidate.gaps = candidate.samples[~covered], candidate.gaps[~covered]
        kept.append(candidate)
    return sorted(kept, key=lambda c: (PAIRS.index(c.pair), -c.dimension, not c.exact,
                                       sp.sstr(c.fixed), sp.sstr(c.relations)))


def summarize(model, crossings, pairs=PAIRS):
    """Readable summary of find_band_crossings() for one model."""
    J = tuple(float(v) for v in (model.Jx, model.Jy, model.Jz))
    lines = [f"{type(model).__name__}  ({model.crystal_system}, J = {J})"]
    for pair in pairs:
        components = [c for c in crossings if c.pair == pair]
        lines.append(f"  {PAIR_NAMES[pair]}:" + ("" if components else " none"))
        lines += [f"    - {c.describe()}" for c in components]
    return "\n".join(lines)
