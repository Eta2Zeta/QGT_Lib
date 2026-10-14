import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from qgt.hamiltonians.minimal_models import MINIMAL_MODELS, MinimalHamSG127_2a2b, MinimalHamSG192_2b, MinimalHamSG194_2a
from qgt.hamiltonians.minimal_models import crossings as cr

REPO = Path(__file__).resolve().parents[1]


def test_spectrum_identity_behind_the_crossing_conditions():
    tx, tz, E = sp.symbols("t_x t_z E", real=True)
    lam = sp.Matrix(sp.symbols("l_x l_y l_z", real=True))
    J = sp.Matrix(sp.symbols("J_x J_y J_z", real=True))
    pauli = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.Matrix([[1, 0], [0, -1]])]
    one = sp.eye(2)
    H = (tx * sp.kronecker_product(pauli[0], one) + tz * sp.kronecker_product(pauli[2], one)
         + sum((lam[i] * sp.kronecker_product(pauli[1], pauli[i]) + J[i] * sp.kronecker_product(pauli[2], pauli[i])
                for i in range(3)), sp.zeros(4)))
    X = tx**2 + tz**2 + lam.dot(lam) + J.dot(J)
    Y2 = tz**2 * J.dot(J) + lam.cross(J).dot(lam.cross(J))
    assert sp.expand((E * sp.eye(4) - H).det() - ((E**2 - X)**2 - 4 * Y2)) == 0
    A, B = tz**2 + lam.dot(lam), J.dot(J)
    assert sp.expand(X**2 - 4 * Y2 - (tx**4 + 2 * tx**2 * (A + B) + (A - B)**2 + 4 * lam.dot(J)**2)) == 0


def _independent_crossings(model, pair, seeds=400):
    """Crossings found by Gauss-Newton on the unfactored conditions, from random starts."""
    conditions = cr.crossing_conditions(model)[pair]
    if not conditions:
        return np.empty((0, 3))
    exprs = [sp.N(e, 17) for e in conditions]
    values = sp.lambdify([cr.K], exprs, "numpy", cse=True)
    jacobian = sp.lambdify([cr.K], [[sp.diff(e, k) for k in cr.K] for e in exprs], "numpy", cse=True)

    def column(v, n):
        return np.broadcast_to(np.asarray(v, dtype=float), (n,))

    def F(x):
        return np.stack([column(v, len(x)) for v in values(x.T)], axis=-1)

    def dF(x):
        return np.stack([np.stack([column(v, len(x)) for v in row], axis=-1) for row in jacobian(x.T)], axis=1)

    x = np.random.default_rng(1).uniform(-1.4 * np.pi, 1.4 * np.pi, (seeds, 3))
    scale = np.maximum(np.abs(F(x)).max(axis=0), 1e-12)
    x, converged = cr._newton(lambda y: F(y) / scale, lambda y: dF(y) / scale[None, :, None], x)
    return cr.wrap_to_bz(x[converged], cr._hexagonal(model))


@pytest.mark.parametrize("name", sorted(MINIMAL_MODELS))
def test_every_minimal_model_crossings_are_degenerate_and_complete(name):
    model = MINIMAL_MODELS[name]()
    found = cr.find_band_crossings(model)
    assert found
    for component in found:
        assert component.max_gap < 1e-10, component.describe()
        assert np.all(component.contains(component.samples, tol=1e-5)), component.describe()
    for pair in cr.PAIRS:
        points = _independent_crossings(model, pair)
        components = [c for c in found if c.pair == pair]
        on = np.zeros(len(points), bool)
        near = np.zeros(len(points), bool)
        for component in components:
            on |= component.contains(points)
            near |= component.contains(points, tol=1e-2)
        # zeros of high order are found only to about eps^(1/order) by the independent search
        loose = ~on & near & (cr.band_gaps(model, points, pair) < 1e-12)
        assert np.all(on | loose), (pair, points[~(on | loose)][:5])


def test_exact_lines_and_planes():
    found = cr.find_band_crossings(MinimalHamSG127_2a2b())
    lines = [(c.fixed, c.relations) for c in found if c.exact and c.dimension == 1]
    assert ({"z": 0}, {"y": cr.KX}) in lines and ({"x": sp.pi, "y": sp.pi}, {}) in lines
    hexagonal = cr.find_band_crossings(MinimalHamSG192_2b(), pairs=("1-2,3-4",))
    relations = [c.relations.get("y") for c in hexagonal if c.exact]
    for slope in (sp.sqrt(3), -sp.sqrt(3), sp.sqrt(3) / 3, -sp.sqrt(3) / 3):
        assert any(r is not None and sp.simplify(r - slope * cr.KX) == 0 for r in relations), slope
    screw = cr.find_band_crossings(MinimalHamSG194_2a(), pairs=("1-2,3-4",))
    assert any(c.exact and c.dimension == 2 and c.fixed == {"z": sp.pi} for c in screw)


def test_zero_exchange_and_in_plane_exchange():
    found = cr.find_band_crossings(MinimalHamSG127_2a2b(Jz=0.0))
    assert any(c.pair == "1-2,3-4" and c.dimension == 3 for c in found)
    assert all(c.max_gap < 1e-10 for c in found)
    model = MinimalHamSG127_2a2b(Jx=0.15, Jz=0.1)
    found = cr.find_band_crossings(model)
    assert all(c.max_gap < 1e-10 for c in found)
    for pair in cr.PAIRS:
        points = _independent_crossings(model, pair, seeds=200)
        assert all(any(c.contains(points[i:i + 1], tol=1e-2)[0] for c in found if c.pair == pair)
                   for i in range(len(points)))


def test_crossings_in_the_planes_of_the_minimal_model_runs():
    # SG127: tz = t4 sin kx sin ky (cos kx - cos ky); lambda_x, lambda_y carry sin kz.
    # On kz = 0 only tz = 0 remains (lines); on kz = 2 pi/100 lambda_x = lambda_y = 0 too (points).
    model = MinimalHamSG127_2a2b()
    flat = cr.find_band_crossings(model, kz=0.0)
    assert {(sp.sstr(c.fixed), sp.sstr(c.relations)) for c in flat} == {
        ("{x: 0, z: 0}", "{}"), ("{x: pi, z: 0}", "{}"), ("{y: 0, z: 0}", "{}"), ("{y: pi, z: 0}", "{}"),
        ("{z: 0}", "{y: k_x}"), ("{z: 0}", "{y: -k_x}")}
    tilted = cr.find_band_crossings(model, kz=2 * np.pi / 100)
    assert {(sp.sstr(c.fixed["x"]), sp.sstr(c.fixed["y"])) for c in tilted} == {("0", "0"), ("pi", "pi")}
    assert all(c.dimension == 0 and c.exact and c.fixed["z"] == sp.pi / 50 for c in tilted)
    # SG11: 5 sin kx + 3 sin kz = 0 at ky = pi crosses kz = 2 pi/100 at two exact points
    points = cr.find_band_crossings(MINIMAL_MODELS["MinimalHamSG11_2a2b2c2d"](), kz="2*pi/100")
    assert len(points) == 2 and all(c.exact and c.max_gap < 1e-12 for c in points)
    values = sorted(float(c.fixed["x"]) % (2 * np.pi) for c in points)
    expected = sorted(v % (2 * np.pi) for v in (-np.arcsin(0.6 * np.sin(np.pi / 50)),
                                                  np.pi + np.arcsin(0.6 * np.sin(np.pi / 50))))
    assert np.allclose(values, expected, atol=1e-12)


@pytest.mark.parametrize("name", ["MinimalHamSG163_2b", "MinimalHamSG192_2b", "MinimalHamSG223_2a"])
def test_plane_crossings_agree_with_the_full_zone(name):
    model = MINIMAL_MODELS[name]()
    full = cr.find_band_crossings(model)
    for kz in (0.0, 2 * np.pi / 100):
        for component in cr.find_band_crossings(model, kz=kz):
            assert component.max_gap < 1e-10
            assert np.allclose(component.samples[:, 2], kz)
            others = [c for c in full if c.pair == component.pair]
            on = np.zeros(len(component.samples), bool)
            for other in others:
                on |= other.contains(component.samples, tol=1e-5)
            assert on.all(), component.describe()


def test_wrap_to_bz_keeps_points_equivalent():
    k = np.random.default_rng(0).uniform(-9, 9, (500, 3))
    for hexagonal in (False, True):
        wrapped = cr.wrap_to_bz(k, hexagonal)
        theta, back = cr._theta_of_k(k, hexagonal), cr._theta_of_k(wrapped, hexagonal)
        difference = (back - theta)[:, None, :] - cr._lattice_shifts(hexagonal)[None]
        assert np.all(np.min(np.abs(np.sin(difference)).max(axis=2), axis=1) < 1e-9)
        if hexagonal:
            assert np.all(np.linalg.norm(wrapped[:, :2], axis=1) <= 4 * np.pi / 3 + 1e-9)
        else:
            assert np.all(np.abs(wrapped) <= np.pi + 1e-12)


def test_matches_the_gwave_nodal_network_tracer():
    pytest.importorskip("contourpy")
    path = REPO / "scripts/figures/gWave_Nodal_Network/Plot_gWave_Nodal_Network_3D.py"
    spec = importlib.util.spec_from_file_location("gwave_nodal_network", path)
    tracer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tracer
    spec.loader.exec_module(tracer)
    curves, junctions = tracer.trace_network(tracer.PARAMETERS, grid=201)
    traced = np.concatenate([np.asarray(c.xyz) for c in curves] + [junctions])
    found = cr.find_band_crossings(MinimalHamSG194_2a(**tracer.PARAMETERS), pairs=("2-3",))
    on = np.zeros(len(traced), bool)
    for component in found:
        on |= component.contains(traced, tol=1e-6)
    assert on.all()


def test_run_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("QGT_DATA", str(tmp_path))
    from qgt.workflows.band_crossings import band_crossings

    run_dir = band_crossings(MinimalHamSG127_2a2b(), plot=True)
    data = json.loads(Path(run_dir, "crossings.json").read_text())
    assert data["model"] == "MinimalHamSG127_2a2b" and data["components"]
    assert {"crossings.html", "summary.txt", "meta.json", "run.json"} <= {p.name for p in Path(run_dir).iterdir()}
