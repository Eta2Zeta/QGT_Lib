import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "Plot_Specific"
    / "Polynomial_Basis_ND_Sweep"
    / "Polynomial_Basis_ND_Sweep.py"
)
SPEC = importlib.util.spec_from_file_location("polynomial_basis_nd_sweep", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_basis_pair_order_is_explicitly_x_then_y():
    """The original D3d evaluator remains unchanged."""
    terms = MODULE.polynomial_basis_terms(2.0, 3.0, 5.0)

    assert terms["a"]["x"] == -5.0
    assert terms["a"]["y"] == 12.0
    assert terms["b"]["x"] == 10.0
    assert terms["b"]["y"] == 15.0
    assert terms["c"]["x"] == -119.0
    assert terms["c"]["y"] == -120.0


def test_d6h_e2g_pairs_use_cartesian_harmonic_normalization():
    terms = MODULE.d6h_e2g_basis_terms(2.0, 3.0, 5.0)

    assert terms["a"] == {"x": -5.0, "y": 12.0}
    assert terms["b"] == {"x": -119.0, "y": -120.0}
    assert terms["c"] == {"x": -125.0, "y": 300.0}
    assert terms["d"] == {"x": -2975.0, "y": -3000.0}
    assert terms["e"] == {"x": -3125.0, "y": 7500.0}


def test_cartesian_harmonics_equal_r_to_m_times_angular_harmonics():
    x, y = 0.7, -0.4
    radius = np.hypot(x, y)
    phi = np.arctan2(y, x)
    harmonics = MODULE.cartesian_harmonics(x, y)

    for order in (1, 2, 4, 5, 6):
        assert harmonics[order]["C"] == pytest.approx(
            radius**order * np.cos(order * phi)
        )
        assert harmonics[order]["S"] == pytest.approx(
            radius**order * np.sin(order * phi)
        )


def test_symmetry_constrained_mode_uses_aligned_covariants_and_invariants():
    x, y = 0.7, -0.4
    coefficients = {
        name: 0.0 for name in MODULE.SYMMETRY_COEFFICIENT_DEFAULTS
    }
    coefficients.update({"a0": 1.2, "a1": -0.3, "b0": 0.5, "b2": 0.25})
    n_powers = dict(MODULE.SYMMETRY_N_DEFAULTS)
    q_powers = dict(MODULE.SYMMETRY_Q_DEFAULTS)
    n_powers.update({"a0": 0, "a1": 1, "b0": 2, "b2": 0})
    q_powers.update({"a0": 0, "a1": 1, "b0": 0, "b2": 2})

    actual = MODULE.evaluate_d6h_e2g_symmetry_constrained(
        x,
        y,
        coefficients,
        n_powers=n_powers,
        q_powers=q_powers,
    )
    harmonics = MODULE.cartesian_harmonics(x, y)
    r_squared = x * x + y * y
    c6 = harmonics[6]["C"]
    f2 = 1.2 - 0.3 * r_squared * c6
    f4 = 0.5 * r_squared**2 + 0.25 * c6**2
    expected = (
        f2 * harmonics[2]["C"] + f4 * harmonics[4]["C"],
        f2 * harmonics[2]["S"] - f4 * harmonics[4]["S"],
    )

    np.testing.assert_allclose(actual, expected)


def test_symmetry_constrained_mode_is_independent_of_z():
    coefficients = dict(MODULE.SYMMETRY_COEFFICIENT_DEFAULTS)
    at_zero = MODULE.evaluate_basis_polynomial(
        0.3,
        -0.8,
        0.0,
        coefficients,
        basis_mode=MODULE.D6H_E2G_SYMMETRY_MODE,
    )
    away_from_zero = MODULE.evaluate_basis_polynomial(
        0.3,
        -0.8,
        97.0,
        coefficients,
        basis_mode=MODULE.D6H_E2G_SYMMETRY_MODE,
    )

    np.testing.assert_allclose(at_zero, away_from_zero)

    with pytest.raises(ValueError, match="nonnegative integer"):
        MODULE.evaluate_d6h_e2g_symmetry_constrained(
            0.3,
            -0.8,
            coefficients,
            n_powers={"a0": 0.5},
        )


def test_each_basis_pair_has_an_independent_even_radial_power():
    coefficients = dict(MODULE.COEFFICIENT_DEFAULTS)
    coefficients["a"] = 0.0
    coefficients["b"] = 2.0
    x_component, y_component = MODULE.evaluate_polynomial(
        2.0,
        3.0,
        5.0,
        coefficients,
        radial_powers={"b": 1},
    )

    # r^2 = x^2 + y^2 = 13, so this is 2 * (r^2)^1 * (xz, yz).
    assert x_component == 260.0
    assert y_component == 390.0


def test_percentile_limit_clips_colors_without_changing_field_values():
    values = np.array([-100.0, -2.0, -1.0, 0.0, 1.0, 2.0])

    assert MODULE._symmetric_limit(values, None) == 100.0
    assert MODULE._symmetric_limit(values, 50.0) == np.percentile(
        np.abs(values),
        50.0,
    )
    np.testing.assert_array_equal(
        values,
        np.array([-100.0, -2.0, -1.0, 0.0, 1.0, 2.0]),
    )


def test_sign_filter_maps_components_to_minus_one_zero_or_plus_one():
    grid = MODULE.make_polar_grid(r_range=(0.0, 1.0), n_r=4, n_phi=8)
    figure = MODULE.plot_polynomial_basis(grid, show=False)
    viewer = figure.polynomial_viewer
    selected_x, selected_y = MODULE.evaluate_basis_polynomial(
        grid["x"],
        grid["y"],
        MODULE.Z_VALUE,
        MODULE.COEFFICIENT_DEFAULTS,
        basis_mode=MODULE.DEFAULT_BASIS_MODE,
    )
    assert MODULE.DEFAULT_BASIS_MODE == MODULE.D6H_E2G_MODE

    np.testing.assert_array_equal(
        np.asarray(viewer["meshes"][0].get_array()).ravel(),
        np.sign(selected_x).ravel(),
    )
    np.testing.assert_array_equal(
        np.asarray(viewer["meshes"][1].get_array()).ravel(),
        np.sign(selected_y).ravel(),
    )

    viewer["sign_filter_toggle"].set_active(0)
    assert viewer["display_options"]["sign_filter"] is False
    np.testing.assert_allclose(viewer["meshes"][0].get_array(), selected_x.ravel())
    np.testing.assert_allclose(viewer["meshes"][1].get_array(), selected_y.ravel())
    MODULE.plt.close(figure)


def test_basis_mode_toggle_switches_fields_and_active_controls():
    grid = MODULE.make_polar_grid(r_range=(0.0, 1.0), n_r=4, n_phi=8)
    coefficients = dict(MODULE.COEFFICIENT_DEFAULTS)
    coefficients["b"] = 0.5
    figure = MODULE.plot_polynomial_basis(
        grid,
        initial_coefficients=coefficients,
        sign_filter_components=False,
        show=False,
    )
    viewer = figure.polynomial_viewer

    expected_d6h = MODULE.evaluate_d6h_e2g_polynomial(
        grid["x"], grid["y"], MODULE.Z_VALUE, coefficients
    )
    np.testing.assert_allclose(
        np.asarray(viewer["meshes"][0].get_array()).ravel(),
        expected_d6h[0].ravel(),
    )
    np.testing.assert_allclose(
        np.asarray(viewer["meshes"][1].get_array()).ravel(),
        expected_d6h[1].ravel(),
    )
    assert viewer["display_options"]["basis_mode"] == MODULE.D6H_E2G_MODE
    assert viewer["coefficient_sliders"]["f"].ax.get_visible() is False
    assert viewer["coefficient_sliders"]["a"].valstep == MODULE.COEFFICIENT_STEP

    viewer["basis_mode_toggle"].set_active(1)
    expected_d3d = MODULE.evaluate_polynomial(
        grid["x"], grid["y"], MODULE.Z_VALUE, coefficients
    )
    np.testing.assert_allclose(viewer["meshes"][0].get_array(), expected_d3d[0].ravel())
    np.testing.assert_allclose(viewer["meshes"][1].get_array(), expected_d3d[1].ravel())
    assert viewer["display_options"]["basis_mode"] == MODULE.D3D_EG_MODE
    assert viewer["coefficient_sliders"]["f"].ax.get_visible() is True
    assert "n_i" in viewer["equation_text"].get_text()
    MODULE.plt.close(figure)


def test_symmetry_constrained_toggle_exposes_continuous_coefficients_and_integer_powers():
    grid = MODULE.make_polar_grid(r_range=(0.0, 1.0), n_r=4, n_phi=12)
    figure = MODULE.plot_polynomial_basis(
        grid,
        sign_filter_components=False,
        show=False,
    )
    viewer = figure.polynomial_viewer

    viewer["basis_mode_toggle"].set_active(2)
    expected = MODULE.evaluate_d6h_e2g_symmetry_constrained(
        grid["x"],
        grid["y"],
        viewer["symmetry_coefficients"],
        n_powers=viewer["symmetry_n_powers"],
        q_powers=viewer["symmetry_q_powers"],
    )

    assert viewer["display_options"]["basis_mode"] == MODULE.D6H_E2G_SYMMETRY_MODE
    assert viewer["coefficient_sliders"]["a"].ax.get_visible() is False
    assert viewer["symmetry_coefficient_sliders"]["a0"].ax.get_visible() is True
    assert (
        viewer["symmetry_coefficient_sliders"]["a0"].valstep
        == MODULE.COEFFICIENT_STEP
    )
    assert viewer["symmetry_n_sliders"]["a0"].valstep == 1
    assert viewer["symmetry_q_sliders"]["a0"].valstep == 1
    np.testing.assert_allclose(viewer["meshes"][0].get_array(), expected[0].ravel())
    np.testing.assert_allclose(viewer["meshes"][1].get_array(), expected[1].ravel())
    assert r"\mathbf{h}_4=(C_4,-S_4)" in viewer["equation_text"].get_text()
    visible_orders = [
        viewer["harmonic_table"][(row, 0)].get_text().get_text()
        for row in range(1, 6)
        if viewer["harmonic_table"][(row, 0)].get_visible()
    ]
    assert visible_orders == [r"$2$", r"$4$", r"$6$"]
    MODULE.plt.close(figure)


def test_realtime_viewer_recalculates_separate_x_and_y_components(tmp_path):
    grid = MODULE.make_polar_grid(r_range=(0.0, 1.0), n_r=4, n_phi=8)
    ranges = {"a": (0.0, 2.0), "b": (0.0, 2.0)}
    preview_path = tmp_path / "preview.png"
    figure = MODULE.plot_polynomial_basis(
        grid,
        ranges=ranges,
        sign_filter_components=False,
        output_png=preview_path,
        show=False,
    )
    viewer = figure.polynomial_viewer

    viewer["coefficient_sliders"]["b"].set_val(2.0)
    viewer["radial_power_sliders"]["b"].set_val(2)
    expected_coefficients = dict(MODULE.COEFFICIENT_DEFAULTS)
    expected_coefficients["b"] = 2.0
    expected_x, expected_y = MODULE.evaluate_basis_polynomial(
        grid["x"],
        grid["y"],
        MODULE.Z_VALUE,
        expected_coefficients,
        basis_mode=MODULE.DEFAULT_BASIS_MODE,
        radial_powers={"b": 2},
    )
    np.testing.assert_allclose(viewer["meshes"][0].get_array(), expected_x.ravel())
    np.testing.assert_allclose(viewer["meshes"][1].get_array(), expected_y.ravel())
    assert viewer["coefficients"]["b"] == 2.0
    assert viewer["radial_powers"]["b"] == 2
    assert preview_path.exists()
    MODULE.plt.close(figure)

    metadata_path = MODULE.save_configuration(grid, tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["evaluation"] == "real_time"
    assert metadata["component_order"] == ["x", "y"]
    assert metadata["default_basis_mode"] == MODULE.D6H_E2G_MODE
    assert metadata["basis_modes"][MODULE.D6H_E2G_MODE]["point_group"] == "D6h"
    assert metadata["basis_modes"][MODULE.D6H_E2G_MODE]["irrep"] == "E2g"
    assert metadata["basis_modes"][MODULE.D6H_E2G_MODE]["active_coefficients"] == list("abcde")
    assert metadata["basis_modes"][MODULE.D6H_E2G_MODE]["basis_pairs"]["b"] == {
        "x": "C_4 = (x^2-y^2)^2-4x^2y^2",
        "y": "S_4 = 4xy(x^2-y^2)",
    }
    assert metadata["basis_modes"][MODULE.D3D_EG_MODE]["point_group"] == "D3d"
    constrained = metadata["basis_modes"][MODULE.D6H_E2G_SYMMETRY_MODE]
    assert constrained["z_plane"] == 0.0
    assert constrained["terms_per_family"] == 3
    assert constrained["aligned_covariants"]["h4"]["y"] == (
        "-S_4 = -4xy(x^2-y^2)"
    )
    assert constrained["n_range"] == [0, 6]
    assert constrained["q_range"] == [0, 4]
    assert metadata["radial_factor"] == "(r^2)^n = (x^2+y^2)^n"
    assert metadata["coefficient_defaults"]["a"] == 1.0
    assert metadata["coefficient_step"] == 0.1
    assert metadata["radial_power_defaults"]["a"] == 0
    assert metadata["berry_lim_percentile"] == MODULE.BERRY_LIM_PERCENTILE
    assert metadata["sign_filter_components"] is True
    assert metadata["harmonic_notation"] == {
        "C_m": "r^m cos(m phi)",
        "S_m": "r^m sin(m phi)",
        "C_6": "r^6 cos(6 phi)",
        "r_squared": "x^2+y^2",
    }
    assert set(metadata["param_ranges"]) == set("abcdefghi")
    assert all(values == [-4.0, 4.0] for values in metadata["param_ranges"].values())
    assert metadata["basis_pairs"]["a"] == {
        "x": "C_2 = x^2-y^2",
        "y": "S_2 = 2xy",
    }
