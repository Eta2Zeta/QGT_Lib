import json
from pathlib import Path
import re

import numpy as np
import pytest

from Plot_QGT_2D_Dynamic_ND_sweep_matplotlib import (
    dynamic_nd_field_with_bands_matplotlib,
    plot_berry_components_nd_matplotlib,
)
from Plot_QGT_2D_Dynamic_ND_sweep import (
    _berry_component_grids,
    _pick_field_grid,
    dynamic_nd_field_with_bands_html,
    plot_berry_components_nd_html,
)


def _write_bundle(root: Path, *, polar: bool, include_winding: bool = True):
    parameter_values = np.array([-0.2, 0.2])
    if polar:
        radius_values = np.linspace(0.0, 1.0, 5)
        phi_values = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        ki, kj = np.meshgrid(radius_values, phi_values)
        kx = ki * np.cos(kj)
        ky = ki * np.sin(kj)
        order = "xpz"
        phi_periodic = True
    else:
        coordinate_values = np.linspace(-1.0, 1.0, 5)
        ki, kj = np.meshgrid(coordinate_values, coordinate_values)
        kx, ky = ki, kj
        order = "xyz"
        phi_periodic = False

    omega_x = np.empty((2,) + ki.shape)
    omega_y = np.empty_like(omega_x)
    omega_z = np.empty_like(omega_x)
    for index, parameter in enumerate(parameter_values):
        amplitude = 1.0 + parameter
        omega_x[index] = amplitude * kx
        omega_y[index] = amplitude * ky
        omega_z[index] = parameter + 0.25 * (kx**2 - ky**2)

    k_dist = np.array([0.0, 0.5, 1.0])
    eigenvalues = np.empty((2, 3, 4))
    for index, parameter in enumerate(parameter_values):
        eigenvalues[index] = np.column_stack(
            [
                -2.0 + parameter + k_dist,
                -1.0 + parameter + 0.5 * k_dist,
                1.0 + parameter - 0.5 * k_dist,
                2.0 + parameter - k_dist,
            ]
        )

    saved_winding = {}
    if polar and include_winding:
        winding_grid = np.ones((parameter_values.size, radius_values.size))
        winding_grid[:, 0] = np.nan
        saved_winding = {
            "winding_radius": radius_values,
            "winding_grid": winding_grid,
        }

    np.savez_compressed(
        root / "qgt_nd_bundle.npz",
        names=np.array(["lamb_z"], dtype=object),
        shape=np.array([2], dtype=int),
        axis_0_lamb_z=parameter_values,
        ki=ki,
        kj=kj,
        kx=kx,
        ky=ky,
        kz=np.zeros_like(kx),
        order=np.array(order),
        kk=np.float64(0.0),
        coordinate_system=np.array("cylindrical" if polar else "cartesian"),
        coordinate_labels=np.array(
            ["r", "phi", "kz"] if polar else ["kx", "ky", "kz"],
            dtype=object,
        ),
        phi_periodic=np.bool_(phi_periodic),
        g_xy_imag_grid=-0.5 * omega_z,
        g_xz_imag_grid=0.5 * omega_y,
        g_yz_imag_grid=-0.5 * omega_x,
        trace_grid=omega_x**2 + omega_y**2 + omega_z**2,
        eigenvalues_sym_grid=eigenvalues,
        k_dist=k_dist,
        node_indices=np.array([0, 1, 2], dtype=int),
        path_labels=np.array(["G", "M", "K"], dtype=object),
        path_points=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.03],
                [0.5, 0.5, np.pi],
            ]
        ),
        **saved_winding,
    )

    with open(root / "parameters.json", "w", encoding="utf-8") as stream:
        json.dump(
            {
                "hamiltonian_name": "SyntheticBerryHamiltonian",
                "parameters": {"Jz": 0.2},
                "scan_ranges": {"lamb_z": {"min": -0.2, "max": 0.2}},
                "scan_spacing": {
                    "lamb_z": {"count": 2, "scale": "linear"}
                },
                "band_index": 0,
                "k_grid": {
                    "coordinate_system": (
                        "cylindrical" if polar else "cartesian"
                    ),
                    "coordinate_labels": (
                        ["r", "phi", "kz"]
                        if polar
                        else ["kx", "ky", "kz"]
                    ),
                    "order": order,
                    "ki_domain": [float(np.min(ki)), float(np.max(ki))],
                    "kj_domain": [0.0, float(2.0 * np.pi)] if polar else [-1.0, 1.0],
                    "fixed_coordinate": 0.0,
                    "kx_min": float(np.min(kx)),
                    "kx_max": float(np.max(kx)),
                    "ky_min": float(np.min(ky)),
                    "ky_max": float(np.max(ky)),
                    "mesh": int(ki.shape[0]),
                },
            },
            stream,
        )

    return omega_x, omega_y, omega_z


def test_berry_component_conversion_uses_pseudovector_convention(tmp_path):
    expected = _write_bundle(tmp_path, polar=False)
    with np.load(tmp_path / "qgt_nd_bundle.npz", allow_pickle=True) as data:
        actual = _berry_component_grids(data)

    for actual_component, expected_component in zip(actual, expected):
        np.testing.assert_allclose(actual_component, expected_component)


@pytest.mark.parametrize(
    ("order", "expected_value"),
    [
        ("xpz", 3.0),
        ("ypz", 3.0),
        ("xpy", 2.0),
        ("zpy", 2.0),
        ("ypx", 1.0),
        ("zpx", 1.0),
    ],
)
def test_scalar_polar_berry_selects_plane_normal_component(
    tmp_path,
    order,
    expected_value,
):
    shape = (1, 2, 2)
    np.savez_compressed(
        tmp_path / "components.npz",
        g_xy_imag_grid=-0.5 * np.full(shape, 3.0),
        g_xz_imag_grid=0.5 * np.full(shape, 2.0),
        g_yz_imag_grid=-0.5 * np.full(shape, 1.0),
    )
    with np.load(tmp_path / "components.npz", allow_pickle=True) as data:
        berry = _pick_field_grid(data, "berry", order=order)

    np.testing.assert_allclose(berry, expected_value)


def test_cartesian_berry_components_have_bands_and_three_maps(tmp_path):
    _write_bundle(tmp_path, polar=False)
    output = tmp_path / "cartesian_components.html"

    plot_berry_components_nd_html(
        tmp_path,
        output_html=output,
        berry_zlim_percentile=100.0,
    )
    html = output.read_text(encoding="utf-8")

    assert "const isPolar = false" in html
    assert 'id="radius-slider"' not in html
    assert "Omega_x" in html
    assert "Omega_y" in html
    assert "Omega_z" in html
    assert "const windingData = null" in html


def test_polar_berry_components_add_ring_cuts_winding_and_magnitude(tmp_path):
    _write_bundle(tmp_path, polar=True)
    output = tmp_path / "polar_components.html"

    plot_berry_components_nd_html(
        tmp_path,
        output_html=output,
        berry_zlim_percentile=100.0,
        cartesian_resolution=9,
    )
    html = output.read_text(encoding="utf-8")

    assert "const isPolar = true" in html
    assert 'id="radius-slider"' in html
    assert "Winding number versus radius" in html
    assert "In-plane Berry magnitude versus phi" in html
    assert "function updateRadiusDependent()" in html
    assert "function updateParameterPlot()" in html
    assert "const circleTraceIndices = [" in html
    assert "Plotly.restyle('qgt-plot', {z: [map]}" in html


def test_existing_scalar_viewer_interpolates_polar_data_to_cartesian(tmp_path):
    _write_bundle(tmp_path, polar=True)
    output = tmp_path / "polar_trace.html"

    dynamic_nd_field_with_bands_html(
        tmp_path,
        quantity="trace",
        output_html=output,
        cartesian_resolution=9,
        trace_zmax_percentile=100.0,
    )
    html = output.read_text(encoding="utf-8")
    match = re.search(r"const fieldData = (.*?);\n", html)

    assert match is not None
    embedded_field = np.asarray(json.loads(match.group(1)), dtype=float)
    assert embedded_field.shape == (2, 9, 9)
    assert 'id="radius-slider"' not in html
    assert "const hasRatioHover = false" in html


def test_missing_off_plane_qgt_components_requests_recalculation(tmp_path):
    axis = np.linspace(-1.0, 1.0, 3)
    kx, ky = np.meshgrid(axis, axis)
    np.savez_compressed(
        tmp_path / "qgt_nd_bundle.npz",
        names=np.array(["V"], dtype=object),
        shape=np.array([1], dtype=int),
        axis_0_V=np.array([0.0]),
        kx=kx,
        ky=ky,
        g_xy_imag_grid=np.zeros((1, 3, 3)),
    )

    with pytest.raises(KeyError, match="Recalculate this N-D bundle"):
        plot_berry_components_nd_html(tmp_path)


def test_cartesian_matplotlib_viewer_updates_maps_and_bands(tmp_path):
    _write_bundle(tmp_path, polar=False)
    figure = plot_berry_components_nd_matplotlib(
        tmp_path,
        bands_to_plot=[0, 1],
        berry_zlim_percentile=100.0,
        show=False,
    )
    viewer = figure.qgt_viewer

    assert viewer["is_polar"] is False
    assert len(viewer["map_images"]) == 3
    assert len(viewer["band_lines"]) == 2
    assert viewer["radius_slider"] is None

    initial_map = np.asarray(viewer["map_images"][0].get_array()).copy()
    initial_band = np.asarray(viewer["band_lines"][0].get_ydata()).copy()
    viewer["parameter_sliders"][0].set_val(0)

    assert viewer["current_indices"] == [0]
    assert not np.allclose(viewer["map_images"][0].get_array(), initial_map)
    assert not np.allclose(viewer["band_lines"][0].get_ydata(), initial_band)

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_polar_matplotlib_viewer_updates_radius_diagnostics(tmp_path):
    _write_bundle(tmp_path, polar=True)
    figure = plot_berry_components_nd_matplotlib(
        tmp_path,
        bands_to_plot=[0, 1],
        berry_zlim_percentile=100.0,
        cartesian_resolution=9,
        show=False,
    )
    viewer = figure.qgt_viewer

    assert viewer["is_polar"] is True
    assert viewer["order"] == "xpz"
    assert len(viewer["map_images"]) == 3
    assert len(viewer["ring_lines"]) == 6
    assert len(viewer["component_cut_lines"]) == 3
    assert viewer["winding_line"] is not None
    assert viewer["magnitude_line"] is not None
    assert viewer["symmetry_point_labels"] == ["G", "M"]
    assert len(viewer["symmetry_point_artists"]) == 3
    for artist in viewer["symmetry_point_artists"]:
        assert artist.get_offsets().shape == (2, 2)
    assert viewer["symmetry_slice_width"] == pytest.approx(0.02 * np.pi)

    initial_ring_x = np.asarray(viewer["ring_lines"][0].get_xdata()).copy()
    initial_cut = np.asarray(
        viewer["component_cut_lines"][0].get_ydata()
    ).copy()
    viewer["radius_slider"].set_val(2)

    assert viewer["radius_position"] == [2]
    assert not np.allclose(viewer["ring_lines"][0].get_xdata(), initial_ring_x)
    assert not np.allclose(
        viewer["component_cut_lines"][0].get_ydata(),
        initial_cut,
    )
    selected_radius = float(viewer["ring_lines"][0].get_xdata()[0])
    marker_x = np.asarray(viewer["winding_radius_line"].get_xdata())
    np.testing.assert_allclose(marker_x, selected_radius)

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_polar_matplotlib_ring_limits_match_percentile_clipped_maps(tmp_path):
    components = _write_bundle(tmp_path, polar=True)
    percentile = 60.0
    figure = plot_berry_components_nd_matplotlib(
        tmp_path,
        berry_zlim_percentile=percentile,
        cartesian_resolution=9,
        show=False,
    )
    viewer = figure.qgt_viewer

    for component, image, line in zip(
        components,
        viewer["map_images"],
        viewer["component_cut_lines"],
    ):
        limit = float(np.percentile(np.abs(component), percentile))
        expected_limits = (-limit, limit)
        assert image.get_clim() == pytest.approx(expected_limits)
        assert line.axes.get_ylim() == pytest.approx(expected_limits)

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_polar_matplotlib_viewer_requires_saved_winding_data(tmp_path):
    _write_bundle(tmp_path, polar=True, include_winding=False)

    with pytest.raises(KeyError, match="saved winding data"):
        plot_berry_components_nd_matplotlib(
            tmp_path,
            cartesian_resolution=9,
            show=False,
        )

    with pytest.raises(KeyError, match="saved winding data"):
        plot_berry_components_nd_html(
            tmp_path,
            cartesian_resolution=9,
        )


@pytest.mark.parametrize("polar", [False, True])
def test_scalar_matplotlib_viewer_uses_same_nd_bundle(tmp_path, polar):
    _write_bundle(tmp_path, polar=polar)
    figure = dynamic_nd_field_with_bands_matplotlib(
        tmp_path,
        quantity="trace",
        bands_to_plot=[0, 1],
        trace_zmax_percentile=100.0,
        cartesian_resolution=9,
        show=False,
    )
    viewer = figure.qgt_viewer

    assert viewer["is_polar"] is polar
    assert viewer["quantity"] == "trace"
    assert len(viewer["map_images"]) == 1
    assert len(viewer["band_lines"]) == 2
    if polar:
        assert np.asarray(viewer["map_images"][0].get_array()).shape == (9, 9)
    assert viewer["symmetry_point_labels"] == ["G", "M"]
    assert len(viewer["symmetry_point_artists"]) == 1

    initial_map = np.asarray(viewer["map_images"][0].get_array()).copy()
    viewer["parameter_sliders"][0].set_val(0)
    assert not np.allclose(
        viewer["map_images"][0].get_array(),
        initial_map,
        equal_nan=True,
    )

    import matplotlib.pyplot as plt

    plt.close(figure)
