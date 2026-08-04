import inspect
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from Library.plotting_qgt_2d import (
    DEFAULT_BERRY_HEATMAP_PERCENTILE,
    get_coordinate_axis_labels,
    get_asymmetric_plot_limits,
    get_symmetric_plot_limits,
    plot_berry_irrep_projection_heatmaps,
    plot_qgt_eigenvalue_berry_component_heatmaps,
    plot_qgt_eigenvalue_berry_trace_heatmaps,
)


class TestQGTBerryHeatmapBounds(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_all_berry_heatmap_apis_default_to_percentile_scaling(self):
        for function in (
            plot_qgt_eigenvalue_berry_trace_heatmaps,
            plot_qgt_eigenvalue_berry_component_heatmaps,
            plot_berry_irrep_projection_heatmaps,
        ):
            default = inspect.signature(function).parameters[
                "zlim_percentile"
            ].default
            self.assertEqual(default, DEFAULT_BERRY_HEATMAP_PERCENTILE)

    def test_polar_order_uses_radial_angular_axis_labels(self):
        self.assertEqual(
            get_coordinate_axis_labels("xpz", backend="matplotlib"),
            (r"$r$", r"$\phi$ (rad)"),
        )
        self.assertEqual(
            get_coordinate_axis_labels("xpz", backend="plotly"),
            ("r", "phi (rad)"),
        )

        radius, phi = np.meshgrid(
            np.linspace(0.0, 1.0, 3),
            np.linspace(0.0, 2.0 * np.pi, 3, endpoint=False),
        )
        eigenvalues = np.ones((3, 3, 1))
        qgt_component = np.ones((3, 3))

        with patch.object(plt, "show"), patch.object(plt, "close"):
            plot_qgt_eigenvalue_berry_trace_heatmaps(
                radius,
                phi,
                eigenvalues,
                qgt_component,
                qgt_component,
                order="xpz",
            )
            figure = plt.gcf()

        for axis in figure.axes[:3]:
            self.assertEqual(axis.get_xlabel(), r"$r$")
            self.assertEqual(axis.get_ylabel(), r"$\phi$ (rad)")
            self.assertEqual(axis.get_aspect(), "auto")

    def test_absolute_percentile_is_symmetric_and_respects_cap(self):
        data = np.array([-4.0, -2.0, 1.0, 3.0, 100.0])
        percentile_limit = np.percentile(np.abs(data), 80)

        self.assertEqual(
            get_symmetric_plot_limits(data, zlim_percentile=80),
            (-percentile_limit, percentile_limit),
        )
        self.assertEqual(
            get_symmetric_plot_limits(data, limit=10, zlim_percentile=80),
            (-10.0, 10.0),
        )

    def test_collapsed_percentile_uses_full_data_extrema(self):
        data = np.array([0.0, 0.0, 0.0, 100.0])

        self.assertEqual(
            get_symmetric_plot_limits(data, zlim_percentile=50),
            (-100.0, 100.0),
        )
        self.assertEqual(
            get_asymmetric_plot_limits(data, zlim_percentile=50),
            (0.0, 100.0),
        )
        self.assertEqual(
            get_symmetric_plot_limits(data, limit=0),
            (-100.0, 100.0),
        )
        self.assertEqual(
            get_asymmetric_plot_limits(data, limit=0),
            (0.0, 100.0),
        )

    def test_constant_data_uses_exact_data_extrema(self):
        self.assertEqual(
            get_symmetric_plot_limits(
                np.zeros(4),
                zlim_percentile=99,
            ),
            (0.0, 0.0),
        )
        self.assertEqual(
            get_asymmetric_plot_limits(
                np.full(4, 7.0),
                zlim_percentile=99,
            ),
            (7.0, 7.0),
        )

    def test_trace_and_component_omega_z_use_the_same_percentile_bounds(self):
        axis = np.arange(3, dtype=float)
        kx, ky = np.meshgrid(axis, axis)
        omega_z = np.array(
            [
                [-4.0, -3.0, -2.0],
                [-1.0, 1.0, 2.0],
                [3.0, 4.0, 100.0],
            ]
        )
        g_xy_imag = -0.5 * omega_z
        eigenvalues = np.ones((3, 3, 1))
        trace = np.ones((3, 3))
        percentile = 80
        expected_limit = np.percentile(np.abs(omega_z), percentile)

        with patch.object(plt, "show"), patch.object(plt, "close"):
            plot_qgt_eigenvalue_berry_trace_heatmaps(
                kx,
                ky,
                eigenvalues,
                g_xy_imag,
                trace,
                zlim_percentile=percentile,
            )
            trace_figure = plt.gcf()
            trace_berry_norm = trace_figure.axes[1].collections[0].norm
            trace_berry_title = trace_figure.axes[1].get_title()

        with patch.object(plt, "show"), patch.object(plt, "close"):
            plot_qgt_eigenvalue_berry_component_heatmaps(
                kx,
                ky,
                eigenvalues,
                g_xy_imag,
                np.full((3, 3), 0.5),
                np.full((3, 3), -0.5),
                zlim_percentile=percentile,
            )
            component_figure = plt.gcf()
            component_berry_z_norm = component_figure.axes[3].collections[0].norm
            component_titles = [
                component_figure.axes[index].get_title()
                for index in range(1, 4)
            ]

        for norm in (trace_berry_norm, component_berry_z_norm):
            self.assertAlmostEqual(norm.vmin, -expected_limit)
            self.assertAlmostEqual(norm.vmax, expected_limit)
            self.assertEqual(norm.vcenter, 0.0)

        self.assertEqual(
            trace_berry_title,
            r"Berry Curvature $\Omega_{xy}$",
        )
        self.assertEqual(
            component_titles,
            [
                r"Berry Curvature $\Omega_x=\Omega_{yz}$",
                r"Berry Curvature $\Omega_y=\Omega_{zx}$",
                r"Berry Curvature $\Omega_z=\Omega_{xy}$",
            ],
        )


if __name__ == "__main__":
    unittest.main()
