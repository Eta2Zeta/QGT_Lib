import os
import tempfile
import unittest

import numpy as np

from Library.plotting_qgt_polar import (
    _polar_component_to_cartesian,
    plot_berry_components_vs_phi_radius_slider,
    plot_berry_winding_vs_radius,
)
from Library.topology import (
    berry_curvature_components_from_qgt,
    integrate_berry_flux_2d,
    winding_numbers_vs_radius,
)
from Library.dimension_lib import (
    create_2d_coordinate_grid_from_ranges,
    cylindrical_order_axes,
)


class TestPolarQGTWinding(unittest.TestCase):
    def setUp(self):
        self.radius_values = np.array([0.0, 0.25, 0.75])
        self.phi_values = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        self.radius, self.phi = np.meshgrid(
            self.radius_values,
            self.phi_values,
        )

    def _harmonic_field(self, winding):
        field = np.exp(1j * winding * self.phi)
        return field.real, field.imag

    def test_winding_is_calculated_for_every_positive_radius(self):
        field_real, field_imag = self._harmonic_field(-2)
        radius, winding = winding_numbers_vs_radius(
            self.radius,
            self.phi,
            field_real,
            field_imag,
        )

        np.testing.assert_allclose(radius, self.radius_values)
        self.assertTrue(np.isnan(winding[0]))
        np.testing.assert_allclose(winding[1:], -2.0, atol=1e-12)

    def test_common_zeros_make_winding_undefined(self):
        field = np.sin(3.0 * self.phi) * np.exp(1j * self.phi)
        _, winding = winding_numbers_vs_radius(
            self.radius,
            self.phi,
            field.real,
            field.imag,
        )
        self.assertTrue(np.all(np.isnan(winding)))

    def test_qgt_to_berry_component_signs(self):
        omega_x, omega_y, omega_z = berry_curvature_components_from_qgt(
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
        )
        np.testing.assert_allclose(omega_x, [-6.0])
        np.testing.assert_allclose(omega_y, [4.0])
        np.testing.assert_allclose(omega_z, [-2.0])

    def test_constant_normal_field_has_correct_flux_for_every_polar_order(self):
        expected_flux = 2.0 * np.pi * 1.5**2
        for order in ("xpz", "ypz", "xpy", "zpy", "ypx", "zpx"):
            radius, phi, info = create_2d_coordinate_grid_from_ranges(
                (0.0, 1.5),
                (0.0, 2.0 * np.pi),
                80,
                order=order,
            )
            components = {
                axis: np.zeros_like(radius)
                for axis in "xyz"
            }
            _, _, _, fixed_axis = cylindrical_order_axes(order)
            components[fixed_axis].fill(2.0)

            flux = integrate_berry_flux_2d(
                components["x"],
                components["y"],
                components["z"],
                radius,
                phi,
                order=order,
                phi_periodic=info["phi_periodic"],
            )
            self.assertAlmostEqual(flux, expected_flux, places=10)

    def test_periodic_polar_field_is_interpolated_to_cartesian_plane(self):
        radial_field = self.radius**2
        x_values, y_values, cartesian_field, plane_axes = (
            _polar_component_to_cartesian(
                self.radius_values,
                self.phi_values,
                radial_field,
                order="xpz",
                resolution=7,
            )
        )

        self.assertEqual(plane_axes, ("x", "y"))
        x_index = int(np.argmin(np.abs(x_values - 0.25)))
        y_index = int(np.argmin(np.abs(y_values)))
        self.assertAlmostEqual(cartesian_field[y_index, x_index], 0.25**2)
        self.assertTrue(np.isnan(cartesian_field[0, 0]))

    def test_polar_plots_are_saved(self):
        omega_x, omega_y = self._harmonic_field(1)
        omega_z = np.cos(2.0 * self.phi)
        _, winding = winding_numbers_vs_radius(
            self.radius,
            self.phi,
            omega_x,
            omega_y,
        )

        with tempfile.TemporaryDirectory() as output_dir:
            winding_path = plot_berry_winding_vs_radius(
                self.radius_values,
                winding,
                band_index=0,
                results_dir=output_dir,
            )
            slider_path = plot_berry_components_vs_phi_radius_slider(
                self.radius,
                self.phi,
                omega_x,
                omega_y,
                omega_z,
                band_index=0,
                results_dir=output_dir,
            )

            self.assertTrue(os.path.exists(winding_path))
            self.assertTrue(os.path.exists(slider_path))
            with open(slider_path, "r", encoding="utf-8") as slider_file:
                slider_html = slider_file.read()
            self.assertIn("Radius r:", slider_html)
            self.assertIn("Omega_x", slider_html)
            self.assertIn("Cartesian Berry curvature and polar-ring cuts", slider_html)
            self.assertIn("Selected radius", slider_html)
            self.assertIn("k_x", slider_html)
            self.assertIn("k_y", slider_html)
            self.assertIn('"redraw":false', slider_html)
            self.assertIn('"images":[', slider_html)
            self.assertIn("base64", slider_html)
            self.assertIn("Winding number versus radius", slider_html)
            self.assertIn("In-plane Berry magnitude versus phi", slider_html)
            self.assertIn("sqrt(Omega_x^2 + Omega_y^2)", slider_html)


if __name__ == "__main__":
    unittest.main()
