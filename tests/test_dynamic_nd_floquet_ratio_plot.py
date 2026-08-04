import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from Plot_QGT_2D_Dynamic_ND_sweep import (
    _floquet_ratio_summary_grid,
    _prepare_floquet_ratio_field,
    dynamic_nd_field_with_bands_html,
    plot_floquet_max_ratio_nd_html,
)


def _diagnostic_arrays():
    ratios = np.array(
        [
            [
                [[0.1, 0.7], [np.inf, 0.2]],
                [[0.0, 0.0], [np.nan, np.nan]],
            ],
            [
                [[0.3, 0.4], [0.5, 0.2]],
                [[0.6, 0.8], [0.9, 0.1]],
            ],
        ],
        dtype=float,
    )
    indices = np.full(ratios.shape + (2,), (-1, 0), dtype=np.int32)
    indices[0, 0, 0, 0] = (1, 1)
    indices[0, 0, 0, 1] = (0, -2)
    indices[0, 0, 1, 0] = (1, 1)
    indices[0, 0, 1, 1] = (0, -1)
    indices[1, ..., 0, :] = (1, 1)
    indices[1, ..., 1, :] = (0, -1)
    return ratios, indices


def _write_bundle(root_dir):
    ratios, indices = _diagnostic_arrays()
    axis = np.array([-1.0, 1.0])
    kx, ky = np.meshgrid(axis, axis)
    np.savez_compressed(
        root_dir / "qgt_nd_bundle.npz",
        names=np.array(["V"], dtype=object),
        shape=np.array([2], dtype=int),
        axis_0_V=np.array([0.0, 1.0]),
        kx=kx,
        ky=ky,
        dkx=2.0,
        dky=2.0,
        trace_grid=np.arange(8, dtype=float).reshape(2, 2, 2),
        floquet_max_ratio_grid=ratios,
        floquet_max_ratio_indices_grid=indices,
    )
    with open(root_dir / "parameters.json", "w", encoding="utf-8") as stream:
        json.dump(
            {
                "hamiltonian_name": "SyntheticFloquetHamiltonian",
                "scan_ranges": {"V": {"min": 0.0, "max": 1.0}},
                "scan_spacing": {"V": {"count": 2, "scale": "linear"}},
                "k_grid": {
                    "kx_min": -1.0,
                    "kx_max": 1.0,
                    "ky_min": -1.0,
                    "ky_max": 1.0,
                    "mesh": 2,
                },
                "floquet_diagnostic": {
                    "max_l": 10,
                    "band_basis": "zero_fourier_harmonic_energy_order",
                    "index_order": ["coupled_band", "photon_index_l"],
                    "includes_same_band": False,
                },
            },
            stream,
        )
    return ratios, indices


class TestDynamicNDFloquetRatioPlot(unittest.TestCase):
    def test_reduction_and_hover_follow_the_maximizing_source_band(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ratios, _ = _write_bundle(root)
            with np.load(root / "qgt_nd_bundle.npz", allow_pickle=True) as data:
                field, hover = _prepare_floquet_ratio_field(data)

        available = np.any(~np.isnan(ratios), axis=-1)
        expected = np.max(np.where(np.isnan(ratios), -np.inf, ratios), axis=-1)
        expected = np.where(available, expected, np.nan)
        np.testing.assert_allclose(field, expected, equal_nan=True)
        self.assertEqual(hover[0, 0, 0].tolist(), ["0.7", 1, 0, -2])
        self.assertEqual(hover[0, 0, 1].tolist(), ["∞", 0, 1, 1])
        self.assertEqual(
            hover[0, 1, 0].tolist(),
            ["0", "none", "none", "none"],
        )
        self.assertEqual(
            hover[0, 1, 1].tolist(),
            ["unavailable", "none", "none", "none"],
        )

    def test_summary_excludes_infinity_from_standard_deviation(self):
        field = np.array([[[1.0, 3.0], [np.inf, np.nan]]])

        summary = _floquet_ratio_summary_grid(field)[0]

        self.assertIn("max=∞", summary)
        self.assertIn("finite std=1.000e+00", summary)
        self.assertIn("exact-resonance k-points=1", summary)

    def test_html_caps_infinity_but_preserves_diagnostic_hover(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_bundle(root)
            output_path = root / "ratio.html"

            plot_floquet_max_ratio_nd_html(
                root,
                output_html=output_path,
                ratio_zmax_percentile=100.0,
            )
            html = output_path.read_text(encoding="utf-8")

        match = re.search(r"const fieldData = (.*?);\n", html)
        self.assertIsNotNone(match)
        embedded_field = json.loads(match.group(1))
        self.assertEqual(embedded_field[0][0][1], 0.9)
        self.assertIsNone(embedded_field[0][1][1])
        self.assertIn("Max ratio: %{customdata[0]}", html)
        self.assertIn("Source H", html)
        self.assertIn("Coupled H", html)
        self.assertIn("Photon index", html)
        self.assertIn("\\u221e", html)
        self.assertIn("update.customdata", html)
        self.assertIn("nestedGet(ratioSummaryData, current)", html)
        self.assertIn("Floquet Diagnostic", html)
        self.assertRegex(html, r'"zmin":0(?:\.0)?[,}]')
        self.assertRegex(html, r'"zmax":0\.9(?:0*)?[,}]')

    def test_symmetric_color_scale_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_bundle(root)

            with self.assertRaisesRegex(ValueError, "nonnegative"):
                plot_floquet_max_ratio_nd_html(
                    root,
                    output_html=root / "ratio.html",
                    symmetric_cbar=True,
                )

    def test_existing_trace_field_still_uses_the_shared_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_bundle(root)
            output_path = root / "trace.html"

            dynamic_nd_field_with_bands_html(
                root,
                quantity="trace",
                output_html=output_path,
                trace_zmax_percentile=100.0,
            )
            html = output_path.read_text(encoding="utf-8")

        self.assertIn("const ratioHoverData = null", html)
        self.assertIn("const isFloquetRatio = false", html)
        self.assertIn("QGT Trace", html)


if __name__ == "__main__":
    unittest.main()
