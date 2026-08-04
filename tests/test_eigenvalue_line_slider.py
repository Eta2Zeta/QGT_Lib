import json

import numpy as np
import pytest

from Library.plotting_eigenvalues_2d import plot_eigenvalue_line_slider


@pytest.fixture
def sample_grid():
    first_values = np.array([1.0, 2.0, 3.0])
    second_values = np.array([10.0, 20.0])
    ki, kj = np.meshgrid(first_values, second_values)
    eigenvalues = np.empty((2, 3, 2), dtype=float)
    for second_index in range(2):
        for first_index in range(3):
            for band in range(2):
                eigenvalues[second_index, first_index, band] = (
                    100.0 * second_index + 10.0 * first_index + band
                )
    return ki, kj, eigenvalues


def test_ij_uses_first_axis_for_lines_and_second_axis_for_slider(sample_grid):
    ki, kj, eigenvalues = sample_grid
    figure = plot_eigenvalue_line_slider(
        ki,
        kj,
        eigenvalues,
        axis_order="ij",
        first_axis_label="r",
        second_axis_label="phi (rad)",
        show=False,
    )

    assert np.allclose(figure.data[0].x, ki[0, :])
    assert np.allclose(figure.data[0].y, eigenvalues[0, :, 0])
    assert len(figure.frames) == kj.shape[0]
    assert np.allclose(figure.frames[1].data[0].y, eigenvalues[1, :, 0])
    assert figure.layout.xaxis.title.text == "r"
    assert figure.layout.showlegend is True
    assert figure.layout.legend.itemclick == "toggle"
    assert figure.layout.legend.itemdoubleclick == "toggleothers"
    assert all(trace.showlegend is True for trace in figure.data)
    assert figure.layout.yaxis.autorange is True
    assert figure.layout.yaxis.range is None


def test_ji_uses_second_axis_for_lines_and_first_axis_for_slider(sample_grid):
    ki, kj, eigenvalues = sample_grid
    figure = plot_eigenvalue_line_slider(
        ki,
        kj,
        eigenvalues,
        axis_order="ji",
        first_axis_label="r",
        second_axis_label="phi (rad)",
        show=False,
    )

    assert np.allclose(figure.data[0].x, kj[:, 0])
    assert np.allclose(figure.data[0].y, eigenvalues[:, 0, 0])
    assert len(figure.frames) == ki.shape[1]
    assert np.allclose(figure.frames[2].data[0].y, eigenvalues[:, 2, 0])
    assert figure.layout.xaxis.title.text == "phi (rad)"


def test_line_slider_saves_html_with_metadata(tmp_path, sample_grid):
    ki, kj, eigenvalues = sample_grid
    with open(tmp_path / "meta.json", "w", encoding="utf-8") as meta_file:
        json.dump({"order": "xpz", "hamiltonian_params": {"M": 3.697}}, meta_file)

    figure = plot_eigenvalue_line_slider(
        ki,
        kj,
        eigenvalues,
        axis_order="ij",
        results_dir=tmp_path,
        save_fig=True,
        show=False,
    )

    output_path = tmp_path / "eigenvalue_line_slider_ij.html"
    assert output_path.is_file()
    assert "Run Metadata" in output_path.read_text(encoding="utf-8")
    assert figure.layout.annotations[0].text.startswith("<b>Run Metadata</b>")


def test_line_slider_rejects_unknown_axis_order(sample_grid):
    ki, kj, eigenvalues = sample_grid
    with pytest.raises(ValueError):
        plot_eigenvalue_line_slider(
            ki,
            kj,
            eigenvalues,
            axis_order="xy",
            show=False,
        )
