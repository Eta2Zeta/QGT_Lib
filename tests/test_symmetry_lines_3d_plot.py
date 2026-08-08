from pathlib import Path

import numpy as np

from Library.Hamiltonian import (
    ChiralHamiltonian,
    MinimalHamSG127_2a2b,
)
from Plot_Symmetry_Lines_3D import (
    build_symmetry_path_3d_figure,
    plot_symmetry_path_3d,
)


def test_two_dimensional_symmetry_path_is_embedded_at_kz_zero():
    figure = build_symmetry_path_3d_figure(
        ChiralHamiltonian(),
        num_points_per_segment=3,
    )

    path_traces = figure.data[:-1]
    assert path_traces
    for trace in path_traces:
        np.testing.assert_allclose(np.asarray(trace.z, dtype=float), 0.0)
        assert trace.opacity == 0.45
    assert figure.data[-1].textfont.size == 20
    assert list(figure.layout.meta["path_labels"]) == ["G", "K", "M", "G"]


def test_three_dimensional_symmetry_path_preserves_kz_coordinates():
    figure = build_symmetry_path_3d_figure(
        MinimalHamSG127_2a2b(),
        num_points_per_segment=3,
    )

    node_trace = figure.data[-1]
    assert np.max(np.asarray(node_trace.z, dtype=float)) == np.pi
    assert "Z" in list(node_trace.text)


def test_symmetry_path_html_is_written(tmp_path):
    output_path = plot_symmetry_path_3d(
        MinimalHamSG127_2a2b(),
        output_html=tmp_path / "symmetry_path.html",
        num_points_per_segment=3,
        include_plotlyjs="cdn",
    )

    output = Path(output_path)
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    assert "high-symmetry path in k-space" in html
