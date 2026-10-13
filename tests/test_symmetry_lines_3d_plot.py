from pathlib import Path

import numpy as np

from Library.Hamiltonian import ChiralHamiltonian
from Library.Hamiltonian.MinimalModels import MinimalHamSG127_2a2b
from Plot_Symmetry_Lines_3D import (
    SEGMENT_COLOR_PALETTE,
    build_symmetry_path_3d_figure,
    plot_symmetry_path_3d,
)


def test_two_dimensional_symmetry_path_is_embedded_at_kz_zero():
    figure = build_symmetry_path_3d_figure(
        ChiralHamiltonian(),
    )

    path_traces = figure.data[:-1]
    assert path_traces
    for trace in path_traces:
        assert trace.type == "scatter3d"
        assert trace.mode == "lines"
        assert len(trace.x) == 2
        assert trace.opacity == 0.45
        assert trace.meta["symmetrySegment"] is True
        assert trace.meta["startPoint"][2] == 0.0
        assert trace.meta["endPoint"][2] == 0.0
    assert figure.layout.uirevision == "symmetry-path-segment-colors"
    assert figure.layout.clickmode == "event"
    assert figure.layout.scene.uirevision == "symmetry-path-segment-colors"
    assert figure.data[-1].textfont.size == 20
    assert list(figure.layout.meta["path_labels"]) == ["G", "K", "M", "G"]


def test_three_dimensional_symmetry_path_preserves_kz_coordinates():
    figure = build_symmetry_path_3d_figure(
        MinimalHamSG127_2a2b(),
    )

    node_trace = figure.data[-1]
    assert np.max(np.asarray(node_trace.z, dtype=float)) == np.pi
    assert "Z" in list(node_trace.text)


def test_symmetry_path_html_is_written(tmp_path):
    output_path = plot_symmetry_path_3d(
        MinimalHamSG127_2a2b(),
        output_html=tmp_path / "symmetry_path.html",
        include_plotlyjs="cdn",
    )

    output = Path(output_path)
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    assert "high-symmetry path in k-space" in html
    assert "segment-color-sidebar" in html
    assert "symmetry-segment-color-select" in html
    assert "plot.on('plotly_click'" not in html
    assert "Plotly.restyle" in html
    assert "'line.color': selectedColor" in html
    assert len(SEGMENT_COLOR_PALETTE) == 10
    for color_name, color_value in SEGMENT_COLOR_PALETTE:
        assert color_name in html
        assert color_value in html
