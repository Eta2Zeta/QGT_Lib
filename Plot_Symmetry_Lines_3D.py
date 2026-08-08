"""Plot a Hamiltonian's high-symmetry path in interactive 3D k-space."""

import inspect
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from Library.Hamiltonian import gWaveAltermagnetHamiltonian


def _display_symmetry_label(label):
    return "\u0393" if str(label) == "G" else str(label)


def _resolve_symmetry_path(hamiltonian, path=None):
    get_sym_path = getattr(hamiltonian, "get_sym_path", None)
    if not callable(get_sym_path):
        raise TypeError(
            f"{hamiltonian.__class__.__name__} does not provide get_sym_path()"
        )

    if path is None:
        symmetry_points, path_labels = get_sym_path()
    else:
        parameters = inspect.signature(get_sym_path).parameters
        if "path" not in parameters:
            raise TypeError(
                f"{hamiltonian.__class__.__name__}.get_sym_path() does not "
                "accept a custom path"
            )
        symmetry_points, path_labels = get_sym_path(path=path)

    if not isinstance(symmetry_points, dict):
        raise TypeError("get_sym_path() must return a dictionary of symmetry points")

    path_labels = [str(label) for label in path_labels]
    if len(path_labels) < 2:
        raise ValueError("A symmetry path must contain at least two point labels")

    missing_labels = [
        label for label in path_labels if label not in symmetry_points
    ]
    if missing_labels:
        raise KeyError(
            f"Symmetry path references undefined labels: {missing_labels}"
        )

    point_dimensions = {
        np.asarray(symmetry_points[label], dtype=float).size
        for label in path_labels
    }
    if not point_dimensions.issubset({2, 3}):
        raise ValueError("Symmetry-point coordinates must have length 2 or 3")

    path_points = []
    for label in path_labels:
        point = np.asarray(symmetry_points[label], dtype=float).reshape(-1)
        if point.size == 2:
            point = np.append(point, 0.0)
        if not np.all(np.isfinite(point)):
            raise ValueError(f"Symmetry point {label!r} contains non-finite values")
        path_points.append(point)

    return path_labels, np.asarray(path_points, dtype=float)


def _interpolate_path(path_points, num_points_per_segment):
    num_points_per_segment = int(num_points_per_segment)
    if num_points_per_segment < 2:
        raise ValueError("num_points_per_segment must be at least 2")

    segments = []
    for start, end in zip(path_points[:-1], path_points[1:]):
        segments.append(
            np.linspace(start, end, num_points_per_segment, endpoint=True)
        )
    return segments


def _unique_path_nodes(path_labels, path_points):
    unique_labels = []
    unique_points = []
    seen = set()
    for label, point in zip(path_labels, path_points):
        if label in seen:
            continue
        seen.add(label)
        unique_labels.append(label)
        unique_points.append(point)
    return unique_labels, np.asarray(unique_points, dtype=float)


def _path_annotation(path_labels, labels_per_line=12):
    displayed = [_display_symmetry_label(label) for label in path_labels]
    lines = []
    for start in range(0, len(displayed), labels_per_line):
        lines.append(" -> ".join(displayed[start:start + labels_per_line]))
    return "<br>".join(lines)


def _scene_aspect(path_points):
    spans = np.ptp(path_points, axis=0)
    largest_span = float(np.max(spans))
    if largest_span == 0.0:
        return {"x": 1.0, "y": 1.0, "z": 1.0}
    normalized = spans / largest_span
    return {
        axis: max(0.18, float(value))
        for axis, value in zip("xyz", normalized)
    }


def build_symmetry_path_3d_figure(
    hamiltonian,
    *,
    path=None,
    num_points_per_segment=40,
    line_color="#1f4e79",
    line_opacity=0.45,
    node_color="#d1495b",
    node_label_size=20,
):
    """Build an interactive Plotly figure from ``hamiltonian.get_sym_path()``."""
    path_labels, path_points = _resolve_symmetry_path(hamiltonian, path=path)
    path_segments = _interpolate_path(path_points, num_points_per_segment)

    figure = go.Figure()
    for segment_index, segment in enumerate(path_segments):
        start_label = _display_symmetry_label(path_labels[segment_index])
        end_label = _display_symmetry_label(path_labels[segment_index + 1])
        segment_label = f"{start_label} -> {end_label}"
        figure.add_trace(
            go.Scatter3d(
                x=segment[:, 0],
                y=segment[:, 1],
                z=segment[:, 2],
                mode="lines",
                name="Symmetry path",
                legendgroup="symmetry_path",
                showlegend=segment_index == 0,
                line={"color": line_color, "width": 7},
                opacity=float(line_opacity),
                customdata=np.full(segment.shape[0], segment_label),
                hovertemplate=(
                    "%{customdata}<br>"
                    "k<sub>x</sub> = %{x:.6g}<br>"
                    "k<sub>y</sub> = %{y:.6g}<br>"
                    "k<sub>z</sub> = %{z:.6g}<extra></extra>"
                ),
            )
        )

    node_labels, node_points = _unique_path_nodes(path_labels, path_points)
    displayed_node_labels = [
        _display_symmetry_label(label) for label in node_labels
    ]
    node_hover = [
        (
            f"{displayed}<br>"
            f"k<sub>x</sub> = {point[0]:.6g}<br>"
            f"k<sub>y</sub> = {point[1]:.6g}<br>"
            f"k<sub>z</sub> = {point[2]:.6g}"
        )
        for displayed, point in zip(displayed_node_labels, node_points)
    ]
    figure.add_trace(
        go.Scatter3d(
            x=node_points[:, 0],
            y=node_points[:, 1],
            z=node_points[:, 2],
            mode="markers+text",
            name="Symmetry points",
            marker={
                "size": 6,
                "color": node_color,
                "line": {"color": "white", "width": 1.2},
            },
            text=displayed_node_labels,
            textposition="top center",
            textfont={"size": int(node_label_size), "color": "#172033"},
            hovertext=node_hover,
            hoverinfo="text",
        )
    )

    hamiltonian_name = getattr(
        hamiltonian,
        "name",
        hamiltonian.__class__.__name__,
    )
    figure.update_layout(
        title={
            "text": f"{hamiltonian_name}: high-symmetry path in k-space",
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        scene={
            "xaxis_title": "k<sub>x</sub>",
            "yaxis_title": "k<sub>y</sub>",
            "zaxis_title": "k<sub>z</sub>",
            "aspectmode": "manual",
            "aspectratio": _scene_aspect(path_points),
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.15}},
            "dragmode": "orbit",
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.02,
            "yanchor": "bottom",
            "groupclick": "togglegroup",
        },
        annotations=[
            {
                "text": "Path: " + _path_annotation(path_labels),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.08,
                "xanchor": "center",
                "yanchor": "top",
                "showarrow": False,
                "align": "center",
            }
        ],
        meta={
            "hamiltonian_name": hamiltonian_name,
            "path_labels": path_labels,
            "path_points": path_points.tolist(),
        },
        margin={"l": 10, "r": 10, "b": 105, "t": 95},
    )
    return figure


def _default_output_path(hamiltonian):
    hamiltonian_name = getattr(
        hamiltonian,
        "name",
        hamiltonian.__class__.__name__,
    )
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(hamiltonian_name))
    return (
        Path.cwd()
        / "results"
        / "Symmetry_Path_Plots"
        / safe_name
        / "symmetry_path_3d.html"
    )


def plot_symmetry_path_3d(
    hamiltonian,
    *,
    path=None,
    output_html=None,
    num_points_per_segment=40,
    include_plotlyjs=True,
):
    """Save a Hamiltonian's interactive 3D symmetry path and return its path."""
    figure = build_symmetry_path_3d_figure(
        hamiltonian,
        path=path,
        num_points_per_segment=num_points_per_segment,
    )
    output_path = (
        Path(output_html).expanduser().resolve()
        if output_html is not None
        else _default_output_path(hamiltonian).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path,
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        auto_open=False,
    )
    print(f"Saved interactive 3D symmetry path to:\n > {output_path}")
    return str(output_path)


# Edit this Hamiltonian or pass a custom path below when needed.
Hamiltonian_Obj = gWaveAltermagnetHamiltonian()
custom_path = None


if __name__ == "__main__":
    plot_symmetry_path_3d(Hamiltonian_Obj, path=custom_path)
