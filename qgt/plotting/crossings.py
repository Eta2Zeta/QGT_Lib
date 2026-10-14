"""Interactive 3D view of band crossings in the first Brillouin zone."""

import numpy as np

from qgt.hamiltonians.minimal_models.crossings import PAIR_NAMES

PAIR_COLORS = {"2-3": ("#d62728", "#ff7f0e", "#8c564b", "#e377c2", "#bcbd22"),
               "1-2,3-4": ("#1f77b4", "#2ca02c", "#9467bd", "#17becf", "#7f7f7f")}


def zone_edges(hexagonal):
    """Edges of the first Brillouin zone as a list of (2, 3) segments."""
    if hexagonal:
        angles = np.deg2rad(np.arange(0, 360, 60))
        corners = 4 * np.pi / 3 * np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        corners = np.pi * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    edges = []
    for n, corner in enumerate(corners):
        following = corners[(n + 1) % len(corners)]
        for z in (-np.pi, np.pi):
            edges.append(np.array([[*corner, z], [*following, z]]))
        edges.append(np.array([[*corner, -np.pi], [*corner, np.pi]]))
    return edges


def plot_crossings(model, crossings, path=None):
    """Plot the samples of each crossing component; write HTML to path if given."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for edge in zone_edges(model.crystal_system in ("trigonal", "hexagonal")):
        fig.add_trace(go.Scatter3d(x=edge[:, 0], y=edge[:, 1], z=edge[:, 2], mode="lines",
                                   line=dict(color="#999999", width=2), showlegend=False, hoverinfo="skip"))
    counts = {}
    for component in crossings:
        n = counts[component.pair] = counts.get(component.pair, -1) + 1
        color = PAIR_COLORS.get(component.pair, ("#333333",))[n % 5]
        text = component.describe(width=120)
        label = f"{PAIR_NAMES.get(component.pair, component.pair)}: {text.split('  [')[0]}"
        k = component.samples
        fig.add_trace(go.Scatter3d(
            x=k[:, 0], y=k[:, 1], z=k[:, 2], mode="markers", name=label[:90],
            marker=dict(size=5 if component.dimension == 0 else 2, color=color),
            hovertext=text, hoverinfo="text+x+y+z"))
    J = tuple(float(v) for v in (model.Jx, model.Jy, model.Jz))
    fig.update_layout(
        title=f"{type(model).__name__}: band crossings (J = {J})",
        scene=dict(xaxis_title="k_x", yaxis_title="k_y", zaxis_title="k_z", aspectmode="data"),
        legend=dict(font=dict(size=10)), margin=dict(l=0, r=0, t=40, b=0))
    if path is not None:
        fig.write_html(path, include_plotlyjs="cdn")
    return fig
