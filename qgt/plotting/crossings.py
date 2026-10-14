"""Band crossings: an interactive view of the first Brillouin zone, and an overlay for
k_x-k_y heatmaps of one band."""

import numpy as np

from qgt.hamiltonians.minimal_models.crossings import PAIR_NAMES, _hex_vectors

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
    """Plot the samples of each crossing component (in 3D, or in 2D when they all share one
    k_z); write HTML to path if given."""
    import plotly.graph_objects as go

    samples = [c.samples for c in crossings]
    heights = np.concatenate([k[:, 2] for k in samples]) if samples else np.empty(0)
    flat = len(heights) > 0 and np.ptp(heights) < 1e-9
    fig = go.Figure()
    for edge in zone_edges(model.crystal_system in ("trigonal", "hexagonal")):
        if flat:
            if edge[0, 2] == edge[1, 2] == np.pi:
                fig.add_trace(go.Scatter(x=edge[:, 0], y=edge[:, 1], mode="lines", line=dict(color="#999999"),
                                         showlegend=False, hoverinfo="skip"))
            continue
        fig.add_trace(go.Scatter3d(x=edge[:, 0], y=edge[:, 1], z=edge[:, 2], mode="lines",
                                   line=dict(color="#999999", width=2), showlegend=False, hoverinfo="skip"))
    counts = {}
    for component in crossings:
        n = counts[component.pair] = counts.get(component.pair, -1) + 1
        color = PAIR_COLORS.get(component.pair, ("#333333",))[n % 5]
        text = component.describe(width=120)
        label = f"{PAIR_NAMES.get(component.pair, component.pair)}: {text.split('  [')[0]}"
        k = component.samples
        if flat:
            fig.add_trace(go.Scatter(x=k[:, 0], y=k[:, 1], mode="markers", name=label[:90],
                                     marker=dict(size=9 if component.dimension == 0 else 4, color=color),
                                     hovertext=text, hoverinfo="text+x+y"))
        else:
            fig.add_trace(go.Scatter3d(
                x=k[:, 0], y=k[:, 1], z=k[:, 2], mode="markers", name=label[:90],
                marker=dict(size=5 if component.dimension == 0 else 2, color=color),
                hovertext=text, hoverinfo="text+x+y+z"))
    J = tuple(float(v) for v in (model.Jx, model.Jy, model.Jz))
    title = f"{type(model).__name__}: band crossings (J = {J})"
    if flat:
        fig.update_layout(title=f"{title}, k_z = {heights[0]:.6g}", xaxis_title="k_x", yaxis_title="k_y",
                          yaxis=dict(scaleanchor="x"), legend=dict(font=dict(size=10)))
    else:
        fig.update_layout(title=title,
                          scene=dict(xaxis_title="k_x", yaxis_title="k_y", zaxis_title="k_z", aspectmode="data"),
                          legend=dict(font=dict(size=10)), margin=dict(l=0, r=0, t=40, b=0))
    if path is not None:
        fig.write_html(path, include_plotlyjs="cdn")
    return fig


# --------------------------------------------------------------------------- overlay on kx-ky heatmaps

OVERLAY_STYLE = {"1-2,3-4": "black", "2-3": "#d4008f"}


def crossings_for_band(crossings, band):
    """The crossing components where band (0-based) touches a neighbouring band."""
    pairs = {"1-2,3-4"} if band in (0, 3) else {"1-2,3-4", "2-3"}
    return [c for c in crossings if c.pair in pairs]


def _partner(pair, band):
    """1-based number of the band that `band` (0-based) touches in this pair."""
    if pair == "2-3":
        return 3 if band == 1 else 2
    return {0: 2, 1: 1, 2: 4, 3: 3}[band]


def _inside_zone(points, hexagonal, tol=1e-6):
    if hexagonal:
        normals = np.array([[np.cos(a), np.sin(a)] for a in np.deg2rad(np.arange(30, 360, 60))])
        return np.all(points @ normals.T <= 2 * np.pi / np.sqrt(3) + tol, axis=1)
    return np.all(np.abs(points) <= np.pi + tol, axis=1)


def zone_images(k, hexagonal):
    """Copies of in-plane momenta (N, 2) under reciprocal lattice vectors, each with the
    points outside the first zone set to NaN (so lines break there)."""
    if hexagonal:
        b1, b2 = _hex_vectors()
        shifts = [m * b1 + n * b2 for m in (-1, 0, 1) for n in (-1, 0, 1)]
    else:
        shifts = [np.array([m, n]) * 2 * np.pi for m in (-1, 0, 1) for n in (-1, 0, 1)]
    images = []
    for shift in shifts:
        image = np.asarray(k, dtype=float)[:, :2] + shift
        inside = _inside_zone(image, hexagonal)
        if inside.any():
            image[~inside] = np.nan
            images.append(image)
    return images


def _broken_at_jumps(points):
    """Insert NaN rows where consecutive samples jump (wrapping), so a line is not drawn across."""
    if len(points) < 3:
        return points
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    typical = np.nanmedian(steps)
    cut = np.flatnonzero(~(steps <= 4 * typical))
    return np.insert(points, cut + 1, np.nan, axis=0)


def overlay_band_crossings(axes, crossings, band, *, legend=True):
    """Draw the crossings of `band` (0-based) with its neighbours on kx-ky axes.

    Exact lines are dashed, curves sampled from exact equations dotted, points
    circled; black for bands 1-2 / 3-4, magenta for bands 2-3. Copies across the
    zone boundary are included. Returns the components drawn.
    """
    from matplotlib.lines import Line2D

    plot_axes = (axes,) if hasattr(axes, "plot") else tuple(np.asarray(axes, dtype=object).reshape(-1))
    selected = crossings_for_band(crossings, band)
    handles = {}
    for component in selected:
        color = OVERLAY_STYLE[component.pair]
        partner = _partner(component.pair, band)
        hexagonal = component.hexagonal
        if component.dimension >= 2:
            for ax in plot_axes:
                ax.text(0.5, 0.02, f"degenerate with band {partner} on the whole plane", color=color,
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
            continue
        for image in zone_images(component.samples, hexagonal):
            for ax in plot_axes:
                if component.dimension == 0:
                    ax.plot(image[:, 0], image[:, 1], linestyle="none", marker="o", markersize=7,
                            markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.6, zorder=6)
                elif component.exact:
                    line = _broken_at_jumps(image)
                    ax.plot(line[:, 0], line[:, 1], color=color, linewidth=1.5, linestyle=(0, (5, 3)), zorder=5)
                else:
                    ax.plot(image[:, 0], image[:, 1], linestyle="none", marker=".", markersize=2.2,
                            color=color, zorder=5)
        kind = ("point", "line", "curve")[0 if component.dimension == 0 else (1 if component.exact else 2)]
        key = (component.pair, kind)
        if key not in handles:
            style = {"point": dict(linestyle="none", marker="o", markerfacecolor="none", markeredgecolor=color,
                                   markeredgewidth=1.6),
                     "line": dict(color=color, linestyle=(0, (5, 3)), linewidth=1.5),
                     "curve": dict(color=color, linestyle=":", linewidth=2)}[kind]
            handles[key] = Line2D([], [], label=f"touches band {partner} ({kind}{'s' if kind != 'curve' else ''})",
                                  **style)
    if legend and handles:
        plot_axes[0].legend(handles=list(handles.values()), loc="upper right", fontsize=7, framealpha=0.85)
    return selected
