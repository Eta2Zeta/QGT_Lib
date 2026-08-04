"""Plots derived from QGT data sampled on a polar momentum grid."""

import base64
import io
import os

import matplotlib.pyplot as plt
import numpy as np

from Library.dimension_lib import cylindrical_order_axes


def _polar_grid_values(radius_grid, phi_grid):
    radius_grid = np.asarray(radius_grid, dtype=float)
    phi_grid = np.asarray(phi_grid, dtype=float)
    if radius_grid.ndim != 2 or phi_grid.ndim != 2:
        raise ValueError("radius_grid and phi_grid must be two-dimensional")
    if radius_grid.shape != phi_grid.shape:
        raise ValueError("radius_grid and phi_grid must have matching shapes")

    radius_values = radius_grid[0, :]
    phi_values = phi_grid[:, 0]
    if not np.allclose(radius_grid, radius_values[np.newaxis, :]):
        raise ValueError("radius_grid must vary only along axis 1")
    if not np.allclose(phi_grid, phi_values[:, np.newaxis]):
        raise ValueError("phi_grid must vary only along axis 0")
    return radius_values, phi_values


def _polar_plane_coordinates(radius, phi, order):
    """Map polar coordinates to the two Cartesian axes normal to the fixed axis."""
    reference_axis, tangent_axis, tangent_sign, fixed_axis = (
        cylindrical_order_axes(order)
    )
    plane_axes = tuple(axis for axis in "xyz" if axis != fixed_axis)

    radius = np.asarray(radius, dtype=float)
    phi = np.asarray(phi, dtype=float)
    shape = np.broadcast_shapes(radius.shape, phi.shape)
    coordinates = {
        axis: np.zeros(shape, dtype=float)
        for axis in "xyz"
    }
    coordinates[reference_axis] = radius * np.cos(phi)
    coordinates[tangent_axis] = tangent_sign * radius * np.sin(phi)
    return coordinates[plane_axes[0]], coordinates[plane_axes[1]], plane_axes


def _polar_component_to_cartesian(
    radius_values,
    phi_values,
    component,
    *,
    order,
    resolution=None,
):
    """Linearly interpolate a periodic ``(phi, r)`` field onto a Cartesian grid."""
    from scipy.interpolate import RegularGridInterpolator

    radius_values = np.asarray(radius_values, dtype=float)
    phi_values = np.asarray(phi_values, dtype=float)
    component = np.asarray(component, dtype=float)
    expected_shape = (phi_values.size, radius_values.size)
    if component.shape != expected_shape:
        raise ValueError(
            f"Polar component shape must be {expected_shape}, got {component.shape}"
        )
    if np.any(np.diff(radius_values) <= 0.0):
        raise ValueError("Polar radii must be strictly increasing")
    if np.any(np.diff(phi_values) <= 0.0):
        raise ValueError("Polar angles must be strictly increasing")

    if resolution is None:
        resolution = max(80, min(401, 2 * radius_values.size - 1))
    resolution = int(resolution)
    if resolution < 3:
        raise ValueError("Cartesian interpolation resolution must be at least 3")

    period = 2.0 * np.pi
    periodic_phi = np.concatenate(
        ([phi_values[-1] - period], phi_values, [phi_values[0] + period])
    )
    periodic_component = np.concatenate(
        (component[-1:, :], component, component[:1, :]),
        axis=0,
    )
    interpolator = RegularGridInterpolator(
        (periodic_phi, radius_values),
        periodic_component,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    radius_max = float(radius_values[-1])
    cartesian_values = np.linspace(-radius_max, radius_max, resolution)
    horizontal_grid, vertical_grid = np.meshgrid(
        cartesian_values,
        cartesian_values,
    )

    reference_axis, tangent_axis, tangent_sign, fixed_axis = (
        cylindrical_order_axes(order)
    )
    plane_axes = tuple(axis for axis in "xyz" if axis != fixed_axis)
    cartesian_coordinates = {
        "x": np.zeros_like(horizontal_grid),
        "y": np.zeros_like(horizontal_grid),
        "z": np.zeros_like(horizontal_grid),
    }
    cartesian_coordinates[plane_axes[0]] = horizontal_grid
    cartesian_coordinates[plane_axes[1]] = vertical_grid

    reference_coordinate = cartesian_coordinates[reference_axis]
    oriented_tangent_coordinate = (
        tangent_sign * cartesian_coordinates[tangent_axis]
    )
    query_radius = np.hypot(reference_coordinate, oriented_tangent_coordinate)
    query_phi = np.mod(
        np.arctan2(oriented_tangent_coordinate, reference_coordinate),
        period,
    )
    query_points = np.column_stack((query_phi.ravel(), query_radius.ravel()))
    cartesian_component = interpolator(query_points).reshape(horizontal_grid.shape)
    cartesian_component[query_radius > radius_max] = np.nan
    cartesian_component[query_radius < radius_values[0]] = np.nan

    return cartesian_values, cartesian_values, cartesian_component, plane_axes


def _component_image_data_uri(component, component_limit):
    """Render a Cartesian component as a transparent PNG data URI."""
    component = np.asarray(component, dtype=float)
    flipped_component = np.flipud(component)
    finite = np.isfinite(flipped_component)
    normalized = np.zeros(flipped_component.shape, dtype=float)
    normalized[finite] = (
        flipped_component[finite] + component_limit
    ) / (2.0 * component_limit)

    rgba = plt.get_cmap("RdBu_r")(np.clip(normalized, 0.0, 1.0))
    rgba[~finite, 3] = 0.0
    image_buffer = io.BytesIO()
    plt.imsave(image_buffer, rgba, format="png")
    encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded_image}"


def plot_berry_winding_vs_radius(
    radius_values,
    winding_numbers,
    *,
    band_index,
    component_labels=("Omega_x", "Omega_y"),
    results_dir=None,
    filename=None,
    save_fig=True,
    show=False,
):
    """Plot a band's Berry-texture winding number as a function of radius."""
    radius_values = np.asarray(radius_values, dtype=float)
    winding_numbers = np.asarray(winding_numbers, dtype=float)
    if radius_values.ndim != 1 or winding_numbers.ndim != 1:
        raise ValueError("radius_values and winding_numbers must be one-dimensional")
    if radius_values.shape != winding_numbers.shape:
        raise ValueError("radius_values and winding_numbers must have matching shapes")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(radius_values, winding_numbers, color="tab:blue", linewidth=1.6)
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel("Winding number W")
    ax.set_title(
        f"Berry-curvature winding, band {band_index}: "
        f"{component_labels[0]} + i {component_labels[1]}"
    )
    ax.grid(alpha=0.25)

    if not np.any(np.isfinite(winding_numbers)):
        ax.text(
            0.5,
            0.5,
            "Winding undefined on every sampled ring",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    output_path = None
    if save_fig:
        if results_dir is None:
            raise ValueError("results_dir is required when save_fig=True")
        os.makedirs(results_dir, exist_ok=True)
        if filename is None:
            filename = f"berry_winding_vs_radius_band_{band_index}.png"
        output_path = os.path.join(results_dir, filename)
        fig.savefig(output_path, dpi=240, bbox_inches="tight")
        print(f"Saved Berry winding plot to: {output_path}")

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_berry_components_vs_phi_radius_slider(
    radius_grid,
    phi_grid,
    omega_x,
    omega_y,
    omega_z,
    *,
    band_index,
    winding_numbers=None,
    order="xpz",
    cartesian_resolution=None,
    results_dir=None,
    filename=None,
    save_fig=True,
    show=False,
):
    """Plot Berry maps, ring cuts, winding, and magnitude with one radius slider."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    radius_values, phi_values = _polar_grid_values(radius_grid, phi_grid)
    components = [
        np.asarray(omega_x, dtype=float),
        np.asarray(omega_y, dtype=float),
        np.asarray(omega_z, dtype=float),
    ]
    for component in components:
        if component.shape != radius_grid.shape:
            raise ValueError(
                "Every Berry-curvature component must match the polar-grid shape"
            )

    radius_indices = np.flatnonzero(radius_values > 0.0)
    if radius_indices.size == 0:
        raise ValueError("The polar grid must contain at least one positive radius")

    omega_by_axis = {
        "x": components[0],
        "y": components[1],
        "z": components[2],
    }
    if winding_numbers is None:
        from Library.topology import winding_numbers_vs_radius

        reference_axis, tangent_axis, tangent_sign, _ = (
            cylindrical_order_axes(order)
        )
        _, winding_numbers = winding_numbers_vs_radius(
            radius_grid,
            phi_grid,
            omega_by_axis[reference_axis],
            tangent_sign * omega_by_axis[tangent_axis],
        )
    winding_numbers = np.asarray(winding_numbers, dtype=float)
    if winding_numbers.shape != radius_values.shape:
        raise ValueError("winding_numbers must contain one value per radius")

    omega_xy_magnitude = np.hypot(components[0], components[1])
    component_names = ["Omega_x", "Omega_y", "Omega_z"]
    component_titles = [
        "Omega_x = Omega_yz",
        "Omega_y = Omega_zx",
        "Omega_z = Omega_xy",
    ]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    initial_index = int(radius_indices[0])

    cartesian_maps = []
    plane_axes = None
    for component in components:
        x_values, y_values, cartesian_component, current_plane_axes = (
            _polar_component_to_cartesian(
                radius_values,
                phi_values,
                component,
                order=order,
                resolution=cartesian_resolution,
            )
        )
        if plane_axes is None:
            plane_axes = current_plane_axes
        cartesian_maps.append(cartesian_component)

    phi_plot = np.concatenate((phi_values, [phi_values[0] + 2.0 * np.pi]))

    def ring_values(component, radius_index):
        return np.concatenate(
            (component[:, radius_index], [component[0, radius_index]])
        )

    fig = make_subplots(
        rows=3,
        cols=6,
        specs=[
            [
                {"colspan": 2}, None,
                {"colspan": 2}, None,
                {"colspan": 2}, None,
            ],
            [
                {"colspan": 2}, None,
                {"colspan": 2}, None,
                {"colspan": 2}, None,
            ],
            [
                {"colspan": 3}, None, None,
                {"colspan": 3}, None, None,
            ],
        ],
        row_heights=[0.45, 0.27, 0.28],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
        subplot_titles=(
            component_titles
            + ["", "", ""]
            + [
                "Winding number versus radius",
                "In-plane Berry magnitude versus phi",
            ]
        ),
    )
    radius_max = float(radius_values[-1])
    component_grid_columns = [1, 3, 5]
    colorbar_x_positions = [0.318, 0.665, 1.015]
    for plot_index, (grid_column, cartesian_component, name) in enumerate(
        zip(component_grid_columns, cartesian_maps, component_names),
    ):
        finite = cartesian_component[np.isfinite(cartesian_component)]
        component_limit = float(np.max(np.abs(finite))) if finite.size else 0.0
        if component_limit == 0.0:
            component_limit = 1.0
        fig.add_trace(
            go.Heatmap(
                x=x_values,
                y=y_values,
                z=cartesian_component,
                zmin=-component_limit,
                zmax=component_limit,
                colorscale="RdBu_r",
                name=name,
                opacity=0.0,
                showscale=True,
                colorbar={
                    "x": colorbar_x_positions[plot_index],
                    "y": 0.81,
                    "len": 0.36,
                    "thickness": 11,
                    "outlinewidth": 0,
                },
                hovertemplate=(
                    f"k_{plane_axes[0]}: %{{x:.6g}}<br>"
                    f"k_{plane_axes[1]}: %{{y:.6g}}<br>"
                    f"{name}: %{{z:.6g}}<extra></extra>"
                ),
            ),
            row=1,
            col=grid_column,
        )
        axis_number = plot_index + 1
        axis_suffix = "" if axis_number == 1 else str(axis_number)
        fig.add_layout_image(
            {
                "source": _component_image_data_uri(
                    cartesian_component,
                    component_limit,
                ),
                "xref": f"x{axis_suffix}",
                "yref": f"y{axis_suffix}",
                "x": -radius_max,
                "y": radius_max,
                "sizex": 2.0 * radius_max,
                "sizey": 2.0 * radius_max,
                "xanchor": "left",
                "yanchor": "top",
                "sizing": "stretch",
                "layer": "below",
            }
        )

    initial_circle_x, initial_circle_y, _ = _polar_plane_coordinates(
        radius_values[initial_index],
        np.linspace(0.0, 2.0 * np.pi, 361),
        order,
    )
    circle_trace_indices = []
    for grid_column in component_grid_columns:
        for width, color in ((5.0, "rgba(0,0,0,0.78)"), (2.2, "#ffd84d")):
            fig.add_trace(
                go.Scatter(
                    x=initial_circle_x,
                    y=initial_circle_y,
                    mode="lines",
                    line={"color": color, "width": width},
                    name="Selected radius",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=grid_column,
            )
            circle_trace_indices.append(len(fig.data) - 1)

    line_trace_indices = []
    for grid_column, component, name, color in zip(
        component_grid_columns,
        components,
        component_names,
        colors,
    ):
        fig.add_trace(
            go.Scatter(
                x=phi_plot,
                y=ring_values(component, initial_index),
                mode="lines",
                name=name,
                line={"color": color, "width": 2},
                showlegend=False,
                hovertemplate=(
                    "phi: %{x:.6g}<br>"
                    f"{name}: %{{y:.6g}}<extra></extra>"
                ),
            ),
            row=2,
            col=grid_column,
        )
        line_trace_indices.append(len(fig.data) - 1)

        finite = component[np.isfinite(component)]
        component_limit = float(np.max(np.abs(finite))) if finite.size else 0.0
        if component_limit == 0.0:
            component_limit = 1.0
        fig.update_yaxes(
            title_text=name,
            range=[-1.05 * component_limit, 1.05 * component_limit],
            zeroline=True,
            zerolinecolor="rgba(100,100,100,0.45)",
            row=2,
            col=grid_column,
        )

    for plot_index, grid_column in enumerate(component_grid_columns, start=1):
        x_axis_id = "x" if plot_index == 1 else f"x{plot_index}"
        fig.update_xaxes(
            title_text=f"k_{plane_axes[0]}",
            range=[-radius_max, radius_max],
            constrain="domain",
            row=1,
            col=grid_column,
        )
        fig.update_yaxes(
            title_text=f"k_{plane_axes[1]}",
            range=[-radius_max, radius_max],
            scaleanchor=x_axis_id,
            scaleratio=1,
            constrain="domain",
            row=1,
            col=grid_column,
        )
        fig.update_xaxes(
            title_text="phi (rad)",
            range=[float(phi_plot[0]), float(phi_plot[-1])],
            row=2,
            col=grid_column,
        )

    finite_winding = winding_numbers[np.isfinite(winding_numbers)]
    if finite_winding.size:
        winding_min = float(np.min(finite_winding))
        winding_max = float(np.max(finite_winding))
        winding_span = winding_max - winding_min
        winding_padding = (
            0.08 * winding_span
            if winding_span > 0.0
            else max(0.5, 0.1 * abs(winding_min))
        )
        winding_y_range = [
            winding_min - winding_padding,
            winding_max + winding_padding,
        ]
    else:
        winding_y_range = [-1.0, 1.0]

    fig.add_trace(
        go.Scatter(
            x=radius_values,
            y=winding_numbers,
            mode="lines+markers",
            line={"color": "#2f3e56", "width": 2},
            marker={"size": 4},
            name="Winding number",
            showlegend=False,
            connectgaps=False,
            hovertemplate="r: %{x:.6g}<br>W: %{y:.6g}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[radius_values[initial_index], radius_values[initial_index]],
            y=winding_y_range,
            mode="lines",
            line={"color": "#d62728", "width": 2, "dash": "dash"},
            name="Selected radius",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=3,
        col=1,
    )
    winding_marker_trace_index = len(fig.data) - 1

    magnitude_finite = omega_xy_magnitude[np.isfinite(omega_xy_magnitude)]
    magnitude_limit = (
        float(np.max(magnitude_finite))
        if magnitude_finite.size
        else 0.0
    )
    if magnitude_limit == 0.0:
        magnitude_limit = 1.0
    fig.add_trace(
        go.Scatter(
            x=phi_plot,
            y=ring_values(omega_xy_magnitude, initial_index),
            mode="lines",
            line={"color": "#7a3db8", "width": 2},
            name="In-plane Berry magnitude",
            showlegend=False,
            hovertemplate=(
                "phi: %{x:.6g}<br>"
                "sqrt(Omega_x^2 + Omega_y^2): %{y:.6g}<extra></extra>"
            ),
        ),
        row=3,
        col=4,
    )
    magnitude_trace_index = len(fig.data) - 1

    fig.update_xaxes(
        title_text="r",
        range=[float(radius_values[0]), radius_max],
        row=3,
        col=1,
    )
    fig.update_yaxes(
        title_text="Winding number W",
        range=winding_y_range,
        zeroline=True,
        zerolinecolor="rgba(100,100,100,0.45)",
        row=3,
        col=1,
    )
    fig.update_xaxes(
        title_text="phi (rad)",
        range=[float(phi_plot[0]), float(phi_plot[-1])],
        row=3,
        col=4,
    )
    fig.update_yaxes(
        title_text="|Omega_(x,y)|",
        range=[0.0, 1.05 * magnitude_limit],
        row=3,
        col=4,
    )

    frames = []
    circle_phi = np.linspace(0.0, 2.0 * np.pi, 361)
    dynamic_trace_indices = (
        circle_trace_indices
        + line_trace_indices
        + [winding_marker_trace_index, magnitude_trace_index]
    )
    for radius_index in radius_indices:
        radius_index = int(radius_index)
        circle_x, circle_y, _ = _polar_plane_coordinates(
            radius_values[radius_index],
            circle_phi,
            order,
        )
        frame_data = []
        for _ in range(3):
            frame_data.extend(
                [
                    go.Scatter(x=circle_x, y=circle_y),
                    go.Scatter(x=circle_x, y=circle_y),
                ]
            )
        frame_data.extend(
            go.Scatter(
                x=phi_plot,
                y=ring_values(component, radius_index),
            )
            for component in components
        )
        frame_data.extend(
            [
                go.Scatter(
                    x=[radius_values[radius_index], radius_values[radius_index]],
                    y=winding_y_range,
                ),
                go.Scatter(
                    x=phi_plot,
                    y=ring_values(omega_xy_magnitude, radius_index),
                ),
            ]
        )
        frames.append(
            go.Frame(
                data=frame_data,
                traces=dynamic_trace_indices,
                name=str(radius_index),
            )
        )
    fig.frames = frames

    slider_steps = [
        {
            "method": "animate",
            "args": [
                [str(int(radius_index))],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": False},
                    "transition": {"duration": 0},
                },
            ],
            "label": f"{radius_values[radius_index]:.4g}",
        }
        for radius_index in radius_indices
    ]

    fig.update_layout(
        title=f"Cartesian Berry curvature and polar-ring cuts, band {band_index}",
        height=1220,
        width=1550,
        hovermode="closest",
        margin={"l": 75, "r": 115, "t": 90, "b": 180},
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Radius r: "},
                "pad": {"t": 55},
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "showactive": False,
                "x": 0.02,
                "y": -0.15,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 90, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                            },
                        ],
                    },
                ],
            }
        ],
    )

    output_path = None
    if save_fig:
        if results_dir is None:
            raise ValueError("results_dir is required when save_fig=True")
        os.makedirs(results_dir, exist_ok=True)
        if filename is None:
            filename = (
                f"berry_components_vs_phi_radius_slider_band_{band_index}.html"
            )
        output_path = os.path.join(results_dir, filename)
        fig.write_html(output_path, include_plotlyjs="cdn")
        print(f"Saved polar Berry-component slider to: {output_path}")

    if show:
        fig.show()
    return output_path
