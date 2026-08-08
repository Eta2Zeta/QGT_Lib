"""Interactive Matplotlib viewer for N-D Berry-curvature component sweeps."""

import os

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np

from Library.dimension_lib import cylindrical_order_axes, is_cylindrical_order
from Library.plotting_qgt_polar import (
    _polar_component_to_cartesian,
    _polar_grid_values,
    _polar_plane_coordinates,
)
from Library.plotting_utils import (
    bz_wigner_seitz_from_bvecs,
    draw_symmetry_point_overlay,
    get_bvecs_from_meta,
    symmetry_points_on_fixed_slice,
)
from Plot_QGT_2D_Dynamic_ND_sweep import (
    BERRY_COMPONENT_QUANTITIES,
    FLOQUET_RATIO_QUANTITIES,
    _berry_component_grids,
    _berry_component_plot_limits,
    _bundle_coordinate_order,
    _label_from_quantity,
    _load_nd_bundle,
    _load_saved_winding_grid,
    _pick_field_grid,
    _prepare_floquet_ratio_field,
    _prepare_nd_band_panel,
)


_COMPONENT_NAMES = ("Omega_x", "Omega_y", "Omega_z")
_COMPONENT_TITLES = (
    r"$\Omega_x=\Omega_{yz}$",
    r"$\Omega_y=\Omega_{zx}$",
    r"$\Omega_z=\Omega_{xy}$",
)
_COMPONENT_COLORS = ("#d62728", "#1f77b4", "#2ca02c")


def _closed_ring(values, radius_index):
    ring = np.asarray(values[:, radius_index], dtype=float)
    return np.concatenate((ring, ring[:1]))


def _global_absolute_limit(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 1.0
    limit = float(np.max(np.abs(finite)))
    return limit if limit > 0.0 else 1.0


def _winding_plot_range(winding_grid):
    finite = np.asarray(winding_grid, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return -1.0, 1.0

    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    span = maximum - minimum
    padding = 0.08 * span if span > 0.0 else max(0.5, 0.1 * abs(minimum))
    return minimum - padding, maximum + padding


def _parameter_title(base_title, names, axes, indices, radius=None):
    parameter_text = ", ".join(
        f"{name}={float(axes[position][indices[position]]):.6g}"
        for position, name in enumerate(names)
    )
    title = base_title
    if parameter_text:
        title += f" | {parameter_text}"
    if radius is not None:
        title += f" | r={float(radius):.6g}"
    return title


def _draw_band_panel(axis, band_panel):
    axis.set_title("Symmetry-line eigenvalues")
    axis.set_xlabel("k path")
    axis.set_ylabel("Eigenvalue")
    axis.set_ylim(*band_panel["range"])
    axis.grid(alpha=0.2)

    if not band_panel["available"]:
        axis.text(
            0.5,
            0.5,
            "Symmetry-line eigenvalues unavailable",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        return []

    lines = []
    for band in band_panel["bands"]:
        (line,) = axis.plot(
            band_panel["k_dist"],
            band_panel["initial"][:, band],
            linewidth=1.35,
            label=f"band {band}",
        )
        lines.append(line)

    for node_index in band_panel["node_indices"]:
        axis.axvline(
            float(band_panel["k_dist"][node_index]),
            color="0.45",
            linestyle="--",
            linewidth=0.8,
            alpha=0.45,
        )

    if band_panel["node_indices"].size:
        ticks = [
            float(band_panel["k_dist"][index])
            for index in band_panel["node_indices"]
        ]
        labels = [
            band_panel["path_labels"][position]
            if position < len(band_panel["path_labels"])
            else ""
            for position in range(len(ticks))
        ]
        axis.set_xticks(ticks, labels)

    if lines:
        axis.legend(loc="best", ncols=min(4, len(lines)), fontsize=8)
    return lines


def _build_cartesian_maps(
    components,
    parameter_shape,
    radius_values,
    phi_values,
    order,
    resolution,
):
    cartesian_maps = np.empty(
        (3,) + tuple(parameter_shape) + (resolution, resolution),
        dtype=float,
    )
    map_x_values = None
    map_y_values = None
    plane_axes = None

    for parameter_index in np.ndindex(tuple(parameter_shape)):
        for component_index, component in enumerate(components):
            (
                current_x,
                current_y,
                cartesian_component,
                current_plane_axes,
            ) = _polar_component_to_cartesian(
                radius_values,
                phi_values,
                component[parameter_index],
                order=order,
                resolution=resolution,
            )
            cartesian_maps[(component_index,) + parameter_index] = (
                cartesian_component
            )
            if plane_axes is None:
                map_x_values = current_x
                map_y_values = current_y
                plane_axes = current_plane_axes

    return cartesian_maps, map_x_values, map_y_values, plane_axes


def _build_scalar_cartesian_maps(
    field_grid,
    parameter_shape,
    radius_values,
    phi_values,
    order,
    resolution,
):
    cartesian_maps = np.empty(
        tuple(parameter_shape) + (resolution, resolution),
        dtype=float,
    )
    map_x_values = None
    map_y_values = None
    plane_axes = None

    for parameter_index in np.ndindex(tuple(parameter_shape)):
        (
            current_x,
            current_y,
            cartesian_field,
            current_plane_axes,
        ) = _polar_component_to_cartesian(
            radius_values,
            phi_values,
            field_grid[parameter_index],
            order=order,
            resolution=resolution,
        )
        cartesian_maps[parameter_index] = cartesian_field
        if plane_axes is None:
            map_x_values = current_x
            map_y_values = current_y
            plane_axes = current_plane_axes

    return cartesian_maps, map_x_values, map_y_values, plane_axes


def plot_berry_components_nd_matplotlib(
    root_dir,
    *,
    bands_to_plot=None,
    title=None,
    berry_zlim=None,
    berry_zlim_percentile=99.0,
    cartesian_resolution=None,
    cmap="RdBu_r",
    show_symmetry_points=True,
    symmetry_slice_tolerance=0.01,
    show=True,
):
    """Show an interactive Matplotlib N-D Berry-component viewer.

    Cartesian bundles display the symmetry-line bands and all three Berry
    maps. Polar bundles additionally display the selected-radius circle,
    three angular cuts, winding versus radius, and in-plane magnitude. One
    integer slider is created for every sweep parameter; polar data receives
    a shared radius slider and play/pause button.

    The returned figure stores the widgets and artists in ``figure.qgt_viewer``
    so they remain alive and can also be controlled programmatically.
    """
    (
        root_dir,
        data,
        names,
        axes,
        shape,
        kx,
        ky,
        meta,
        _params_json,
    ) = _load_nd_bundle(root_dir)

    order = _bundle_coordinate_order(data, meta)
    is_polar = is_cylindrical_order(order)
    if "ki" in data.files and "kj" in data.files:
        ki = np.asarray(data["ki"], dtype=float)
        kj = np.asarray(data["kj"], dtype=float)
    elif is_polar:
        raise KeyError(
            "Polar Berry-component plotting requires ki and kj in the N-D "
            "bundle. Recalculate the dataset with the polar-aware N-D calculation."
        )
    else:
        ki = np.asarray(kx, dtype=float)
        kj = np.asarray(ky, dtype=float)

    components = tuple(
        np.asarray(component, dtype=float)
        for component in _berry_component_grids(data)
    )
    expected_shape = tuple(shape) + tuple(ki.shape)
    for component_name, component in zip(_COMPONENT_NAMES, components):
        if component.shape != expected_shape:
            raise ValueError(
                f"{component_name} must have shape {expected_shape}; "
                f"received {component.shape}"
            )

    initial_indices = tuple(axis.size // 2 for axis in axes)
    current_indices = list(initial_indices)
    band_panel = _prepare_nd_band_panel(data, initial_indices, bands_to_plot)
    component_limits = [
        _berry_component_plot_limits(
            component,
            berry_zlim,
            berry_zlim_percentile,
        )
        for component in components
    ]

    radius_values = None
    phi_values = None
    phi_plot = None
    radius_indices = np.asarray([], dtype=int)
    radius_position = [0]
    winding_grid = None
    magnitude_grid = None
    unit_circle_x = None
    unit_circle_y = None

    if is_polar:
        phi_periodic = (
            bool(np.asarray(data["phi_periodic"]).item())
            if "phi_periodic" in data.files
            else False
        )
        if not phi_periodic:
            raise ValueError(
                "Polar Berry-component ring plots require a complete 2*pi "
                "angular grid"
            )

        radius_values, phi_values = _polar_grid_values(ki, kj)
        radius_indices = np.flatnonzero(radius_values > 0.0)
        if not radius_indices.size:
            raise ValueError("The polar grid must contain at least one positive radius")

        if cartesian_resolution is None:
            cartesian_resolution = max(80, min(301, radius_values.size))
        cartesian_resolution = int(cartesian_resolution)
        if cartesian_resolution < 3:
            raise ValueError("cartesian_resolution must be at least 3")

        map_data, map_x_values, map_y_values, plane_axes = (
            _build_cartesian_maps(
                components,
                shape,
                radius_values,
                phi_values,
                order,
                cartesian_resolution,
            )
        )

        _, _, _, fixed_axis = cylindrical_order_axes(order)
        omega_by_axis = dict(zip("xyz", components))
        winding_grid = _load_saved_winding_grid(data, shape, radius_values)

        plane_component_axes = tuple(
            axis for axis in "xyz" if axis != fixed_axis
        )
        magnitude_grid = np.hypot(
            omega_by_axis[plane_component_axes[0]],
            omega_by_axis[plane_component_axes[1]],
        )
        phi_plot = np.concatenate(
            (phi_values, [phi_values[0] + 2.0 * np.pi])
        )
        circle_phi = np.linspace(0.0, 2.0 * np.pi, 361)
        unit_circle_x, unit_circle_y, _ = _polar_plane_coordinates(
            1.0,
            circle_phi,
            order,
        )
    else:
        map_data = np.stack(components, axis=0)
        map_x_values = np.asarray(ki[0, :], dtype=float)
        map_y_values = np.asarray(kj[:, 0], dtype=float)
        plane_axes = (order[0], order[1])

    number_of_controls = len(names) + int(is_polar)
    controls_height = 0.036 * number_of_controls
    bottom_margin = min(0.42, 0.045 + controls_height)
    base_title = title or "Berry Curvature Components"

    if is_polar:
        figure = plt.figure(figsize=(16, 14.5))
        grid = figure.add_gridspec(
            4,
            6,
            height_ratios=(0.70, 1.45, 0.90, 0.90),
        )
        band_axis = figure.add_subplot(grid[0, :])
        map_axes = [
            figure.add_subplot(grid[1, 0:2]),
            figure.add_subplot(grid[1, 2:4]),
            figure.add_subplot(grid[1, 4:6]),
        ]
        cut_axes = [
            figure.add_subplot(grid[2, 0:2]),
            figure.add_subplot(grid[2, 2:4]),
            figure.add_subplot(grid[2, 4:6]),
        ]
        winding_axis = figure.add_subplot(grid[3, 0:3])
        magnitude_axis = figure.add_subplot(grid[3, 3:6])
        figure.subplots_adjust(
            left=0.065,
            right=0.965,
            top=0.93,
            bottom=bottom_margin,
            hspace=0.62,
            wspace=0.60,
        )
    else:
        figure = plt.figure(figsize=(16, 9.5))
        grid = figure.add_gridspec(2, 3, height_ratios=(0.78, 1.50))
        band_axis = figure.add_subplot(grid[0, :])
        map_axes = [figure.add_subplot(grid[1, index]) for index in range(3)]
        cut_axes = []
        winding_axis = None
        magnitude_axis = None
        figure.subplots_adjust(
            left=0.065,
            right=0.965,
            top=0.91,
            bottom=bottom_margin,
            hspace=0.42,
            wspace=0.58,
        )

    band_lines = _draw_band_panel(band_axis, band_panel)
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("#d9dde5")
    map_images = []
    initial_parameter_index = tuple(current_indices)
    extent = (
        float(map_x_values[0]),
        float(map_x_values[-1]),
        float(map_y_values[0]),
        float(map_y_values[-1]),
    )

    for component_index, (axis, component_title) in enumerate(
        zip(map_axes, _COMPONENT_TITLES)
    ):
        vmin, vmax = component_limits[component_index]
        image = axis.imshow(
            map_data[(component_index,) + initial_parameter_index],
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(component_title)
        axis.set_xlabel(rf"$k_{plane_axes[0]}$")
        axis.set_ylabel(rf"$k_{plane_axes[1]}$")
        axis.set_facecolor("#d9dde5")
        figure.colorbar(image, ax=axis, pad=0.025, fraction=0.048)
        map_images.append(image)

    if not is_polar and order == "xyz":
        b1, b2 = get_bvecs_from_meta(meta)
        bz_x, bz_y = bz_wigner_seitz_from_bvecs(b1, b2)
        path_points = (
            np.asarray(data["path_points"], dtype=float)
            if "path_points" in data.files
            else None
        )
        for map_axis in map_axes:
            map_axis.plot(bz_x, bz_y, color="white", linewidth=1.7)
            if path_points is not None:
                map_axis.plot(
                    path_points[:, 0],
                    path_points[:, 1],
                    color="#ff4545",
                    linewidth=1.2,
                    linestyle="--",
                )

    symmetry_point_labels = []
    symmetry_point_coordinates = np.empty((0, 2), dtype=float)
    symmetry_point_artists = []
    symmetry_point_annotations = []
    symmetry_slice_width = 0.0
    if show_symmetry_points:
        (
            symmetry_point_labels,
            symmetry_point_coordinates,
            symmetry_slice_width,
        ) = symmetry_points_on_fixed_slice(
            data,
            order,
            plane_axes,
            tolerance_fraction=symmetry_slice_tolerance,
        )
        (
            symmetry_point_artists,
            symmetry_point_annotations,
        ) = draw_symmetry_point_overlay(
            map_axes,
            symmetry_point_labels,
            symmetry_point_coordinates,
        )

    ring_lines = []
    component_cut_lines = []
    winding_line = None
    winding_radius_line = None
    magnitude_line = None

    if is_polar:
        initial_radius_index = int(radius_indices[radius_position[0]])
        initial_radius = float(radius_values[initial_radius_index])
        circle_x = initial_radius * unit_circle_x
        circle_y = initial_radius * unit_circle_y
        radius_max = float(radius_values[-1])
        for map_axis in map_axes:
            map_axis.set_xlim(-radius_max, radius_max)
            map_axis.set_ylim(-radius_max, radius_max)
            (outer_line,) = map_axis.plot(
                circle_x,
                circle_y,
                color="black",
                linewidth=4.2,
                alpha=0.78,
            )
            (inner_line,) = map_axis.plot(
                circle_x,
                circle_y,
                color="#ffd84d",
                linewidth=2.0,
            )
            ring_lines.extend((outer_line, inner_line))

        for component_index, (axis, component, color, component_name) in enumerate(
            zip(cut_axes, components, _COMPONENT_COLORS, _COMPONENT_NAMES)
        ):
            (line,) = axis.plot(
                phi_plot,
                _closed_ring(
                    component[initial_parameter_index],
                    initial_radius_index,
                ),
                color=color,
                linewidth=1.7,
            )
            axis.axhline(0.0, color="0.55", linewidth=0.8)
            axis.set_xlim(float(phi_plot[0]), float(phi_plot[-1]))
            axis.set_ylim(*component_limits[component_index])
            axis.set_xlabel(r"$\phi$ (rad)")
            axis.set_ylabel(component_name)
            axis.grid(alpha=0.18)
            component_cut_lines.append(line)

        current_winding = winding_grid[initial_parameter_index]
        (winding_line,) = winding_axis.plot(
            radius_values,
            current_winding,
            color="#2f3e56",
            linewidth=1.7,
            marker="o",
            markersize=2.8,
        )
        winding_axis.axhline(0.0, color="0.65", linewidth=0.8)
        winding_axis.set_xlim(
            float(radius_values[0]),
            float(radius_values[-1]),
        )
        winding_axis.set_ylim(*_winding_plot_range(winding_grid))
        winding_axis.set_xlabel(r"$r$")
        winding_axis.set_ylabel("Winding number W")
        winding_axis.set_title("Winding number versus radius")
        winding_axis.grid(alpha=0.2)
        winding_radius_line = winding_axis.axvline(
            initial_radius,
            color="#d62728",
            linewidth=1.6,
            linestyle="--",
        )

        initial_magnitude = _closed_ring(
            magnitude_grid[initial_parameter_index],
            initial_radius_index,
        )
        (magnitude_line,) = magnitude_axis.plot(
            phi_plot,
            initial_magnitude,
            color="#7a3db8",
            linewidth=1.7,
        )
        magnitude_axis.set_xlim(float(phi_plot[0]), float(phi_plot[-1]))
        magnitude_axis.set_ylim(0.0, 1.05 * _global_absolute_limit(magnitude_grid))
        magnitude_axis.set_xlabel(r"$\phi$ (rad)")
        magnitude_axis.set_ylabel(
            rf"$|\Omega_{{{plane_axes[0]}{plane_axes[1]}}}|$"
        )
        magnitude_axis.set_title("In-plane Berry magnitude versus phi")
        magnitude_axis.grid(alpha=0.2)

    parameter_sliders = []
    radius_slider = None
    play_button = None
    timer = None
    playing = [False]

    def current_parameter_index():
        return tuple(current_indices)

    def current_radius_index():
        return int(radius_indices[radius_position[0]])

    def update_title():
        radius = (
            float(radius_values[current_radius_index()])
            if is_polar
            else None
        )
        figure.suptitle(
            _parameter_title(
                base_title,
                names,
                axes,
                current_indices,
                radius,
            ),
            fontsize=14,
        )

    def update_radius_artists():
        if not is_polar:
            return

        parameter_index = current_parameter_index()
        radius_index = current_radius_index()
        radius = float(radius_values[radius_index])
        circle_x = radius * unit_circle_x
        circle_y = radius * unit_circle_y
        for ring_line in ring_lines:
            ring_line.set_data(circle_x, circle_y)

        for line, component in zip(component_cut_lines, components):
            line.set_ydata(
                _closed_ring(component[parameter_index], radius_index)
            )
        magnitude_line.set_ydata(
            _closed_ring(magnitude_grid[parameter_index], radius_index)
        )
        winding_radius_line.set_xdata([radius, radius])

    def update_parameter_artists():
        parameter_index = current_parameter_index()
        for component_index, image in enumerate(map_images):
            image.set_data(map_data[(component_index,) + parameter_index])

        if band_panel["available"]:
            eigenvalues = band_panel["eigenvalues"][
                parameter_index + (slice(None), slice(None))
            ]
            for line, band in zip(band_lines, band_panel["bands"]):
                line.set_ydata(eigenvalues[:, band])

        if is_polar:
            winding_line.set_ydata(winding_grid[parameter_index])
            update_radius_artists()

    def redraw():
        update_title()
        figure.canvas.draw_idle()

    slider_y = 0.018
    if is_polar:
        play_axis = figure.add_axes([0.05, slider_y - 0.003, 0.075, 0.026])
        radius_axis = figure.add_axes([0.20, slider_y, 0.70, 0.018])
        play_button = Button(play_axis, "Play")
        radius_slider = Slider(
            radius_axis,
            "Radius r",
            valmin=0,
            valmax=len(radius_indices) - 1,
            valinit=radius_position[0],
            valstep=1,
            valfmt="%d",
        )
        radius_slider.valtext.set_text(
            f"{float(radius_values[current_radius_index()]):.6g}"
        )
        slider_y += 0.036

        def on_radius_change(value):
            radius_position[0] = int(round(value))
            radius_slider.valtext.set_text(
                f"{float(radius_values[current_radius_index()]):.6g}"
            )
            update_radius_artists()
            redraw()

        radius_slider.on_changed(on_radius_change)

        timer = figure.canvas.new_timer(interval=100)

        def advance_radius():
            next_position = (radius_position[0] + 1) % len(radius_indices)
            radius_slider.set_val(next_position)

        timer.add_callback(advance_radius)

        def toggle_play(_event):
            playing[0] = not playing[0]
            if playing[0]:
                play_button.label.set_text("Pause")
                timer.start()
            else:
                play_button.label.set_text("Play")
                timer.stop()
            figure.canvas.draw_idle()

        play_button.on_clicked(toggle_play)

    for parameter_position, (name, axis_values) in enumerate(zip(names, axes)):
        slider_axis = figure.add_axes([0.20, slider_y, 0.70, 0.018])
        slider = Slider(
            slider_axis,
            name,
            valmin=0,
            valmax=len(axis_values) - 1,
            valinit=current_indices[parameter_position],
            valstep=1,
            valfmt="%d",
        )
        slider.valtext.set_text(
            f"{float(axis_values[current_indices[parameter_position]]):.6g}"
        )

        def on_parameter_change(
            value,
            position=parameter_position,
            current_slider=slider,
            values=axis_values,
        ):
            current_indices[position] = int(round(value))
            current_slider.valtext.set_text(
                f"{float(values[current_indices[position]]):.6g}"
            )
            update_parameter_artists()
            redraw()

        slider.on_changed(on_parameter_change)
        parameter_sliders.append(slider)
        slider_y += 0.036

    if timer is not None:
        def stop_timer(_event):
            timer.stop()

        figure.canvas.mpl_connect("close_event", stop_timer)

    update_title()
    figure.qgt_viewer = {
        "parameter_sliders": parameter_sliders,
        "radius_slider": radius_slider,
        "play_button": play_button,
        "timer": timer,
        "current_indices": current_indices,
        "radius_position": radius_position,
        "map_images": map_images,
        "band_lines": band_lines,
        "ring_lines": ring_lines,
        "component_cut_lines": component_cut_lines,
        "winding_line": winding_line,
        "winding_radius_line": winding_radius_line,
        "magnitude_line": magnitude_line,
        "symmetry_point_labels": symmetry_point_labels,
        "symmetry_point_coordinates": symmetry_point_coordinates,
        "symmetry_point_artists": symmetry_point_artists,
        "symmetry_point_annotations": symmetry_point_annotations,
        "symmetry_slice_width": symmetry_slice_width,
        "is_polar": is_polar,
        "order": order,
    }

    if show:
        plt.show()
    return figure


def dynamic_nd_field_with_bands_matplotlib(
    root_dir,
    *,
    quantity="trace",
    bands_to_plot=None,
    convert_berry_from_imQ=True,
    cmap="inferno",
    symmetric_cbar=None,
    title=None,
    show_integral=True,
    trace_zmax_percentile=99.0,
    berry_zlim=None,
    berry_zlim_percentile=99.0,
    ratio_zmax=None,
    ratio_zmax_percentile=99.0,
    berry_component_cmap="RdBu_r",
    cartesian_resolution=None,
    show_symmetry_points=True,
    symmetry_slice_tolerance=0.01,
    show=True,
):
    """Show the N-D QGT viewer directly with Matplotlib.

    The arguments mirror :func:`dynamic_nd_field_with_bands_html`, except
    that no output path is needed. ``quantity='berry_components'`` dispatches
    to :func:`plot_berry_components_nd_matplotlib`; scalar quantities use a
    symmetry-line band panel above one two-dimensional field map.
    """
    normalized_quantity = quantity.lower()
    if normalized_quantity in BERRY_COMPONENT_QUANTITIES:
        return plot_berry_components_nd_matplotlib(
            root_dir,
            bands_to_plot=bands_to_plot,
            title=title,
            berry_zlim=berry_zlim,
            berry_zlim_percentile=berry_zlim_percentile,
            cartesian_resolution=cartesian_resolution,
            cmap=berry_component_cmap,
            show_symmetry_points=show_symmetry_points,
            symmetry_slice_tolerance=symmetry_slice_tolerance,
            show=show,
        )

    (
        _root_dir,
        data,
        names,
        axes,
        shape,
        kx,
        ky,
        meta,
        _params_json,
    ) = _load_nd_bundle(root_dir)
    order = _bundle_coordinate_order(data, meta)
    is_polar = is_cylindrical_order(order)
    if is_polar:
        if "ki" not in data.files or "kj" not in data.files:
            raise KeyError("Polar N-D plotting requires ki and kj in the bundle")
        sampling_ki = np.asarray(data["ki"], dtype=float)
        sampling_kj = np.asarray(data["kj"], dtype=float)
    else:
        sampling_ki = (
            np.asarray(data["ki"], dtype=float)
            if "ki" in data.files
            else np.asarray(kx, dtype=float)
        )
        sampling_kj = (
            np.asarray(data["kj"], dtype=float)
            if "kj" in data.files
            else np.asarray(ky, dtype=float)
        )

    is_floquet_ratio = normalized_quantity in FLOQUET_RATIO_QUANTITIES
    if is_floquet_ratio:
        raw_field_grid, _ratio_hover_grid = _prepare_floquet_ratio_field(data)
    else:
        raw_field_grid = _pick_field_grid(
            data,
            quantity,
            convert_berry_from_imQ=convert_berry_from_imQ,
            order=order,
        )
    raw_field_grid = np.asarray(raw_field_grid, dtype=float)
    expected_shape = tuple(shape) + tuple(sampling_ki.shape)
    if raw_field_grid.shape != expected_shape:
        raise ValueError(
            f"Field '{quantity}' must have shape {expected_shape}; "
            f"received {raw_field_grid.shape}"
        )

    finite = raw_field_grid[np.isfinite(raw_field_grid)]
    berry_quantities = ("berry", "berry_curvature", "omega")
    if is_floquet_ratio:
        if ratio_zmax is not None:
            vmax = float(ratio_zmax)
            if not np.isfinite(vmax) or vmax <= 0.0:
                raise ValueError("ratio_zmax must be a finite positive number")
        elif finite.size and ratio_zmax_percentile is not None:
            percentile = float(ratio_zmax_percentile)
            if not 0.0 <= percentile <= 100.0:
                raise ValueError(
                    "ratio_zmax_percentile must be between 0 and 100"
                )
            vmax = float(np.percentile(finite, percentile))
        elif finite.size:
            vmax = float(np.max(finite))
        else:
            vmax = 1.0
        if vmax <= 0.0:
            positive = finite[finite > 0.0]
            vmax = float(np.max(positive)) if positive.size else 1.0
        vmin = 0.0
    else:
        if not finite.size:
            raise ValueError(f"Field '{quantity}' contains no finite values")
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))

    if symmetric_cbar is None:
        symmetric_cbar = normalized_quantity != "trace" and not is_floquet_ratio
    elif is_floquet_ratio and symmetric_cbar:
        raise ValueError(
            "Floquet perturbative ratios are nonnegative; "
            "symmetric_cbar must be False or None"
        )

    if normalized_quantity in berry_quantities:
        symmetric_limits = _berry_component_plot_limits(
            finite,
            berry_zlim,
            berry_zlim_percentile,
        )
        if symmetric_cbar:
            vmin, vmax = symmetric_limits
        else:
            clipped_min = max(vmin, symmetric_limits[0])
            clipped_max = min(vmax, symmetric_limits[1])
            if clipped_min < clipped_max:
                vmin, vmax = clipped_min, clipped_max
    elif normalized_quantity == "trace" and trace_zmax_percentile is not None:
        vmax = float(np.nanpercentile(finite, trace_zmax_percentile))
        if vmax <= vmin:
            vmax = float(np.max(finite))
    if symmetric_cbar and normalized_quantity not in berry_quantities:
        absolute_maximum = max(abs(vmin), abs(vmax))
        if absolute_maximum == 0.0:
            absolute_maximum = 1.0
        vmin, vmax = -absolute_maximum, absolute_maximum
    if vmin == vmax:
        padding = 1.0 if vmin == 0.0 else 0.05 * abs(vmin)
        vmin -= padding
        vmax += padding

    field_grid = np.array(raw_field_grid, dtype=float, copy=True)
    field_grid[np.isposinf(field_grid)] = vmax
    field_grid[np.isneginf(field_grid)] = vmin
    if is_polar:
        radius_values, phi_values = _polar_grid_values(
            sampling_ki,
            sampling_kj,
        )
        if cartesian_resolution is None:
            cartesian_resolution = max(80, min(301, radius_values.size))
        cartesian_resolution = int(cartesian_resolution)
        if cartesian_resolution < 3:
            raise ValueError("cartesian_resolution must be at least 3")
        map_data, map_x_values, map_y_values, plane_axes = (
            _build_scalar_cartesian_maps(
                field_grid,
                shape,
                radius_values,
                phi_values,
                order,
                cartesian_resolution,
            )
        )
    else:
        map_data = field_grid
        map_x_values = np.asarray(sampling_ki[0, :], dtype=float)
        map_y_values = np.asarray(sampling_kj[:, 0], dtype=float)
        plane_axes = (order[0], order[1])

    initial_indices = tuple(axis.size // 2 for axis in axes)
    current_indices = list(initial_indices)
    band_panel = _prepare_nd_band_panel(data, initial_indices, bands_to_plot)
    bottom_margin = min(0.38, 0.055 + 0.036 * len(names))

    figure = plt.figure(figsize=(10.5, 10.0))
    grid = figure.add_gridspec(2, 1, height_ratios=(0.85, 1.65))
    band_axis = figure.add_subplot(grid[0, 0])
    map_axis = figure.add_subplot(grid[1, 0])
    figure.subplots_adjust(
        left=0.10,
        right=0.88,
        top=0.91,
        bottom=bottom_margin,
        hspace=0.42,
    )

    band_lines = _draw_band_panel(band_axis, band_panel)
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("#d9dde5")
    extent = (
        float(map_x_values[0]),
        float(map_x_values[-1]),
        float(map_y_values[0]),
        float(map_y_values[-1]),
    )
    image = map_axis.imshow(
        map_data[initial_indices],
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
    )
    label = title or _label_from_quantity(quantity)
    label = label.replace("<sub>", "_").replace("</sub>", "")
    if is_polar and title is None and normalized_quantity in berry_quantities:
        _, _, _, fixed_axis = cylindrical_order_axes(order)
        label = f"Berry Curvature Omega_{fixed_axis}"
    map_axis.set_title(label)
    map_axis.set_xlabel(rf"$k_{plane_axes[0]}$")
    map_axis.set_ylabel(rf"$k_{plane_axes[1]}$")
    map_axis.set_facecolor("#d9dde5")
    colorbar = figure.colorbar(image, ax=map_axis, pad=0.025, fraction=0.048)
    colorbar.set_label(label)

    if not is_polar and order == "xyz":
        b1, b2 = get_bvecs_from_meta(meta)
        bz_x, bz_y = bz_wigner_seitz_from_bvecs(b1, b2)
        map_axis.plot(bz_x, bz_y, color="white", linewidth=1.7)
        if "path_points" in data.files:
            path_points = np.asarray(data["path_points"], dtype=float)
            map_axis.plot(
                path_points[:, 0],
                path_points[:, 1],
                color="#ff4545",
                linewidth=1.2,
                linestyle="--",
            )

    symmetry_point_labels = []
    symmetry_point_coordinates = np.empty((0, 2), dtype=float)
    symmetry_point_artists = []
    symmetry_point_annotations = []
    symmetry_slice_width = 0.0
    if show_symmetry_points:
        (
            symmetry_point_labels,
            symmetry_point_coordinates,
            symmetry_slice_width,
        ) = symmetry_points_on_fixed_slice(
            data,
            order,
            plane_axes,
            tolerance_fraction=symmetry_slice_tolerance,
        )
        (
            symmetry_point_artists,
            symmetry_point_annotations,
        ) = draw_symmetry_point_overlay(
            [map_axis],
            symmetry_point_labels,
            symmetry_point_coordinates,
        )

    if not is_polar:
        if "dkx" in data.files:
            first_spacing = float(data["dkx"])
        else:
            first_spacing = float(sampling_ki[0, 1] - sampling_ki[0, 0])
        if "dky" in data.files:
            second_spacing = float(data["dky"])
        else:
            second_spacing = float(sampling_kj[1, 0] - sampling_kj[0, 0])
        area_element = first_spacing * second_spacing
    else:
        area_element = 0.0

    def current_parameter_index():
        return tuple(current_indices)

    def title_for_current_state():
        index = current_parameter_index()
        frame = raw_field_grid[index]
        finite_frame = frame[np.isfinite(frame)]
        if is_floquet_ratio:
            positive_infinities = int(np.count_nonzero(np.isposinf(frame)))
            if positive_infinities:
                summary = f"max=inf | exact resonances={positive_infinities}"
            elif finite_frame.size:
                summary = f"max={float(np.max(finite_frame)):.6g}"
            else:
                summary = "unavailable"
        else:
            standard_deviation = (
                float(np.std(finite_frame)) if finite_frame.size else np.nan
            )
            summary = f"std={standard_deviation:.3e}"
            if (
                show_integral
                and not is_polar
                and normalized_quantity
                in ("trace_minus_berry", "trace_minus_omega")
            ):
                integral = float(np.nansum(frame) * area_element)
                summary += f" | integral={integral:.6g}"

        parameter_text = ", ".join(
            f"{name}={float(axes[position][current_indices[position]]):.6g}"
            for position, name in enumerate(names)
        )
        if parameter_text:
            return f"{label} | {parameter_text} | {summary}"
        return f"{label} | {summary}"

    def update_parameter_artists():
        index = current_parameter_index()
        image.set_data(map_data[index])
        if band_panel["available"]:
            eigenvalues = band_panel["eigenvalues"][
                index + (slice(None), slice(None))
            ]
            for line, band in zip(band_lines, band_panel["bands"]):
                line.set_ydata(eigenvalues[:, band])
        figure.suptitle(title_for_current_state(), fontsize=13)
        figure.canvas.draw_idle()

    parameter_sliders = []
    slider_y = 0.02
    for parameter_position, (name, axis_values) in enumerate(zip(names, axes)):
        slider_axis = figure.add_axes([0.20, slider_y, 0.66, 0.018])
        slider = Slider(
            slider_axis,
            name,
            valmin=0,
            valmax=len(axis_values) - 1,
            valinit=current_indices[parameter_position],
            valstep=1,
            valfmt="%d",
        )
        slider.valtext.set_text(
            f"{float(axis_values[current_indices[parameter_position]]):.6g}"
        )

        def on_parameter_change(
            value,
            position=parameter_position,
            current_slider=slider,
            values=axis_values,
        ):
            current_indices[position] = int(round(value))
            current_slider.valtext.set_text(
                f"{float(values[current_indices[position]]):.6g}"
            )
            update_parameter_artists()

        slider.on_changed(on_parameter_change)
        parameter_sliders.append(slider)
        slider_y += 0.036

    figure.suptitle(title_for_current_state(), fontsize=13)
    figure.qgt_viewer = {
        "parameter_sliders": parameter_sliders,
        "radius_slider": None,
        "play_button": None,
        "current_indices": current_indices,
        "map_images": [image],
        "band_lines": band_lines,
        "symmetry_point_labels": symmetry_point_labels,
        "symmetry_point_coordinates": symmetry_point_coordinates,
        "symmetry_point_artists": symmetry_point_artists,
        "symmetry_point_annotations": symmetry_point_annotations,
        "symmetry_slice_width": symmetry_slice_width,
        "is_polar": is_polar,
        "order": order,
        "quantity": normalized_quantity,
    }

    if show:
        plt.show()
    return figure


if __name__ == "__main__":
    dataset_dir = os.path.join(
        "results",
        "2D_QGT_ND",
        "gWaveAltermagnetHamiltonian",
        "dataset7",
    )
    dynamic_nd_field_with_bands_matplotlib(
        dataset_dir,
        quantity="berry_components",
        berry_zlim_percentile=90.0,
        bands_to_plot=[0, 1, 2, 3],
    )
