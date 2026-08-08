import json
import os
import pickle
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import Library.Hamiltonian.Hamiltonian
from Library.Hamiltonian.ChiralHamiltonian import ChiralHamiltonian
from Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected import (
    ChiralHamiltonianChiralBasisProjected,
)
from Library.dimension_lib import (
    cylindrical_order_axes,
    is_cylindrical_order,
    normalize_coordinate_order,
)
from Library.plotting_qgt_2d import get_symmetric_plot_limits
from Library.plotting_qgt_polar import (
    _polar_component_to_cartesian,
    _polar_grid_values,
    _polar_plane_coordinates,
)
from Library.plotting_utils import bz_wigner_seitz_from_bvecs, get_bvecs_from_meta
from Library.topology import (
    berry_curvature_components_from_qgt,
)


FLOQUET_RATIO_QUANTITIES = frozenset(
    {
        "floquet_max_ratio",
        "max_floquet_ratio",
        "floquet_ratio",
        "perturbative_ratio",
    }
)

BERRY_COMPONENT_QUANTITIES = frozenset(
    {
        "berry_components",
        "berry components",
        "berry-components",
        "omega_components",
        "omega components",
    }
)


# Backward compatibility for old pickled Hamiltonians.
sys.modules["Library.Hamiltonian_v2"] = Library.Hamiltonian.Hamiltonian
Library.Hamiltonian.Hamiltonian.ChiralHamiltonian = ChiralHamiltonian
Library.Hamiltonian.Hamiltonian.RhombohedralGrapheneHamiltonian = (
    ChiralHamiltonianChiralBasisProjected
)


def _resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)


def _load_nd_bundle(root_dir):
    root_dir = _resolve_path(root_dir)
    bundle_path = os.path.join(root_dir, "qgt_nd_bundle.npz")
    meta_path = os.path.join(root_dir, "meta.pkl")
    json_path = os.path.join(root_dir, "parameters.json")

    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Cannot find bundle at: {bundle_path}")

    data = np.load(bundle_path, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    shape = tuple(int(x) for x in data["shape"])
    axes = [np.asarray(data[f"axis_{i}_{names[i]}"]) for i in range(len(names))]
    kx = np.asarray(data["kx"])
    ky = np.asarray(data["ky"])

    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

    params_json = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            params_json = json.load(f)

    return root_dir, data, names, axes, shape, kx, ky, meta, params_json


def _bundle_coordinate_order(data, meta):
    if "order" in data.files:
        return normalize_coordinate_order(str(np.asarray(data["order"]).item()))
    if isinstance(meta, dict) and "order" in meta:
        return normalize_coordinate_order(meta["order"])
    return "xyz"


def _berry_component_grids(data):
    required = (
        "g_xy_imag_grid",
        "g_xz_imag_grid",
        "g_yz_imag_grid",
    )
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(
            "Berry-component plotting requires all three imaginary QGT "
            f"components; missing {missing}. Recalculate this N-D bundle."
        )

    return berry_curvature_components_from_qgt(
        np.asarray(data["g_xy_imag_grid"], dtype=float),
        np.asarray(data["g_xz_imag_grid"], dtype=float),
        np.asarray(data["g_yz_imag_grid"], dtype=float),
    )


def _load_saved_winding_grid(data, parameter_shape, radius_values):
    """Load and validate winding numbers produced by the N-D calculation."""
    required = ("winding_radius", "winding_grid")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(
            "Polar winding plots require saved winding data; missing "
            f"{missing}. Recalculate this dataset with the polar-aware N-D "
            "calculation."
        )

    saved_radius = np.asarray(data["winding_radius"], dtype=float)
    radius_values = np.asarray(radius_values, dtype=float)
    if saved_radius.shape != radius_values.shape or not np.allclose(
        saved_radius,
        radius_values,
    ):
        raise ValueError(
            "winding_radius must match the radial coordinate stored in ki"
        )

    winding_grid = np.asarray(data["winding_grid"], dtype=float)
    expected_shape = tuple(parameter_shape) + (radius_values.size,)
    if winding_grid.shape != expected_shape:
        raise ValueError(
            "winding_grid must have shape "
            f"{expected_shape}; received {winding_grid.shape}"
        )
    return winding_grid


def _berry_component_plot_limits(component, zlim, percentile):
    limits = get_symmetric_plot_limits(component, zlim, percentile)
    return (-1.0, 1.0) if limits is None else tuple(float(v) for v in limits)


def _normal_berry_grid(data, order, convert_berry_from_imQ):
    if "berry_grid" in data.files:
        return np.asarray(data["berry_grid"])

    if is_cylindrical_order(order):
        _, _, _, fixed_axis = cylindrical_order_axes(order)
        try:
            omega_by_axis = dict(zip("xyz", _berry_component_grids(data)))
        except KeyError:
            if fixed_axis != "z":
                raise
        else:
            return omega_by_axis[fixed_axis]

    gxy_imag = np.asarray(data["g_xy_imag_grid"])
    return (-2.0 * gxy_imag) if convert_berry_from_imQ else gxy_imag


def _pick_field_grid(
    data,
    quantity="trace",
    *,
    convert_berry_from_imQ=True,
    order="xyz",
):
    q = quantity.lower()

    if q == "trace":
        return np.asarray(data["trace_grid"])

    if q in ("berry", "berry_curvature", "omega"):
        return _normal_berry_grid(data, order, convert_berry_from_imQ)

    if q in ("imqxy", "im(q_xy)", "im_qxy"):
        return np.asarray(data["g_xy_imag_grid"])

    if q in ("trace_minus_berry", "trace_minus_omega"):
        trace = np.asarray(data["trace_grid"])
        berry = _normal_berry_grid(data, order, convert_berry_from_imQ)
        return trace - berry

    if q in FLOQUET_RATIO_QUANTITIES:
        field_grid, _ = _prepare_floquet_ratio_field(data)
        return field_grid

    if q in BERRY_COMPONENT_QUANTITIES:
        raise ValueError(
            "Berry components use the dedicated three-component renderer. "
            "Call dynamic_nd_field_with_bands_html with "
            "quantity='berry_components'."
        )

    raise ValueError(f"Unknown quantity '{quantity}'.")


def _prepare_floquet_ratio_field(data):
    """Reduce the per-band ratio grid and build its diagnostic hover data.

    Returns
    -------
    field_grid : ndarray
        Maximum ratio over the source-band axis, with shape
        ``(*parameter_shape, Ny, Nx)``.
    hover_grid : ndarray of object
        Last axis contains the displayed ratio, source band, coupled band,
        and photon index ``l`` for the maximizing transition.
    """
    ratio_key = "floquet_max_ratio_grid"
    index_key = "floquet_max_ratio_indices_grid"
    missing = [key for key in (ratio_key, index_key) if key not in data.files]
    if missing:
        raise KeyError(
            "The N-D bundle does not contain the Floquet diagnostic arrays: "
            f"{missing}. Recalculate the bundle with Floquet-ratio storage enabled."
        )

    ratios = np.asarray(data[ratio_key], dtype=float)
    indices = np.asarray(data[index_key])
    if ratios.ndim < 3:
        raise ValueError(
            "floquet_max_ratio_grid must have shape "
            "(*parameter_shape, Ny, Nx, number_of_bands)"
        )
    if indices.shape != ratios.shape + (2,):
        raise ValueError(
            "floquet_max_ratio_indices_grid must have shape "
            "floquet_max_ratio_grid.shape + (2,); received "
            f"{indices.shape} for ratio shape {ratios.shape}"
        )
    if np.any(ratios[np.isfinite(ratios)] < 0.0) or np.any(
        np.isneginf(ratios)
    ):
        raise ValueError("Floquet perturbative ratios must be nonnegative")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("Floquet ratio indices must use an integer dtype")

    available = np.any(~np.isnan(ratios), axis=-1)
    comparison = np.where(np.isnan(ratios), -np.inf, ratios)
    source_bands = np.argmax(comparison, axis=-1)
    field_grid = np.take_along_axis(
        ratios,
        source_bands[..., None],
        axis=-1,
    )[..., 0]
    field_grid = np.where(available, field_grid, np.nan)

    gather_indices = np.broadcast_to(
        source_bands[..., None, None],
        source_bands.shape + (1, 2),
    )
    maximizing_indices = np.take_along_axis(
        indices,
        gather_indices,
        axis=-2,
    )[..., 0, :]
    coupled_bands = maximizing_indices[..., 0]
    photon_indices = maximizing_indices[..., 1]
    has_transition = available & (coupled_bands >= 0) & (photon_indices != 0)

    safe_ratios = np.where(np.isfinite(field_grid), field_grid, 0.0)
    ratio_display = np.char.mod("%.6g", safe_ratios).astype(object)
    ratio_display[np.isposinf(field_grid)] = "∞"
    ratio_display[np.isneginf(field_grid)] = "−∞"
    ratio_display[np.isnan(field_grid)] = "unavailable"

    # Keep valid indices numeric so large N-D HTML files stay materially
    # smaller than an all-string customdata representation.
    source_display = source_bands.astype(object)
    coupled_display = coupled_bands.astype(object)
    photon_display = photon_indices.astype(object)
    source_display[~has_transition] = "none"
    coupled_display[~has_transition] = "none"
    photon_display[~has_transition] = "none"

    hover_grid = np.stack(
        (
            ratio_display,
            source_display,
            coupled_display,
            photon_display,
        ),
        axis=-1,
    )
    return field_grid, hover_grid


def _label_from_quantity(quantity):
    q = quantity.lower()
    if q in FLOQUET_RATIO_QUANTITIES:
        return "Maximum Floquet Perturbative Ratio"
    if q in BERRY_COMPONENT_QUANTITIES:
        return "Berry Curvature Components"
    return {
        "trace": "QGT Trace",
        "berry": "Berry Curvature Ω<sub>xy</sub>",
        "berry_curvature": "Berry Curvature Ω<sub>xy</sub>",
        "omega": "Berry Curvature Ω<sub>xy</sub>",
        "imqxy": "Im(Q<sub>xy</sub>)",
        "im(q_xy)": "Im(Q<sub>xy</sub>)",
        "im_qxy": "Im(Q<sub>xy</sub>)",
        "trace_minus_berry": "Tr(g) − Ω<sub>xy</sub>",
        "trace_minus_omega": "Tr(g) − Ω<sub>xy</sub>",
    }.get(q, "Field")


def _json_script_value(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _json_ready_numeric_array(values):
    """Convert non-finite numeric entries to JSON/Plotly-compatible nulls."""
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.number):
        return values.tolist()
    output = values.astype(object)
    output[~np.isfinite(values)] = None
    return output.tolist()


def _floquet_ratio_summary_grid(field_grid):
    """Build one finite-statistics summary string per parameter point."""
    parameter_shape = field_grid.shape[:-2]
    summaries = np.empty(parameter_shape, dtype=object)
    for index in np.ndindex(parameter_shape):
        frame = np.asarray(field_grid[index], dtype=float)
        finite = frame[np.isfinite(frame)]
        positive_infinities = int(np.count_nonzero(np.isposinf(frame)))
        if positive_infinities:
            maximum_text = "∞"
        elif finite.size:
            maximum_text = f"{float(np.max(finite)):.6g}"
        else:
            maximum_text = "unavailable"

        finite_std_text = (
            f"{float(np.std(finite)):.3e}" if finite.size else "unavailable"
        )
        summary = f"max={maximum_text} | finite std={finite_std_text}"
        if positive_infinities:
            summary += (
                " | exact-resonance k-points="
                f"{positive_infinities}"
            )
        summaries[index] = summary
    return summaries


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return str(v)


def _row(key, val):
    return f'<div class="row"><span class="key">{key}</span><span class="val">{val}</span></div>'


def _section(title, rows):
    return f'<div class="section"><div class="section-title">{title}</div>{rows}</div>'


def _sidebar_html(params_json, names, axes, band_index_text, band_cut_text):
    ham_name = params_json.get("hamiltonian_name", "ND QGT Bundle")

    ham_rows = "".join(
        _row(k, _fmt(v))
        for k, v in sorted(params_json.get("parameters", {}).items())
    )
    if not ham_rows:
        ham_rows = _row("source", "qgt_nd_bundle.npz")

    scan_ranges = params_json.get("scan_ranges", {})
    scan_spacing = params_json.get("scan_spacing", {})
    scan_rows = ""
    for name in names:
        if name in scan_ranges:
            info = scan_ranges[name]
            spacing = scan_spacing.get(name, {})
            scan_rows += _row(
                name,
                f'{_fmt(info.get("min"))} to {_fmt(info.get("max"))}'
                + f' ({spacing.get("count", len(axes[names.index(name)]))} pts, '
                + f'{spacing.get("scale", "unknown")})',
            )
        else:
            scan_rows += _row(name, f"{len(axes[names.index(name)])} pts")

    kg = params_json.get("k_grid", {})
    if kg:
        if kg.get("coordinate_system") == "cylindrical":
            labels = kg.get("coordinate_labels", ["r", "phi", "k_fixed"])
            ki_domain = kg.get("ki_domain", kg.get("ki_range", [None, None]))
            kj_domain = kg.get("kj_domain", kg.get("kj_range", [None, None]))
            grid_rows = (
                _row("order", _fmt(kg.get("order")))
                + _row(labels[0], f"{_fmt(ki_domain[0])} to {_fmt(ki_domain[1])}")
                + _row(labels[1], f"{_fmt(kj_domain[0])} to {_fmt(kj_domain[1])}")
                + _row(labels[2], _fmt(kg.get("fixed_coordinate", kg.get("kk"))))
                + _row("mesh", kg.get("mesh"))
            )
        else:
            grid_rows = (
                _row("k<sub>x</sub>", f'{_fmt(kg.get("kx_min"))} to {_fmt(kg.get("kx_max"))}')
                + _row("k<sub>y</sub>", f'{_fmt(kg.get("ky_min"))} to {_fmt(kg.get("ky_max"))}')
                + _row("mesh", kg.get("mesh"))
            )
    else:
        grid_rows = _row("grid", "from bundle")

    floquet_diagnostic = params_json.get("floquet_diagnostic", {})
    if floquet_diagnostic:
        diagnostic_rows = (
            _row("max |ℓ|", _fmt(floquet_diagnostic.get("max_l")))
            + _row("band basis", _fmt(floquet_diagnostic.get("band_basis")))
            + _row("index order", _fmt(floquet_diagnostic.get("index_order")))
            + _row(
                "same-band replicas",
                _fmt(floquet_diagnostic.get("includes_same_band")),
            )
        )
        diagnostic_section = _section("Floquet Diagnostic", diagnostic_rows)
    else:
        diagnostic_section = ""

    return f"""
<aside class="sidebar">
  <div class="sidebar-header">{ham_name}</div>
  {_section("Hamiltonian Parameters", ham_rows)}
  {_section("Sweep Axes", scan_rows)}
  {_section("k-Grid", grid_rows + _row("band index", band_index_text) + _row("band cut", band_cut_text))}
  {diagnostic_section}
  <div class="section">
    <div class="section-title">Controls</div>
    <div id="controls"></div>
  </div>
</aside>
"""


def _prepare_nd_band_panel(data, init_idx, bands_to_plot):
    required = ("eigenvalues_sym_grid", "k_dist", "node_indices", "path_labels")
    if not all(key in data.files for key in required):
        return {
            "available": False,
            "eigenvalues": None,
            "k_dist": np.asarray([0.0, 1.0]),
            "node_indices": np.asarray([], dtype=int),
            "path_labels": [],
            "bands": [],
            "initial": None,
            "range": (-1.0, 1.0),
        }

    eigenvalues = np.asarray(data["eigenvalues_sym_grid"], dtype=float)
    k_dist = np.asarray(data["k_dist"], dtype=float)
    node_indices = np.asarray(data["node_indices"], dtype=int)
    path_labels = [str(value) for value in np.asarray(data["path_labels"])]
    number_of_bands = int(eigenvalues.shape[-1])

    if bands_to_plot is None:
        bands = list(range(number_of_bands))
    elif isinstance(bands_to_plot, int):
        bands = [int(bands_to_plot)]
    else:
        bands = [int(band) for band in bands_to_plot]
    invalid = [band for band in bands if not 0 <= band < number_of_bands]
    if invalid:
        raise IndexError(
            f"Out-of-range bands {invalid}; valid range is [0,{number_of_bands - 1}]"
        )

    selected = eigenvalues[..., bands]
    finite = selected[np.isfinite(selected)]
    if finite.size:
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        padding = 0.05 * (maximum - minimum) if maximum > minimum else 1.0
        energy_range = (minimum - padding, maximum + padding)
    else:
        energy_range = (-1.0, 1.0)

    return {
        "available": True,
        "eigenvalues": eigenvalues,
        "k_dist": k_dist,
        "node_indices": node_indices,
        "path_labels": path_labels,
        "bands": bands,
        "initial": eigenvalues[init_idx + (slice(None), slice(None))],
        "range": energy_range,
    }


def _dynamic_nd_berry_components_html(
    root_dir,
    *,
    bands_to_plot=None,
    title=None,
    output_html=None,
    berry_zlim=None,
    berry_zlim_percentile=99.0,
    cartesian_resolution=None,
    cmap="RdBu_r",
):
    """Render all Cartesian Berry components for an N-D parameter sweep."""
    (
        root_dir,
        data,
        names,
        axes,
        shape,
        kx,
        ky,
        meta,
        params_json,
    ) = _load_nd_bundle(root_dir)

    order = _bundle_coordinate_order(data, meta)
    is_polar = is_cylindrical_order(order)
    if "ki" in data.files and "kj" in data.files:
        ki = np.asarray(data["ki"], dtype=float)
        kj = np.asarray(data["kj"], dtype=float)
    elif is_polar:
        raise KeyError(
            "Polar Berry-component plotting requires ki and kj in the N-D bundle. "
            "Recalculate the dataset with the polar-aware N-D calculation."
        )
    else:
        ki = np.asarray(kx, dtype=float)
        kj = np.asarray(ky, dtype=float)

    components = tuple(np.asarray(value, dtype=float) for value in _berry_component_grids(data))
    expected_shape = tuple(shape) + tuple(ki.shape)
    for name, component in zip(("Omega_x", "Omega_y", "Omega_z"), components):
        if component.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}; received {component.shape}"
            )

    component_stack = np.stack(components, axis=0)
    component_names = ["Omega_x", "Omega_y", "Omega_z"]
    component_titles = [
        "Ω<sub>x</sub> = Ω<sub>yz</sub>",
        "Ω<sub>y</sub> = Ω<sub>zx</sub>",
        "Ω<sub>z</sub> = Ω<sub>xy</sub>",
    ]
    component_colors = ["#d62728", "#1f77b4", "#2ca02c"]
    component_limits = [
        _berry_component_plot_limits(component, berry_zlim, berry_zlim_percentile)
        for component in components
    ]
    component_line_limits = []
    for component in components:
        finite_component = component[np.isfinite(component)]
        absolute_maximum = (
            float(np.max(np.abs(finite_component)))
            if finite_component.size
            else 0.0
        )
        component_line_limits.append(
            1.0 if absolute_maximum == 0.0 else absolute_maximum
        )

    init_idx = tuple(axis.size // 2 for axis in axes)
    band_panel = _prepare_nd_band_panel(data, init_idx, bands_to_plot)

    radius_values = None
    phi_values = None
    phi_plot = None
    radius_indices = np.asarray([], dtype=int)
    initial_radius_index = None
    winding_grid = None
    magnitude_grid = None
    winding_range = (-1.0, 1.0)
    magnitude_limit = 1.0
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
                "Polar Berry-component ring plots require a complete 2*pi angular grid"
            )

        radius_values, phi_values = _polar_grid_values(ki, kj)
        radius_indices = np.flatnonzero(radius_values > 0.0)
        if not radius_indices.size:
            raise ValueError("The polar grid must contain at least one positive radius")
        initial_radius_index = int(radius_indices[0])

        if cartesian_resolution is None:
            cartesian_resolution = max(80, min(301, radius_values.size))
        cartesian_resolution = int(cartesian_resolution)
        cartesian_maps = np.empty(
            (3,) + tuple(shape) + (cartesian_resolution, cartesian_resolution),
            dtype=float,
        )
        plane_axes = None
        map_x_values = None
        map_y_values = None
        for parameter_index in np.ndindex(shape):
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
                    resolution=cartesian_resolution,
                )
                cartesian_maps[(component_index,) + parameter_index] = cartesian_component
                if plane_axes is None:
                    plane_axes = current_plane_axes
                    map_x_values = current_x
                    map_y_values = current_y
        map_data = cartesian_maps

        _, _, _, fixed_axis = cylindrical_order_axes(order)
        omega_by_axis = dict(zip("xyz", components))
        winding_grid = _load_saved_winding_grid(data, shape, radius_values)

        plane_component_axes = tuple(axis for axis in "xyz" if axis != fixed_axis)
        magnitude_grid = np.hypot(
            omega_by_axis[plane_component_axes[0]],
            omega_by_axis[plane_component_axes[1]],
        )
        finite_winding = winding_grid[np.isfinite(winding_grid)]
        if finite_winding.size:
            winding_min = float(np.min(finite_winding))
            winding_max = float(np.max(finite_winding))
            winding_span = winding_max - winding_min
            winding_padding = (
                0.08 * winding_span
                if winding_span > 0.0
                else max(0.5, 0.1 * abs(winding_min))
            )
            winding_range = (
                winding_min - winding_padding,
                winding_max + winding_padding,
            )
        finite_magnitude = magnitude_grid[np.isfinite(magnitude_grid)]
        if finite_magnitude.size and float(np.max(finite_magnitude)) > 0.0:
            magnitude_limit = 1.05 * float(np.max(finite_magnitude))

        phi_plot = np.concatenate((phi_values, [phi_values[0] + 2.0 * np.pi]))
        circle_phi = np.linspace(0.0, 2.0 * np.pi, 361)
        unit_circle_x, unit_circle_y, _ = _polar_plane_coordinates(
            1.0,
            circle_phi,
            order,
        )
    else:
        map_data = component_stack
        map_x_values = np.asarray(ki[0, :], dtype=float)
        map_y_values = np.asarray(kj[:, 0], dtype=float)
        plane_axes = (order[0], order[1])

    if is_polar:
        subplot_specs = [
            [{"colspan": 6}, None, None, None, None, None],
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
        ]
        subplot_titles = (
            "Symmetry-line eigenvalues",
            *component_titles,
            "", "", "",
            "Winding number versus radius",
            "In-plane Berry magnitude versus phi",
        )
        row_heights = [0.17, 0.37, 0.24, 0.22]
        vertical_spacing = 0.055
        figure_height = 1460
        map_columns = [1, 3, 5]
        map_row = 2
    else:
        subplot_specs = [
            [{"colspan": 3}, None, None],
            [{}, {}, {}],
        ]
        subplot_titles = ("Symmetry-line eigenvalues", *component_titles)
        row_heights = [0.30, 0.70]
        vertical_spacing = 0.08
        figure_height = 900
        map_columns = [1, 2, 3]
        map_row = 2

    fig = make_subplots(
        rows=len(subplot_specs),
        cols=len(subplot_specs[0]),
        specs=subplot_specs,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        horizontal_spacing=0.045,
        vertical_spacing=vertical_spacing,
    )

    band_trace_indices = []
    if band_panel["available"]:
        for band in band_panel["bands"]:
            fig.add_trace(
                go.Scatter(
                    x=band_panel["k_dist"],
                    y=band_panel["initial"][:, band],
                    mode="lines",
                    line={"width": 1.4},
                    name=f"band {band}",
                    hovertemplate="k: %{x:.5g}<br>E: %{y:.5g}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            band_trace_indices.append(len(fig.data) - 1)
        for node_index in band_panel["node_indices"]:
            fig.add_vline(
                x=float(band_panel["k_dist"][node_index]),
                line_width=1,
                line_dash="dash",
                line_color="rgba(0,0,0,0.32)",
                row=1,
                col=1,
            )
    else:
        fig.add_annotation(
            text="Symmetry-line eigenvalues unavailable",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="x domain",
            yref="y domain",
            row=1,
            col=1,
        )

    map_trace_indices = []
    colorbar_x = [0.305, 0.655, 1.005]
    if is_polar:
        colorbar_y = 0.645
        colorbar_length = 0.29
    else:
        colorbar_y = 0.322
        colorbar_length = 0.59

    for component_index, (column, component_name, component_title) in enumerate(
        zip(map_columns, component_names, component_titles)
    ):
        zmin, zmax = component_limits[component_index]
        initial_map = map_data[(component_index,) + init_idx]
        fig.add_trace(
            go.Heatmap(
                x=map_x_values,
                y=map_y_values,
                z=initial_map,
                zmin=zmin,
                zmax=zmax,
                colorscale=cmap,
                showscale=True,
                colorbar={
                    "title": component_title,
                    "x": colorbar_x[component_index],
                    "y": colorbar_y,
                    "len": colorbar_length,
                    "thickness": 12,
                    "outlinewidth": 0,
                },
                name=component_name,
                showlegend=False,
                hovertemplate=(
                    f"k<sub>{plane_axes[0]}</sub>: %{{x:.6g}}<br>"
                    f"k<sub>{plane_axes[1]}</sub>: %{{y:.6g}}<br>"
                    f"{component_title}: %{{z:.6g}}<extra></extra>"
                ),
            ),
            row=map_row,
            col=column,
        )
        map_trace_indices.append(len(fig.data) - 1)

        fig.update_xaxes(
            title_text=f"k<sub>{plane_axes[0]}</sub>",
            row=map_row,
            col=column,
        )
        fig.update_yaxes(
            title_text=f"k<sub>{plane_axes[1]}</sub>",
            scaleanchor=f"x{component_index + 2}",
            scaleratio=1,
            row=map_row,
            col=column,
        )
        if is_polar:
            radius_max = float(radius_values[-1])
            fig.update_xaxes(range=[-radius_max, radius_max], row=map_row, col=column)
            fig.update_yaxes(range=[-radius_max, radius_max], row=map_row, col=column)

    overlay_trace_indices = []
    if not is_polar and order == "xyz":
        b1, b2 = get_bvecs_from_meta(meta)
        bz_x, bz_y = bz_wigner_seitz_from_bvecs(b1, b2)
        has_path = "path_points" in data.files
        path_points = np.asarray(data["path_points"], dtype=float) if has_path else None
        for map_index, column in enumerate(map_columns):
            fig.add_trace(
                go.Scatter(
                    x=bz_x,
                    y=bz_y,
                    mode="lines",
                    line={"color": "white", "width": 2},
                    name="BZ",
                    legendgroup="bz",
                    showlegend=map_index == 0,
                    hoverinfo="skip",
                ),
                row=map_row,
                col=column,
            )
            overlay_trace_indices.append(len(fig.data) - 1)
            if has_path:
                fig.add_trace(
                    go.Scatter(
                        x=path_points[:, 0],
                        y=path_points[:, 1],
                        mode="lines",
                        line={"color": "#ff4545", "width": 1.4, "dash": "dash"},
                        name="Sym Path",
                        legendgroup="sym-path",
                        showlegend=map_index == 0,
                        hoverinfo="skip",
                    ),
                    row=map_row,
                    col=column,
                )
                overlay_trace_indices.append(len(fig.data) - 1)

    circle_trace_indices = []
    line_trace_indices = []
    winding_trace_index = None
    winding_marker_trace_index = None
    magnitude_trace_index = None
    if is_polar:
        initial_radius = float(radius_values[initial_radius_index])
        initial_circle_x = initial_radius * unit_circle_x
        initial_circle_y = initial_radius * unit_circle_y
        for column in map_columns:
            for width, color in ((5.0, "rgba(0,0,0,0.78)"), (2.2, "#ffd84d")):
                fig.add_trace(
                    go.Scatter(
                        x=initial_circle_x,
                        y=initial_circle_y,
                        mode="lines",
                        line={"color": color, "width": width},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=2,
                    col=column,
                )
                circle_trace_indices.append(len(fig.data) - 1)

        def initial_ring(component):
            values = component[init_idx + (slice(None), initial_radius_index)]
            return np.concatenate((values, [values[0]]))

        for component_index, (column, component, component_name, color) in enumerate(
            zip(map_columns, components, component_names, component_colors)
        ):
            fig.add_trace(
                go.Scatter(
                    x=phi_plot,
                    y=initial_ring(component),
                    mode="lines",
                    line={"color": color, "width": 2},
                    name=component_name,
                    showlegend=False,
                    hovertemplate=(
                        "phi: %{x:.6g}<br>"
                        f"{component_name}: %{{y:.6g}}<extra></extra>"
                    ),
                ),
                row=3,
                col=column,
            )
            line_trace_indices.append(len(fig.data) - 1)
            component_limit = component_line_limits[component_index]
            fig.update_xaxes(
                title_text="phi (rad)",
                range=[float(phi_plot[0]), float(phi_plot[-1])],
                row=3,
                col=column,
            )
            fig.update_yaxes(
                title_text=component_name,
                range=[-1.05 * component_limit, 1.05 * component_limit],
                zeroline=True,
                zerolinecolor="rgba(100,100,100,0.45)",
                row=3,
                col=column,
            )

        initial_winding = winding_grid[init_idx + (slice(None),)]
        fig.add_trace(
            go.Scatter(
                x=radius_values,
                y=initial_winding,
                mode="lines+markers",
                line={"color": "#2f3e56", "width": 2},
                marker={"size": 4},
                connectgaps=False,
                showlegend=False,
                hovertemplate="r: %{x:.6g}<br>W: %{y:.6g}<extra></extra>",
            ),
            row=4,
            col=1,
        )
        winding_trace_index = len(fig.data) - 1
        fig.add_trace(
            go.Scatter(
                x=[initial_radius, initial_radius],
                y=list(winding_range),
                mode="lines",
                line={"color": "#d62728", "width": 2, "dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=4,
            col=1,
        )
        winding_marker_trace_index = len(fig.data) - 1

        initial_magnitude = magnitude_grid[
            init_idx + (slice(None), initial_radius_index)
        ]
        initial_magnitude = np.concatenate(
            (initial_magnitude, [initial_magnitude[0]])
        )
        fig.add_trace(
            go.Scatter(
                x=phi_plot,
                y=initial_magnitude,
                mode="lines",
                line={"color": "#7a3db8", "width": 2},
                showlegend=False,
                hovertemplate=(
                    "phi: %{x:.6g}<br>"
                    "In-plane magnitude: %{y:.6g}<extra></extra>"
                ),
            ),
            row=4,
            col=4,
        )
        magnitude_trace_index = len(fig.data) - 1

        fig.update_xaxes(
            title_text="r",
            range=[float(radius_values[0]), float(radius_values[-1])],
            row=4,
            col=1,
        )
        fig.update_yaxes(
            title_text="Winding number W",
            range=list(winding_range),
            zeroline=True,
            row=4,
            col=1,
        )
        fig.update_xaxes(
            title_text="phi (rad)",
            range=[float(phi_plot[0]), float(phi_plot[-1])],
            row=4,
            col=4,
        )
        fig.update_yaxes(
            title_text=(
                f"|Omega_({plane_axes[0]},{plane_axes[1]})|"
            ),
            range=[0.0, magnitude_limit],
            row=4,
            col=4,
        )

    fig.update_xaxes(title_text="k path", row=1, col=1)
    fig.update_yaxes(
        title_text="Eigenvalue",
        range=list(band_panel["range"]),
        row=1,
        col=1,
    )
    if band_panel["available"] and band_panel["node_indices"].size:
        tick_values = [
            float(band_panel["k_dist"][index])
            for index in band_panel["node_indices"]
        ]
        tick_text = [
            band_panel["path_labels"][index]
            if index < len(band_panel["path_labels"])
            else ""
            for index in range(len(tick_values))
        ]
        fig.update_xaxes(
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_text,
            row=1,
            col=1,
        )

    base_title = title or "Berry Curvature Components"
    initial_parameters = ", ".join(
        f"{names[index]}={axes[index][init_idx[index]]:.6g}"
        for index in range(len(names))
    )
    initial_title = f"{base_title}<br>{initial_parameters}"
    if is_polar:
        initial_title += f" | r={radius_values[initial_radius_index]:.6g}"
    fig.update_layout(
        title=initial_title,
        width=1510,
        height=figure_height,
        margin={"l": 70, "r": 120, "t": 105, "b": 80},
        hovermode="closest",
        plot_bgcolor="#e5e8ee",
        paper_bgcolor="#ffffff",
        showlegend=True,
    )

    plot_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id="qgt-plot",
        config={"responsive": True},
    )

    band_index = params_json.get("band_index")
    if band_index is None and isinstance(meta, dict):
        band_index = meta.get("band")
    sidebar = _sidebar_html(
        params_json,
        names,
        axes,
        _fmt(band_index) if band_index is not None else "unknown",
        "available" if band_panel["available"] else "unavailable",
    )

    if output_html is None:
        output_html = os.path.join(root_dir, "dynamic_nd_berry_components_multipara.html")

    axes_json = [axis.astype(float).tolist() for axis in axes]
    js_map_data = _json_script_value(_json_ready_numeric_array(map_data))
    js_component_data = (
        _json_script_value(_json_ready_numeric_array(component_stack))
        if is_polar
        else "null"
    )
    js_winding_data = (
        _json_script_value(_json_ready_numeric_array(winding_grid))
        if is_polar
        else "null"
    )
    js_magnitude_data = (
        _json_script_value(_json_ready_numeric_array(magnitude_grid))
        if is_polar
        else "null"
    )
    js_eigen_data = (
        _json_script_value(_json_ready_numeric_array(band_panel["eigenvalues"]))
        if band_panel["available"]
        else "null"
    )
    js_radius_values = (
        _json_script_value(radius_values.astype(float).tolist())
        if is_polar
        else "null"
    )
    js_radius_indices = (
        _json_script_value(radius_indices.astype(int).tolist())
        if is_polar
        else "[]"
    )
    js_phi_plot = (
        _json_script_value(phi_plot.astype(float).tolist())
        if is_polar
        else "null"
    )
    js_unit_circle_x = (
        _json_script_value(unit_circle_x.astype(float).tolist())
        if is_polar
        else "null"
    )
    js_unit_circle_y = (
        _json_script_value(unit_circle_y.astype(float).tolist())
        if is_polar
        else "null"
    )

    radius_control = ""
    if is_polar:
        radius_control = f"""
<div class="radius-control">
  <button id="radius-play" type="button">Play</button>
  <label for="radius-slider">Radius r</label>
  <input id="radius-slider" type="range" min="0" max="{len(radius_indices) - 1}" step="1" value="0"/>
  <output id="radius-output">{radius_values[initial_radius_index]:.6g}</output>
</div>
"""

    document_title = (title or "Berry Curvature Components").replace("<", "").replace(">", "")
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{document_title} multiparameter sweep</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  font-family: 'Segoe UI', Arial, sans-serif;
  background: #f0f2f7;
  margin: 0;
  padding: 20px;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  gap: 20px;
}}
.plot-wrapper {{ flex: 1 1 auto; min-width: 0; max-width: 1510px; }}
.sidebar {{
  flex: 0 0 300px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  overflow: hidden;
  margin-top: 8px;
}}
.sidebar-header {{
  background: #34405f;
  color: #fff;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.03em;
  padding: 14px 16px;
  word-break: break-word;
}}
.section {{ padding: 10px 16px 4px 16px; }}
.section-title {{
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #7b849c;
  margin-bottom: 6px;
  border-bottom: 1px solid #e8ecf4;
  padding-bottom: 4px;
}}
.row {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 10px;
  padding: 3px 0;
  font-size: 12px;
  border-bottom: 1px solid #f3f5fb;
}}
.row:last-child {{ border-bottom: none; }}
.key {{ color: #4a5568; font-weight: 600; white-space: nowrap; }}
.val {{ color: #2d3748; text-align: right; font-variant-numeric: tabular-nums; word-break: break-all; }}
.control {{ margin-bottom: 12px; }}
.control-head {{
  display: flex;
  justify-content: space-between;
  gap: 10px;
  font-size: 12px;
  margin-bottom: 4px;
}}
.control label {{ color: #4a5568; font-weight: 700; }}
.control output {{ color: #2d3748; font-variant-numeric: tabular-nums; text-align: right; }}
input[type=range] {{ width: 100%; }}
.radius-control {{
  display: grid;
  grid-template-columns: auto auto minmax(240px, 1fr) 90px;
  align-items: center;
  gap: 12px;
  background: #fff;
  border: 1px solid #dfe4ee;
  border-radius: 6px;
  margin: 0 70px 18px 70px;
  padding: 10px 14px;
  font-size: 12px;
}}
.radius-control button {{
  border: 1px solid #aeb8ca;
  border-radius: 4px;
  background: #fff;
  color: #2d3748;
  padding: 5px 12px;
  cursor: pointer;
}}
.radius-control label {{ font-weight: 700; color: #4a5568; }}
.radius-control output {{ font-variant-numeric: tabular-nums; text-align: right; }}
@media (max-width: 1100px) {{
  body {{ flex-direction: column; padding: 10px; }}
  .sidebar {{ width: 100%; flex-basis: auto; }}
  .plot-wrapper {{ width: 100%; overflow-x: auto; }}
}}
</style>
</head>
<body>
<div class="plot-wrapper">{plot_html}{radius_control}</div>
{sidebar}
<script>
const mapData = {js_map_data};
const polarComponentData = {js_component_data};
const windingData = {js_winding_data};
const magnitudeData = {js_magnitude_data};
const eigenData = {js_eigen_data};
const names = {_json_script_value(names)};
const axes = {_json_script_value(axes_json)};
const bands = {_json_script_value(band_panel["bands"])};
const isPolar = {str(is_polar).lower()};
const radiusValues = {js_radius_values};
const radiusIndices = {js_radius_indices};
const phiPlot = {js_phi_plot};
const unitCircleX = {js_unit_circle_x};
const unitCircleY = {js_unit_circle_y};
const mapTraceIndices = {_json_script_value(map_trace_indices)};
const bandTraceIndices = {_json_script_value(band_trace_indices)};
const circleTraceIndices = {_json_script_value(circle_trace_indices)};
const lineTraceIndices = {_json_script_value(line_trace_indices)};
const windingTraceIndex = {_json_script_value(winding_trace_index)};
const windingMarkerTraceIndex = {_json_script_value(winding_marker_trace_index)};
const magnitudeTraceIndex = {_json_script_value(magnitude_trace_index)};
const baseTitle = {_json_script_value(base_title)};

let current = axes.map(axis => Math.floor(axis.length / 2));
let radiusPosition = 0;
let radiusTimer = null;

function nestedGet(array, indices) {{
  let value = array;
  for (const index of indices) value = value[index];
  return value;
}}

function formatValue(value) {{
  if (typeof value !== 'number') return String(value);
  return Number(value).toPrecision(6).replace(/\\.?0+$/, '');
}}

function currentRadiusIndex() {{
  return isPolar ? radiusIndices[radiusPosition] : null;
}}

function titleForCurrentState() {{
  const parameters = names
    .map((name, index) => `${{name}}=${{formatValue(axes[index][current[index]])}}`)
    .join(', ');
  const radiusText = isPolar
    ? ` | r=${{formatValue(radiusValues[currentRadiusIndex()])}}`
    : '';
  return `${{baseTitle}}<br>${{parameters}}${{radiusText}}`;
}}

function closedRing(matrix, radiusIndex) {{
  const values = matrix.map(row => row[radiusIndex]);
  if (values.length) values.push(values[0]);
  return values;
}}

function updateRadiusDependent() {{
  if (!isPolar) {{
    Plotly.relayout('qgt-plot', {{'title.text': titleForCurrentState()}});
    return;
  }}

  const radiusIndex = currentRadiusIndex();
  const radius = radiusValues[radiusIndex];
  const circleX = unitCircleX.map(value => radius * value);
  const circleY = unitCircleY.map(value => radius * value);
  circleTraceIndices.forEach(traceIndex => {{
    Plotly.restyle('qgt-plot', {{x: [circleX], y: [circleY]}}, [traceIndex]);
  }});

  for (let componentIndex = 0; componentIndex < 3; componentIndex += 1) {{
    const component = nestedGet(polarComponentData[componentIndex], current);
    Plotly.restyle(
      'qgt-plot',
      {{y: [closedRing(component, radiusIndex)]}},
      [lineTraceIndices[componentIndex]],
    );
  }}

  const magnitude = nestedGet(magnitudeData, current);
  Plotly.restyle(
    'qgt-plot',
    {{y: [closedRing(magnitude, radiusIndex)]}},
    [magnitudeTraceIndex],
  );
  Plotly.restyle(
    'qgt-plot',
    {{x: [[radius, radius]]}},
    [windingMarkerTraceIndex],
  );
  document.getElementById('radius-output').textContent = formatValue(radius);
  Plotly.relayout('qgt-plot', {{'title.text': titleForCurrentState()}});
}}

function updateParameterPlot() {{
  mapTraceIndices.forEach((traceIndex, componentIndex) => {{
    const map = nestedGet(mapData[componentIndex], current);
    Plotly.restyle('qgt-plot', {{z: [map]}}, [traceIndex]);
  }});

  if (eigenData !== null) {{
    const eigenvalues = nestedGet(eigenData, current);
    bands.forEach((band, position) => {{
      Plotly.restyle(
        'qgt-plot',
        {{y: [eigenvalues.map(row => row[band])]}},
        [bandTraceIndices[position]],
      );
    }});
  }}

  if (isPolar) {{
    Plotly.restyle(
      'qgt-plot',
      {{y: [nestedGet(windingData, current)]}},
      [windingTraceIndex],
    );
  }}
  updateRadiusDependent();
}}

function buildParameterControls() {{
  const root = document.getElementById('controls');
  names.forEach((name, index) => {{
    const wrapper = document.createElement('div');
    wrapper.className = 'control';
    const head = document.createElement('div');
    head.className = 'control-head';
    const label = document.createElement('label');
    label.textContent = name;
    const output = document.createElement('output');
    output.textContent = formatValue(axes[index][current[index]]);
    head.appendChild(label);
    head.appendChild(output);

    const input = document.createElement('input');
    input.type = 'range';
    input.min = 0;
    input.max = axes[index].length - 1;
    input.step = 1;
    input.value = current[index];
    input.addEventListener('input', () => {{
      current[index] = Number(input.value);
      output.textContent = formatValue(axes[index][current[index]]);
      updateParameterPlot();
    }});
    wrapper.appendChild(head);
    wrapper.appendChild(input);
    root.appendChild(wrapper);
  }});
}}

function buildRadiusControl() {{
  if (!isPolar) return;
  const slider = document.getElementById('radius-slider');
  const playButton = document.getElementById('radius-play');
  slider.addEventListener('input', () => {{
    radiusPosition = Number(slider.value);
    updateRadiusDependent();
  }});
  playButton.addEventListener('click', () => {{
    if (radiusTimer !== null) {{
      clearInterval(radiusTimer);
      radiusTimer = null;
      playButton.textContent = 'Play';
      return;
    }}
    playButton.textContent = 'Pause';
    radiusTimer = setInterval(() => {{
      radiusPosition = (radiusPosition + 1) % radiusIndices.length;
      slider.value = radiusPosition;
      updateRadiusDependent();
    }}, 100);
  }});
}}

buildParameterControls();
buildRadiusControl();
</script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as stream:
        stream.write(full_html)
    print(f"Saved interactive Berry-component N-D HTML to: {output_html}")
    return output_html


def dynamic_nd_field_with_bands_html(
    root_dir,
    *,
    quantity="trace",
    bands_to_plot=None,
    convert_berry_from_imQ=True,
    cmap="inferno",
    symmetric_cbar=None,
    title=None,
    show_integral=True,
    output_html=None,
    trace_zmax_percentile=99.0,
    berry_zlim=None,
    berry_zlim_percentile=99.0,
    ratio_zmax=None,
    ratio_zmax_percentile=99.0,
    berry_component_cmap="RdBu_r",
    cartesian_resolution=None,
):
    """
    Write a Plotly/HTML version of the N-D multiparameter QGT viewer.

    This mirrors the Matplotlib ``dynamic_nd_field_with_bands`` workflow:
    a top symmetry-line eigenvalue panel when available, a bottom 2D field
    heatmap, BZ/path overlays when available, and one slider per parameter.

    ``quantity='floquet_max_ratio'`` reduces the stored per-band diagnostic
    over its source-band axis. ``ratio_zmax`` sets an explicit positive color
    limit; otherwise ``ratio_zmax_percentile`` is evaluated globally over all
    finite parameter and momentum points. Exact resonances remain infinite in
    hover data but are saturated at the selected color limit for rendering.

    ``quantity='berry_components'`` renders all three Cartesian Berry-vector
    components. Polar bundles additionally receive radius-ring cuts, winding,
    and in-plane magnitude panels controlled by a shared radius slider.
    """
    q = quantity.lower()
    if q in BERRY_COMPONENT_QUANTITIES:
        return _dynamic_nd_berry_components_html(
            root_dir,
            bands_to_plot=bands_to_plot,
            title=title,
            output_html=output_html,
            berry_zlim=berry_zlim,
            berry_zlim_percentile=berry_zlim_percentile,
            cartesian_resolution=cartesian_resolution,
            cmap=berry_component_cmap,
        )

    (
        root_dir,
        data,
        names,
        axes,
        shape,
        kx,
        ky,
        meta,
        params_json,
    ) = _load_nd_bundle(root_dir)

    order = _bundle_coordinate_order(data, meta)
    is_polar_grid = is_cylindrical_order(order)
    if is_polar_grid:
        if "ki" not in data.files or "kj" not in data.files:
            raise KeyError(
                "Polar N-D plotting requires ki and kj in the saved bundle"
            )
        sampling_ki = np.asarray(data["ki"], dtype=float)
        sampling_kj = np.asarray(data["kj"], dtype=float)
    else:
        sampling_ki = np.asarray(data["ki"], dtype=float) if "ki" in data.files else kx
        sampling_kj = np.asarray(data["kj"], dtype=float) if "kj" in data.files else ky

    is_floquet_ratio = q in FLOQUET_RATIO_QUANTITIES
    if is_floquet_ratio:
        raw_field_grid, ratio_hover_grid = _prepare_floquet_ratio_field(data)
    else:
        raw_field_grid = _pick_field_grid(
            data,
            quantity,
            convert_berry_from_imQ=convert_berry_from_imQ,
            order=order,
        )
        ratio_hover_grid = None

    expected_field_shape = tuple(shape) + tuple(kx.shape)
    if raw_field_grid.shape != expected_field_shape:
        raise ValueError(
            f"Field '{quantity}' must have shape {expected_field_shape}; "
            f"received {raw_field_grid.shape}"
        )

    label_q = title or _label_from_quantity(quantity)
    if is_polar_grid and title is None and q in ("berry", "berry_curvature", "omega"):
        _, _, _, fixed_axis = cylindrical_order_axes(order)
        label_q = f"Berry Curvature Ω<sub>{fixed_axis}</sub>"
    document_title = label_q.replace("<sub>", "_").replace("</sub>", "")

    if is_polar_grid:
        area_element = 0.0
    else:
        dkx = float(data["dkx"]) if "dkx" in data.files else float(sampling_ki[0, 1] - sampling_ki[0, 0])
        dky = float(data["dky"]) if "dky" in data.files else float(sampling_kj[1, 0] - sampling_kj[0, 0])
        area_element = dkx * dky

    if symmetric_cbar is None:
        symmetric_cbar = q != "trace" and not is_floquet_ratio
    elif is_floquet_ratio and symmetric_cbar:
        raise ValueError(
            "Floquet perturbative ratios are nonnegative; "
            "symmetric_cbar must be False or None"
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
                raise ValueError("ratio_zmax_percentile must be between 0 and 100")
            vmax = float(np.percentile(finite, percentile))
        elif finite.size:
            vmax = float(np.max(finite))
        else:
            vmax = 1.0

        # An all-zero diagnostic still needs a non-degenerate color range.
        if vmax <= 0.0:
            positive_finite = finite[finite > 0.0]
            vmax = (
                float(np.max(positive_finite))
                if positive_finite.size
                else 1.0
            )
        vmin = 0.0
    else:
        if not finite.size:
            raise ValueError(f"Field '{quantity}' contains no finite values")
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))

    if q in berry_quantities:
        symmetric_limits = get_symmetric_plot_limits(
            finite,
            berry_zlim,
            berry_zlim_percentile,
        )
        if symmetric_cbar:
            vmin, vmax = symmetric_limits
        else:
            clipped_vmin = max(vmin, symmetric_limits[0])
            clipped_vmax = min(vmax, symmetric_limits[1])
            if clipped_vmin < clipped_vmax:
                vmin, vmax = clipped_vmin, clipped_vmax
    elif q == "trace" and trace_zmax_percentile is not None:
        vmax = float(np.nanpercentile(finite, trace_zmax_percentile))
        if vmax <= vmin:
            vmax = float(np.nanmax(finite))
    if symmetric_cbar and q not in berry_quantities:
        amax = max(abs(vmin), abs(vmax))
        vmin, vmax = -amax, amax

    # Plotly's strict JSON path cannot encode infinities. Exact resonances are
    # saturated at the color limit only in z; their hover data remains "∞".
    field_grid = np.array(raw_field_grid, dtype=float, copy=True)
    field_grid[np.isposinf(field_grid)] = vmax
    field_grid[np.isneginf(field_grid)] = vmin

    has_ratio_hover = is_floquet_ratio and not is_polar_grid
    if is_polar_grid:
        radius_values, phi_values = _polar_grid_values(sampling_ki, sampling_kj)
        if cartesian_resolution is None:
            scalar_cartesian_resolution = max(
                80,
                min(301, radius_values.size),
            )
        else:
            scalar_cartesian_resolution = int(cartesian_resolution)
        display_field_grid = np.empty(
            tuple(shape)
            + (scalar_cartesian_resolution, scalar_cartesian_resolution),
            dtype=float,
        )
        plane_axes = None
        plot_x_values = None
        plot_y_values = None
        for parameter_index in np.ndindex(shape):
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
                resolution=scalar_cartesian_resolution,
            )
            display_field_grid[parameter_index] = cartesian_field
            if plane_axes is None:
                plane_axes = current_plane_axes
                plot_x_values = current_x
                plot_y_values = current_y
    else:
        display_field_grid = field_grid
        plane_axes = (order[0], order[1])
        plot_x_values = np.asarray(sampling_ki[0, :], dtype=float)
        plot_y_values = np.asarray(sampling_kj[:, 0], dtype=float)

    init_idx = tuple(ax.size // 2 for ax in axes)
    z0 = display_field_grid[init_idx + (slice(None), slice(None))]
    ratio_hover0 = (
        ratio_hover_grid[init_idx + (slice(None), slice(None), slice(None))]
        if has_ratio_hover
        else None
    )
    ratio_summary_grid = (
        _floquet_ratio_summary_grid(raw_field_grid)
        if is_floquet_ratio
        else None
    )

    has_band_cut = all(
        key in data.files
        for key in ("eigenvalues_sym_grid", "k_dist", "node_indices", "path_labels")
    )
    if has_band_cut:
        eigenvalues_sym_grid = np.asarray(data["eigenvalues_sym_grid"])
        k_dist = np.asarray(data["k_dist"], dtype=float)
        node_indices = np.asarray(data["node_indices"], dtype=int)
        path_labels = [str(x) for x in np.asarray(data["path_labels"])]
        num_bands = int(eigenvalues_sym_grid.shape[-1])

        if bands_to_plot is None:
            bands = list(range(num_bands))
        elif isinstance(bands_to_plot, int):
            bands = [bands_to_plot]
        else:
            bands = list(bands_to_plot)
        bad = [b for b in bands if not (0 <= b < num_bands)]
        if bad:
            raise IndexError(f"Out-of-range bands {bad}; valid range is [0,{num_bands - 1}]")

        ev0 = eigenvalues_sym_grid[init_idx + (slice(None), slice(None))]
        ev_selected = eigenvalues_sym_grid[..., bands]
        ev_min = float(np.nanmin(ev_selected))
        ev_max = float(np.nanmax(ev_selected))
        ev_pad = 0.05 * (ev_max - ev_min) if ev_max > ev_min else 1.0
    else:
        eigenvalues_sym_grid = None
        k_dist = np.asarray([0.0, 1.0])
        node_indices = np.asarray([], dtype=int)
        path_labels = []
        bands = []
        ev0 = None
        ev_min, ev_max, ev_pad = -1.0, 1.0, 0.0

    has_path_overlay = "path_points" in data.files and "path_labels" in data.files
    path_points = np.asarray(data["path_points"], dtype=float) if has_path_overlay else None

    row_heights = [0.25, 0.75]
    vertical_spacing = 0.10
    heatmap_domain_height = (1.0 - vertical_spacing) * row_heights[1] / sum(row_heights)
    heatmap_domain_center = 0.5 * heatmap_domain_height

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=row_heights,
        vertical_spacing=vertical_spacing,
        subplot_titles=("Symmetry-line eigenvalues", label_q),
    )

    if has_band_cut:
        for b in bands:
            fig.add_trace(
                go.Scatter(
                    x=k_dist.tolist(),
                    y=ev0[:, b].tolist(),
                    mode="lines",
                    line=dict(width=1.4),
                    name=f"band {b}",
                    hovertemplate="k: %{x:.4g}<br>E: %{y:.4g}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        for idx in node_indices:
            fig.add_vline(x=float(k_dist[idx]), line_width=1, line_dash="dash", line_color="rgba(0,0,0,0.35)", row=1, col=1)
    else:
        fig.add_annotation(
            text="Symmetry-line eigenvalues unavailable",
            x=0.5,
            y=0.88,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="rgba(70,70,70,0.75)"),
        )

    heatmap_trace_index = len(fig.data)
    if has_ratio_hover:
        hovertemplate = (
            "k<sub>x</sub>: %{x:.4g}<br>"
            "k<sub>y</sub>: %{y:.4g}<br>"
            "Max ratio: %{customdata[0]}<br>"
            "Source H<sub>0</sub> band: %{customdata[1]}<br>"
            "Coupled H<sub>0</sub> band: %{customdata[2]}<br>"
            "Photon index ℓ: %{customdata[3]}<extra></extra>"
        )
    elif is_floquet_ratio:
        hovertemplate = (
            f"k<sub>{plane_axes[0]}</sub>: %{{x:.4g}}<br>"
            f"k<sub>{plane_axes[1]}</sub>: %{{y:.4g}}<br>"
            "Max ratio: %{z:.6g}<extra></extra>"
        )
    else:
        hovertemplate = (
            f"k<sub>{plane_axes[0]}</sub>: %{{x:.4g}}<br>"
            f"k<sub>{plane_axes[1]}</sub>: %{{y:.4g}}<br>"
            "Value: %{z:.4g}<extra></extra>"
        )
    fig.add_trace(
        go.Heatmap(
            z=z0.tolist(),
            x=plot_x_values.tolist(),
            y=plot_y_values.tolist(),
            customdata=(ratio_hover0.tolist() if has_ratio_hover else None),
            colorscale=cmap,
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(
                title=label_q,
                thickness=18,
                len=heatmap_domain_height,
                y=heatmap_domain_center,
                yanchor="middle",
            ),
            hovertemplate=hovertemplate,
            name=label_q,
        ),
        row=2,
        col=1,
    )

    if not is_polar_grid:
        b1, b2 = get_bvecs_from_meta(meta)
        bx, by = bz_wigner_seitz_from_bvecs(b1, b2)
        fig.add_trace(
            go.Scatter(
                x=bx.tolist(),
                y=by.tolist(),
                mode="lines",
                line=dict(color="white", width=2),
                name="BZ",
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    if has_path_overlay and not is_polar_grid:
        fig.add_trace(
            go.Scatter(
                x=path_points[:, 0].tolist(),
                y=path_points[:, 1].tolist(),
                mode="lines",
                line=dict(color="red", width=1.5, dash="dash"),
                name="Sym Path",
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

        label_indices = node_indices if has_band_cut else np.arange(min(len(path_labels), len(path_points)))
        label_indices = label_indices[label_indices < len(path_points)]
        if len(label_indices) and path_labels:
            text = [path_labels[i] if i < len(path_labels) else "" for i in range(len(label_indices))]
            fig.add_trace(
                go.Scatter(
                    x=path_points[label_indices, 0].tolist(),
                    y=path_points[label_indices, 1].tolist(),
                    mode="markers+text",
                    marker=dict(color="red", size=7),
                    text=text,
                    textposition="top right",
                    name="Sym Points",
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )

    if show_integral and not is_polar_grid and q in ("trace_minus_berry", "trace_minus_omega"):
        integral0 = float(np.nansum(z0) * area_element)
        extra0 = f" | integral={integral0:.6g}"
    else:
        extra0 = ""

    init_label = ", ".join(f"{names[i]}={axes[i][init_idx[i]]:.6g}" for i in range(len(names)))
    if is_floquet_ratio:
        initial_summary = str(ratio_summary_grid[init_idx])
    else:
        initial_summary = f"std={float(np.nanstd(z0)):.3e}{extra0}"
    fig.update_layout(
        title=f"{label_q}<br>{init_label} | {initial_summary}",
        width=900,
        height=900,
        margin=dict(l=70, r=110, t=100, b=80),
        showlegend=True,
    )
    fig.update_xaxes(title_text="k path", row=1, col=1)
    fig.update_yaxes(title_text="Eigenvalue", range=[ev_min - ev_pad, ev_max + ev_pad], row=1, col=1)
    if has_band_cut and len(node_indices):
        tick_vals = [float(k_dist[i]) for i in node_indices]
        tick_text = [path_labels[i] if i < len(path_labels) else "" for i in range(len(tick_vals))]
        fig.update_xaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_text, row=1, col=1)
    fig.update_xaxes(title_text=f"k<sub>{plane_axes[0]}</sub>", row=2, col=1)
    fig.update_yaxes(
        title_text=f"k<sub>{plane_axes[1]}</sub>",
        scaleanchor="x2",
        scaleratio=1,
        row=2,
        col=1,
    )

    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="qgt-plot")

    axes_json = [axis.astype(float).tolist() for axis in axes]
    band_cut_text = "available" if has_band_cut else "unavailable"
    band_index = params_json.get("band_index")
    if band_index is None and isinstance(meta, dict):
        band_index = meta.get("band")
    if is_floquet_ratio:
        band_index_text = "all source bands (maximum)"
    else:
        band_index_text = _fmt(band_index) if band_index is not None else "unknown"
    sidebar = _sidebar_html(params_json, names, axes, band_index_text, band_cut_text)

    if output_html is None:
        output_html = os.path.join(root_dir, f"dynamic_nd_{q}_multipara.html")

    js_band_data = (
        _json_script_value(eigenvalues_sym_grid.tolist()) if has_band_cut else "null"
    )
    js_k_dist = _json_script_value(k_dist.astype(float).tolist())
    js_bands = _json_script_value(bands)
    js_field_data = _json_script_value(_json_ready_numeric_array(display_field_grid))
    js_ratio_hover_data = (
        _json_script_value(ratio_hover_grid.tolist())
        if has_ratio_hover
        else "null"
    )
    js_ratio_summary_data = (
        _json_script_value(ratio_summary_grid.tolist())
        if is_floquet_ratio
        else "null"
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{document_title} multiparameter sweep</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  font-family: 'Segoe UI', Arial, sans-serif;
  background: #f0f2f7;
  margin: 0;
  padding: 24px;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  gap: 22px;
}}
.plot-wrapper {{ flex: 0 0 auto; }}
.sidebar {{
  flex: 0 0 300px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  overflow: hidden;
  margin-top: 8px;
}}
.sidebar-header {{
  background: #34405f;
  color: #fff;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.03em;
  padding: 14px 16px;
  word-break: break-word;
}}
.section {{ padding: 10px 16px 4px 16px; }}
.section-title {{
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #7b849c;
  margin-bottom: 6px;
  border-bottom: 1px solid #e8ecf4;
  padding-bottom: 4px;
}}
.row {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 10px;
  padding: 3px 0;
  font-size: 12px;
  border-bottom: 1px solid #f3f5fb;
}}
.row:last-child {{ border-bottom: none; }}
.key {{ color: #4a5568; font-weight: 600; white-space: nowrap; }}
.val {{ color: #2d3748; text-align: right; font-variant-numeric: tabular-nums; word-break: break-all; }}
.control {{ margin-bottom: 12px; }}
.control-head {{
  display: flex;
  justify-content: space-between;
  gap: 10px;
  font-size: 12px;
  margin-bottom: 4px;
}}
.control label {{ color: #4a5568; font-weight: 700; }}
.control output {{ color: #2d3748; font-variant-numeric: tabular-nums; text-align: right; }}
input[type=range] {{ width: 100%; }}
</style>
</head>
<body>
<div class="plot-wrapper">{plot_html}</div>
{sidebar}
<script>
const fieldData = {js_field_data};
const ratioHoverData = {js_ratio_hover_data};
const ratioSummaryData = {js_ratio_summary_data};
const eigenData = {js_band_data};
const names = {_json_script_value(names)};
const axes = {_json_script_value(axes_json)};
const bands = {js_bands};
const heatmapTraceIndex = {heatmap_trace_index};
const bandTraceStart = 0;
const hasBandCut = {str(has_band_cut).lower()};
const areaElement = {area_element:.17g};
const quantity = {_json_script_value(q)};
const labelQ = {_json_script_value(label_q)};
const isFloquetRatio = {str(is_floquet_ratio).lower()};
const hasRatioHover = {str(has_ratio_hover).lower()};
const showIntegral = {str(show_integral and not is_polar_grid).lower()};

let current = axes.map(axis => Math.floor(axis.length / 2));

function nestedGet(arr, idxs) {{
  let out = arr;
  for (const idx of idxs) out = out[idx];
  return out;
}}

function formatValue(value) {{
  if (typeof value !== 'number') return String(value);
  return Number(value).toPrecision(6).replace(/\\.?0+$/, '');
}}

function calcStd(matrix) {{
  let n = 0, sum = 0, sum2 = 0;
  for (const row of matrix) {{
    for (const value of row) {{
      if (Number.isFinite(value)) {{
        n += 1;
        sum += value;
        sum2 += value * value;
      }}
    }}
  }}
  if (!n) return NaN;
  const mean = sum / n;
  return Math.sqrt(Math.max(0, sum2 / n - mean * mean));
}}

function calcIntegral(matrix) {{
  let sum = 0;
  for (const row of matrix) {{
    for (const value of row) {{
      if (Number.isFinite(value)) sum += value;
    }}
  }}
  return sum * areaElement;
}}

function titleFor(matrix) {{
  const params = names.map((name, i) => `${{name}}=${{formatValue(axes[i][current[i]])}}`).join(', ');
  if (isFloquetRatio) {{
    return `${{labelQ}}<br>${{params}} | ${{nestedGet(ratioSummaryData, current)}}`;
  }}
  let extra = `std=${{calcStd(matrix).toExponential(3)}}`;
  if (showIntegral && (quantity === 'trace_minus_berry' || quantity === 'trace_minus_omega')) {{
    extra += ` | integral=${{formatValue(calcIntegral(matrix))}}`;
  }}
  return `${{labelQ}}<br>${{params}} | ${{extra}}`;
}}

function updatePlot() {{
  const z = nestedGet(fieldData, current);
  const update = {{ z: [z] }};
  if (hasRatioHover) {{
    update.customdata = [nestedGet(ratioHoverData, current)];
  }}
  Plotly.restyle('qgt-plot', update, [heatmapTraceIndex]);

  if (hasBandCut && eigenData !== null) {{
    const ev = nestedGet(eigenData, current);
    bands.forEach((band, j) => {{
      const y = ev.map(row => row[band]);
      Plotly.restyle('qgt-plot', {{ y: [y] }}, [bandTraceStart + j]);
    }});
  }}

  Plotly.relayout('qgt-plot', {{ title: {{ text: titleFor(z) }} }});
}}

function buildControls() {{
  const root = document.getElementById('controls');
  names.forEach((name, i) => {{
    const wrapper = document.createElement('div');
    wrapper.className = 'control';

    const head = document.createElement('div');
    head.className = 'control-head';

    const label = document.createElement('label');
    label.textContent = name;

    const output = document.createElement('output');
    output.id = `control-output-${{i}}`;
    output.textContent = formatValue(axes[i][current[i]]);

    head.appendChild(label);
    head.appendChild(output);

    const input = document.createElement('input');
    input.type = 'range';
    input.min = 0;
    input.max = axes[i].length - 1;
    input.step = 1;
    input.value = current[i];
    input.addEventListener('input', () => {{
      current[i] = Number(input.value);
      output.textContent = formatValue(axes[i][current[i]]);
      updatePlot();
    }});

    wrapper.appendChild(head);
    wrapper.appendChild(input);
    root.appendChild(wrapper);
  }});
}}

buildControls();
</script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Saved interactive multiparameter HTML to: {output_html}")
    return output_html


def plot_floquet_max_ratio_nd_html(root_dir, **kwargs):
    """Plot the maximum Floquet perturbative ratio with diagnostic hover data.

    This is a discoverable convenience entry point; all rendering remains in
    :func:`dynamic_nd_field_with_bands_html`.
    """
    if "quantity" in kwargs:
        raise TypeError(
            "plot_floquet_max_ratio_nd_html fixes quantity='floquet_max_ratio'"
        )
    return dynamic_nd_field_with_bands_html(
        root_dir,
        quantity="floquet_max_ratio",
        **kwargs,
    )


def plot_berry_components_nd_html(root_dir, **kwargs):
    """Plot all Berry components, with polar ring diagnostics when available."""
    if "quantity" in kwargs:
        raise TypeError(
            "plot_berry_components_nd_html fixes quantity='berry_components'"
        )
    return dynamic_nd_field_with_bands_html(
        root_dir,
        quantity="berry_components",
        **kwargs,
    )


if __name__ == "__main__":
    dataset_dir = "results/2D_QGT_ND/gWaveAltermagnetHamiltonian/dataset5"

    dynamic_nd_field_with_bands_html(
        dataset_dir,
        quantity="berry_components",
        bands_to_plot=[0, 1, 2, 3],
        output_html=os.path.join(dataset_dir, "qgt_trace_nd_sweep.html"),
    )

    # plot_floquet_max_ratio_nd_html(
    #     dataset_dir,
    #     bands_to_plot=[0, 1, 2, 3, 4, 5],
    #     output_html=os.path.join(
    #         dataset_dir,
    #         "floquet_max_ratio_nd_sweep.html",
    #     ),
    # )
