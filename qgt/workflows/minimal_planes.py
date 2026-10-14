"""Batch the existing band -> numerical QGT pipeline over minimal models.

Default: 22 newly added models, four bands, 150 x 150 points, kz=0 and
kz=2*pi/100. Run with the QGT Python environment, e.g.:

    python scripts/compute/minimal_planes.py --workers 6

To replace the existing calculations and plots in their current folders:

    python scripts/compute/minimal_planes.py --overwrite --workers 6

To keep the previous datasets and update only selected models in the gallery:

    python scripts/compute/minimal_planes.py --models MinimalHamSG11_2a2b2c2d --force-new

Band and QGT plots are generated automatically using the current library
plotting routines. To add plots to existing datasets without recalculation:

    python scripts/compute/minimal_planes.py --plots-only --workers 2

Results are self-contained and accepted by qgt.plotting.qgt_2d_comparison.plot_qgt_from_directory.
The five older classes are included only with --all-models. Model parameters
are their class defaults; the Hamiltonian and full parameters are saved.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

# Avoid nested BLAS thread pools when parallelizing independent models.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from qgt.hamiltonians.minimal_models import MINIMAL_MODELS
from qgt.geometry.pointwise_3d import quantum_geometric_tensor_3d_num_phase_corrected
from qgt.geometry.qgt_2d import QGT_grid_num
from qgt.io.metadata import dump_metadata, pick_or_create_result_dir_simple
from qgt.kspace.grids import create_2d_coordinate_grid_from_ranges
from qgt.spectra.eigenvalues import grid_eigenvalues_eigenfunctions
from qgt.topology import berry_curvature_components_from_qgt
from qgt.io.paths import legacy_dir


ROOT = Path(__file__).resolve().parents[2]  # repository root
DEFAULT_OUTPUT = Path(legacy_dir("MinimalModels_QGT_planes"))
LEGACY_MODELS = frozenset({
    "MinimalHamSG124_2b2d", "MinimalHamSG127_2a2b", "MinimalHamSG127_2c2d",
    "MinimalHamSG192_2b", "MinimalHamSG194_2a",
})
NEW_MODELS = tuple(name for name in MINIMAL_MODELS if name not in LEGACY_MODELS)
PLANES = (("kz_0", 0.0), ("kz_0p01_BZ", 0.01))
QGT_KEYS = (
    "g_xx", "g_yy", "g_zz", "g_xy_real", "g_xy_imag",
    "g_xz_real", "g_xz_imag", "g_yz_real", "g_yz_imag",
)
REQUIRED_FILES = [
    "eigenvalues.npy", "eigenfunctions.npy", "ki.npy", "kj.npy",
    "bz_mask.npy", "band_gap.npy", "qgt_valid_mask.npy", "trace.npy",
    "berry_x.npy", "berry_y.npy", "berry_z.npy",
    "floquet_max_ratio.npy", "floquet_max_ratio_indices.npy",
    "meta.json", "meta_info.pkl", "summary.json",
] + [name + ".npy" for name in QGT_KEYS]


def inside_bz(kx, ky, crystal_system):
    """Hexagonal Wigner-Seitz cell, or the model's reciprocal phase box."""
    if crystal_system in ("hexagonal", "trigonal"):
        return ((np.abs(ky) <= 2 * np.pi / np.sqrt(3) + 1e-12)
                & (np.sqrt(3) * np.abs(kx) + np.abs(ky)
                   <= 4 * np.pi / np.sqrt(3) + 1e-12))
    return (np.abs(kx) <= np.pi + 1e-12) & (np.abs(ky) <= np.pi + 1e-12)


def make_grid(model, mesh_spacing):
    hexagonal = model.crystal_system in ("hexagonal", "trigonal")
    xmax = 4 * np.pi / 3 if hexagonal else np.pi
    ymax = 2 * np.pi / np.sqrt(3) if hexagonal else np.pi
    ki, kj, info = create_2d_coordinate_grid_from_ranges(
        (-xmax, xmax), (-ymax, ymax), mesh_spacing,
        order="xyz", include_endpoints=True,
    )
    return ki, kj, inside_bz(ki, kj, model.crystal_system), info


def band_gaps(eigenvalues):
    """Nearest other-band energy separation; last axis is the band index."""
    gaps = np.abs(eigenvalues[..., :, None] - eigenvalues[..., None, :])
    index = np.arange(eigenvalues.shape[-1])
    gaps[..., index, index] = np.inf
    return np.min(gaps, axis=-1)


def isolated_band_qgt(
    model, kx, ky, kz, delta_k, eigenvalues, eigenfunctions, band_index,
    *, gap_tolerance,
):
    """Mask undefined single-band values; delegate all QGT math to the library."""
    if not inside_bz(kx, ky, model.crystal_system):
        return (np.nan,) * 9
    other = np.delete(eigenvalues, band_index)
    if np.min(np.abs(eigenvalues[band_index] - other)) <= gap_tolerance:
        return (np.nan,) * 9
    return quantum_geometric_tensor_3d_num_phase_corrected(
        model, kx, ky, kz, delta_k, eigenvalues, eigenfunctions, band_index,
    )


def source_hashes(model_name):
    """Make cached datasets sensitive to changes in models and numerical code."""
    files = [
        Path(__file__),
        ROOT / f"qgt/hamiltonians/minimal_models/{model_name}.py",
    ]
    files += [ROOT / "qgt" / filename for filename in (
        "hamiltonians/minimal_models/MinimalAltermagnetHamiltonian.py",
        "hamiltonians/base.py",
        "hamiltonians/evaluate.py", "spectra/eigenvalues.py", "spectra/diagonalization.py",
        "spectra/eigenvectors.py", "geometry/derivatives.py", "geometry/pointwise_3d.py",
        "geometry/qgt_2d.py", "kspace/grids.py", "topology.py",
    )]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def run_plane(
    model_name, plane_name, bz_fraction, *, output_root=DEFAULT_OUTPUT,
    mesh_spacing=150, delta_k=1e-5, gap_tolerance=1e-6, force_new=False,
    overwrite=False, make_plots=True, params=None, variant=None,
):
    """Save bands first, then the full Cartesian QGT on one fixed-kz slice.

    params: constructor arguments that differ from the model's defaults; variant: a name
    for such a parameter set, kept in the manifest next to the default calculation.
    """
    if mesh_spacing < 2 or not np.isfinite(delta_k) or delta_k <= 0:
        raise ValueError("mesh_spacing >= 2 and finite delta_k > 0 are required")
    if not np.isfinite(gap_tolerance) or gap_tolerance < 1e-6:
        raise ValueError("gap_tolerance must be at least the library's 1e-6 block tolerance")
    if overwrite and force_new:
        raise ValueError("overwrite and force_new cannot be combined")
    model = MINIMAL_MODELS[model_name](**(params or {}))
    kz = float(2 * np.pi * bz_fraction)
    ki, kj, bz_mask, grid_info = make_grid(model, mesh_spacing)
    meta = {
        **grid_info,
        "runner": __name__, "runner_schema": 1,
        "hamiltonian_name": model.name,
        "hamiltonian_params": model.get_parameters_dict(parameter="2D"),
        "space_group": model.space_group,
        "wyckoff_positions": list(model.wyckoff_positions),
        "crystal_system": model.crystal_system,
        "mesh_spacing": int(mesh_spacing), "kk": kz, "plane_name": plane_name,
        "kz_bz_fraction": bz_fraction, "kz_bz_full_width": float(2 * np.pi),
        "method_name": "numerical_phase_corrected", "delta_k": delta_k,
        "gap_tolerance": gap_tolerance, "band_index": "ALL",
        "band_indices": list(range(model.dim)), "n_bands": model.dim,
        "array_axes_bands": ["ky", "kx", "band"],
        "array_axes_qgt": ["band", "ky", "kx"],
        "mask_policy": "QGT is NaN outside BZ or when nearest band gap <= gap_tolerance",
        "berry_convention": "Omega = (-2 Im Qyz, +2 Im Qxz, -2 Im Qxy)",
        "bz_domain": ("hexagonal Wigner-Seitz cell" if model.crystal_system in
                      ("hexagonal", "trigonal") else "[-pi, pi]^2 reciprocal phase box"),
        "source_sha256": source_hashes(model_name),
    }
    if variant:
        meta["variant"] = variant
    base_root = Path(output_root).resolve() / model_name / plane_name
    if overwrite:
        existing = [path for path in base_root.glob("dataset_*") if path.is_dir()]
        if len(existing) > 1:
            raise ValueError(f"Cannot choose which dataset to overwrite in {base_root}")
        result_dir = existing[0] if existing else base_root / "dataset_1"
        result_dir.mkdir(parents=True, exist_ok=True)
        reused = False
    else:
        result_dir, reused = pick_or_create_result_dir_simple(
            base_root=str(base_root), required_params=meta,
            required_files=REQUIRED_FILES, force_new=force_new,
        )
        result_dir = Path(result_dir)
    if reused:
        summary = json.loads((result_dir / "summary.json").read_text())
        print(f"Reusing {model_name} {plane_name}: {result_dir}", flush=True)
        if make_plots:
            from qgt.plotting.minimal_planes import plot_dataset
            plot_dataset(result_dir)
        return summary

    started = time.monotonic()
    print(f"Starting {model_name} {plane_name}: bands on {mesh_spacing} x {mesh_spacing}", flush=True)
    # An interrupted replacement must not look like a completed calculation.
    (result_dir / "summary.json").unlink(missing_ok=True)
    (result_dir / "plots.json").unlink(missing_ok=True)
    dump_metadata(meta, str(result_dir / "meta.json"))
    with (result_dir / "meta_info.pkl").open("wb") as stream:
        pickle.dump({**meta, "Hamiltonian_Obj": model, "ki": ki, "kj": kj}, stream)
    eigenvalues, eigenfunctions, _, _, ratios, ratio_indices = grid_eigenvalues_eigenfunctions(
        model, ki, kj, mesh_spacing, dim=model.dim, kk=kz, order="xyz",
        show_progress=False, store_hamiltonians=False,
    )
    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenfunctions).all():
        raise RuntimeError(f"Nonfinite eigenpairs in {model_name} {plane_name}")
    analytic_error = float(np.max(np.abs(
        eigenvalues - model.get_analytical_eigenvalues(ki, kj, kz)
    )))
    if analytic_error > 1e-10:
        raise RuntimeError(f"Numerical bands disagree with analytic spectrum: {analytic_error}")
    gaps = band_gaps(eigenvalues)
    valid = np.moveaxis((gaps > gap_tolerance) & bz_mask[..., None], -1, 0)
    for name, value in {
        "eigenvalues": eigenvalues, "eigenfunctions": eigenfunctions,
        "ki": ki, "kj": kj, "bz_mask": bz_mask,
        "band_gap": np.moveaxis(gaps, -1, 0), "qgt_valid_mask": valid,
        "floquet_max_ratio": ratios, "floquet_max_ratio_indices": ratio_indices,
    }.items():
        np.save(result_dir / (name + ".npy"), value)

    components = {name: [] for name in QGT_KEYS}
    callback = partial(isolated_band_qgt, gap_tolerance=gap_tolerance)
    for band in range(model.dim):
        print(f"  {model_name} {plane_name}: QGT band {band + 1}/{model.dim}", flush=True)
        # A fully degenerate plane has no isolated-band QGT to evaluate.
        arrays = (QGT_grid_num(
            ki, kj, eigenvalues, eigenfunctions, callback, model,
            delta_k=delta_k, band_index=band, kk=kz, order="xyz", show_progress=False,
        ) if valid[band].any() else tuple(np.full(ki.shape, np.nan) for _ in QGT_KEYS))
        for name, array in zip(QGT_KEYS, arrays):
            if not np.isfinite(array[valid[band]]).all():
                raise RuntimeError(f"Nonfinite {name} on isolated band {band} of {model_name}")
            components[name].append(array)
    components = {name: np.stack(arrays) for name, arrays in components.items()}
    components["trace"] = components["g_xx"] + components["g_yy"] + components["g_zz"]
    for axis, value in zip("xyz", berry_curvature_components_from_qgt(
        components["g_xy_imag"], components["g_xz_imag"], components["g_yz_imag"],
    )):
        components["berry_" + axis] = value
    for name, value in components.items():
        np.save(result_dir / (name + ".npy"), value)
    bz_count = int(bz_mask.sum())
    summary = {
        "model": model_name, "plane": plane_name, "kz": kz,
        "dataset_dir": str(result_dir), "mesh_spacing": mesh_spacing,
        "bz_grid_points": bz_count,
        "valid_qgt_points_per_band": valid.sum(axis=(1, 2)).tolist(),
        "degenerate_points_per_band": (bz_count - valid.sum(axis=(1, 2))).tolist(),
        "status": "complete" if valid.any() else "degenerate_plane_no_isolated_band_qgt",
        "analytic_eigenvalue_max_error": analytic_error,
        "elapsed_seconds": time.monotonic() - started,
    }
    if variant:
        summary["variant"] = variant
    # Written last: only fully written datasets can be reused.
    dump_metadata(summary, str(result_dir / "summary.json"))
    if make_plots:
        from qgt.plotting.minimal_planes import plot_dataset
        plot_dataset(result_dir)
    print(f"Finished {model_name} {plane_name} in {summary['elapsed_seconds']:.1f}s", flush=True)
    return summary


def run_model(model_name, **settings):
    return [run_plane(model_name, plane, fraction, **settings) for plane, fraction in PLANES]


def save_manifest(path, settings, results, failures):
    manifest = {"settings": settings, "expected_calculations": 2 * len(settings["models"]),
                "completed_calculations": len(results), "results": results, "failures": failures}
    temporary = path.with_suffix(".json.tmp")
    dump_metadata(manifest, str(temporary))
    temporary.replace(path)
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "plane", "kz", "status", "bz_grid_points",
                         "valid_band_0", "valid_band_1", "valid_band_2", "valid_band_3", "dataset_dir"])
        for result in results:
            writer.writerow([result["model"], result["plane"], result["kz"], result["status"],
                             result["bz_grid_points"], *result["valid_qgt_points_per_band"], result["dataset_dir"]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-spacing", type=int, default=150)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--delta-k", type=float, default=1e-5)
    parser.add_argument("--gap-tolerance", type=float, default=1e-6)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument("--force-new", action="store_true")
    storage.add_argument("--overwrite", action="store_true",
                         help="Recalculate in the existing dataset folders and replace their plots")
    plotting = parser.add_mutually_exclusive_group()
    plotting.add_argument("--plots-only", action="store_true",
                          help="Plot datasets in the existing manifest without recalculating bands or QGT")
    plotting.add_argument("--no-plots", action="store_true",
                          help="Explicitly skip the automatic band and QGT plots")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all-models", action="store_true")
    group.add_argument("--models", nargs="+", choices=list(MINIMAL_MODELS))
    parser.add_argument("--params", type=json.loads, default=None,
                        help='JSON dict of model parameters, e.g. \'{"t3_add": 0}\' (needs --models and --variant)')
    parser.add_argument("--variant", default=None,
                        help="name of this parameter set; kept in the manifest next to the default run")
    args = parser.parse_args(argv)
    if (args.params is not None or args.variant) and not (args.models and args.variant):
        parser.error("--params needs --models and --variant")
    if args.workers < 1 or args.mesh_spacing < 2:
        parser.error("workers must be >= 1 and mesh-spacing must be >= 2")
    if not np.isfinite(args.delta_k) or args.delta_k <= 0:
        parser.error("delta-k must be finite and positive")
    if not np.isfinite(args.gap_tolerance) or args.gap_tolerance < 1e-6:
        parser.error("gap-tolerance must be finite and >= 1e-6")
    if args.overwrite and args.plots_only:
        parser.error("overwrite recalculates bands and QGT; use --force-new with --plots-only to refresh just plots")
    models = list(dict.fromkeys(args.models or (MINIMAL_MODELS if args.all_models else NEW_MODELS)))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        from qgt.plotting.minimal_planes import plot_batch
        report = plot_batch(output_root, workers=args.workers, force=args.force_new,
                            model_names=args.models)
        return 1 if report["failures"] else 0
    settings = {"output_root": str(output_root), "mesh_spacing": args.mesh_spacing,
                "delta_k": args.delta_k, "gap_tolerance": args.gap_tolerance,
                "force_new": args.force_new, "overwrite": args.overwrite,
                "make_plots": not args.no_plots}
    if args.variant:
        settings.update(params=args.params or {}, variant=args.variant)
    manifest = output_root / "manifest.json"
    run_settings = {**settings, "models": models, "workers": args.workers}
    results, failures = [], []
    catalog_models = models
    if args.models and manifest.exists():
        previous = json.loads(manifest.read_text())
        results = [result for result in previous["results"] if result["status"] in
                   ("complete", "degenerate_plane_no_isolated_band_qgt")]
        failures = [failure for failure in previous.get("failures", [])
                    if failure["model"] not in models]
        catalog_models = list(dict.fromkeys(
            previous["settings"]["models"] + [result["model"] for result in results] + models
        ))
        # These settings still describe the earlier, unselected calculations.
        # Each dataset also records its actual parameters in its own meta.json.
        run_settings = {**previous["settings"], "models": catalog_models,
                        "latest_subset_run": run_settings}
    results.sort(key=lambda result: (catalog_models.index(result["model"]), result["kz"]))
    save_manifest(manifest, run_settings, results, failures)
    print(f"Running {len(models)} models x 2 planes, {args.workers} workers. Output: {output_root}", flush=True)
    refreshed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_model, name, **settings): name for name in models}
        for future in as_completed(futures):
            name = futures[future]
            try:
                completed = future.result()
                # Publish the new selection only after both planes have completed.
                # An interrupted or failed subset run keeps its prior datasets.
                results = [result for result in results
                           if (result["model"], result.get("variant")) != (name, args.variant)]
                results.extend(completed)
                refreshed += len(completed)
                print(f"Completed {refreshed}/{2 * len(models)} requested calculations; "
                      f"catalog has {len(results)}/{2 * len(catalog_models)}", flush=True)
            except Exception as exc:
                failures.append({"model": name, "error": repr(exc)})
                print(f"FAILED {name}: {exc}", flush=True)
            results.sort(key=lambda result: (catalog_models.index(result["model"]), result["kz"]))
            save_manifest(manifest, run_settings, results, failures)
    print(f"Manifest: {manifest}", flush=True)
    if not args.no_plots:
        from qgt.plotting.minimal_planes import write_gallery
        print(f"Plots: {write_gallery(output_root, results)}", flush=True)
    return 1 if failures else 0
