"""End-to-end checks for the minimal-model plane calculation runner."""

import json
from concurrent.futures import Future
from pathlib import Path
import pickle

import numpy as np
import pytest

from qgt.workflows import minimal_planes as runner


def test_new_model_catalog_and_requested_planes():
    legacy = {
        "MinimalHamSG124_2b2d", "MinimalHamSG127_2a2b", "MinimalHamSG127_2c2d",
        "MinimalHamSG192_2b", "MinimalHamSG194_2a",
    }
    assert len(runner.NEW_MODELS) == 22
    assert set(runner.NEW_MODELS) == set(runner.MINIMAL_MODELS) - legacy
    assert len(runner.PLANES) == 2
    np.testing.assert_allclose(
        [2 * np.pi * fraction for _, fraction in runner.PLANES],
        [0, 2 * np.pi / 100],
    )


@pytest.mark.parametrize(
    "model_name,extent,expected_points",
    [
        ("MinimalHamSG136_2a2b", (np.pi, np.pi), 25),
        ("MinimalHamSG163_2b", (4 * np.pi / 3, 2 * np.pi / np.sqrt(3)), 17),
    ],
)
def test_grid_domain_and_hexagonal_boundary(model_name, extent, expected_points):
    model = runner.MINIMAL_MODELS[model_name]()
    kx, ky, mask, info = runner.make_grid(model, 5)
    assert kx.shape == ky.shape == mask.shape == (5, 5)
    np.testing.assert_allclose(kx[0], np.linspace(-extent[0], extent[0], 5))
    np.testing.assert_allclose(ky[:, 0], np.linspace(-extent[1], extent[1], 5))
    assert mask.sum() == expected_points
    assert mask[2, 2]
    assert info["order"] == "xyz"
    np.testing.assert_allclose([info["dki"], info["dkj"]], np.array(extent) / 2)
    if model.crystal_system == "trigonal":
        # Hexagon vertices are included; its bounding rectangle's corners are not.
        assert mask[2, 0] and mask[2, -1]
        assert mask[0, 1] and mask[-1, -2]
        assert not mask[0, 0] and not mask[-1, -1]


def test_real_band_then_geometry_pipeline_and_cache(tmp_path, monkeypatch):
    model_name = "MinimalHamSG136_2a2b"
    plane_name, fraction = runner.PLANES[1]
    target = tmp_path / model_name / plane_name / "dataset_1"
    calls = []
    calculate_bands = runner.grid_eigenvalues_eigenfunctions
    calculate_qgt = runner.quantum_geometric_tensor

    def record_bands(*args, **kwargs):
        calls.append("bands")
        return calculate_bands(*args, **kwargs)

    def record_qgt(*args, **kwargs):
        # One Kubo evaluation of every band, numbered as in the saved band results.
        assert args[2] is None
        np.testing.assert_array_equal(kwargs["eigenvectors"], np.load(target / "eigenfunctions.npy"))
        calls.append("qgt")
        return calculate_qgt(*args, **kwargs)

    monkeypatch.setattr(runner, "grid_eigenvalues_eigenfunctions", record_bands)
    monkeypatch.setattr(runner, "quantum_geometric_tensor", record_qgt)
    summary = runner.run_plane(
        model_name, plane_name, fraction, output_root=tmp_path, mesh_spacing=5, make_plots=False,
    )
    assert calls == ["bands", "qgt"]
    assert Path(summary["dataset_dir"]) == target
    assert summary["status"] == "complete"
    assert summary["analytic_eigenvalue_max_error"] < 1e-10
    assert all((target / name).is_file() for name in runner.REQUIRED_FILES)

    eigenvalues = np.load(target / "eigenvalues.npy")
    eigenvectors = np.load(target / "eigenfunctions.npy")
    assert eigenvalues.shape == (5, 5, 4)
    assert eigenvectors.shape == (5, 5, 4, 4)
    assert np.isfinite(eigenvalues).all()
    gaps = np.load(target / "band_gap.npy")
    valid = np.load(target / "qgt_valid_mask.npy")
    assert gaps.shape == valid.shape == (4, 5, 5)
    assert valid.any() and (~valid).any()
    np.testing.assert_array_equal(valid, gaps > 1e-6)
    for name in runner.QGT_KEYS:
        component = np.load(target / f"{name}.npy")
        assert component.shape == (4, 5, 5)
        assert np.isfinite(component[valid]).all()
        assert np.isnan(component[~valid]).all()

    arrays = {name: np.load(target / f"{name}.npy") for name in runner.QGT_KEYS}
    np.testing.assert_allclose(
        np.load(target / "trace.npy"),
        arrays["g_xx"] + arrays["g_yy"] + arrays["g_zz"], equal_nan=True,
    )
    for axis, expected in zip(
        "xyz", (-2 * arrays["g_yz_imag"], 2 * arrays["g_xz_imag"], -2 * arrays["g_xy_imag"]),
    ):
        np.testing.assert_allclose(
            np.load(target / f"berry_{axis}.npy"), expected, equal_nan=True,
        )

    meta = json.loads((target / "meta.json").read_text())
    assert meta["kk"] == pytest.approx(2 * np.pi / 100)
    assert meta["band_indices"] == [0, 1, 2, 3]
    assert meta["array_axes_qgt"] == ["band", "ky", "kx"]
    with (target / "meta_info.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    assert type(saved["Hamiltonian_Obj"]) is runner.MINIMAL_MODELS[model_name]
    np.testing.assert_array_equal(saved["ki"], np.load(target / "ki.npy"))
    np.testing.assert_array_equal(saved["kj"], np.load(target / "kj.npy"))

    def unexpected_recalculation(*args, **kwargs):
        pytest.fail("Completed matching datasets must be reused without recalculation")

    monkeypatch.setattr(runner, "grid_eigenvalues_eigenfunctions", unexpected_recalculation)
    monkeypatch.setattr(runner, "quantum_geometric_tensor", unexpected_recalculation)
    repeated = runner.run_plane(
        model_name, plane_name, fraction, output_root=tmp_path, mesh_spacing=5, make_plots=False,
    )
    assert repeated == summary
    assert list(target.parent.iterdir()) == [target]


def test_hexagonal_degenerate_plane_and_displaced_plane(tmp_path, monkeypatch):
    # Keep an explicitly restricted model to exercise undefined-band masking.
    # The full default model now enables the terms that lift this degeneracy.
    model_class = runner.MINIMAL_MODELS["MinimalHamSG163_2b"]
    monkeypatch.setitem(
        runner.MINIMAL_MODELS, "MinimalHamSG163_2b",
        lambda: model_class(t3_alt=0, t4_alt=0, lamb2=0, lamb_z_alt=0),
    )
    zero, displaced = runner.run_model(
        "MinimalHamSG163_2b", output_root=tmp_path, mesh_spacing=5, make_plots=False,
    )
    assert zero["status"] == "degenerate_plane_no_isolated_band_qgt"
    assert zero["valid_qgt_points_per_band"] == [0, 0, 0, 0]
    assert zero["degenerate_points_per_band"] == [zero["bz_grid_points"]] * 4
    zero_dir = Path(zero["dataset_dir"])
    assert np.isfinite(np.load(zero_dir / "eigenvalues.npy")).all()
    assert np.isfinite(np.load(zero_dir / "eigenfunctions.npy")).all()
    for name in (*runner.QGT_KEYS, "trace", "berry_x", "berry_y", "berry_z"):
        assert np.isnan(np.load(zero_dir / f"{name}.npy")).all()

    assert displaced["status"] == "complete"
    assert displaced["kz"] == pytest.approx(2 * np.pi / 100)
    displaced_dir = Path(displaced["dataset_dir"])
    inside = np.load(displaced_dir / "bz_mask.npy")
    valid = np.load(displaced_dir / "qgt_valid_mask.npy")
    gaps = np.load(displaced_dir / "band_gap.npy")
    assert inside.any() and (~inside).any()
    assert valid.any()
    np.testing.assert_array_equal(valid, (gaps > 1e-6) & inside[None])
    for name in runner.QGT_KEYS:
        component = np.load(displaced_dir / f"{name}.npy")
        assert np.isnan(component[:, ~inside]).all()
        assert np.isnan(component[~valid]).all()
        assert np.isfinite(component[valid]).all()


def test_automatic_plotting_for_new_and_cached_calculations(tmp_path, monkeypatch):
    from qgt.plotting import minimal_planes as plotting

    plotted = []

    def record_plot(dataset_dir):
        dataset_dir = Path(dataset_dir)
        assert all((dataset_dir / name).is_file() for name in runner.REQUIRED_FILES)
        plotted.append(dataset_dir)

    monkeypatch.setattr(plotting, "plot_dataset", record_plot)
    settings = dict(output_root=tmp_path, mesh_spacing=4)
    first = runner.run_plane("MinimalHamSG136_2a2b", *runner.PLANES[1], **settings)
    repeated = runner.run_plane("MinimalHamSG136_2a2b", *runner.PLANES[1], **settings)
    assert first == repeated
    assert plotted == [Path(first["dataset_dir"])] * 2


def test_overwrite_recalculates_changed_parameters_in_same_folder(tmp_path, monkeypatch):
    model_name = "MinimalHamSG84_2a2b"
    settings = dict(output_root=tmp_path, mesh_spacing=5, make_plots=False)
    first = runner.run_plane(model_name, *runner.PLANES[1], **settings)
    target = Path(first["dataset_dir"])
    old_bands = np.load(target / "eigenvalues.npy")
    old_berry = np.load(target / "berry_z.npy")
    (target / "plots.json").write_text('{"status": "old plots"}')
    model_class = runner.MINIMAL_MODELS[model_name]
    monkeypatch.setitem(runner.MINIMAL_MODELS, model_name,
                        lambda: model_class(t4_alt=0.43, lamb2=0.13))

    replaced = runner.run_plane(model_name, *runner.PLANES[1], overwrite=True, **settings)
    assert replaced["dataset_dir"] == first["dataset_dir"]
    assert list(target.parent.iterdir()) == [target]
    assert not (target / "plots.json").exists()
    assert not np.allclose(old_bands, np.load(target / "eigenvalues.npy"))
    assert not np.allclose(old_berry, np.load(target / "berry_z.npy"), equal_nan=True)
    meta = json.loads((target / "meta.json").read_text())
    assert meta["hamiltonian_params"]["t4_alt"] == 0.43
    assert meta["hamiltonian_params"]["lamb2"] == 0.13
    assert all((target / name).is_file() for name in runner.REQUIRED_FILES)


def test_plots_only_preserves_calculation_manifest(tmp_path, monkeypatch):
    from qgt.plotting import minimal_planes as plotting

    manifest = tmp_path / "manifest.json"
    original = '{"completed_calculations": 44, "results": []}\n'
    manifest.write_text(original)
    calls = []

    def record_batch(output_root, **kwargs):
        calls.append((output_root, kwargs))
        return {"failures": []}

    def forbidden_calculation(*args, **kwargs):
        pytest.fail("plots-only must not recalculate bands or QGT")

    monkeypatch.setattr(plotting, "plot_batch", record_batch)
    monkeypatch.setattr(runner, "run_model", forbidden_calculation)
    assert runner.main(["--plots-only", "--workers", "1", "--output-root", str(tmp_path)]) == 0
    assert calls == [(tmp_path, {"workers": 1, "force": False, "model_names": None})]
    assert manifest.read_text() == original


def test_subset_force_new_preserves_old_sets_and_full_gallery(tmp_path, monkeypatch):
    from qgt.plotting import minimal_planes as plotting

    selected = "MinimalHamSG11_2a2b2c2d"
    model_class = runner.MINIMAL_MODELS[selected]
    monkeypatch.setitem(runner.MINIMAL_MODELS, selected,
                        lambda: model_class(t3_add=0, lamb_z_add=0, lamb_z_xz_add=0))
    old_selected = runner.run_model(selected, output_root=tmp_path,
                                    mesh_spacing=4, make_plots=False)
    monkeypatch.setitem(runner.MINIMAL_MODELS, selected, model_class)
    catalog = list(runner.MINIMAL_MODELS)
    previous_results = []
    original_arrays = {}
    for index, name in enumerate(catalog):
        for plane, fraction in runner.PLANES:
            if name == selected:
                result = next(item for item in old_selected if item["plane"] == plane)
                directory = Path(result["dataset_dir"])
            else:
                # Existing unselected datasets are opaque to a subset rerun.
                directory = tmp_path / name / plane / "dataset_1"
                directory.mkdir(parents=True)
                np.save(directory / "eigenvalues.npy", np.array([index, fraction]))
                result = {**old_selected[0], "model": name, "plane": plane,
                          "kz": 2 * np.pi * fraction, "dataset_dir": str(directory)}
                (directory / "summary.json").write_text(json.dumps(result))
            previous_results.append(result)
            for path in directory.glob("*.npy"):
                original_arrays[path] = path.read_bytes()
    original_settings = {"models": catalog, "mesh_spacing": 150, "delta_k": 1e-5,
                         "workers": 6, "make_plots": True}
    manifest_path = tmp_path / "manifest.json"
    runner.save_manifest(manifest_path, original_settings, previous_results, [])
    galleries = []

    def assert_old_selection():
        current = json.loads(manifest_path.read_text())
        assert current["expected_calculations"] == current["completed_calculations"] == 54
        assert [item for item in current["results"] if item["model"] == selected] == old_selected

    class InlinePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, name, **settings):
            assert name == selected
            assert_old_selection()
            future = Future()
            future.set_result(function(name, **settings))
            return future

    def record_plot(directory):
        assert Path(directory).name == "dataset_2"
        assert_old_selection()

    def record_gallery(output_root, results):
        galleries.append(results.copy())
        return output_root / "index.html"

    monkeypatch.setattr(runner, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(plotting, "plot_dataset", record_plot)
    monkeypatch.setattr(plotting, "write_gallery", record_gallery)
    assert runner.main(["--models", selected, "--force-new", "--mesh-spacing", "4",
                        "--workers", "1", "--output-root", str(tmp_path)]) == 0

    current = json.loads(manifest_path.read_text())
    assert current["expected_calculations"] == current["completed_calculations"] == 54
    assert current["settings"]["models"] == catalog
    assert current["settings"]["mesh_spacing"] == 150
    assert current["settings"]["workers"] == 6
    subset_settings = current["settings"]["latest_subset_run"]
    assert subset_settings["models"] == [selected]
    assert subset_settings["mesh_spacing"] == 4
    assert subset_settings["workers"] == 1
    assert subset_settings["force_new"] is True
    new_selected = [item for item in current["results"] if item["model"] == selected]
    assert len(new_selected) == 2
    for old, new in zip(old_selected, new_selected):
        new_dir = Path(new["dataset_dir"])
        assert new_dir.name == "dataset_2"
        assert sorted(path.name for path in new_dir.parent.iterdir()) == ["dataset_1", "dataset_2"]
        assert not np.allclose(np.load(Path(old["dataset_dir"]) / "eigenvalues.npy"),
                               np.load(new_dir / "eigenvalues.npy"))
        meta = json.loads((new_dir / "meta.json").read_text())
        assert meta["hamiltonian_params"]["t3_add"] != 0
        assert meta["hamiltonian_params"]["lamb_z_add"] != 0
        assert meta["hamiltonian_params"]["lamb_z_xz_add"] != 0
    assert [item for item in current["results"] if item["model"] != selected] == [
        item for item in previous_results if item["model"] != selected
    ]
    assert all(path.read_bytes() == content for path, content in original_arrays.items())
    assert galleries == [current["results"]]


def test_failed_subset_keeps_previous_dataset_selection(tmp_path, monkeypatch):
    selected = "MinimalHamSG11_2a2b2c2d"
    previous = runner.run_model(selected, output_root=tmp_path,
                                mesh_spacing=3, make_plots=False)
    manifest_path = tmp_path / "manifest.json"
    runner.save_manifest(manifest_path, {"models": [selected], "mesh_spacing": 150},
                         previous, [])

    def partially_complete_model(name, **settings):
        runner.run_plane(name, *runner.PLANES[0], **settings)
        raise RuntimeError("second plane failed")

    class InlinePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, name, **settings):
            future = Future()
            try:
                future.set_result(function(name, **settings))
            except Exception as exc:
                future.set_exception(exc)
            return future

    monkeypatch.setattr(runner, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(runner, "run_model", partially_complete_model)
    assert runner.main(["--models", selected, "--force-new", "--mesh-spacing", "3",
                        "--workers", "1", "--no-plots", "--output-root", str(tmp_path)]) == 1
    current = json.loads(manifest_path.read_text())
    assert current["results"] == previous
    assert current["completed_calculations"] == current["expected_calculations"] == 2
    assert current["failures"][0]["model"] == selected
    assert "second plane failed" in current["failures"][0]["error"]
    assert (tmp_path / selected / runner.PLANES[0][0] / "dataset_2").is_dir()
