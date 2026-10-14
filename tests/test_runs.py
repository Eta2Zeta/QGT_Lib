import csv
import json
import os
import re

import pytest

from Library import runs


@pytest.fixture
def data(monkeypatch, tmp_path):
    monkeypatch.setenv("QGT_DATA", str(tmp_path))
    return tmp_path


def _write_meta(run_dir, meta, files=()):
    with open(os.path.join(run_dir, "meta.json"), "w") as handle:
        json.dump(meta, handle)
    for name in files:
        open(os.path.join(run_dir, name), "w").close()


def test_new_run_creates_timestamped_folder_with_run_info(data):
    run_dir, reused = runs.new_run("eigen_2d", "Model A", {"mesh": 10}, tag="kz0")
    assert not reused
    rel = os.path.relpath(run_dir, data / "runs")
    assert re.fullmatch(r"eigen_2d/Model_A/\d{8}-\d{6}_kz0", rel)
    info = json.load(open(os.path.join(run_dir, runs.RUN_INFO)))
    assert info["kind"] == "eigen_2d" and info["model"] == "Model A" and info["tag"] == "kz0"
    assert info["inputs"] == {}


def test_matching_complete_run_is_reused(data):
    first, _ = runs.new_run("eigen_2d", "M", {"mesh": 10, "t": 0.3}, required_files=["eigenvalues.npy"])
    _write_meta(first, {"mesh": 10, "t": 0.3, "extra": 1}, ["eigenvalues.npy"])
    again, reused = runs.new_run("eigen_2d", "M", {"mesh": 10, "t": 0.3}, required_files=["eigenvalues.npy"])
    assert reused and again == first
    other, reused = runs.new_run("eigen_2d", "M", {"mesh": 20, "t": 0.3}, required_files=["eigenvalues.npy"])
    assert not reused and other != first


def test_incomplete_or_forced_runs_are_not_reused(data):
    first, _ = runs.new_run("eigen_2d", "M", {"mesh": 10}, required_files=["eigenvalues.npy"])
    _write_meta(first, {"mesh": 10})  # crashed before saving its arrays
    second, reused = runs.new_run("eigen_2d", "M", {"mesh": 10}, required_files=["eigenvalues.npy"])
    assert not reused and second != first
    _write_meta(second, {"mesh": 10}, ["eigenvalues.npy"])
    third, reused = runs.new_run("eigen_2d", "M", {"mesh": 10}, required_files=["eigenvalues.npy"], force_new=True)
    assert not reused and third not in (first, second)


def test_latest_run_follows_the_most_recent_create_or_reuse(data):
    assert runs.latest_run("eigen_2d") is None
    a, _ = runs.new_run("eigen_2d", "A", {"mesh": 1})
    _write_meta(a, {"mesh": 1})
    b, _ = runs.new_run("eigen_2d", "B", {"mesh": 1})
    assert runs.latest_run("eigen_2d") == b
    reused, was_reused = runs.new_run("eigen_2d", "A", {"mesh": 1})
    assert was_reused and runs.latest_run("eigen_2d") == a


def test_input_run_prefers_explicit_run_and_explains_when_missing(data):
    with pytest.raises(FileNotFoundError, match="No eigen_2d run found"):
        runs.input_run("eigen_2d")
    a, _ = runs.new_run("eigen_2d", "A", None)
    b, _ = runs.new_run("eigen_2d", "B", None)
    assert runs.input_run("eigen_2d") == b
    assert runs.input_run("eigen_2d", a) == a


def test_record_inputs_and_find_runs(data):
    eigen, _ = runs.new_run("eigen_2d", "M", None)
    qgt, _ = runs.new_run("qgt_2d", "M", None)
    runs.record_inputs(qgt, eigen_run=eigen)
    info = json.load(open(os.path.join(qgt, runs.RUN_INFO)))
    assert info["inputs"] == {"eigen_run": os.path.relpath(eigen, data / "runs")}
    _write_meta(eigen, {"mesh": 5})
    assert runs.find_runs("eigen_2d", "M", mesh=5) == [eigen]
    assert runs.find_runs("eigen_2d", "M", mesh=6) == []


def test_write_index_lists_runs(data):
    run_dir, _ = runs.new_run("qgt_2d", "M", None, tag="t")
    _write_meta(run_dir, {"hamiltonian_params": {"J": 0.5}})
    path = runs.write_index()
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 1
    assert rows[0]["kind"] == "qgt_2d" and rows[0]["model"] == "M" and rows[0]["complete"] == "True"
    assert json.loads(rows[0]["hamiltonian_params"]) == {"J": 0.5}
