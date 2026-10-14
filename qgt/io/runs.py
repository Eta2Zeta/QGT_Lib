"""Run folders for new calculations.

Each calculation writes into ``Results/runs/<kind>/<model>/<YYYYMMDD-HHMMSS>[_tag]/``.
``new_run`` creates the folder, or reuses the newest earlier run whose ``meta.json``
matches the requested parameters (same rule as before), and writes ``run.json``
with the kind, model, creation time, git commit and the script that made it. The
calculation itself still writes ``meta.json``, ``meta_info.pkl`` and its arrays.

``latest_run(kind)`` returns the run most recently created or reused for a kind;
this replaces the old ``temp/`` copies, e.g. a QGT calculation reads the latest
eigenvalue run. ``write_index()`` (``scripts/index_runs.py``) lists all runs in
``runs/index.csv``.
"""

import csv
import datetime
import json
import os
import re
import subprocess
import sys

from qgt.io.metadata import meta_matcher_all_fields
from qgt.io.paths import REPO_ROOT, data_root

RUN_INFO = "run.json"
_LATEST = "latest.txt"
_CONFIG = {"config": None, "file": None}


def runs_dir(*parts):
    """Return a path inside ``Results/runs``."""
    return os.path.join(data_root(), "runs", *parts)


def use_config(config, path):
    """Record ``config`` (loaded from ``path``) in every run created until it is cleared with None."""
    _CONFIG["config"] = config
    _CONFIG["file"] = os.path.abspath(path) if path else None


def _safe(name):
    return re.sub(r"[^\w.\-]", "_", str(name))


def _git_state():
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD") or None
        dirty = bool(git("status", "--porcelain", "--untracked-files=no")) if commit else None
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _read_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _matches(run_dir, required_params, required_files):
    existing = _read_json(os.path.join(run_dir, "meta.json"))
    if existing is None:
        return False
    for key, value in required_params.items():
        if key not in existing or not meta_matcher_all_fields(existing[key], value):
            return False
    return all(os.path.exists(os.path.join(run_dir, name)) for name in (required_files or ()))


def _run_dirs(kind, model=None):
    """Run folders of ``kind`` (optionally one model), newest first."""
    base = runs_dir(kind)
    if not os.path.isdir(base):
        return []
    models = [_safe(model)] if model is not None else sorted(os.listdir(base))
    found = []
    for name in models:
        model_dir = os.path.join(base, name)
        if os.path.isdir(model_dir):
            found += [os.path.join(model_dir, run) for run in os.listdir(model_dir)
                      if os.path.isfile(os.path.join(model_dir, run, RUN_INFO))]
    return sorted(found, key=os.path.basename, reverse=True)


def _mark_latest(kind, run_dir):
    os.makedirs(runs_dir(kind), exist_ok=True)
    with open(runs_dir(kind, _LATEST), "w") as handle:
        handle.write(os.path.relpath(run_dir, runs_dir()) + "\n")


def new_run(kind, model, required_params=None, *, required_files=None, force_new=False, tag=None):
    """Create or reuse a run folder; return ``(run_dir, reused)``.

    Unless ``force_new``, the newest run of this kind and model whose ``meta.json``
    contains every key of ``required_params`` with a matching value, and which has
    all ``required_files``, is reused.
    """
    if not force_new and required_params is not None:
        for run_dir in _run_dirs(kind, model):
            if _matches(run_dir, required_params, required_files):
                _mark_latest(kind, run_dir)
                return run_dir, True

    now = datetime.datetime.now()
    name = now.strftime("%Y%m%d-%H%M%S") + (f"_{_safe(tag)}" if tag else "")
    base = runs_dir(kind, _safe(model))
    run_dir, suffix = os.path.join(base, name), 2
    while os.path.exists(run_dir):
        run_dir, suffix = os.path.join(base, f"{name}-{suffix}"), suffix + 1
    os.makedirs(run_dir)
    commit, dirty = _git_state()
    info = {
        "kind": kind, "model": str(model), "tag": tag,
        "created": now.isoformat(timespec="seconds"),
        "git_commit": commit, "git_dirty": dirty,
        "runner": os.path.basename(sys.argv[0]) or None,
        "inputs": {}, "schema": 1,
    }
    if _CONFIG["config"] is not None:
        info["config_file"] = _CONFIG["file"]
        info["config"] = _CONFIG["config"]
    with open(os.path.join(run_dir, RUN_INFO), "w") as handle:
        json.dump(info, handle, indent=2)
    _mark_latest(kind, run_dir)
    return run_dir, False


def record_inputs(run_dir, **inputs):
    """Note in ``run.json`` which runs this one was computed from (e.g. ``eigen_run=...``)."""
    path = os.path.join(run_dir, RUN_INFO)
    info = _read_json(path)
    if info is None:
        return
    for name, value in inputs.items():
        info.setdefault("inputs", {})[name] = os.path.relpath(value, runs_dir()) if value else None
    with open(path, "w") as handle:
        json.dump(info, handle, indent=2)


def latest_run(kind):
    """The run most recently created or reused for ``kind``, or None."""
    try:
        with open(runs_dir(kind, _LATEST)) as handle:
            path = os.path.join(runs_dir(), handle.read().strip())
        if os.path.isdir(path):
            return path
    except OSError:
        pass
    found = _run_dirs(kind)
    return found[0] if found else None


def input_run(kind, run=None):
    """The run to read input from: ``run`` if given, else ``latest_run(kind)``."""
    path = run or latest_run(kind)
    if path is None or not os.path.isdir(path):
        raise FileNotFoundError(
            f"No {kind} run found under {runs_dir(kind)}; run that calculation first "
            f"or pass the run folder explicitly."
        )
    return path


def find_runs(kind, model=None, **params):
    """Run folders of ``kind`` (newest first) whose ``meta.json`` matches ``params``."""
    return [run for run in _run_dirs(kind, model) if not params or _matches(run, params, None)]


def write_index():
    """Write ``runs/index.csv``: one row per run, newest first. Returns its path."""
    rows = []
    root = runs_dir()
    for kind in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        for run_dir in _run_dirs(kind):
            info = _read_json(os.path.join(run_dir, RUN_INFO)) or {}
            meta = _read_json(os.path.join(run_dir, "meta.json"))
            params = (meta or {}).get("hamiltonian_params", (meta or {}).get("parameters", {}))
            size = sum(os.path.getsize(os.path.join(folder, name))
                       for folder, _, names in os.walk(run_dir) for name in names)
            rows.append({
                "created": info.get("created"), "kind": kind, "model": info.get("model"),
                "tag": info.get("tag") or "", "complete": meta is not None,
                "size_mb": round(size / 2**20, 1),
                "git_commit": (info.get("git_commit") or "")[:8], "git_dirty": info.get("git_dirty"),
                "runner": info.get("runner"),
                "inputs": json.dumps(info.get("inputs") or {}, sort_keys=True),
                "hamiltonian_params": json.dumps(params, sort_keys=True, default=str),
                "path": os.path.relpath(run_dir, root),
            })
    rows.sort(key=lambda row: row["created"] or "", reverse=True)
    path = os.path.join(root, "index.csv")
    os.makedirs(root, exist_ok=True)
    fields = ["created", "kind", "model", "tag", "complete", "size_mb", "git_commit", "git_dirty",
              "runner", "inputs", "hamiltonian_params", "path"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path
