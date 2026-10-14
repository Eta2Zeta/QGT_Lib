"""Run calculations and plots from TOML config files.

A config file describes one setup::

    [model]                          # optional: the Hamiltonian, built by class name
    class = "MinimalHamSG127_2a2b"
    argument = "Hamiltonian_Obj"     # optional: keyword to pass it as (default: first positional)
    [model.params]                   # constructor arguments
    Jz = 0.2
    [model.attributes]               # optional: attributes set after construction
    polarization = "right"

    [arguments]                      # keyword arguments of the script's function
    mesh_spacing = 150

    [[calls]]                        # optional: call the function once per entry,
    quantity = "trace"               # each entry adding to [arguments]

String values may refer to data folders:
``"@legacy/<path>"`` (the pre-reorganization results), ``"@runs/<path>"``, and
``"@latest/<kind>[/<path>]"`` (the run most recently created or reused for a kind).

Scripts call ``run(function, config_path)``. For a module with several functions,
``run(module, config_path)`` calls the one named by the config's ``function`` key.
The config and its file are recorded in the ``run.json`` of every run the call creates.
"""

import copy
import inspect
import os
import tomllib

from qgt.io import runs
from qgt.io.paths import REPO_ROOT, legacy_dir

CONFIG_DIR = REPO_ROOT / "configs"


def config_path(name_or_path, subdir=""):
    """Resolve a config name (``eigen_2d.toml``) against ``configs/<subdir>``; paths are kept."""
    path = os.fspath(name_or_path)
    if os.path.isabs(path) or os.path.exists(path):
        return os.path.abspath(path)
    return str(CONFIG_DIR / subdir / path)


def load_config(path):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def resolve_value(value):
    """Expand ``@legacy/``, ``@runs/`` and ``@latest/`` references, recursively."""
    if isinstance(value, dict):
        return {key: resolve_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_value(item) for item in value]
    if not isinstance(value, str) or not value.startswith("@"):
        return value
    kind, _, rest = value[1:].partition("/")
    if kind == "legacy":
        return legacy_dir(rest)
    if kind == "runs":
        return runs.runs_dir(rest)
    if kind == "latest":
        run_kind, _, subpath = rest.partition("/")
        run_dir = runs.input_run(run_kind)
        return os.path.join(run_dir, subpath) if subpath else run_dir
    raise ValueError(f"Unknown data reference {value!r}; use @legacy/, @runs/ or @latest/")


def model_class(name):
    """Find a Hamiltonian class by name in qgt.hamiltonians or its minimal models."""
    import qgt.hamiltonians as hamiltonians
    import qgt.hamiltonians.minimal_models as minimal_models

    for package in (hamiltonians, minimal_models):
        candidate = getattr(package, name, None)
        if isinstance(candidate, type):
            return candidate
    raise ValueError(f"No Hamiltonian class named {name!r} in qgt.hamiltonians")


def build_model(spec):
    model = model_class(spec["class"])(**resolve_value(spec.get("params", {})))
    for name, value in resolve_value(spec.get("attributes", {})).items():
        setattr(model, name, value)
    return model


def run(target, path, *, prepare=None):
    """Call ``target`` (a function, or a module plus ``function`` in the config) as the
    config at ``path`` describes; return the result(s).

    ``prepare(model, arguments)`` may adjust the arguments before each call
    (e.g. to derive a value from the model).
    """
    config = load_config(path)
    function = getattr(target, config["function"]) if inspect.ismodule(target) else target
    runs.use_config(config, path)
    try:
        results = []
        for call in config.get("calls", [{}]):
            arguments = resolve_value({**copy.deepcopy(config.get("arguments", {})), **call})
            args = []
            model = None
            if "model" in config:
                model = build_model(config["model"])
                keyword = config["model"].get("argument")
                if keyword:
                    arguments[keyword] = model
                else:
                    args.append(model)
            if prepare is not None:
                arguments = prepare(model, arguments)
            results.append(function(*args, **arguments))
        return results[0] if "calls" not in config else results
    finally:
        runs.use_config(None, None)
