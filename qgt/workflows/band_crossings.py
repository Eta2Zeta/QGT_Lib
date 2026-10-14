"""Exact band crossings of minimal altermagnet models; writes one band_crossings run per model.

Each run folder holds:
- crossings.json: the exact conditions, and each component's fixed momenta, relations,
  remaining equations, samples and largest gap
- summary.txt: the same in one line per component
- crossings.html: interactive 3D view of the first Brillouin zone
"""

import json
import os

import sympy as sp

from qgt.hamiltonians.minimal_models import MINIMAL_MODELS
from qgt.hamiltonians.minimal_models.crossings import PAIRS, crossing_conditions, find_band_crossings, summarize
from qgt.io import runs


def band_crossings(model, *, pairs=PAIRS, resolution=64, tag=None, plot=True):
    """Find the crossings of one model and save them as a band_crossings run; return the run folder."""
    pairs = tuple(pairs)
    crossings = find_band_crossings(model, pairs, resolution=resolution)
    name = type(model).__name__
    meta = {"model": name, "crystal_system": model.crystal_system,
            "hamiltonian_params": model.get_parameters_dict(parameter="2D"),
            "pairs": list(pairs), "resolution": resolution}
    run_dir, _ = runs.new_run("band_crossings", name, None, force_new=True, tag=tag)
    with open(os.path.join(run_dir, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2, default=str)
    conditions = {pair: [sp.sstr(e) for e in equations]
                  for pair, equations in crossing_conditions(model).items() if pair in pairs}
    with open(os.path.join(run_dir, "crossings.json"), "w") as handle:
        json.dump({**meta, "conditions": conditions, "components": [c.to_dict() for c in crossings]},
                  handle, indent=1, default=str)
    text = summarize(model, crossings, pairs)
    with open(os.path.join(run_dir, "summary.txt"), "w") as handle:
        handle.write(text + "\n")
    if plot:
        from qgt.plotting.crossings import plot_crossings
        plot_crossings(model, crossings, os.path.join(run_dir, "crossings.html"))
    print(text)
    print(f"  -> {run_dir}\n")
    return run_dir


def band_crossings_all(*, models="all", params=None, pairs=PAIRS, resolution=64, tag=None, plot=True):
    """band_crossings for several minimal models (by class name, or "all"), each with its
    default parameters updated by params."""
    names = list(MINIMAL_MODELS) if models == "all" else list(models)
    return [band_crossings(MINIMAL_MODELS[name](**(params or {})), pairs=pairs, resolution=resolution,
                           tag=tag, plot=plot) for name in names]
