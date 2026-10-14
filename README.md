# QGT_Lib

Calculations of the quantum geometric tensor (QGT), Berry curvature and band
structure for tight-binding and Floquet-driven Hamiltonians.

## Setup

```bash
pip install -e ".[dev]"          # the qgt package, its dependencies and pytest
pip install -e ".[dev,plot3d]"   # also trimesh, for scripts/plot/qgt_3d.py
pytest                           # runs tests/ only
```

The scripts also run without installing: each adds the repository to `sys.path`.

## Layout

```
qgt/                    the library
  hamiltonians/         base class, one file per model, minimal_models/ (27 altermagnet families)
  spectra/              eigenvalue solvers, eigenvector alignment
  geometry/             QGT on grids and at single k-points, wavefunction derivatives
  symmetry/             point groups, QGT group orbits, Brillouin-zone zones
  kspace/               grids and coordinate orders, indexing, symmetry paths
  workflows/            complete calculations that save a run (eigen_2d, qgt_2d, qgt_nd_sweep, ...)
  plotting/             band, QGT, sweep and 3D plots
  io/                   data folders, run folders, metadata, old results
  topology.py           Berry curvature, flux, Chern and winding numbers
  config.py             runs a workflow or plot from a TOML config
scripts/compute/        one entry point per calculation
scripts/plot/           one entry point per plot
scripts/figures/        one-off figure scripts for specific comparisons
scripts/index_runs.py   writes Results/runs/index.csv
configs/compute/, configs/plot/   the settings each entry point reads
tests/
docs/
Library/                compatibility stub for results saved before the rename to qgt
```

## Running a calculation

Each entry point reads the config with its own name, or a config file given as
the first argument:

```bash
python scripts/compute/eigen_2d.py                       # configs/compute/eigen_2d.toml
python scripts/compute/qgt_2d.py                         # reads the latest eigen_2d run
python scripts/plot/qgt_sym_lines.py my_comparison.toml  # any config file
```

To try another setup, copy a config, change it, and pass the copy. A config looks like:

```toml
[model]                       # the Hamiltonian, by class name
class = "MinimalHamSG127_2a2b"
[model.params]                # constructor arguments
Jz = 0.2
t1 = 0.8

[arguments]                   # arguments of the script's function
mesh_spacing = 150
kk = 0.031415926535897934     # TOML has no π: write the number, note the expression
```

Other keys:
- `[model.attributes]` sets attributes after construction.
- `argument = "hamiltonian_template"` under `[model]` passes the model as that keyword instead of first.
- `function = "..."` picks one of several functions in a plotting module.
- `[[calls]]` entries call the function once each, adding to `[arguments]`.

Data folders are written as references:
- `@legacy/<path>`: the results from before the reorganization
- `@runs/<path>`: new run folders
- `@latest/<kind>`: the run most recently created or reused for a kind, e.g. `@latest/eigen_2d`

The two MinimalModels tools take command-line options instead of a config, e.g.
`python scripts/compute/minimal_planes.py --workers 6`; see `--help`.

## Data

Results live outside the repository, in `Results/` next to it
(`~/Documents/Quantum_Geometric_Tensor/Results`, backed up by iCloud). Set
`QGT_DATA` to use another folder.

```
Results/
  legacy/        everything from the old QGT_Lib/results/, unchanged (see its README.md)
  runs/<kind>/<model>/<YYYYMMDD-HHMMSS>[_tag]/
  runs/index.csv one row per run (python scripts/index_runs.py)
```

Every run folder has:
- `meta.json`: the parameters, used to reuse a matching earlier run
- `meta_info.pkl`: the same plus the Hamiltonian object
- `run.json`: when and by which script and git commit it was made, the config, and the input runs
- the arrays and figures

A QGT or spin-density calculation reads the eigenvalue run most recently created
or reused for its kind (`eigen_2d` or `eigen_3d`), unless the config names one
with `eigen_run`.

## Old results

Old result pickles record module paths such as `Library.Hamiltonian_v2` or
`Library.Hamiltonian.MinimalModels.MinimalHamSG194_2a`. `qgt.io.legacy_pickles`
serves those paths with the current classes, so `pickle.load` and
`np.load(allow_pickle=True)` work on every file in `Results/legacy/`. If you
rename or move a module in qgt, add its old path to `MOVED_MODULES` there; the
test `tests/test_legacy_pickles.py` loads every old pickle when `Results/` is present.

## Minimal altermagnet models

All 27 explicit Hamiltonian families from Tables III-IV of *Minimal models for
altermagnetism* are in `qgt/hamiltonians/minimal_models`. See the
[model catalog and examples](qgt/hamiltonians/minimal_models/README.md) for
classes, parameter conventions and source coverage.

## To do

- Berry curvature calculation -> Chern number
- Polarization: change A sin(wt) to -A sin(wt)
