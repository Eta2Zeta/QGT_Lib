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
  geometry/             QGT by the Kubo formula at any set of k-points, grids, lines
  symmetry/             point groups, QGT group orbits, Brillouin-zone zones
  kspace/               grids and coordinate orders, indexing, symmetry paths
  workflows/            complete calculations that save a run (eigen_2d, qgt_2d, qgt_nd_sweep, ...)
  plotting/             band, QGT, sweep and 3D plots
  io/                   data folders, run folders, metadata, old results
  topology.py           Berry curvature, flux, Chern numbers (link variables), monopole charges, winding numbers
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

Results live outside the repository, in
`~/Documents/Education/Research/QGT_Results` (backed up by iCloud). Set
`QGT_DATA` to use another folder.

```
QGT_Results/
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

`qgt.hamiltonians.minimal_models.crossings.find_band_crossings(model)` finds
where the bands of a minimal model cross, exactly from its form factors
(planes, lines and linear relations in closed form, other curves from exact
equations). `python scripts/compute/band_crossings.py` saves them with a 3D view;
see [Exact band crossings](qgt/hamiltonians/minimal_models/README.md#exact-band-crossings).

## How the QGT is calculated

`qgt.geometry.kubo.quantum_geometric_tensor(model, k, bands)` evaluates the
Kubo (sum-over-states) formula

    Q_ij = sum_{m != n} <n|d_i H|m> <m|d_j H|n> / (E_n - E_m)^2,

with `g_ij = Re Q_ij` and `Omega_ij = -2 Im Q_ij`, at momenta `k` of any shape
`(..., 3)`: one point, a line, a plane, a volume, a sphere or scattered points.
It is vectorized over the points. Only `H` and its central difference enter, so
no eigenvector phase or order has to be fixed; a driven model uses its Floquet
effective Hamiltonian.

- `bands` is one band, a list of bands treated as one group (finite where the
  bands of the group touch each other), or `None` for every band separately.
- Bands are numbered by energy. Pass `eigenvectors=` from an eigen run to number
  them as that run does, e.g. bands tracked through crossings.
- `QGT_grid_kubo` (planes, in any coordinate order), `QGT_3d_kubo` (volumes) and
  `QGT_line` return the nine component arrays the workflows save; workflows
  and configs call this method `"kubo"`.
- `analytic` (closed forms on a model) and `semi_numerical` (the chiral model's
  pseudo-eigenvectors, which are not eigenstates of `H`) remain separate methods.

`qgt.topology.chern_number_from_links` and `monopole_charge` count Berry flux
with link variables (Fukui, Hatsugai and Suzuki 2005): integers already on
coarse meshes, with no derivative. `compute_chern_number` integrates a saved
curvature grid instead.

## To do

- Polarization: change A sin(wt) to -A sin(wt)
