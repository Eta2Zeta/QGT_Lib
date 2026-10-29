# QGT_Lib: organization proposal

*Written 2026-10-13 and updated the same day with your decisions. This is a plan only: nothing has been moved, renamed or deleted.*

## Decisions

| Topic | Decision |
|---|---|
| Old results | **Keep all of them** and leave every format as it is. They move to `~/Documents/Quantum_Geometric_Tensor/Results/`, where iCloud backs them up. The old plot scripts must still be able to replot them. |
| Legacy code | Delete `Legacy/`. It stays recoverable from the `pre-reorg` git tag. |
| Duplicate functions | If two copies do the same thing, keep one copy in a sensible place and import it everywhere. If they do different things, keep both, but give them names that say how they differ. |
| New runs | A consistent format, used only for runs made after the change. |
| Package | Rename `Library` → `qgt` and make it installable (my recommendation, which you accepted). |
| Run folder names | `<YYYYMMDD-HHMMSS>[_tag]` plus an index (my recommendation, which you accepted). |
| Git history | Rewrite it to drop the 2.5 GB of old result files (my recommendation, which you accepted). |
| Run configuration | Config files, one per setup, in `configs/` (§6). |
| `tmp/` | Deleted (done). |

## Summary

QGT_Lib takes 79 GB:
- 29 entry-point scripts at the root
- about 45 library modules, plus 27 minimal-model files
- 25 test files
- 77 GB of results

The plan, in order:

1. **Separate code from data.** Move `results/` unchanged to `../Results/legacy/`. New runs go to `../Results/runs/`, and the repo shrinks to a few MB.
2. **Keep old results replottable.**
   - One path helper replaces every hard-coded `"results/..."` string.
   - One loader opens old pickled metadata whose module paths have since moved. It replaces the `sys.modules` patches now copied into four scripts.
3. **Give new runs one shape:** JSON metadata, timestamped folders, an index, and no hidden handoff through `temp/`.
4. **Turn `Library/` into the `qgt` package,** organized into subpackages by topic, with the duplicates resolved under your rule.
5. **Make the scripts thin entry points** under `scripts/`, each reading a config file.
6. **Clean up git:** a fixed `.gitignore`, untracked results, and a rewritten history.

The order of work is in §9.

---

## 1. Current state (for reference)

| Path | Size | Contents | Issue |
|---|---|---|---|
| root `*.py` | — | 14 `Calc_*`/`Fit_*` and 15 `Plot_*`/`plot_*` scripts | Mixed naming; 25 of 29 are configured by editing the file. Tests import 4 of them. |
| `Library/` | 3 MB | 30 flat modules, plus `Hamiltonian/` (12 model modules and `MinimalModels/`), `GroupTheory/` and `Geometry/` | §4 |
| `tests/` | 1 MB | 25 files | §7 |
| `results/` | **77 GB** | 18 folders, five folder-naming schemes, three metadata formats | §3 |
| `temp/` | 1.0 GB | copies of the most recent run's arrays | Hidden handoff between scripts (§5) |
| `Legacy/` | 1.2 MB | Graphene, Lieb, Rashba and THF scripts, old QGT and phase-diagram calculators | Delete. Nothing imports it. |
| `hello.txt`, root `__pycache__/` | — | | Delete |
| `README.md` | — | Tells you to use `Hamiltonian_v2.py`, `Eigenvalues_Calc.py` and `QGT_Calc.py` | None of those files exist |

## 2. Target layout

```
~/Documents/Quantum_Geometric_Tensor/
├── QGT_Lib/                         code only (git → GitHub)
│   ├── README.md                    rewritten: quick start + map
│   ├── pyproject.toml               installable package, dependencies, pytest config
│   ├── qgt/                         ← Library/
│   │   ├── hamiltonians/            base class, one file per model family, minimal_models/
│   │   ├── spectra/                 eigen-solvers, eigenvector alignment
│   │   ├── geometry/                QGT, wavefunction derivatives, projectors
│   │   ├── topology.py              Berry curvature, flux, Chern, winding
│   │   ├── symmetry/                point groups, QGT group orbits, BZ zones
│   │   ├── kspace.py                grids, coordinate orders, paths, indexing
│   │   ├── io/                      data paths, new runs, finding runs, old-metadata loader
│   │   └── plotting/
│   ├── scripts/
│   │   ├── compute/                 ← Calc_*.py
│   │   ├── plot/                    ← Plot_*.py
│   │   └── figures/                 ← the 10 scripts now in results/Plot_Specific/
│   ├── configs/                     one file per run you want to repeat (§6)
│   ├── tests/
│   └── docs/
│
└── Results/                         data (backed up by iCloud, not in git)
    ├── legacy/                      today's results/, moved as is: same 18 folders, same files
    │   ├── 2D_QGT_ND/  3D_QGT_results/  phase_diagram/  …
    │   └── temp_snapshot/           today's temp/
    ├── runs/<kind>/<Model>/<YYYYMMDD-HHMMSS>[_tag]/      new runs only
    └── index.csv                    one row per new run
```

The code finds the data folder through one setting. It defaults to `Results/` next to the repo and can be overridden with the `QGT_DATA` environment variable, e.g. for a cluster.

## 3. Results

### Moving them

- **What moves:** `results/` becomes `Results/legacy/` and `temp/` becomes `Results/legacy/temp_snapshot/`. Nothing inside them changes.
- Both folders are on the same disk, so the move is a rename. It's instant and needs no extra space.
- Then remove `results/` and `temp/` from git tracking with `git rm -r --cached`; the files on disk aren't touched.
- **`Plot_Specific/` is the exception:** its 10 scripts move to `scripts/figures/`, and their outputs move with the rest of the data.

### Keeping old results replottable

The move changes paths, and some old metadata can't be loaded even today. Two fixes cover both problems:

**1. A path helper.** About 17 scripts and 5 library modules build paths from the string `"results/..."`. Replace each with `qgt.io.legacy_dir("2D_QGT_ND", ...)`, which points into `Results/legacy/`. This is a mechanical change, and the old data format is irrelevant to it.

**2. A loader for old metadata.** Hundreds of old `meta_info.pkl`/`meta.pkl` files (248 near the top of `results/` alone) store pickled Hamiltonian objects, so each pickle records the module path its class lived at when it was saved. Many of those paths no longer exist. A single `qgt.io.load_old_meta(path)` will handle all of them. It is a `pickle.Unpickler` that translates old paths through one table. These paths come from sampling one pickle per results folder:

| Path stored in old pickles | Loads as | Status today |
|---|---|---|
| `Library.Hamiltonian_v2.<Class>` (1D_QGT, phase_diagram, Legacy) | the current class of the same name | 4 plot scripts each copy a `sys.modules` patch for this |
| `Library.Hamiltonian.Altermagnet_Hamiltonian.AltermagnetHamiltonian` (3D_Eigen, 3D_QGT) | `AltermagnetHamiltonian` | **already broken:** that module was renamed earlier |
| `Library.Hamiltonian.gWaveAltermagnetHamiltonian.…` (the 26 GB of gWave runs, Spin_Density) | `MinimalHamSG194_2a` | loads at HEAD through an alias file, but **your uncommitted change deletes that alias** |
| `Library.Hamiltonian.MinimalHamSG*.…` (older 2D_Eigen runs) | the same class in `minimal_models/` | **your uncommitted move breaks these** |
| `Library.Hamiltonian.Chiral_Hamiltonian_Projected` | its current Chiral counterpart | patched in `Plot_QGT_2D_Dynamic_Omega_sweep.py` only |
| `Library.…` in general, after the rename | `qgt.…` | — |

How the loader treats classes:
- A class that has no successor loads as a placeholder that keeps all the saved attributes, which is enough for plotting.
- Classes that do have a successor map to the real class. That matters because some plotting code, such as the symmetry-path overlays, calls methods like `get_sym_path()` on the Hamiltonian.

Until the loader exists, keep the old import paths working: either don't commit the deletions, or leave one-line alias files behind (§9, step 0).

**How to verify:**
1. Load one pickle from every folder in `Results/legacy/`.
2. Run each old plot script on one dataset before and after the change, and compare the output.

Old results are **not** added to the index, because their formats vary too much. Instead, a one-time `Results/legacy/README.md` lists each folder's size, date range, and the scripts that read it.

### New runs

`Calc_QGT_MinimalModels_Planes.py` already does most of this. It writes a `meta.json` (`runner`, `runner_schema`, `hamiltonian_name`, `hamiltonian_params`, `space_group`, …) and a `summary.json` (`status`, `elapsed_seconds`, error checks). That becomes the pattern for every new run:

- **One API in `qgt.io`:**
  - `new_run(kind, model, config) → folder`, replacing the 13 `setup_*_results_directory` functions spread across four modules
  - `find_runs(kind, model, **filters)` and `latest_run(...)`, so no script hard-codes a dataset number again
- **Metadata is JSON only, never pickled objects.** It records:
  - the model's import path and parameters, so the Hamiltonian can be rebuilt from them
  - the full run config
  - the git commit, plus a flag for uncommitted changes
  - the runner's name and schema version
  - a timestamp
- **Folder name:** `<YYYYMMDD-HHMMSS>[_tag]`, e.g. `20261014-0930_kz0-bz`. It is unique and sorts by date, and the tag is optional. Parameters live in `meta.json`, not in the folder name.
- **Reuse:** the current "reuse a run if its parameters match" behavior stays, but it compares a hash of the config rather than parsing folder names.
- **`index.csv`:** one row per new run (kind, model, key parameters, tag, size, status, date), regenerated by `scripts/index_runs.py`. To find a run, you filter this table.

### iCloud notes

- iCloud's "Optimize Mac Storage" may move old results to the cloud, in which case the first read downloads them. To keep a folder you're actively using on disk, right-click it in Finder and choose "Keep Downloaded".
- While a run is writing, iCloud sometimes creates `name 2` duplicate copies. Four already exist in results, such as `Jx_0 2/` and `summary 2.json`. `find_runs` will ignore them.
- The code repo also sits in iCloud. Once the data is out it's small, and GitHub is its real backup, so push regularly.

## 4. `Library/` → `qgt`

### Where each module goes

| Today | Proposed |
|---|---|
| `Hamiltonian/Hamiltonian.py` (base class `hamiltonian` + `TestHamiltonian`, `GrapheneHamiltonian`, `RashbaHamiltonian`) | `hamiltonians/base.py`; each concrete model gets its own file |
| `Hamiltonian/*.py`, `Hamiltonian/MinimalModels/` | `hamiltonians/`; the three Chiral variants go together in `chiral.py`, and `minimal_models/` is unchanged |
| `Hamiltonian_v1.py` (`H_THF` functions) and `Hamiltonian/THF_Hamiltonian.py` (`THF_Hamiltonian` + `THF_Hamiltonian_Legacy`) | `hamiltonians/thf.py`. The duplicate rule decides whether these are one implementation or two. |
| `Hamiltonian_helper.py` (`get_Hamiltonian`) | `hamiltonians/__init__.py` |
| `diagnalization.py`, `eigenvalue_calc_lib.py`, `eigenvalue_calc_lib_1d.py`, `Eigenvector.py` | `spectra/` |
| `QGT_lib.py`, `QGT_Calc_3d_lib.py`, `QGT_calc_functions_2comp.py`, `QGT_calc_functions_3comp.py`, `calculus_lib.py` | `geometry/` |
| `topology.py` | `topology.py` |
| `GroupTheory/`, `Geometry/zones.py` | `symmetry/` |
| `dimension_lib.py`, `indexing_lib.py`, the k-path generators in `utilities.py` | `kspace.py` |
| `data_management_utils_{common,1d,2d,3d,nd}.py`, `output_utils.py` | `io/` |
| `plotting_*.py` | `plotting/` |
| `basic_math.py` (one 5-line function, used once) | inlined where it's used |
| `test.py` (sympy scratch, unused) | delete |

### Duplicates, using your rule

I checked each pair of definitions.

| Function | Finding | Action |
|---|---|---|
| `line_eigenvalues_eigenfunctions`: `eigenvalue_calc_lib.py` and `eigenvalue_calc_lib_1d.py` | **identical** (49 lines) | one copy in `spectra/`, imported where needed |
| `projection_operator`: `QGT_calc_functions_2comp.py` and `…_3comp.py` | **identical** | one copy in `geometry/` |
| `pick_or_create_result_dir_simple`: `utilities.py` and `data_management_utils_common.py` | **different.** The `utilities` version matches runs on `parameters.json` and names folders `datasetN`; the `common` version matches on `meta.json` with float tolerance and names them `dataset_N`. Every caller uses the `common` version. | The `utilities` version and its sibling `pick_or_create_result_dir` are unused old code, so delete both |
| `_pick_field_grid`: `Plot_QGT_2D_Dynamic_ND_sweep.py` and `Plot_idealness_2D_phase_diagram.py` | **different** (35 vs 20 lines, about 30% similar) | rename each after what it does |
| `calculus_lib.py`: `dpsi_dx_num`, `dpsi_dy_num`, `dpsi_dz_num` | **the same computation**, differing only in which coordinate is shifted by `delta_k` | one `dpsi_num(axis, ...)` |
| `calculus_lib.py`: `_num` vs `_num_eigenvector_ordered` vs `_num_phase_corrected` | **different:** they align eigenvectors between neighboring k-points in different ways | stay separate, and each takes an `axis` argument. The 11 functions in `calculus_lib` should come down to about 4. |
| QGT functions: `QGT_grid_num`, `_semi_num`, `_analytic`, `QGT_3d_num`, `quantum_geometric_tensor_*` and their `_3d_` twins | not yet checked pair by pair | apply the same rule during the refactor |

Also:
- **Replace the 28 star imports** (`from Library… import *`) in the root scripts with explicit imports, so each script shows where every function comes from.
- **Make names consistent:** `diagnalization` → `diagonalization`; `QGT_lib` and `Eigenvector` → snake_case. Describe dimensions one way only; today the code says both `2comp`/`3comp` and `2d`/`3d`.

### Making the change without breaking scripts

1. Move code into the new modules, leaving each old module as a one-line re-export that emits a `DeprecationWarning`.
2. Update the scripts at your own pace.
3. Remove the shims once nothing warns.

The rename to `qgt` is safe for old data, because the loader in §3 translates the `Library.*` paths stored in pickles. Installing the package with `pip install -e .` also removes the need for `.env` (`PYTHONPATH`) and the `sys.path` hacks in two files.

## 5. The `temp/` handoff

`Calc_Eigenvalues_2D/3D`, `Calc_QGT_2D/3D` and `Calc_Spin_Density` copy their latest arrays to `<current folder>/temp`. `Calc_QGT_2D` then reads `temp/eigenvalues.npy` and `temp/meta_info.pkl` as its input. As a result:
- Two runs overwrite each other.
- Launching from a different folder silently uses different data.
- No run records which eigen-run its QGT came from.

The fix: the QGT step takes the eigen-run's folder as an explicit input and records it in its `meta.json`, or it computes the eigenvectors itself. The current `temp/` contents are kept in `Results/legacy/temp_snapshot/`.

## 6. Scripts and how runs are configured

### Layout
- **Location:** `scripts/compute/` and `scripts/plot/`, named `<quantity>_<geometry>`, e.g. `compute/eigen_2d.py`, `compute/qgt_2d.py`, `compute/qgt_nd_sweep.py`, `compute/qgt_minimal_planes.py`, `plot/qgt_3d.py`.
- **Thin scripts:** read a config, call `qgt`, save a run. Logic that a test needs moves into `qgt/`. Tests currently import four scripts directly.
- **Old plot scripts stay as they are, apart from the path helper and the loader,** because they're how you replot old results. That includes the three ND sweep plotters (`Plot_QGT_2D_Dynamic_ND_sweep.py`, `…_matplotlib.py`, `Plot_QGT_2D_Dynamic_Omega_sweep.py`, about 4,700 lines together). The duplicate rule still applies to the helper functions inside them.

### Run configuration (decided: config files)

Today you change a parameter by opening the script, editing the numbers in it, and running it. Nothing records which numbers a past run used, except what the script happened to save. There are two common alternatives:

**argparse flags.** You give the parameters on the command line when you run the script, and the script never changes:

```bash
python scripts/compute/qgt_2d.py --model MinimalHamSG194_2a --J 0.5 --t1 0.3 --mesh 200
```

This is fine for two or three parameters. Your runs have a dozen or more (model parameters, k ranges, sweep ranges, spacing), so the commands would get long and hard to repeat.

**Config files (chosen).** The parameters live in a small text file, one file per setup you care about:

```toml
# configs/sg194_kz0.toml
model = "MinimalHamSG194_2a"
tag   = "kz0-bz"

[params]
J  = 0.5
t1 = 0.3

[grid]
kx   = [-3.14, 3.14]
ky   = [-3.14, 3.14]
mesh = 200
```

```bash
python scripts/compute/qgt_2d.py configs/sg194_kz0.toml
```

- Each setup is a file you can keep, copy, compare and rerun.
- The config is copied into the run's `meta.json`, so every result records exactly what produced it.
- **VS Code still works:** your `launch.json` runs the current file without arguments, so each script gets one `DEFAULT_CONFIG = "configs/…"` line that is used when no file is given. Pressing Run still works, and switching setups means changing that one line.

## 7. Tests

- **Add a pytest config** with `testpaths = ["tests"]` and `MPLBACKEND=Agg`. Without it, running `pytest` from the root walks the whole results tree. Even collecting only `tests/` didn't finish within 4 minutes; it sat at 0% CPU, possibly because an import opens a plot window.
- **`hamiltonian_harmonics_test.py` imports `Hamiltonian.Hamiltonian_v2`,** which doesn't exist. Fix it or delete it.
- **`V_k_test.py` and `derivative_testing.py` are scratch plots, not tests.** The latter uses `scipy.misc.derivative`, which SciPy removed in 1.12. Delete both, since you said legacy code can go.
- **`test_polynomial_basis_nd_sweep.py` loads its code from `results/Plot_Specific/`.** Point it at `scripts/figures/`.
- **Add a "legacy replot" test** that loads one old pickle per `Results/legacy/` folder through `load_old_meta`. It catches future moves that would break old data. It runs only when `Results/` is present.

## 8. Git

- **Untrack results.** 2,307 of the 2,500 tracked files are under `results/` or `temp/`. After the move, `git rm -r --cached` them.
- **Rewrite `.gitignore`.** It has unresolved stash-conflict markers and doesn't cover `results/`, `temp/`, `tmp/`, `.pytest_cache/` or `*.pyc` properly.
- **Fix the case-only renames.** Git's index has `Calc_QGT_2d.py`/`Calc_QGT_3d.py`, while the files on disk are `…2D.py`/`…3D.py`. Fix with `git mv -f`.
- **Rewrite history (accepted).** Once the data is in `Results/` and confirmed there:
  1. Run `git filter-repo --path results --path temp --path tmp --invert-paths` to drop 2.5 GB of old blobs; the pack goes from 1.17 GB to a few MB.
  2. Force-push to `Eta2Zeta/QGT_Lib`.

  Consequences:
  - Any other clone, on another machine or the cluster, must be re-cloned.
  - Keep the old `.git` as a single backup file (`git bundle`, about 1.2 GB) in `Results/legacy/` until you're satisfied.
  - The 10 `Plot_Specific` scripts lose their history from before the move; their current contents are kept.
  - `Legacy/` and `tmp/` stay in history (at `pre-reorg`), so they're still recoverable after they're deleted.
- **Commit messages and tags.** Write descriptive commit messages, and keep versions as tags (`git tag v0.1.15`). Delete the one leftover Codex checkpoint ref.

## 9. Order of work

Run the tests after each step, and make each step its own commit.

0. **Secure what's there. ✅ Done 2026-10-13.**
   - `915a6191`: committed the MinimalModels work, keeping the old import paths alive as one-line re-exports. All five old paths the move removed load again; checked on sample pickles from Spin_Density, 2D_Eigen and the gWave runs. The 122 minimal-model and symmetry-line tests pass.
   - `3d7db4e9`: fixed the case-only renames.
   - `647ae504`: removed `tmp/`.
   - Pushed to GitHub, and tagged `pre-reorg` at `3d7db4e9`.
1. **Infrastructure. ✅ Done 2026-10-13.**
   - `ed8d16ee`: added `pyproject.toml` (package, dependencies, pytest config), `tests/conftest.py` (Agg backend), a fixed import in `hamiltonian_harmonics_test.py` and a rewritten `.gitignore`. Also untracked the `.DS_Store` files.
   - `181c8b71`: deleted `Legacy/`, including 5 Mathematica notebooks, plus the unused old result-dir block in `utilities.py`, `Library/test.py`, `hello.txt` and the two scratch tests. `Calc_QGT_3D.py` now imports `json` itself; it used to get it through the star import.
   - Deleted the iCloud duplicate `.pyc` files.
   - `pytest` now collects 230 tests in about 10 s, where it used to hang. 228 pass; the 2 failures predate this work (§10).
2. **Data. ✅ Done 2026-10-13.**
   - `7af4a322`:
     - Added `Library/data_paths.py` (`data_root`, `legacy_dir`, `QGT_DATA`). It moves to `qgt.io` in step 5; `runs_dir` comes with step 4.
     - Routed every hard-coded `results/` path in 18 scripts and 5 library modules through `legacy_dir`.
     - Moved the 10 `Plot_Specific` scripts to `scripts/figures/` (pulled forward from step 6, so they stay in git); they now write their figures to `Results/legacy/Plot_Specific/`.
     - `conftest.py` points `QGT_DATA` at a temporary folder during tests.
   - **Data move:** `results/` → `Results/legacy/` and `temp/` → `Results/legacy/temp_snapshot/`. All 16,631 files, 1,797 folders and 29 temp files arrived, and every one of the 2,286 formerly tracked files is present.
   - `765e52c9`: stopped tracking `results/` and `temp/`.
   - **Checks:**
     - Every path that pointed at existing data resolves.
     - Old pickles load from the new location.
     - Two old plot functions (`Plot_QGT_2D`, `Plot_QGT_2D_sym_lines`) replot correctly from the moved data.
     - Tests: 228 pass, plus the 2 known failures.
   - Until step 4, the next `Calc_QGT_2D/3D` run needs a fresh eigenvalue run first, because the old `temp/` handoff files are now in `temp_snapshot/`.
3. **Old metadata. ✅ Done 2026-10-13.**
   - `101bf0cb`: added `Library/legacy_pickles.py`, a table of moved modules served through an import hook that `Library/__init__.py` installs. It replaces the `sys.modules` patches in four scripts and the step-0 alias files. Plain `pickle.load` now opens all 1,583 old metadata pickles; before, it could open 224.
   - `RhombohedralGrapheneHamiltonian` maps to `ChiralHamiltonian`, its v0.0.3 rename. Two scripts mapped it to `ChiralHamiltonianChiralBasisProjected`, but no plot code calls methods on the loaded object.
   - `tests/test_legacy_pickles.py` loads every old pickle whenever `Results/` is present.
   - Wrote `Results/legacy/README.md`: size, dates and the code that uses each folder.
   - `12de2d3d`: **fixed a regression from step 2.** MinimalModels metadata records absolute `QGT_Lib/results/...` paths; `relocate_legacy_path` maps them to `Results/legacy/...`, and the data files are untouched.
4. **New runs. ✅ Done 2026-10-13** (`51aefa56`). You chose JSON plus the pickle, with MinimalModels staying in `legacy/` for now.
   - `Library/runs.py`: `new_run` (the same reuse rule as before), `run.json` provenance, `latest_run`, `input_run`, `find_runs`, `record_inputs`, `write_index` (`scripts/index_runs.py`).
   - All setup functions, the D3d projection and the symmetry sweep now write `Results/runs/<kind>/<model>/<timestamp>[_tag]/`.
   - The `temp/` handoff is gone. QGT and spin-density calculations read the most recent eigen run (or an explicit `eigen_run=`) and record it.
   - **Check:** an end-to-end eigen → QGT run on a 12×12 mesh in a throwaway folder recorded its input, reused correctly and produced finite arrays.
5. **Package. ✅ Done 2026-10-13** (`eccc1a40`).
   - `Library/` is now `qgt/` with topic subpackages (moved with `git mv`). Chiral variants stay in three files (`chiral.py`, `chiral_basis_projected.py`, `chiral_sw_projected.py`) instead of one.
   - No star imports remain anywhere; each name is imported from where it is defined.
   - Duplicates were resolved with your rule. The derivative merge was checked bit-identical, and so was a full eigen → QGT run before and after the move.
   - `Library/` is a stub: `qgt.io.legacy_pickles` serves every old `Library.*` path, so old pickles and old imports keep working.
   - Hiding behind the star imports were two undefined names in `Calc_QGT_3D.py`'s old serial `calculate_3d` (`mesh_size`, `QGT_grid_3d_num`). They predate this work; see §10.
6. **Scripts. ✅ Done 2026-10-13** (`f6cbdbfd`).
   - The 29 root scripts became modules in `qgt/workflows/` and `qgt/plotting/`, with thin entry points in `scripts/compute/` and `scripts/plot/` that read `configs/<kind>/<name>.toml`. The configs hold the values the scripts had, and commented-out alternatives are kept as comments.
   - `qgt/config.py` builds models by class name and resolves `@legacy/`, `@runs/` and `@latest/<kind>`. It records the config in every run's `run.json`.
   - The two MinimalModels tools keep their command-line options.
   - **One behavior change:** `plot/qgt_3d` had every call commented out; its config activates the last of them.
   - **Check:** `eigen_2d` → `qgt_2d` ran end to end from the scripts.
7. **Finish. ✅ Done 2026-10-13.**
   - `c4fafb4c`: rewrote the README.
   - Refreshed `Results/legacy/README.md`.
   - Backed up the full git history to `Results/legacy/QGT_Lib_git_backup_before_history_rewrite_2026-10-13.bundle` (1.2 GB).
   - Rewrote history without `results/`, `temp/` and `tmp/`, keeping the 10 `Plot_Specific` scripts, and force-pushed `main` and `pre-reorg`. `.git` went from 1.2 GB to 39 MB, and a fresh clone is 40 MB.
   - The `Library/` stub stays permanently, because old pickles need its module paths.

## 10. Still open

None of these block anything. They predate the reorganization unless marked.

- **Failing tests:** `test_d3d_parallel_projection` expects worker results in irrep order, but they arrive in completion order. `test_twoorbitalunspinful_lattice_harmonics` hits a `lambdify` argument-count error in `qgt/hamiltonians/base.py` (`fx11p`).
- **`qgt.workflows.qgt_3d.calculate_3d`** (the old serial path) uses the undefined names `mesh_size` and `QGT_grid_3d_num`. The parallel `calculate_3d_all_bands_parallel` works.
- **Mismatched path:** `compute/sym_nd_sweep` writes runs that `plot/sym_nd` reads through `@latest/sym_nd`. The old pair used different folders (`Sym_Phase_Diagram` vs `Sym_Energy_ND`).
- **Missing data:** `plot/idealness_1d` (via `plotting_utils.load_qgt`) and the commented "chiral projected" inputs of `plot/idealness_2d` point at folders that were never in `results/`.
- **`plot/qgt_3d` needs `trimesh`:** `pip install -e ".[plot3d]"`.
- **MinimalModels:** it still writes to `Results/legacy/MinimalModels_QGT_planes` (your choice). Its reuse key includes source hashes, so the next run recomputes, which takes about 30 min. Switch it to `new_run` when convenient.
- **`plot/symmetry_path_3d`** writes its HTML to `Results/legacy/Symmetry_Path_Plots/` (unchanged from before).
- **Changed by me:** two plot scripts mapped old `RhombohedralGrapheneHamiltonian` pickles to `ChiralHamiltonianChiralBasisProjected`; the shared table maps them to `ChiralHamiltonian`, the class it was renamed to.
- **Not done:** `pip install -e ".[dev]"`. The scripts work without it; installing makes `qgt` importable from anywhere.
