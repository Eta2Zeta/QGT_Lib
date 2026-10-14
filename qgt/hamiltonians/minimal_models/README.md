# Minimal altermagnet Hamiltonians

This package implements all **27 explicit space-group/Wyckoff families** in
Tables III and IV of Roig et al., *Minimal models for altermagnetism*,
Physical Review B **110**, 144412 (2024), DOI: 10.1103/PhysRevB.110.144412.
The source tables are on PDF pages 12-13. The paper's broader classification
in Tables I-II includes centered lattices whose tight-binding coefficients
are not tabulated; those additional cases are not derived here.

Every class is a normal QGT `hamiltonian` object in the sublattice-spin basis
`(A up, A down, B up, B down)`, with

```text
H(k) = eps0(k) I + tx(k) tau_x + tz(k) tau_z
     + tau_y [lambda(k) . sigma] + tau_z [J . sigma].
```

## Use

```python
import numpy as np
from qgt.hamiltonians.minimal_models import MinimalHamSG136_2a2b, MINIMAL_MODELS
from qgt.hamiltonians.evaluate import get_Hamiltonian

model = MinimalHamSG136_2a2b(t3=0.5, t4=-0.15, Jz=0.2, lamb=0.1)
H = model.compute_static(0.3, 0.2, 0.4)  # (4, 4)
H_effective, H_prime = get_Hamiltonian(model, 0.3, 0.2, 0.4)

kx = np.linspace(-np.pi, np.pi, 31)[:, None]
ky = np.linspace(-np.pi, np.pi, 41)[None, :]
H_grid = model.compute_static_vectorized(kx, ky, 0.4)  # (31, 41, 4, 4)
energies = model.get_analytical_eigenvalues(kx, ky, 0.4)  # (31, 41, 4)

# Discover and instantiate any of the 27 models.
other = MINIMAL_MODELS["MinimalHamSG163_2b"](lamb2=0.03, t4_alt=0.02)
```

Import the models from `qgt.hamiltonians.minimal_models`. Every concrete
class inherits directly from `MinimalAltermagnetHamiltonian`, which holds
the shared parameters, scalar dispersion, and lattice symmetry paths.
Individual model files supply their coefficient functions and any additional
parameters or custom paths, including the SG194 path.

The shared base provides static matrices, vectorized matrices, sorted analytic
static spectra, spin operators, and the existing numerical driven/Floquet
machinery. Analytic spectra describe the **static** model. Floquet settings
(`omega`, `A0`, `polarization`, `magnus_order`, `analytic_magnus`) retain the
library's existing behavior. Momentum arrays follow NumPy broadcasting;
scalar vectorized calls retain one-point shapes `(1, 4, 4)` and `(1, 4)`.
Public parameter attributes remain mutable for the existing sweep code.

## Parameters and conventions

- `t3` multiplies the first listed `tau_x` form factor; `t4` multiplies the
  first listed `tau_z` form factor. `t3_alt` and `t4_alt`, when present,
  independently multiply the second forms in the printed order, with
  nonzero defaults `t3_alt=0.15` and `t4_alt=0.2`.
- `lamb` is the first in-plane SOC amplitude. `lamb2`, when present,
  multiplies the second symmetry-related in-plane pair. Its default is `0.07`.
- For monoclinic and orthorhombic rows, `lamb_y` independently controls the
  y SOC amplitude; `lamb` controls x. `lamb_x_alt`, `lamb_y_alt`, and
  `lamb_z_alt`, when present, control the second listed SOC form factors.
  Their defaults are `0.04`, `0.06`, and `0.08`, respectively.
- `lamb_z` is independent except for cubic SG223, where all three SOC axes
  share `lamb`. An optional `lamb_z` must equal `lamb`; changing the alias
  updates the shared value so parameter sweeps respect cubic symmetry.
- `Jx`, `Jy`, `Jz` give the staggered exchange field. Every model defaults to
  `Jx=Jy=0` and `Jz=0.2`.
- SG11 additionally includes the symmetry-allowed hoppings derived in Note 3-3:
  `t3_add=0.8` multiplies `cos(kx)*cos(ky/2)` in `tx`, and
  `lamb_z_add=0.3` multiplies `cos(kz)*cos(ky/2)` in `lambda_z`, and
  `lamb_z_xz_add=0.3` multiplies `sin(kx)*sin(kz)*cos(ky/2)` in `lambda_z`.
  The `_add`
  suffix identifies terms beyond the paper's explicit model; `_alt` identifies
  independent bases already listed in the paper. Set all three `_add` amplitudes
  to zero to recover the explicit Table III functions. The xz SOC hopping is
  zero at `kz=0`, but its nonzero z derivative permits nonzero `Omega_y` on
  `ky=kz=0` together with `t3_add`, lifting that accidental line node at generic
  `kx`. The mirror-required `Omega_x=Omega_z=0` on `ky=0` remains intact.
- `mu`, `t1`, `t2` set a simple scalar dispersion. The tables specify no
  unique scalar dispersion or parameter fit. These are illustrative models;
  existing material-specific `RuO2Hamiltonian` and other objects stay separate.

All implemented hopping and SOC amplitudes are active by default. Unequal
alternate amplitudes avoid imposing special proportionalities among independent
coefficient functions. `mu=0` sets the energy origin, and `A0=omega=0` keeps the
default calculation static.

For monoclinic, orthorhombic and tetragonal models, the chosen scalar term is
`t1*(cos(kx)+cos(ky)) + t2*cos(kz) - mu`. For trigonal and hexagonal models,
it is `t1*(cos(kx)+2*cos(kx/2)*cos(sqrt(3)*ky/2)) + t2*cos(kz) - mu`.
The cubic scalar term is
`t1*(cos(kx)+cos(ky)+cos(kz)) + t2*(cos(kx)*cos(ky)+cos(ky)*cos(kz)+cos(kz)*cos(kx)) - mu`.

Trigonal and hexagonal form factors use the paper's Cartesian convention:

```text
fx = sin(kx) + sin(kx/2) cos(sqrt(3) ky/2)
fy = sqrt(3) cos(kx/2) sin(sqrt(3) ky/2).
```

Half-angle intersublattice terms can change sign under reciprocal translations:
the Bloch matrix is gauge-covariant, so elementwise 2*pi periodicity is not
required. `get_sym_path()` uses these same dimensionless coordinates.
Monoclinic paths use reciprocal phase-box points; a metric-specific monoclinic
Brillouin-zone path requires lattice information not supplied by the tables.
Class attributes `space_group`, `wyckoff_positions`, `source_table`, and
`crystal_system` describe each family. Grouped Wyckoff labels denote the
alternative sites sharing a table row, not extra orbitals.

SG84 (2c,2d) reproduces both printed z-SOC factors, including their
`cos(kx)-cos(ky)` factors. No unverified correction to the source table is made.
The SG192 docstring's old table attribution has been corrected to Table IV.

## Model catalog

| Space group | Wyckoff positions | Table | Class |
| --- | --- | --- | --- |
| 11 | 2a, 2b, 2c, 2d | III | `MinimalHamSG11_2a2b2c2d` |
| 13 | 2a, 2b, 2c, 2d | IV | `MinimalHamSG13_2a2b2c2d` |
| 14 | 2a, 2b, 2c, 2d | III | `MinimalHamSG14_2a2b2c2d` |
| 49 | 2a, 2b, 2c, 2d | IV | `MinimalHamSG49_2a2b2c2d` |
| 51 | 2a, 2b, 2c, 2d | III | `MinimalHamSG51_2a2b2c2d` |
| 53 | 2a, 2b, 2c, 2d | III | `MinimalHamSG53_2a2b2c2d` |
| 55 | 2a, 2b, 2c, 2d | III | `MinimalHamSG55_2a2b2c2d` |
| 58 | 2a, 2b, 2c, 2d | III | `MinimalHamSG58_2a2b2c2d` |
| 83 | 2e, 2f | IV | `MinimalHamSG83_2e2f` |
| 84 | 2a, 2b | IV | `MinimalHamSG84_2a2b` |
| 84 | 2c, 2d | IV | `MinimalHamSG84_2c2d` |
| 123 | 2e, 2f | IV | `MinimalHamSG123_2e2f` |
| 124 | 2b, 2d | IV | `MinimalHamSG124_2b2d` |
| 127 | 2a, 2b | III | `MinimalHamSG127_2a2b` |
| 127 | 2c, 2d | III | `MinimalHamSG127_2c2d` |
| 128 | 2a, 2b | III | `MinimalHamSG128_2a2b` |
| 131 | 2a, 2b | IV | `MinimalHamSG131_2a2b` |
| 131 | 2c, 2d | IV | `MinimalHamSG131_2c2d` |
| 132 | 2a, 2c | IV | `MinimalHamSG132_2a2c` |
| 136 | 2a, 2b | III | `MinimalHamSG136_2a2b` |
| 163 | 2b | IV | `MinimalHamSG163_2b` |
| 165 | 2b | IV | `MinimalHamSG165_2b` |
| 176 | 2b | III | `MinimalHamSG176_2b` |
| 192 | 2b | IV | `MinimalHamSG192_2b` |
| 193 | 2b | III | `MinimalHamSG193_2b` |
| 194 | 2a | III | `MinimalHamSG194_2a` |
| 223 | 2a | IV | `MinimalHamSG223_2a` |

## Batch bands and QGT on two kz planes

From the library root:

```sh
python Calc_QGT_MinimalModels_Planes.py --workers 6
```

To replace the existing calculations and regenerate their plots in the same
dataset folders, use `--overwrite`:

```sh
python Calc_QGT_MinimalModels_Planes.py --overwrite --workers 6
```

The runner uses the existing `grid_eigenvalues_eigenfunctions` first, followed
by `QGT_grid_num` and `quantum_geometric_tensor_3d_num_phase_corrected` for all
four bands. Defaults are the 22 newly added classes, their default parameters,
150 x 150 points, and kz = 0 and 2*pi/100. The offset is one percent of the
**full** reciprocal z period, not one percent of the Gamma-to-boundary distance.
All nine Cartesian QGT components are retained, including z derivatives.
The derivative step is 1e-5, independent of the plotting-grid spacing.

Use `--all-models` to include the five older models, `--models CLASS ...` for
a subset, `--mesh-spacing N` for another resolution, or `--force-new` for new
datasets. Completed matching datasets are reused by default; source hashes
are included in the cache metadata. Run with `--help` for all options.

Outputs go under `results/MinimalModels_QGT_planes/CLASS/PLANE/dataset_N/`.
Each dataset contains standard eigenvalue/eigenvector and `g_*.npy` files,
the metric trace, Berry components, coordinates, JSON/pickle metadata, and
a summary. `manifest.json` and `manifest.csv` in the batch root index the runs.
The batch automatically generates the same band surfaces, degeneracy map,
band-cut sliders, QGT surfaces, and Berry/metric heatmaps as the current
eigenvalue and all-band QGT scripts: 20 plot files per four-band dataset.
Open `results/MinimalModels_QGT_planes/index.html` to browse the plots.
The batch saves plots without opening hundreds of windows or overwriting
the shared `temp/`. `--no-plots` is an explicit opt-out.

To add or repair plots for the datasets already in the manifest, without
recalculating bands or geometry:

```sh
python Calc_QGT_MinimalModels_Planes.py --plots-only --workers 2
```

This calls the existing plotting helpers through
`Plot_QGT_MinimalModels_Planes.py`. `--force-new` with `--plots-only` regenerates
plots, leaving the numerical arrays and calculation metadata unchanged.
The existing `Plot_QGT_2D.plot_qgt_from_directory` also accepts each dataset.

For trigonal/hexagonal models, the grid covers the Cartesian bounding box of
the hexagonal BZ; `bz_mask.npy` identifies its interior. Other models use the
[-pi, pi]^2 reciprocal phase box. Eigenpairs are saved across the full grid;
QGT outside the BZ is NaN. At a band separation <= 1e-6, individual-band QGT
is also marked NaN, matching the near-degenerate block scale in the existing
phase-correction routine. `band_gap.npy` and `qgt_valid_mask.npy` record this
explicitly, both with shape (4, N, N). No finite QGT values are clipped.

In particular, default SG163 and SG165 have two degenerate doublets everywhere
on kz=0. Those plane datasets contain valid bands and gaps, but no defined
isolated-band QGT. A degenerate-subspace calculation would require a separate
method; the runner does not alter model parameters to lift the degeneracies.

## Verify

From the QGT library root, using a Python environment with the library's
NumPy/SciPy/QuTiP dependencies and pytest:

```sh
python -m pytest -q tests/test_minimal_ham_sg124.py tests/test_minimal_models.py tests/test_minimal_models_table_three.py tests/test_minimal_models_table_four.py tests/test_minimal_models_planes.py
```

The tests check source form factors, independent amplitudes, spectra,
Hermiticity, broadcasting, lattice conventions, serialization, parameter updates,
and integration with the existing eigenvalue, QGT and Floquet routines.
