import numpy as np
from qgt.utils import sign_check
from qgt.kspace.grids import map_k_by_order
from tqdm import tqdm
from qgt.geometry.kubo import qgt_components, quantum_geometric_tensor

def QGT_3d_kubo(kx_vals, ky_vals, kz_vals, hamiltonian, band_index, *, eigenfunctions_3d=None, delta_k=1e-5):
    """
    Calculate the 3D quantum geometric tensor (QGT) components for a kx-ky-kz grid,
    by the Kubo formula (``qgt.geometry.kubo``).

    Parameters:
    - kx_vals, ky_vals, kz_vals: 1D arrays defining the grid.
    - hamiltonian: The Hamiltonian object.
    - band_index: Band index (or a sequence of bands treated as one group).
    - eigenfunctions_3d: Optional eigenfunctions of an eigen run, shape [nkx, nky, nkz, dim, dim];
      bands are then numbered as in that run.
    - delta_k: Step of the central difference of H.

    Returns:
    - 9 arrays of shape [nkx, nky, nkz] for metric components.
    """
    k = np.stack(np.meshgrid(kx_vals, ky_vals, kz_vals, indexing="ij"), axis=-1)
    Q = quantum_geometric_tensor(hamiltonian, k, band_index, delta_k=delta_k, eigenvectors=eigenfunctions_3d)
    return qgt_components(Q)


def QGT_grid_3d_analytic(
    kx_vals, ky_vals, kz_vals, eigenvalues_3d,
    quantum_geometric_tensor_func,
    hamiltonian,
    z_cutoff=None,
    progress_label=None,
    band_index=None,
    **kwargs
):
    """
    Analytic 3D QGT grid that *works for RuO2* (or anything else) when some analytic
    components need the eigenvalue/energy.

    Key behavior:
      - Pulls the energy from eigenvalues_3d at each (i,j,k)
          * If band_index is not None: uses eigenvalues_3d[i,j,k, band_index]
          * Else: uses eigenvalues_3d[i,j,k] (scalar)
      - Passes that as energy=... into quantum_geometric_tensor_func(...)
        (unless you already passed energy in kwargs, in which case your kwargs wins)
      - Stores 0.0 for any component that returns None (same style as your 2D analytic grid)
      - Optional clipping to [-z_cutoff, z_cutoff]

    Expected signature of quantum_geometric_tensor_func:
        (hamiltonian, kx, ky, kz, **point_kwargs)
          -> (g_xx, g_yy, g_zz,
              g_xy_real, g_xy_imag,
              g_xz_real, g_xz_imag,
              g_yz_real, g_yz_imag,
              trace)
    """
    nkx, nky, nkz = len(kx_vals), len(ky_vals), len(kz_vals)

    # allocate
    g_xx_arr = np.zeros((nkx, nky, nkz))
    g_yy_arr = np.zeros((nkx, nky, nkz))
    g_zz_arr = np.zeros((nkx, nky, nkz))

    g_xy_real_arr = np.zeros((nkx, nky, nkz))
    g_xy_imag_arr = np.zeros((nkx, nky, nkz))

    g_xz_real_arr = np.zeros((nkx, nky, nkz))
    g_xz_imag_arr = np.zeros((nkx, nky, nkz))

    g_yz_real_arr = np.zeros((nkx, nky, nkz))
    g_yz_imag_arr = np.zeros((nkx, nky, nkz))


    total_points = nkx * nky * nkz
    desc = f"Analytic 3D QGT [{progress_label}]" if progress_label else "Computing Analytic 3D QGT"

    # small helper: get E at (i,j,k)
    def _get_energy(i, j, k):
        ev = eigenvalues_3d[i, j, k]
        # common cases:
        #   ev is scalar (single band already selected) -> return float(ev)
        #   ev is vector (all bands) -> index it
        if np.ndim(ev) == 0:
            return float(ev)
        if band_index is None:
            raise ValueError(
                "eigenvalues_3d appears to contain multiple bands (ev is not scalar), "
                "so you must pass band_index."
            )
        return float(ev[int(band_index)])

    with tqdm(total=total_points, desc=desc, unit="kpt") as pbar:
        for i, kx in enumerate(kx_vals):
            for j, ky in enumerate(ky_vals):
                for k, kz in enumerate(kz_vals):
                    # energy from eig grid (needed for RuO2.g_xz_imag)
                    E = _get_energy(i, j, k)

                    # per-point kwargs: user kwargs + energy (unless user already supplied it)
                    point_kwargs = dict(kwargs)
                    point_kwargs.setdefault("energy", E)

                    out = quantum_geometric_tensor_func(hamiltonian, kx, ky, kz, **point_kwargs)

                    if (not isinstance(out, (tuple, list))) or (len(out) != 10):
                        raise ValueError(
                            "quantum_geometric_tensor_func must return 10 values:\n"
                            "(g_xx,g_yy,g_zz,g_xy_real,g_xy_imag,g_xz_real,g_xz_imag,g_yz_real,g_yz_imag,trace)"
                        )

                    (g_xx, g_yy, g_zz,
                     g_xy_r, g_xy_i,
                     g_xz_r, g_xz_i,
                     g_yz_r, g_yz_i,
                     tr) = out

                    # None -> 0.0 (your 2D analytic behavior)
                    g_xx_arr[i, j, k] = 0.0 if g_xx is None else g_xx
                    g_yy_arr[i, j, k] = 0.0 if g_yy is None else g_yy
                    g_zz_arr[i, j, k] = 0.0 if g_zz is None else g_zz

                    g_xy_real_arr[i, j, k] = 0.0 if g_xy_r is None else g_xy_r
                    g_xy_imag_arr[i, j, k] = 0.0 if g_xy_i is None else g_xy_i

                    g_xz_real_arr[i, j, k] = 0.0 if g_xz_r is None else g_xz_r
                    g_xz_imag_arr[i, j, k] = 0.0 if g_xz_i is None else g_xz_i

                    g_yz_real_arr[i, j, k] = 0.0 if g_yz_r is None else g_yz_r
                    g_yz_imag_arr[i, j, k] = 0.0 if g_yz_i is None else g_yz_i

                    pbar.update(1)

    if z_cutoff is not None:
        arrays = [
            g_xx_arr, g_yy_arr, g_zz_arr,
            g_xy_real_arr, g_xy_imag_arr,
            g_xz_real_arr, g_xz_imag_arr,
            g_yz_real_arr, g_yz_imag_arr
        ]
        arrays = [np.clip(a, -z_cutoff, z_cutoff) for a in arrays]
        return tuple(arrays)

    return (
        g_xx_arr, g_yy_arr, g_zz_arr,
        g_xy_real_arr, g_xy_imag_arr,
        g_xz_real_arr, g_xz_imag_arr,
        g_yz_real_arr, g_yz_imag_arr
    )