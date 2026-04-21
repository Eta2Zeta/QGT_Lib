import sys
import os
import numpy as np
import pickle
from tqdm import tqdm  # Import tqdm for progress bar

from Library.Hamiltonian_v1 import *
from Library.Hamiltonian.Hamiltonian import * 
from Library.Hamiltonian.THF_Hamiltonian import *
from Library.Hamiltonian.TwoOrbitalUnspinfulHamiltonian import *
from Library.Hamiltonian.SquareLatticeHamiltonian import *
from Library.eigenvalue_calc_lib import *
from Library.QGT_lib import *
from Library.topology import *
from Library.utilities import *
from Library.data_management_utils_1d import setup_QGT_results_directory_1D_single_param

def sweep_single_param_1d(
    hamiltonian,
    param_name="omega",
    *,
    vmin=1.0,
    vmax=1e3,
    spacing="log",             # "log" | "linear"
    num_param_points=30,
    num_k_points=100,
    angle_deg=0.0,
    kx_shift=0.0,
    ky_shift=0.0,
    k_max=None,                # if None -> choose a sensible default
    band=0,
    delta_k=1e-5,              # Default numeric derivative step
    force_new=False
):
    """
    Sweep ONE Hamiltonian attribute across [vmin, vmax] and compute QGT along a 1D k-line.
    Saves list-of-dicts to QGT_1D.npy and meta_info.pkl in a parameterized directory.
    """
    if not hasattr(hamiltonian, param_name):
        h_name = getattr(hamiltonian, "name", type(hamiltonian).__name__)
        raise ValueError(f"Hamiltonian '{h_name}' does not have the parameter '{param_name}'. Cannot sweep.")

    # ---- build k-line ----
    theta = np.deg2rad(angle_deg)
    if k_max is None:
        # simple defaults
        if abs((angle_deg % 180)) in (0.0, 90.0):
            k_max = 0.5*np.pi
        elif abs((angle_deg % 180)) == 45.0:
            k_max = np.sqrt(2.0)*np.pi
        else:
            k_max = np.pi

    k_line = np.linspace(-k_max, k_max, int(num_k_points))
    line_kx = k_line*np.cos(theta) + kx_shift
    line_ky = k_line*np.sin(theta) + ky_shift

    # ---- parameter values ----
    if spacing == "log":
        if vmin <= 0 or vmax <= 0:
            raise ValueError("Log spacing requires positive vmin and vmax.")
        values = np.logspace(np.log10(vmin), np.log10(vmax), int(num_param_points))
    elif spacing == "linear":
        values = np.linspace(vmin, vmax, int(num_param_points))
    else:
        raise ValueError("spacing must be 'log' or 'linear'.")

    # ---- directory ----
    file_paths, used_existing, out_dir = setup_QGT_results_directory_1D_single_param(
        hamiltonian,
        param_name=param_name,
        vmin=vmin, vmax=vmax,
        spacing=spacing,
        num_param_points=num_param_points,
        num_k_points=num_k_points,
        angle_deg=angle_deg,
        kx_shift=kx_shift, ky_shift=ky_shift,
        k_max=k_max,
        force_new=force_new,
    )
    if (not force_new) and used_existing:
        print(f"Using existing results: {out_dir}")
        return

    # ---- sweep & compute ----
    results = []
    for val in tqdm(values, desc=f"Sweeping {param_name}", unit=param_name):
        setattr(hamiltonian, param_name, float(val))
        eigenvalues, permutations, g_xx, g_xy_real, g_xy_imag, g_yy, trace, magnus_operator_norm = QGT_line(
            hamiltonian, line_kx, line_ky, delta_k, band_index=band
        )
        results.append({
            param_name: float(val),
            "g_xx": g_xx,
            "g_xy_real": g_xy_real,
            "g_xy_imag": g_xy_imag,
            "g_yy": g_yy,
            "trace": trace,
            "eigenvalues": eigenvalues,
            "perturbation": permutations,
            "magnus_operator_norm": magnus_operator_norm,
        })

    # ---- save ----
    np.save(file_paths["QGT_1D"], results)
    meta = {
        "param_name": param_name,
        "values": values,
        "spacing": spacing,
        "num_param_points": int(num_param_points),
        "num_k_points": int(num_k_points),
        "angle_deg": float(angle_deg),
        "kx_shift": float(kx_shift),
        "ky_shift": float(ky_shift),
        "k_max": float(k_max),
        "band": int(band),
        "Hamiltonian_Obj": hamiltonian,
    }
    with open(file_paths["meta_info"], "wb") as f:
        pickle.dump(meta, f)

    print(f"✅ Saved 1D QGT sweep for '{param_name}' to: {out_dir}")


if __name__ == '__main__':
    # Default parameters
    Hamiltonian_Obj = TwoOrbitalUnspinfulHamiltonian(zeta=1.0, omega=10.0, A0=0.1, mu=0, magnus_order=1)

    sweep_single_param_1d(
        Hamiltonian_Obj, param_name="omega",
        vmin=5.0, vmax=50.0, spacing="linear",
        num_param_points=20, num_k_points=100,
        angle_deg=0.0, band=0, force_new=True
    )
