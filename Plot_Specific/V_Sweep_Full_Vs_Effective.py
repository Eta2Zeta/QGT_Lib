import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to allow importing from Library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Library.Hamiltonian.ChiralHamiltonian import ChiralHamiltonian
from Library.Hamiltonian.ChiralHamiltonian_SW_Projected import ChiralHamiltonianEffective


def compute_results(n, vF, t1, A0, omega, V_vals, kx_vals, ky=0.0):
    """Compute full and analytic-effective eigenvalues for each V."""
    results = []
    print(f"Sweeping V values: {V_vals}")
    print(f"Parameters: n={n}, A0={A0}")

    for V in V_vals:
        print(f"Processing V = {V}...")

        H_full_obj = ChiralHamiltonian(n=n, vF=vF, t1=t1, V=V, omega=omega, A0=A0)
        H_eff_obj  = ChiralHamiltonianEffective(n=n, vF=vF, t1=t1, V=V, omega=omega, A0=A0)

        evals_full_list     = []
        evals_analytic_list = []

        for kx in kx_vals:
            H_mat_full     = H_full_obj.compute_static(kx, ky)
            e_full         = np.linalg.eigvalsh(H_mat_full)
            evals_full_list.append(np.sort(e_full))

            H_mat_analytic = H_eff_obj.compute_static_analytic(kx, ky, order=1)
            e_analytic     = np.linalg.eigvalsh(H_mat_analytic)
            evals_analytic_list.append(np.sort(e_analytic))

        results.append({
            'V':       V,
            'full':    np.array(evals_full_list),
            'analytic': np.array(evals_analytic_list),
        })

    return results


def plot_three_panels(results, kx_vals, n, t1, vF, save_path):
    """Original 3-panel plot: Full | Analytic | Comparison overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    k_lim  = t1 / vF

    ax1 = axes[0]
    for idx, res in enumerate(results):
        V = res['V']; evals = res['full']; c = colors[idx]
        for b in range(2 * n):
            if b not in (n - 1, n): continue
            label = f"$V={V}$ meV" if b == n else None
            ax1.plot(kx_vals, evals[:, b], color=c, linewidth=1.5, alpha=0.8, label=label)
    ax1.axvline(x=k_lim,  color='k', linestyle=':', alpha=0.5)
    ax1.axvline(x=-k_lim, color='k', linestyle=':', alpha=0.5)
    ax1.set_title(f'Full Hamiltonian ({2*n}×{2*n})')
    ax1.set_xlabel('$k_x$'); ax1.set_ylabel('Energy (meV)')
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim(-50, 50)

    ax2 = axes[1]
    for idx, res in enumerate(results):
        V = res['V']; evals = res['analytic']; c = colors[idx]
        ax2.plot(kx_vals, evals[:, 0], color=c, linewidth=2, label=f"$V={V}$ meV")
        ax2.plot(kx_vals, evals[:, 1], color=c, linewidth=2, linestyle='--')
    ax2.axvline(x=k_lim,  color='k', linestyle=':', alpha=0.5)
    ax2.axvline(x=-k_lim, color='k', linestyle=':', alpha=0.5)
    ax2.set_title('Analytic Effective Hamiltonian (2×2)')
    ax2.set_xlabel('$k_x$'); ax2.set_ylabel('Energy (meV)')
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim(-50, 50)

    ax3 = axes[2]
    for idx, res in enumerate(results):
        V = res['V']; evals_f = res['full']; evals_a = res['analytic']; c = colors[idx]
        ax3.plot(kx_vals, evals_f[:, n - 1], color=c, linewidth=1.5, alpha=0.4)
        ax3.plot(kx_vals, evals_f[:, n],     color=c, linewidth=1.5, alpha=0.4)
        ax3.plot(kx_vals, evals_a[:, 0], color=c, linewidth=2, linestyle=':', label=f"Eff $V={V}$")
        ax3.plot(kx_vals, evals_a[:, 1], color=c, linewidth=2, linestyle=':')
    ax3.axvline(x=k_lim,  color='k', linestyle=':', alpha=0.5)
    ax3.axvline(x=-k_lim, color='k', linestyle=':', alpha=0.5)
    ax3.set_title('Comparison (Overlay)')
    ax3.set_xlabel('$k_x$'); ax3.set_ylabel('Energy (meV)')
    ax3.set_ylim(-50, 50); ax3.legend(fontsize='small'); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")


def plot_overlay(results, kx_vals, n, t1, vF, save_path):
    """Single big comparison overlay: full low bands (solid) vs analytic effective (dashed)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    k_lim = t1 / vF

    for idx, res in enumerate(results):
        V = res['V']
        evals_f = res['full']
        evals_a = res['analytic']
        c = colors[idx]

        # Full low-energy bands — solid, slightly faded
        ax.plot(kx_vals, evals_f[:, n - 1], color=c, linewidth=2.0, alpha=0.5,
                label=f"Full  $V={V}$ meV")
        ax.plot(kx_vals, evals_f[:, n],     color=c, linewidth=2.0, alpha=0.5)

        # Analytic effective — dashed, full opacity
        ax.plot(kx_vals, evals_a[:, 0], color=c, linewidth=2.0, linestyle='--',
                label=f"Analytic $V={V}$ meV")
        ax.plot(kx_vals, evals_a[:, 1], color=c, linewidth=2.0, linestyle='--')

    ax.axvline(x=k_lim,  color='k', linestyle=':', alpha=0.5)
    ax.axvline(x=-k_lim, color='k', linestyle=':', alpha=0.5)
    ax.set_title('Full vs Analytic Effective — V Sweep (Overlay)', fontsize=14)
    ax.set_xlabel('$k_x$', fontsize=13)
    ax.set_ylabel('Energy (meV)', fontsize=13)
    ax.set_ylim(-50, 50)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")


def main():
    # Parameters
    n     = 5
    vF    = 542.1
    t1    = 355.16
    A0    = 0.0
    omega = 2 * np.pi  # undriven

    V_vals  = [5.0, 10.0, 20.0]
    kx_vals = np.linspace(-1.0, 1.0, 300)
    ky      = 0.0

    results = compute_results(n, vF, t1, A0, omega, V_vals, kx_vals, ky)

    print("Calculation done. Plotting...")

    here = os.path.dirname(__file__)

    # Single overlay plot
    plot_overlay(
        results, kx_vals, n, t1, vF,
        save_path=os.path.join(here, 'V_Sweep_Comparison.png')
    )

    # Optionally also save the 3-panel version
    # plot_three_panels(
    #     results, kx_vals, n, t1, vF,
    #     save_path=os.path.join(here, 'V_Sweep_Three_Panel.png')
    # )


if __name__ == "__main__":
    main()
