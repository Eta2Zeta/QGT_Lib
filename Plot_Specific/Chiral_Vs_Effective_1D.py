import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to allow importing from Library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Library.Hamiltonian.ChiralHamiltonian import ChiralHamiltonian
from Library.Hamiltonian.ChiralHamiltonianEffective import ChiralHamiltonianEffective

def main():
    # Parameters
    n = 5
    vF = 542.1
    t1 = 355.16
    V = 10.0
    
    # Initialize Hamiltonians
    # 10x10 Chiral Hamiltonian
    H_full = ChiralHamiltonian(n=n, vF=vF, t1=t1, V=V)
    
    # 2x2 Effective Hamiltonian (Schrieffer-Wolff)
    H_eff = ChiralHamiltonianEffective(n=n, vF=vF, t1=t1, V=V)
    
    # k-points setup (1D cut along kx, ky=0)
    # Range: The projected model is low-energy, so we focus on small k.
    # beta = vF*k/t1 < 1 is required for validity of expansion, but let's go a bit beyond to see divergence if any.
    # vF/t1 ~ 1.5. So k should be < 1/1.5 ~ 0.66.
    k_range = 1
    kx_vals = np.linspace(-k_range, k_range, 200)
    ky = 0.0
    
    # Storage for eigenvalues
    # Full Hamiltonian has 2*n = 10 bands
    evals_full = np.zeros((len(kx_vals), 2*n))
    
    # Effective Hamiltonian has 2 bands
    evals_eff = np.zeros((len(kx_vals), 2))
    
    print("Calculating eigenvalues...")
    
    for i, kx in enumerate(kx_vals):
        # 1. Full Hamiltonian 10x10
        H_mat = H_full.compute_static(kx, ky)
        e_full = np.linalg.eigvalsh(H_mat)
        evals_full[i, :] = np.sort(e_full) # Sort to ensure consistent band ordering
        
        # 2. Effective Hamiltonian 2x2
        # We use compute_static which performs the downfolding numerically
        H_eff_mat = H_eff.compute_static(kx, ky)
        # Note: compute_static returns 2x2 matrix
        e_eff = np.linalg.eigvalsh(H_eff_mat)
        evals_eff[i, :] = np.sort(e_eff)

    print("Calculation done. Plotting...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Calculate divergence limit
    k_lim = t1 / vF
    print(f"Adding vertical lines at k = +/- {k_lim:.4f} (vF*k/t1 = 1)")

    # Plot 1: 10x10 Bands
    ax1 = axes[0]
    for b in range(2*n):
        ax1.plot(kx_vals, evals_full[:, b], color='blue', linewidth=1.5, alpha=0.7)
    # Add vertical lines
    ax1.axvline(x=k_lim, color='k', linestyle=':', label='vF k / t1 = 1')
    ax1.axvline(x=-k_lim, color='k', linestyle=':')
    
    ax1.set_title(f'Full Chiral Hamiltonian ({2*n}x{2*n})')
    ax1.set_xlabel('$k_x$')
    ax1.set_ylabel('Energy (meV)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 2x2 Bands
    ax2 = axes[1]
    # Effective bands
    ax2.plot(kx_vals, evals_eff[:, 0], color='red', linewidth=2, label='Effective -')
    ax2.plot(kx_vals, evals_eff[:, 1], color='red', linewidth=2, linestyle='--', label='Effective +')
    # Add vertical lines
    ax2.axvline(x=k_lim, color='k', linestyle=':', label='Limit')
    ax2.axvline(x=-k_lim, color='k', linestyle=':')
    
    ax2.set_title('Effective Hamiltonian (2x2, SW)')
    ax2.set_xlabel('$k_x$')
    ax2.set_ylabel('Energy (meV)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Comparison
    ax3 = axes[2]
    
    # Plot full bands (background)
    for b in range(2*n):
        # Highlight the middle two bands which correspond to low energy
        if b == n-1 or b == n: # 0-indexed, so 4 and 5 for n=5
             ax3.plot(kx_vals, evals_full[:, b], color='blue', linewidth=2, alpha=0.6, label='Full' if b==n else "")
        else:
             ax3.plot(kx_vals, evals_full[:, b], color='gray', linewidth=0.5, alpha=0.3)
             
    # Plot effective bands (foreground)
    ax3.plot(kx_vals, evals_eff[:, 0], color='red', linewidth=1.5, linestyle='--', label='Eff')
    ax3.plot(kx_vals, evals_eff[:, 1], color='red', linewidth=1.5, linestyle='--')
    
    # Add vertical lines
    ax3.axvline(x=k_lim, color='k', linestyle=':', label='Limit')
    ax3.axvline(x=-k_lim, color='k', linestyle=':')
    
    ax3.set_title('Comparison (Overlay)')
    ax3.set_xlabel('$k_x$')
    ax3.set_ylabel('Energy (meV)')
    
    # Zoom in on the low energy part for comparison
    # The full bandwidth is large, but effective is only valid for low energy.
    ymax = np.max(np.abs(evals_eff)) * 1.5
    # Handle NaNs in evals_eff for ylim determination
    if np.all(np.isnan(ymax)):
         pass # Don't set ylim if all NaNs
    else:
         # Filter out ridiculously large values if divergence happened
         ax3.set_ylim(-200, 200) # Manual limit or smart limit?
         # User asked for vertical lines where it diverges, so let's try not to clamp too hard, 
         # but usually spectral plots have reasonable window. 
         # Let's keep the dynamic limit but guarded.
         pass
         
    # Let's use a fixed reasonable range or 1.5x of Full Hamiltonian low energy bands?
    # For n=5, V=30, typical energy scale is ~10-100 meV.
    ax3.set_ylim(-150, 150)

    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
