import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to allow importing from Library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Library.Hamiltonian.ChiralHamiltonian import ChiralHamiltonian
from Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected import ChiralHamiltonianChiralBasisProjected

def main():
    # Parameters
    n = 5
    vF = 542.1
    t1 = 355.16
    V = 10.0
    A0 = 0.1
    
    # Frequencies to sweep
    omega_vals = [50.0, 100.0, 200.0, 500.0] 
    
    # k-points setup (1D cut along kx, ky=0)
    k_range = 1.0 # Go a bit beyond the limit to see
    kx_vals = np.linspace(-k_range, k_range, 200)
    ky = 0.0
    
    # Storage
    # List of (omega, evals_full, evals_proj)
    results = []
    
    print(f"Sweeping frequencies: {omega_vals}")
    
    for omega in omega_vals:
        print(f"Processing omega = {omega}...")
        
        # 1. Full Hamiltonian (Numerical Magnus)
        # Use magnus_order=1 to get 1st order effective Hamiltonian
        H_full_obj = ChiralHamiltonian(n=n, vF=vF, t1=t1, V=V, omega=omega, A0=A0, magnus_order=1)
        
        # 2. Projected Hamiltonian (Analytic Magnus)
        H_proj_obj = ChiralHamiltonianChiralBasisProjected(n=n, vF=vF, t1=t1, V=V, omega=omega, A0=A0)
        
        evals_full_list = []
        evals_proj_list = []
        
        for kx in kx_vals:
            # Full H: Effective H from base class (H0 + H_Magnus)
            # The base class method effective_hamiltonian returns (H_eff, H_prime)
            H_eff_full, _ = H_full_obj.effective_hamiltonian(kx, ky)
            e_full = np.linalg.eigvalsh(H_eff_full)
            evals_full_list.append(np.sort(e_full))
            
            # Projected H:
            # H_eff = H_static + H_magnus_analytic
            H_static = H_proj_obj.compute_static(kx, ky) # 2x2
            
            H_magnus = H_proj_obj.analytic_magnus_first_term(kx, ky)
            
            H_eff_proj = H_static + H_magnus
            e_proj = np.linalg.eigvalsh(H_eff_proj)
            evals_proj_list.append(np.sort(e_proj))
            
        results.append({
            'omega': omega,
            'full': np.array(evals_full_list),
            'proj': np.array(evals_proj_list)
        })

    print("Calculation done. Plotting...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colors for frequencies
    colors = plt.cm.viridis(np.linspace(0, 1, len(omega_vals)))
    
    # Calculate divergence limit
    k_lim = t1 / vF
    
    # Plot 1: Full Bands (Low Energy Focus)
    ax1 = axes[0]
    for idx, res in enumerate(results):
        omega = res['omega']
        evals = res['full']
        c = colors[idx]
        
        # Only plot the middle bands (low energy)
        for b in range(2*n):
            is_low_energy = (b == n-1 or b == n)
            lw = 1.5 if is_low_energy else 0.5
            alpha = 0.8 if is_low_energy else 0.2
            # Only label one band per frequency to avoid clutter
            if b == n:
                 ax1.plot(kx_vals, evals[:, b], color=c, linewidth=lw, alpha=alpha, label=f"$\omega={omega}$")
            elif b == n-1:
                 ax1.plot(kx_vals, evals[:, b], color=c, linewidth=lw, alpha=alpha)
            else:
                 ax1.plot(kx_vals, evals[:, b], color=c, linewidth=lw, alpha=alpha)

    ax1.axvline(x=k_lim, color='k', linestyle=':', alpha=0.5)
    ax1.axvline(x=-k_lim, color='k', linestyle=':', alpha=0.5)
    ax1.set_title(f'Full Hamiltonian (Magnus 1st Order, {2*n}x{2*n})')
    ax1.set_xlabel('$k_x$')
    ax1.set_ylabel('Energy (meV)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-100, 100) # Zoom in
    
    # Plot 2: Projected Bands
    ax2 = axes[1]
    for idx, res in enumerate(results):
        omega = res['omega']
        evals = res['proj']
        c = colors[idx]
        
        ax2.plot(kx_vals, evals[:, 0], color=c, linewidth=2, label=f"$\omega={omega}$")
        ax2.plot(kx_vals, evals[:, 1], color=c, linewidth=2, linestyle='--')

    ax2.axvline(x=k_lim, color='k', linestyle=':', alpha=0.5)
    ax2.axvline(x=-k_lim, color='k', linestyle=':', alpha=0.5)
    ax2.set_title('Projected Hamiltonian (Analytic Magnus)')
    ax2.set_xlabel('$k_x$')
    ax2.set_ylabel('Energy (meV)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Comparison Overlay
    ax3 = axes[2]
    for idx, res in enumerate(results):
        omega = res['omega']
        evals_f = res['full']
        evals_p = res['proj']
        c = colors[idx]
        
        # Plot Full Low Energy (Solid)
        ax3.plot(kx_vals, evals_f[:, n-1], color=c, linewidth=1.5, alpha=0.4)
        ax3.plot(kx_vals, evals_f[:, n], color=c, linewidth=1.5, alpha=0.4)
        
        # Plot Projected (Dashed)
        ax3.plot(kx_vals, evals_p[:, 0], color=c, linewidth=2, linestyle=':', label=f"Proj $\omega={omega}$")
        ax3.plot(kx_vals, evals_p[:, 1], color=c, linewidth=2, linestyle=':')
    
    ax3.axvline(x=k_lim, color='k', linestyle=':', alpha=0.5)
    ax3.axvline(x=-k_lim, color='k', linestyle=':', alpha=0.5)
    ax3.set_title('Comparison (Overlay)')
    ax3.set_xlabel('$k_x$')
    ax3.set_ylabel('Energy (meV)')
    ax3.set_ylim(-100, 100)
    ax3.legend(fontsize='small')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "Frequency_Sweep_Full_vs_2x2.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
if __name__ == "__main__":
    main()
