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
    V = 10.0 # Using V=10 as per user's latest manual change in the other script
    A0 = 0.1
    
    # Frequencies to sweep
    omega_vals = [50.0, 100.0, 200.0, 500.0] 
    
    # k-points setup (1D cut along kx, ky=0)
    k_range = 1.0 
    kx_vals = np.linspace(-k_range, k_range, 200)
    ky = 0.0
    
    # Storage
    # List of (omega, evals_full, evals_eff)
    results = []
    
    print(f"Sweeping frequencies: {omega_vals}")
    print(f"Parameters: n={n}, V={V}, A0={A0}")
    
    for omega in omega_vals:
        print(f"Processing omega = {omega}...")
        
        # 1. Full Hamiltonian (Numerical Magnus)
        # magnus_order=1 for 1st order effective H
        H_full_obj = ChiralHamiltonian(n=n, vF=vF, t1=t1, V=V, omega=omega, A0=A0, magnus_order=1)
        
        # 2. Effective Hamiltonian (Numerical SW projection of Driven Full)
        # We pass magnus_order=1 so the internal full hamiltonian gets the magnus term before projection
        H_eff_obj = ChiralHamiltonianEffective(n=n, vF=vF, t1=t1, V=V, omega=omega, A0=A0, magnus_order=1)
        
        evals_full_list = []
        evals_eff_list = []
        
        for kx in kx_vals:
            # Full H (10x10)
            # effective_hamiltonian returns (H_eff, H_prime)
            H_mat_full, _ = H_full_obj.effective_hamiltonian(kx, ky)
            e_full = np.linalg.eigvalsh(H_mat_full)
            evals_full_list.append(np.sort(e_full))
            
            # Effective H (2x2 SW)
            # Uses the newly added compute_effective_hamiltonian
            H_mat_eff = H_eff_obj.compute_effective_hamiltonian(kx, ky)
            e_eff = np.linalg.eigvalsh(H_mat_eff)
            evals_eff_list.append(np.sort(e_eff))
            
        results.append({
            'omega': omega,
            'full': np.array(evals_full_list),
            'eff': np.array(evals_eff_list)
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
        # For n, indices n-1 and n are the zero-energy crossing bands
        for b in range(2*n):
            is_low_energy = (b == n-1 or b == n)
            lw = 1.5 if is_low_energy else 0.5
            alpha = 0.8 if is_low_energy else 0.2
            # Only label one band per frequency to avoid clutter
            label = f"$\omega={omega}$" if (b==n and is_low_energy) else None
            
            if is_low_energy and b==n:
                 ax1.plot(kx_vals, evals[:, b], color=c, linewidth=lw, alpha=alpha, label=label)
            else:
                 ax1.plot(kx_vals, evals[:, b], color=c, linewidth=lw, alpha=alpha)

    ax1.axvline(x=k_lim, color='k', linestyle=':', alpha=0.5)
    ax1.axvline(x=-k_lim, color='k', linestyle=':', alpha=0.5)
    ax1.set_title(f'Full Hamiltonian (Magnus Order 1, {2*n}x{2*n})')
    ax1.set_xlabel('$k_x$')
    ax1.set_ylabel('Energy (meV)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-100, 100) # Zoom in
    
    # Plot 2: Effective Bands (SW)
    ax2 = axes[1]
    for idx, res in enumerate(results):
        omega = res['omega']
        evals = res['eff']
        c = colors[idx]
        
        ax2.plot(kx_vals, evals[:, 0], color=c, linewidth=2, label=f"$\omega={omega}$")
        ax2.plot(kx_vals, evals[:, 1], color=c, linewidth=2, linestyle='--')

    ax2.axvline(x=k_lim, color='k', linestyle=':', alpha=0.5)
    ax2.axvline(x=-k_lim, color='k', linestyle=':', alpha=0.5)
    ax2.set_title('Effective Hamiltonian (Numerical SW, 2x2)')
    ax2.set_xlabel('$k_x$')
    ax2.set_ylabel('Energy (meV)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Comparison Overlay
    ax3 = axes[2]
    for idx, res in enumerate(results):
        omega = res['omega']
        evals_f = res['full']
        evals_e = res['eff']
        c = colors[idx]
        
        # Plot Full Low Energy (Solid)
        ax3.plot(kx_vals, evals_f[:, n-1], color=c, linewidth=1.5, alpha=0.4)
        ax3.plot(kx_vals, evals_f[:, n], color=c, linewidth=1.5, alpha=0.4)
        
        # Plot Effective (Dashed)
        ax3.plot(kx_vals, evals_e[:, 0], color=c, linewidth=2, linestyle=':', label=f"Eff $\omega={omega}$")
        ax3.plot(kx_vals, evals_e[:, 1], color=c, linewidth=2, linestyle=':')
    
    ax3.axvline(x=k_lim, color='k', linestyle=':', alpha=0.5)
    ax3.axvline(x=-k_lim, color='k', linestyle=':', alpha=0.5)
    ax3.set_title('Comparison (Overlay)')
    ax3.set_xlabel('$k_x$')
    ax3.set_ylabel('Energy (meV)')
    ax3.set_ylim(-100, 100)
    ax3.legend(fontsize='small')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
