
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Constants ---
v_F = 542.1        # Fermi velocity (v_p)
t_1 = 355.16       # Interlayer coupling
A_0 = 0.10         # Vector potential amplitude
omega = 1e2       # Frequency (approx 2*pi)
n = 5              # Number of layers
V_const = 30.0     # The scalar 'V' on the RHS of the equation

def compute_V_k(kx, ky):
    """
    Computes V(k) based on the formula:
    V(k) = V * [ -(n-1)/2 + ( (n-1)*beta^(2n+2) + beta^2 - n*beta^(2n) ) / ( (1-beta^2)*(1-beta^(2n)) ) ]
    where beta = (v_p * k) / t_1
    and k = sqrt(kx^2 + ky^2)
    """
    k = np.sqrt(kx**2 + ky**2)
    
    # beta = (v_p * k) / t_1
    beta = (v_F * k) / t_1
    
    # Avoid singularities where beta = 1 (or close to 1) 
    # Use a small epsilon or mask, but for now just let numpy handle infs/nans or use masked array
    # However, strictly speaking, we might want to mask points very close to beta=1 if denominator explodes.
    # The term (1 - beta^2) and (1 - beta^(2n)) are in denominator.
    
    # Term 2 numerator: (n-1)*beta^(2n+2) + beta^2 - n*beta^(2n)
    num = (n - 1) * (beta**(2*n + 2)) + (beta**2) - n * (beta**(2*n))
    
    # Term 2 denominator: (1 - beta^2) * (1 - beta^(2n))
    den = (1 - beta**2) * (1 - beta**(2*n))
    
    # Handle division by zero or numerical instability if needed
    # For plotting broad range, just set invalid values to nan
    with np.errstate(divide='ignore', invalid='ignore'):
        term2 = num / den
        
    term1 = -(n - 1) / 2.0
    
    # V(k)
    V_val = V_const * (term1 + term2)
    
    return V_val

def compute_constant_function():
    """
    Constant function: (v_F * A_0)^2 / omega
    """
    return (v_F * A_0)**2 / omega

def main():
    print(f"Parameters used:")
    print(f"v_F = {v_F}")
    print(f"t_1 = {t_1}")
    print(f"A_0 = {A_0}")
    print(f"omega = {omega}")
    print(f"n = {n}")
    print(f"V_const = {V_const}")
    
    # Grid setup
    # Plot on a 100x100 k grid with kx and ky
    # Assuming range -1 to 1 based on typical BZ usage in codebase (or -0.9 to 0.9)
    k_range = 1.0
    kx_vals = np.linspace(-k_range, k_range, 100)
    ky_vals = np.linspace(-k_range, k_range, 100)
    KX, KY = np.meshgrid(kx_vals, ky_vals)
    
    # 1. Constant function value
    C1 = compute_constant_function()
    Z1 = np.full_like(KX, C1)
    
    # 2. V(k)
    Z2 = compute_V_k(KX, KY)
    
    # Setup plot
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Constant function
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(KX, KY, Z1, color='b', alpha=0.6)
    ax1.set_title(r'Constant: $(v_F A_0)^2 / \omega$')
    ax1.set_xlabel('$k_x$')
    ax1.set_ylabel('$k_y$')
    
    # Plot 2: V(k)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    # Use a colormap for V(k) to see structure
    surf2 = ax2.plot_surface(KX, KY, Z2, cmap='viridis', alpha=0.8)
    ax2.set_title(r'$V(k)$')
    ax2.set_xlabel('$k_x$')
    ax2.set_ylabel('$k_y$')
    
    # Plot 3: Both
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    # Plot constant as a wireframe or transparent surface
    ax3.plot_surface(KX, KY, Z1, color='b', alpha=0.3, label='Constant')
    # Plot V(k) as solid
    ax3.plot_surface(KX, KY, Z2, cmap='viridis', alpha=0.8, label='V(k)')
    
    ax3.set_title('Comparison')
    ax3.set_xlabel('$k_x$')
    ax3.set_ylabel('$k_y$')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
