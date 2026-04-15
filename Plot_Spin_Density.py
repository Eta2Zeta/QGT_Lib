import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_spin_component_2x2(kx, ky, spin_data, component_name, results_dir=None):
    """
    Plots a 2x2 grid of the spin density for the 4 bands given a specific spin component array.
    """
    n_bands = spin_data.shape[-1]
    
    if n_bands != 4:
        print(f"Warning: Expected 4 bands to fit a 2x2 grid, but found {n_bands}.")
    
    rows, cols = 2, 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    fig.suptitle(f"{component_name} Expectation Value", fontsize=18, y=0.95)
    
    axes = axes.ravel()
    
    # Calculate global symmetric limits for to keep colors bound universally across bands
    vmax = np.nanmax(np.abs(spin_data))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0
    vmin = -vmax
    
    for b in range(min(n_bands, 4)):
        ax = axes[b]
        Z = spin_data[..., b]
        
        # Plot 2D color mesh
        im = ax.pcolormesh(kx, ky, Z, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
        
        # Formatting
        ax.set_title(f"Band {b+1}", fontsize=14)
        ax.set_xlabel("$k_x$", fontsize=12)
        ax.set_ylabel("$k_y$", fontsize=12)
        ax.set_aspect('equal')
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if results_dir:
        out_path = os.path.join(results_dir, f"{component_name}_density_grid.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved {component_name} to {out_path}")
    
    # Automatically display visually as well unless running headless
    # plt.show()
    plt.close()

def main():
    # Identify the target dataset automatically
    base_dir = os.path.join(os.getcwd(), "results", "Spin_Density_results", "gWaveAltermagnetHamiltonian")
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist. Please run Calc_Spin_Density.py first.")
        return
        
    datasets = [d for d in os.listdir(base_dir) if d.startswith("data_set_")]
    if not datasets:
        print("No datasets found inside Spin_Density_results.")
        return
    
    # Identify newest dataset generically by checking the folder numbering
    datasets.sort(key=lambda x: int(x.split('_')[-1]))
    latest_dataset = datasets[-1]
    
    dataset_dir = os.path.join(base_dir, latest_dataset)
    print(f"Loading data from {dataset_dir}...")
    
    # Load metadata to get exact dimensional k-mesh structures
    meta_pkl_path = os.path.join(dataset_dir, "meta_info.pkl")
    with open(meta_pkl_path, "rb") as f:
        meta = pickle.load(f)
        
    ki = meta["ki"]
    kj = meta["kj"]
    
    # Load array datasets for the explicit operators
    spin_x = np.load(os.path.join(dataset_dir, "spin_x.npy"))
    spin_y = np.load(os.path.join(dataset_dir, "spin_y.npy"))
    spin_z = np.load(os.path.join(dataset_dir, "spin_z.npy"))
    
    # Fire off plotting evaluations internally targeting dataset
    plot_spin_component_2x2(ki, kj, spin_x, "Spin_X", dataset_dir)
    plot_spin_component_2x2(ki, kj, spin_y, "Spin_Y", dataset_dir)
    plot_spin_component_2x2(ki, kj, spin_z, "Spin_Z", dataset_dir)
    
if __name__ == "__main__":
    main()
