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
    else:
        out_path = None
    
    # Automatically display visually as well unless running headless
    # plt.show()
    plt.close()
    return out_path


def plot_all_spin_components(dataset_dir):
    """
    Load a spin-density dataset and save the standard spin-component plots.
    """
    print(f"Loading spin-density data from {dataset_dir}...")

    meta_pkl_path = os.path.join(dataset_dir, "meta_info.pkl")
    with open(meta_pkl_path, "rb") as f:
        meta = pickle.load(f)

    ki = meta["ki"]
    kj = meta["kj"]

    spin_components = {
        "Spin_X": np.load(os.path.join(dataset_dir, "spin_x.npy")),
        "Spin_Y": np.load(os.path.join(dataset_dir, "spin_y.npy")),
        "Spin_Z": np.load(os.path.join(dataset_dir, "spin_z.npy")),
    }

    output_paths = []
    for component_name, spin_data in spin_components.items():
        out_path = plot_spin_component_2x2(ki, kj, spin_data, component_name, dataset_dir)
        if out_path is not None:
            output_paths.append(out_path)

    return output_paths


def latest_spin_density_dataset(hamiltonian_name):
    base_dir = os.path.join(os.getcwd(), "results", "Spin_Density_results", hamiltonian_name)

    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist. Please run Calc_Spin_Density.py first.")
        return None

    datasets = [d for d in os.listdir(base_dir) if d.startswith("dataset_")]
    if not datasets:
        print("No datasets found inside Spin_Density_results.")
        return None

    datasets.sort(key=lambda x: int(x.split('_')[-1]))
    return os.path.join(base_dir, datasets[-1])

def main():
    dataset_dir = latest_spin_density_dataset("MinimalHamSG127_2a2b")
    if dataset_dir is None:
        return
    plot_all_spin_components(dataset_dir)
    
if __name__ == "__main__":
    main()
