import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
import numpy as np
from Library.plotting_lib_3d import plot_isosurface, plot_slice_stack
import sys

def plot_3d_qgt_slices(results_dir, quantity, component="xy", slice_plane="xy",
                       n_slices=1, include_endpoints=True):
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    # --- load meta ---
    meta_file = os.path.join(results_dir, "meta_info.pkl")
    if not os.path.exists(meta_file):
        meta_file = os.path.join(results_dir, "qgt_meta_info.pkl")
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file not found in {results_dir}")
        return

    print(f"Loading data from {results_dir}...")
    with open(meta_file, "rb") as f:
        meta_info = pickle.load(f)

    kx_vals = meta_info["kx_vals"]
    ky_vals = meta_info["ky_vals"]
    kz_vals = meta_info["kz_vals"]

    hamiltonian = meta_info.get("Hamiltonian_Obj")
    hamiltonian_name = hamiltonian.name if hamiltonian else "Hamiltonian"

    # --- helpers ---
    def load_arr(name):
        path = os.path.join(results_dir, f"{name}.npy")
        return np.load(path) if os.path.exists(path) else None

    def load_component_real(comp):
        # try common naming conventions for REAL part
        for nm in (f"g_{comp}", f"g_{comp}_real", f"g_{comp}_re"):
            arr = load_arr(nm)
            if arr is not None:
                return arr
        return None

    def load_component_imag(comp):
        # your files appear to be named like g_xy_imag
        return load_arr(f"g_{comp}_imag")

    comp = component.lower()
    if len(comp) != 2 or comp[0] not in "xyz" or comp[1] not in "xyz" or comp[0] == comp[1]:
        print(f"Error: component must be one of 'xy','xz','yz' (got '{component}')")
        return

    data_3d = None
    title_base = ""

    if quantity == "berry":
        gij_im = load_component_imag(comp)
        if gij_im is None:
            print(f"Error: Could not load g_{comp}_imag.npy in {results_dir}")
            return

        # Ω_ij = -2 Im g_ij  (your prior code used a different "vector + magnitude" thing)
        data_3d = -2.0 * gij_im
        title_base = f"{hamiltonian_name} Berry Curvature Ω_{comp}"

    elif quantity == "metric":
        gij_re = load_component_real(comp)
        if gij_re is None:
            print(f"Error: Could not load real metric component for g_{comp}. "
                  f"Tried g_{comp}.npy, g_{comp}_real.npy, g_{comp}_re.npy")
            return

        data_3d = gij_re
        title_base = f"{hamiltonian_name} Metric g_{comp}"

    else:
        print(f"Error: Unknown quantity '{quantity}'. Use 'metric' or 'berry'.")
        return

    # your existing plotting function
    plot_slice_stack(
        data_3d, kx_vals, ky_vals, kz_vals,
        plane=slice_plane,
        n_slices=n_slices,
        include_endpoints=include_endpoints,
        title=title_base
    )




def plot_3d_qgt(results_dir, plane, quantity, min_val, max_val, count=5):
    """
    Plot 3D QGT isosurfaces based on plane and quantity.
    
    Parameters:
    - results_dir: Path to the directory containing QGT results.
    - plane: 'xy', 'yz', or 'xz'.
    - quantity: 'metric' (Real symmetric part) or 'berry' (Imaginary antisymmetric part).
    - min_val, max_val: Range of values for isosurfaces.
    - count: Number of isosurfaces to plot in the range.
    """
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Check for meta info
    meta_file = os.path.join(results_dir, "meta_info.pkl")
    if not os.path.exists(meta_file):
        meta_file = os.path.join(results_dir, "qgt_meta_info.pkl")
        
    if not os.path.exists(meta_file):
        print(f"Error: Metadata file not found in {results_dir}")
        return

    print(f"Loading data from {results_dir}...")
    with open(meta_file, "rb") as f:
        meta_info = pickle.load(f)
        
    kx_vals = meta_info["kx_vals"]
    ky_vals = meta_info["ky_vals"]
    kz_vals = meta_info["kz_vals"]
    hamiltonian = meta_info.get("Hamiltonian_Obj")
    if hamiltonian is None:
        hamiltonian_name = "Hamiltonian"
    else:
        hamiltonian_name = hamiltonian.name
    
    # Determine which components
    if plane == 'xy':
        comp_11_name = "g_xx"
        comp_22_name = "g_yy"
        comp_12_imag_name = "g_xy_imag"
    elif plane == 'yz':
        comp_11_name = "g_yy"
        comp_22_name = "g_zz"
        comp_12_imag_name = "g_yz_imag"
    elif plane == 'xz':
        comp_11_name = "g_xx"
        comp_22_name = "g_zz"
        comp_12_imag_name = "g_xz_imag"
    else:
        print(f"Error: Unknown plane '{plane}'. Use 'xy', 'yz', or 'xz'.")
        return

    # Load required arrays
    def load_arr(name):
        path = os.path.join(results_dir, f"{name}.npy")
        if not os.path.exists(path):
             print(f"Error: File {name}.npy not found in {results_dir}")
             return None
        return np.load(path)

    if quantity == 'metric':
        val_11 = load_arr(comp_11_name)
        val_22 = load_arr(comp_22_name)
        if val_11 is None or val_22 is None: return
        data_3d = val_11 + val_22
        title_base = f"{hamiltonian_name} Metric Trace ({plane})"
        
    elif quantity == 'berry':
        val_12_imag = load_arr(comp_12_imag_name)
        if val_12_imag is None: return
        data_3d = -2.0 * val_12_imag
        title_base = f"{hamiltonian_name} Berry Curvature ({plane})"
        
    else:
        print(f"Error: Unknown quantity '{quantity}'. Use 'metric' or 'berry'.")
        return


    levels = np.linspace(min_val, max_val, count)
    
    print(f"Plotting {quantity} on {plane} plane over 3D Volume.")
    print(f"Data Range: [{np.min(data_3d):.4f}, {np.max(data_3d):.4f}]")
    print(f"Target Levels: {levels}")
    
    # Create Figure Once
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for the different levels
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    
    for level in levels:
        # Check if level is within range of data, otherwise marching cubes fails
        if level < np.min(data_3d) or level > np.max(data_3d):
            print(f"Skipping level {level:.3f} (out of data range)")
            continue
            
        color = cmap(norm(level))
        print(f"  - plotting level {level:.3f}")
        plot_isosurface(data_3d, level, kx_vals, ky_vals, kz_vals, 
                        title=None, step_size=1, ax=ax, color=color, alpha=0.3)
                        
    ax.set_title(f"{title_base} Isosurfaces\nRange: [{min_val}, {max_val}]")
    
    # Add scalar mappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=f"{quantity} value", shrink=0.7)
    plt.show() 

if __name__ == "__main__":
    # Example usage as requested
    
    # Using the latest result directory found
    # base_results_path = "/Users/home/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/3D_QGT_results/RuO2Hamiltonian"

    # base_results_path = "/Users/home/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/3D_QGT_results/AltermagnetHamiltonian"

    base_results_path = "/Users/home/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/3D_QGT_results/gWaveAltermagnetHamiltonian"
    
    # Placeholder for the specific dataset, using one found in list_dir
    latest_dataset = "data_set_1"
    
    results_dir = os.path.join(base_results_path, latest_dataset)

    print(f"Running 3D QGT Slice Plot on: {results_dir}")
    plot_3d_qgt_slices(results_dir=results_dir, quantity='berry', component='xy', slice_plane="xy", n_slices=3)
