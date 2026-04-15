import os
import pickle
import numpy as np
import plotly.graph_objects as go
from Library.utilities import generate_3d_sym_lines
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_3d_degeneracies(results_dir, threshold=0.02, stride=1, point_opacity=0.5):
    """
    Plots the 3D map of band degeneracies (crossings) using Plotly and saves as HTML.
    
    Parameters:
    - results_dir: Directory containing 'eigenvalues_3d.npy' and 'meta_info.pkl'
    - threshold: Relative gap threshold to consider bands degenerate.
    - stride: Downsampling stride if there are too many points to plot.
    - point_opacity: Opacity of the 3D scatter points.
    """
    if not os.path.exists(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        return

    # Load meta
    meta_file = os.path.join(results_dir, "meta_info.pkl")
    if not os.path.exists(meta_file):
        print(f"Error: {meta_file} not found in {results_dir}")
        return

    print(f"Loading metadata from {meta_file}...")
    with open(meta_file, "rb") as f:
        meta_info = pickle.load(f)

    kx_vals = meta_info["kx_vals"]
    ky_vals = meta_info["ky_vals"]
    kz_vals = meta_info["kz_vals"]
    hamiltonian = meta_info.get("Hamiltonian_Obj", None)
    hamiltonian_name = getattr(hamiltonian, "name", "Hamiltonian")

    # Load eigenvalues_3d
    eig_file = os.path.join(results_dir, "eigenvalues_3d.npy")
    if not os.path.exists(eig_file):
        print(f"Error: {eig_file} not found in {results_dir}")
        return

    print(f"Loading 3D eigenvalues from {eig_file}...")
    eigenvalues = np.load(eig_file) # Expected shape: [nkx, nky, nkz, nbands]
    
    if eigenvalues.ndim != 4:
        print(f"Error: Expected 4D array [nkx, nky, nkz, nbands], but got {eigenvalues.shape}")
        return

    print("Calculating 3D degeneracy map (vectorized)...")
    
    # Vectorized computation matching the 2D logic exactly for high performance
    evals = np.sort(eigenvalues, axis=-1)
    diffs = np.diff(evals, axis=-1)
    
    if diffs.shape[-1] > 0:
        max_gap = np.max(diffs, axis=-1, keepdims=True)
        current_thresh = np.maximum(threshold * max_gap, 1e-10)
        degeneracy_map = np.sum(diffs <= current_thresh, axis=-1)
    else:
        degeneracy_map = np.zeros(evals.shape[:-1], dtype=int)

    print("Generating Plotly 3D scatter plot...")
    
    # Grid coordinates
    X, Y, Z = np.meshgrid(kx_vals, ky_vals, kz_vals, indexing='ij')
    
    mask = degeneracy_map > 0
    
    x_coords = X[mask][::stride]
    y_coords = Y[mask][::stride]
    z_coords = Z[mask][::stride]
    deg_vals = degeneracy_map[mask][::stride]
    
    if len(deg_vals) == 0:
        print(f"No degeneracies found at the given threshold={threshold}.")
        return
        
    print(f"Plotting {len(x_coords)} degenerate points...")
    
    cmap = plt.get_cmap('tab10')
    
    data = []
    unique_degs = np.unique(deg_vals)
    for i, d in enumerate(unique_degs):
        idx = (deg_vals == d)
        color = mcolors.to_hex(cmap(i % 10))
        
        # Actual 3D data points (hidden from legend)
        data.append(go.Scatter3d(
            x=x_coords[idx],
            y=y_coords[idx],
            z=z_coords[idx],
            mode='markers',
            name=f"Degeneracy {d}",
            showlegend=False,
            legendgroup=f"group_{d}",
            marker=dict(
                size=3,
                color=color,
                opacity=point_opacity
            ),
            text=[f"Degeneracy Level: {d}"] * np.sum(idx),
            hoverinfo="x+y+z+text"
        ))
        
        # Dummy 2D trace for square legend item
        data.append(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            name=f"Degeneracy {d}",
            showlegend=True,
            legendgroup=f"group_{d}",
            marker=dict(
                size=15,
                symbol='square',
                color=color,
            )
        ))
        
    # Add Symmetry Lines
    print("Generating High Symmetry Lines...")
    k_path, _, _, path_labels, path_points_arr = generate_3d_sym_lines(num_points_per_segment=2) # low segment count is fine for straight lines
    data.append(go.Scatter3d(
        x=k_path[:, 0],
        y=k_path[:, 1],
        z=k_path[:, 2],
        mode='lines',
        name='Symmetry Path',
        line=dict(color='black', width=4),
        showlegend=True,
    ))
    
    # Optionally mark the symmetry nodes themselves
    data.append(go.Scatter3d(
        x=path_points_arr[:, 0],
        y=path_points_arr[:, 1],
        z=path_points_arr[:, 2],
        mode='markers+text',
        name='Symmetry Nodes',
        marker=dict(size=4, color='black'),
        text=path_labels,
        textposition="top center",
        showlegend=False,
    ))
        
    fig = go.Figure(data=data)
    
    title = f"{hamiltonian_name} 3D Band Degeneracies (Threshold: {threshold})"
    fig.update_layout(
        title=title,
        showlegend=True,
        legend_title_text='Degeneracy Count:',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        scene=dict(
            xaxis_title='kx',
            yaxis_title='ky',
            zaxis_title='kz',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fname = f"{hamiltonian_name}_3D_Degeneracy_Map_thresh{threshold:g}_stride{stride}.html".replace(" ", "_").replace("/", "_")
    out_file = os.path.join(results_dir, fname)
    
    fig.write_html(out_file, include_plotlyjs='cdn')
    print(f"\n✅ Saved interactive degeneracy map to:\n   {out_file}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change the target Hamiltonian name below if you switch models.
    base_results_path = os.path.join(current_dir, "results/3D_Eigen_results/gWaveAltermagnetHamiltonian")
    
    # Specify specific dataset number (e.g., 15) or None for latest
    target_dataset_num = None
    
    if os.path.exists(base_results_path):
        datasets = [d for d in os.listdir(base_results_path) if d.startswith("data_set_")]
        if datasets:
            if target_dataset_num is not None:
                dataset_name = f"data_set_{target_dataset_num}"
                if dataset_name in datasets:
                    results_dir = os.path.join(base_results_path, dataset_name)
                    print(f"Processing specific dataset: {results_dir}")
                else:
                    print(f"Error: {dataset_name} not found in {base_results_path}")
                    exit(1)
            else:
                # Pick the most recent dataset index based on integer parsing
                latest_dataset = sorted(datasets, key=lambda x: int(x.split('_')[-1]))[-1]
                results_dir = os.path.join(base_results_path, latest_dataset)
                print(f"Processing latest dataset: {results_dir}")
                
            plot_3d_degeneracies(results_dir, threshold=0.02, stride=2)
        else:
            print(f"No datasets found in {base_results_path}")
    else:
        print(f"Directory {base_results_path} does not exist. Update the base_results_path in the script.")
