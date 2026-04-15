import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def dynamic_sym_nd(bundle_path, bands=None):
    if not os.path.exists(bundle_path):
        print(f"Error: {bundle_path} does not exist.")
        return
    
    print(f"Loading data from {bundle_path}...")
    data = np.load(bundle_path, allow_pickle=True)
    names = data['names']
    shape = data['shape']
    k_dist = data['k_dist']
    node_indices = data['node_indices']
    path_labels = data['path_labels']
    eigenvalues_grid = data['eigenvalues_grid']
    
    axes_vals = [data[f"axis_{i}_{name}"] for i, name in enumerate(names)]
    
    # Identify variables that actually change (where n > 1)
    active_axes_idx = [i for i, val in enumerate(axes_vals) if len(val) > 1]
    num_active = len(active_axes_idx)
    
    dim = eigenvalues_grid.shape[-1]
    if bands is None:
        bands = list(range(dim))
        
    gmin = np.min(eigenvalues_grid[..., bands])
    gmax = np.max(eigenvalues_grid[..., bands])
    margin = (gmax - gmin) * 0.05 if (gmax - gmin) != 0 else 0.1
    
    # Initial indices (all 0)
    current_indices = [0] * len(names)
    
    # Setup Figure and layout space for sliders
    slider_spacing = 0.04
    slider_height_total = min(num_active * slider_spacing, 0.5) # limit max space so plot doesn't shrink completely
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.subplots_adjust(bottom=0.10 + slider_height_total, top=0.92, left=0.08, right=0.95)
    
    # Determine segment colors
    labels_clean = [l.decode('utf-8') if isinstance(l, bytes) else str(l) for l in path_labels]
    
    top_points = set(['A', 'L', 'H', 'B'])
    mid_points = set(['D', 'U', 'P', 'C'])
    bot_points = set(['G', 'M', 'K', 'E'])
    
    vert_groups = [
        set(['G', 'A', 'D']),
        set(['M', 'L', 'U']),
        set(['K', 'H', 'P']),
        set(['B', 'C', 'E'])
    ]

    segment_colors = []
    for i in range(len(node_indices) - 1):
        p1 = labels_clean[i]
        p2 = labels_clean[i+1]
        
        if p1 in top_points and p2 in top_points:
            color = 'red'
        elif p1 in mid_points and p2 in mid_points:
            color = 'blue'
        elif p1 in bot_points and p2 in bot_points:
            color = 'green'
        elif any(p1 in g and p2 in g for g in vert_groups):
            color = 'purple'
        else:
            color = 'black'
        segment_colors.append(color)
        
    # Plot initial bands
    lines = []
    initial_eigs = eigenvalues_grid[tuple(current_indices)]
    
    for b in bands:
        l, = ax.plot(k_dist, initial_eigs[:, b], label=f"Band {b}")
        lines.append(l)
        
    ax.set_xticks(k_dist[node_indices])
    ax.set_xticklabels(labels_clean)
    
    # Draw colored bar along the x-axis
    for i in range(len(node_indices) - 1):
        start_k = k_dist[node_indices[i]]
        end_k   = k_dist[node_indices[i+1]]
        color   = segment_colors[i]
        
        # Draw a thick line right at the bottom (y=0 in axes coordinates)
        ax.plot([start_k, end_k], [0, 0], color=color, linewidth=6, 
                transform=ax.get_xaxis_transform(), clip_on=False)
    
    for k in k_dist[node_indices]:
        ax.axvline(x=k, color='grey', linestyle='--', alpha=0.3)
        
    ax.set_ylabel("Energy")
    ax.margins(x=0)
    ax.set_ylim(gmin - margin, gmax + margin)
    
    # Helper to construct title
    def get_title(indices):
        title_parts = []
        for count, idx in enumerate(active_axes_idx):
            name = names[idx]
            val = axes_vals[idx][indices[idx]]
            title_parts.append(f"{name}={val:.3f}")
        
        return " | ".join(title_parts) if title_parts else "Static Parameters"
        
    ax.set_title(get_title(current_indices))
    
    # Create Sliders dynamically
    sliders = []
    
    for count, idx in enumerate(active_axes_idx):
        name = names[idx]
        vals = axes_vals[idx]
        
        # Calculate Y position dynamically
        y_pos = 0.05 + slider_spacing * (num_active - 1 - count)
        
        ax_slider = plt.axes([0.15, y_pos, 0.65, 0.025])
        
        slider = Slider(
            ax=ax_slider,
            label=f"{name}",
            valmin=0,
            valmax=len(vals) - 1,
            valinit=0,
            valstep=1,
            valfmt="%d"  # shows index 
        )
        
        # Format the numbers manually to show the actual value rather than index
        slider.valtext.set_text(f"{vals[0]:.3f}")
        
        def update(val, i=idx, s=slider):
            # Update index
            current_idx = int(s.val)
            current_indices[i] = current_idx
            
            # Update slider text to show the actual parameter value
            actual_val = axes_vals[i][current_idx]
            s.valtext.set_text(f"{actual_val:.3f}")
            
            # Update plot
            new_eigs = eigenvalues_grid[tuple(current_indices)]
            for b_idx, b in enumerate(bands):
                lines[b_idx].set_ydata(new_eigs[:, b])
            
            # Update Title
            ax.set_title(get_title(current_indices))
            fig.canvas.draw_idle()
            
        slider.on_changed(update)
        sliders.append(slider)

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        bundle_path = sys.argv[1]
    else:
        # Default placeholder pointing to the latest calculation
        base = os.path.join(os.getcwd(), "results", "Sym_Phase_Diagram")
        # bundle_path = os.path.join(base, "gWaveAltermagnetHamiltonian_Sym_Jx_0.0to1_Jy_0.0to1_Jz_0.0to1_lamb_0.0to0.3_lamb_z_0.0to0.3_1", "sym_nd_bundle.npz")
        bundle_path = os.path.join(base, "gWaveAltermagnetHamiltonian_Sym_Jx_0.0to1_Jy_0.0to1_Jz_0.0to1_lamb_0.0to0.3_lamb_z_0.0to0.3_t1_0.0to0.3_t2_0.0to0.3_t3_0.0to0.3_t4_0.0to0.3", "sym_nd_bundle.npz")
    dynamic_sym_nd(bundle_path)
