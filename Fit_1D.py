
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

def power_law(x, a, b, c, d):
    # y = a * (x - c)^b + d
    # Safety: ensure base is positive. This is usually handled by bounds c < min(x), 
    # but for visualization/residuals we guard.
    base = x - c
    # If standard numpy power is used and base < 0 with float exp, it yields NaN.
    # We rely on optimizer bounds, but return NaNs if violated to penalize.
    return a * np.power(base, b) + d

def exponential(x, a, b, c, d):
    # y = a * exp(b * (x - c)) + d
    return a * np.exp(b * (x - c)) + d


def fit_and_plot(x_data, y_data, x_label, y_label, title, output_dir, file_prefix, power_law_bounds=None):
    """
    Fit multiple models to x_data, y_data.
    Plot data and fits.
    Save plot and print fit parameters.
    """
    # Remove NaNs or Infs
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_clean = x_data[mask]
    y_clean = y_data[mask]

    if len(x_clean) < 4:
        print(f"Not enough data points to fit for {title}")
        return

    # Sort for plotting lines
    sort_idx = np.argsort(x_clean)
    x_sorted = x_clean[sort_idx]
    y_sorted = y_clean[sort_idx]
    
    fit_results = {}

    min_x = np.min(x_sorted)
    max_x = np.max(x_sorted)

    # --- Power Law Fit ---
    # a*(x-c)^b + d
    try:
        # Initial guesses:
        # d (vertical shift) approx min(y)
        d_guess = np.min(y_sorted) - 1e-9
        # c (horizontal shift). Must be < min_x. Let's guess slightly left of min_x?
        # Or 0 if data is far from 0.
        c_guess = min_x - (max_x - min_x) * 0.1 

        # Check if user provided specific bounds
        if power_law_bounds:
            bounds = power_law_bounds
        else:
            # Default Bounds: 
            # a > 0
            # b < 0 (decay)
            # c < min_x (singularity avoidance). Strictly x-c > 0.
            # d > 0 (assuming positive measurements)
            # bounds = ([low_a, low_b, low_c, low_d], [high_a, high_b, high_c, high_d])
            bounds = ([0, -np.inf, -np.inf, 0], [np.inf, 0, min_x - 1e-9, np.inf])
        
        # Adjust initial guess for b
        # If user wants b > 1, guess b=2. If default (b<0), guess -1.
        if power_law_bounds and power_law_bounds[0][1] >= 1:
             b_guess = 2
        else:
             b_guess = -1
        
        p0_pow = [1, b_guess, c_guess, d_guess]
        
        popt, pcov = curve_fit(power_law, x_sorted, y_sorted, p0=p0_pow, 
                               bounds=bounds,
                               maxfev=10000)
        fit_results['Power Law'] = (popt, pcov, power_law, 
                                    f"$y = {popt[0]:.2e}(x - {popt[2]:.2e})^{{{popt[1]:.2f}}} + {popt[3]:.2e}$")
    except Exception as e:
        print(f"Power law fit failed: {e}")

    # --- Exponential Fit ---
    # a * exp(b*(x-c)) + d
    try:
        # Note: c is redundant with a in simple exp, but helps conditioning sometimes or interpretation.
        d_guess = np.min(y_sorted)
        c_guess = min_x
        # Bounds: a>0, b<0, c unconstrained (or near x range), d>0
        p0_exp = [np.max(y_sorted)-d_guess, -0.1, c_guess, d_guess]
        # bounds: a, b, c, d
        bounds_exp = ([0, -np.inf, -np.inf, 0], [np.inf, 0, np.inf, np.inf])
        
        popt, pcov = curve_fit(exponential, x_sorted, y_sorted, p0=p0_exp, 
                               bounds=bounds_exp,
                               maxfev=10000)
        fit_results['Exponential'] = (popt, pcov, exponential, 
                                      f"$y = {popt[0]:.2e}e^{{{popt[1]:.2f}(x - {popt[2]:.2e})}} + {popt[3]:.2e}$")
    except Exception: pass


    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x_sorted, y_sorted, 'ko', label='Data', markersize=4, alpha=0.6)

    # Create dense x-grid for smooth plotting
    x_dense = np.linspace(np.min(x_sorted), np.max(x_sorted), 1000)
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    for i, (name, res) in enumerate(fit_results.items()):
        popt, pcov, func, label_str = res
        try:
            # Evaluate on dense grid for smooth curve
            y_fit = func(x_dense, *popt)
            ls = '-' if name == 'Power Law' else '--'
            ax.plot(x_dense, y_fit, linestyle=ls, lw=2, color=colors[i % len(colors)], label=label_str)
        except Exception: 
            pass # func might fail if params bad

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)

    # Save fit plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{file_prefix}_fit.png")
    fig.savefig(save_path, dpi=200)
    print(f"Saved fit plot to: {save_path}")
    plt.close(fig)

    # Write stats
    txt_path = os.path.join(output_dir, f"{file_prefix}_fit_stats.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Fit Results for: {title}\n")
        f.write(f"Num points: {len(x_clean)}\n\n")
        
        for name, res in fit_results.items():
            popt, pcov, func, label_str = res
            f.write(f"--- {name} ---\n")
            f.write(f"  Params: {popt}\n")
            if pcov is not None:
                perr = np.sqrt(np.diag(pcov))
                f.write(f"  Errors: {perr}\n")
            # Calculate RMSE
            residuals = y_clean - func(x_clean, *popt)
            rmse = np.sqrt(np.mean(residuals**2))
            f.write(f"  RMSE: {rmse:.5e}\n\n")

def process_slice_data(
    input_npz_path,
    output_dir,
    # Criteria for truncation
    horz_x_min=None, horz_x_max=None,
    vert_x_min=None, vert_x_max=None,
    # Options to invert x axis (flip sign)
    invert_horz_x=False,
    invert_vert_x=False
):
    if not os.path.exists(input_npz_path):
        raise FileNotFoundError(f"Input file not found: {input_npz_path}")

    data = np.load(input_npz_path)

    # --- 1. Horizontal Slice (x axis is signed 1/omega) ---
    hx = data['horz_x']
    hy = data['horz_y']
    h_param_val = data['horz_param_val']
    h_param_name = str(data['horz_param_name'])
    quantity = str(data['quantity'])

    # Truncate
    mask_h = np.ones_like(hx, dtype=bool)
    if horz_x_min is not None: mask_h &= (hx >= horz_x_min)
    if horz_x_max is not None: mask_h &= (hx <= horz_x_max)
    
    hx_trunc = hx[mask_h]
    hy_trunc = hy[mask_h]

    # Invert AFTER truncation
    if invert_horz_x:
        hx_trunc = -hx_trunc
    hy_trunc = hy[mask_h]

    # Fit - Custom Bounds from User request:
    # Model: a(x-c)^b + d
    # Previous Req: b > 1, old_c (now d) in [5, 10]
    # New Bounds structure: [a, b, c, d]
    
    # We must enable c (horz shift) to range reasonably. 
    # Important: x-c > 0. So c < min(hx_trunc).
    min_h = np.min(hx_trunc) if len(hx_trunc) > 0 else 0
    
    # bounds = ( [low_a, low_b, low_c, low_d], [high_a, high_b, high_c, high_d] )
    h_bounds = ( 
        [0,       -np.inf,    -np.inf,    -np.inf], 
        [np.inf,  0, np.inf,  np.inf] 
    )

    fit_and_plot(
        hx_trunc, hy_trunc,
        x_label="signed 1/omega",
        y_label=f"Std[{quantity}]",
        title=f"Horizontal Slice (fixed {h_param_name} ~ {h_param_val:.3g})",
        output_dir=output_dir,
        file_prefix=f"horz_slice_{h_param_name}_{h_param_val:.2f}",
        power_law_bounds=h_bounds
    )

    # --- 2. Vertical Slice (x axis is V or other parameter) ---
    vx = data['vert_x']
    vy = data['vert_y']
    v_param_val = data['vert_param_val']  # this is the x-cut value (signed 1/w)
    v_omega_approx = data['vert_omega_approx']

    # Truncate
    mask_v = np.ones_like(vx, dtype=bool)
    if vert_x_min is not None: mask_v &= (vx >= vert_x_min)
    if vert_x_max is not None: mask_v &= (vx <= vert_x_max)

    vx_trunc = vx[mask_v]
    vy_trunc = vy[mask_v]

    if invert_vert_x:
        vx_trunc = -vx_trunc

    fit_and_plot(
        vx_trunc, vy_trunc,
        x_label=h_param_name, # Usually 'V'
        y_label=f"Std[{quantity}]",
        title=f"Vertical Slice (fixed omega ~ {v_omega_approx:.3g})",
        output_dir=output_dir,
        file_prefix=f"vert_slice_omega_{v_omega_approx:.1f}"
    )

if __name__ == "__main__":
    # --- Configuration ---
    # Path to where Plot_idealness.py exported the data
    input_data_path = "results/1d_idealness_resutls/slice_data_export.npz" 
    
    # Where to save fit results
    fit_results_dir = "results/1d_idealness_resutls"

    # Truncation settings (adjust as needed)
    # E.g. for horizontal (1/omega), maybe we only care about the small 1/omega region (high omega)
    # or exclude the singularity at 0.
    h_x_min = 0.00   # e.g. only positive side (right data)
    h_x_max = 0.0125   # e.g. up to some cutoff
    
    # For vertical (V), maybe fitting the tail?
    v_x_min = 15.0
    v_x_max = 50.0

    # Inversion settings
    invert_h = True
    invert_v = False

    print(f"Processing {input_data_path}...")
    try:
        process_slice_data(
            input_data_path,
            fit_results_dir,
            horz_x_min=h_x_min,
            horz_x_max=h_x_max,
            vert_x_min=v_x_min,
            vert_x_max=v_x_max,
            invert_horz_x=invert_h,
            invert_vert_x=invert_v
        )
        print("Done.")
    except FileNotFoundError as e:
        print(e)
        print("Make sure you ran Plot_idealness.py with export_slice_path set!")
