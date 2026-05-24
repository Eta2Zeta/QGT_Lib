import os
import numpy as np
import pickle

import plotly.graph_objects as go
import plotly.offline as pyo
import json
import pickle
import sys
import Library.Hamiltonian.Hamiltonian
import Library.Hamiltonian.ChiralHamiltonian

# Patch for unpickling old data that references Library.Hamiltonian_v2 (direct module)
sys.modules["Library.Hamiltonian_v2"] = Library.Hamiltonian.Hamiltonian

# Also ensure ChiralHamiltonian is available in Hamiltonian_v2 for unpickling
try:
    Library.Hamiltonian.Hamiltonian.ChiralHamiltonian = Library.Hamiltonian.ChiralHamiltonian.ChiralHamiltonian
except AttributeError:
    pass


def _save_dynamic_qgt_plotly(
    filepath, 
    k_line, 
    g_results, 
    bands, 
    y_bounds, 
    param_name="omega",
    title="QGT Dynamic Plot"
):
    """
    Helper to create a Plotly figure with frames and a slider, then save to HTML.
    """
    fig = go.Figure()

    # Determine availability of Magnus
    has_magnus = 'magnus_operator_norm' in g_results[0]
    
    # 1. Base traces (Initial state)
    initial_data = g_results[0]
    ev0 = np.array(initial_data['eigenvalues'])
    if ev0.ndim != 2: ev0 = ev0.reshape(ev0.shape[0], -1)

    # Eigenvalues
    for b in bands:
        fig.add_trace(go.Scatter(
            x=k_line, y=ev0[:, b], name=f'Band {b}', 
            mode='lines', yaxis="y1"
        ))
    # Trace
    fig.add_trace(go.Scatter(
        x=k_line, y=initial_data['trace'], name='Trace', 
        mode='lines', line=dict(dash='dash', color='blue'), yaxis="y2"
    ))
    # Perturbation
    fig.add_trace(go.Scatter(
        x=k_line, y=initial_data['perturbation'], name='Perturbation', 
        mode='lines', line=dict(dash='dot', color='green'), yaxis="y3"
    ))
    # Magnus
    if has_magnus:
        fig.add_trace(go.Scatter(
            x=k_line, y=initial_data['magnus_operator_norm'], name='Magnus Norm', 
            mode='lines', line=dict(color='black', width=1), opacity=0.5, yaxis="y4"
        ))

    # 2. Build Frames
    frames = []
    for i, data in enumerate(g_results):
        ev = np.array(data['eigenvalues'])
        if ev.ndim != 2: ev = ev.reshape(ev.shape[0], -1)
        
        frame_data = []
        # Update Eigenvalues traces
        for b in bands:
            frame_data.append(go.Scatter(y=ev[:, b]))
        # Update Trace
        frame_data.append(go.Scatter(y=data['trace']))
        # Update Perturbation
        frame_data.append(go.Scatter(y=data['perturbation']))
        # Update Magnus
        if has_magnus:
            frame_data.append(go.Scatter(y=data['magnus_operator_norm']))
        
        pv = data.get(param_name, i)
        frame_title = f"{title} — {param_name} = {pv:.6g}"
        frames.append(go.Frame(data=frame_data, name=str(i), layout=dict(title=frame_title)))

    fig.frames = frames

    # 3. Layout with sliders and 4 y-axes
    steps = []
    for i in range(len(g_results)):
        pv = g_results[i].get(param_name, i)
        step = dict(
            method="animate",
            args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=f"{pv:.4g}"
        )
        steps.append(step)

    sliders = [dict(active=0, currentvalue={"prefix": f"{param_name}: "}, pad={"t": 50}, steps=steps)]

    # Configure multiple y-axes
    layout_update = {
        "title": f"{title} — {param_name} = {g_results[0].get(param_name, 0):.6g}",
        "width": 1100,
        "height": 700,
        "xaxis": dict(title="k", domain=[0, 0.8]),
        "yaxis": dict(title=dict(text="Eigenvalues", font=dict(color="red")), range=y_bounds['eigen'], tickfont=dict(color="red")),
        "yaxis2": dict(title=dict(text="Trace", font=dict(color="blue")), range=y_bounds['trace'], overlaying="y", side="right", tickfont=dict(color="blue")),
        "yaxis3": dict(title=dict(text="Perturbation", font=dict(color="green")), range=y_bounds['perturb'], overlaying="y", side="right", anchor="free", position=0.88, tickfont=dict(color="green")),
        "sliders": sliders,
        "updatemenus": [dict(
            type="buttons",
            buttons=[
                dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ],
            direction="left", pad={"r": 10, "t": 87}, showactive=False, x=0.1, xanchor="right", y=0, yanchor="top"
        )]
    }

    if has_magnus:
        layout_update["yaxis4"] = dict(title=dict(text="Magnus Norm", font=dict(color="black")), range=y_bounds['magnus'], overlaying="y", side="right", anchor="free", position=0.96, tickfont=dict(color="black"))


    fig.update_layout(**layout_update)
    
    # Save to HTML
    fig.write_html(filepath, include_plotlyjs='cdn')
    print(f"✅ Saved interactive Plotly plot to: {filepath}")


def dynamic_with_eigenvalues_single_param(result_dir, band_index1=0, band_index2=1):
    """
    Visualize (vs k along the chosen line) for a single-parameter 1D sweep.
    Saves results to a Plotly HTML file.
    """
    # Resolve directory
    if os.path.isdir(result_dir):
        folder_path = result_dir
    else:
        folder_path = os.path.join(os.getcwd(), "results", "1D_QGT_results", result_dir)

    g_results_path = os.path.join(folder_path, "QGT_1D.npy")
    meta_path      = os.path.join(folder_path, "parameters.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(folder_path, "meta_info.pkl")

    if not os.path.exists(g_results_path):
        raise FileNotFoundError(f"Missing '{g_results_path}'")

    # Load metadata + results
    if meta_path.endswith('.json'):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

    param_name   = meta.get("param_name", "omega")
    num_k_points = int(meta["num_k_points"])
    k_max        = float(meta["k_max"])
    k_line = np.linspace(-k_max, k_max, num_k_points)
    g_results = np.load(g_results_path, allow_pickle=True)

    bands = [band_index1, band_index2]
    
    # Global y-limits
    y_min_eval = np.inf; y_max_eval = -np.inf
    y_min_trace = np.inf; y_max_trace = -np.inf
    y_min_pert  = np.inf; y_max_pert  = -np.inf
    has_magnus = all("magnus_operator_norm" in d for d in g_results)
    y_min_mag = np.inf; y_max_mag = -np.inf

    for d in g_results:
        ev = np.asarray(d["eigenvalues"])
        if ev.ndim != 2: ev = ev.reshape(ev.shape[0], -1)
        sel = ev[:, bands]
        y_min_eval = min(y_min_eval, np.nanmin(sel))
        y_max_eval = max(y_max_eval, np.nanmax(sel))
        y_min_trace = min(y_min_trace, np.nanmin(d["trace"]))
        y_max_trace = max(y_max_trace, np.nanmax(d["trace"]))
        y_min_pert  = min(y_min_pert,  np.nanmin(d["perturbation"]))
        y_max_pert  = max(y_max_pert,  np.nanmax(d["perturbation"]))
        if has_magnus:
            y_min_mag = min(y_min_mag, np.nanmin(d["magnus_operator_norm"]))
            y_max_mag = max(y_max_mag, np.nanmax(d["magnus_operator_norm"]))

    eval_buf = 0.1 * (y_max_eval - y_min_eval + 1e-12)
    y_bounds = {
        'eigen': [y_min_eval - eval_buf, y_max_eval + eval_buf],
        'trace': [y_min_trace, y_max_trace],
        'perturb': [y_min_pert, y_max_pert],
        'magnus': [y_min_mag, y_max_mag]
    }

    html_path = os.path.join(folder_path, f"dynamic_1d_sweep_{param_name}.html")
    _save_dynamic_qgt_plotly(html_path, k_line, g_results, bands, y_bounds, param_name=param_name, title=f"1D Sweep vs {param_name}")


if __name__ == "__main__":

    #! Full Chiral Hamiltonian
    # dynamic_with_eigenvalues("ChiralHamiltonian/A00.1_polarizationleft_magnus_order1_analytic_magnusFalse_n5_vF542.1_t1355.16_V30.0_eta1.0_angle0.0_kxshift0.00_kyshift0.00_points100_kmax1.57_omega5.00e_00_5.00e_03_spacing_log_points30_2", bands=[0,1,2,3,4, 5,6,7,8,9])
    # dynamic_with_eigenvalues("ChiralHamiltonian/A0_0.10-V_30.00-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-polarization_right-t1_355.16-vF_542.10_angle0.0_kxshift0.00_kyshift0.00_points100_kmax1.57_omega5.00e_00_5.00e_03_spacing_log_points30_1", bands=[4, 5])
    
    dynamic_with_eigenvalues_single_param("ChiralHamiltonian/A0_0-V_20.00-a_1.00-analytic_magnus_False-eta_1.00-magnus_order_1-n_5-polarization_left-t1_355.16-vF_542.10_angle0.0_kx0.00_ky0.00_kmax1.57_param_V_5_50_spacing_linear_N20_kN100_data_set1", band_index1=4, band_index2=5)
