#!/usr/import/env python3
import os
import sys
import argparse
import pickle
import numpy as np

# Adjust the path to include the QGT_Lib directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from Library.plotting_lib_3d import plot_volumetric_cloud, plot_arbitrary_slice_no_interp

def plot_3d_energy(results_dir, band_idx=2, cloud_opacity=0.2, cloud_stride=2, slice_stride=2, levels=[0]):
    """
    Load 3D eigenvalues and metadata from results_dir, and generate the energy cloud/slice plots.
    """
    meta_path = os.path.join(results_dir, "meta_info.pkl")
    eig_path = os.path.join(results_dir, "eigenvalues_3d.npy")
    
    if not os.path.exists(meta_path) or not os.path.exists(eig_path):
        print(f"Dataset not found or missing files in: {results_dir}")
        return

    print(f"Loading metadata from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta_info = pickle.load(f)
        
    kx_vals = meta_info['kx_vals']
    ky_vals = meta_info['ky_vals']
    kz_vals = meta_info['kz_vals']
    
    # Try to extract hamiltonian name safely
    if 'hamiltonian_name' in meta_info:
        hamiltonian_name = meta_info['hamiltonian_name']
    elif 'target' in meta_info and 'hamiltonian_name' in meta_info['target']:
        hamiltonian_name = meta_info['target']['hamiltonian_name']
    else:
        hamiltonian_name = "Hamiltonian"
    eigenvalues_3d = np.load(eig_path)
    
    nkx, nky, nkz, nbands = eigenvalues_3d.shape
    
    if band_idx >= nbands:
        print(f"Error: Requested band {band_idx}, but only {nbands} dimensions found.")
        return

    print("Generating volumetric cloud plot...")
    eig_band = eigenvalues_3d[:, :, :, band_idx] # [x, y, z]
    
    cloud_fname = f"{hamiltonian_name}_3D_Energy_Cloud_Band{band_idx}.html".replace(" ", "_").replace("/", "_")
    cloud_out_file = os.path.join(results_dir, cloud_fname)
    
    plot_volumetric_cloud(
        eig_band, kx_vals, ky_vals, kz_vals,
        opacity=cloud_opacity, 
        levels=levels,
        stride=cloud_stride,
        title=f"{hamiltonian_name} Energy Cloud (Band {band_idx})",
        filename=cloud_out_file
    )

    print("Generating slice plots...")
    orientations = ['z']
    shift_val = 0
    
    for orientation in orientations:
        slice_fname = f"{hamiltonian_name}_3D_Energy_Slice_{orientation}_Band{band_idx}.png".replace(" ", "_").replace("/", "_")
        slice_out_file = os.path.join(results_dir, slice_fname)
        
        plot_arbitrary_slice_no_interp(
            eigenvalues_3d, orientation, shift_val, kx_vals, ky_vals, kz_vals, 
            title=f"{hamiltonian_name} Slice {orientation} (shift={shift_val})",
            stride=slice_stride,
            filename=slice_out_file,
            show=False
        )

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change the target Hamiltonian name below if you switch models.
    base_results_path = os.path.join(current_dir, "results/3D_Eigen_results/gWaveAltermagnetHamiltonian")
    
    # Specify specific dataset number (e.g., 15) or None for latest
    target_dataset = None  
    
    if os.path.exists(base_results_path):
        datasets = [d for d in os.listdir(base_results_path) if d.startswith("data_set_")]
        
        if datasets:
            if target_dataset is not None:
                dataset_name = f"data_set_{target_dataset}"
                if dataset_name in datasets:
                    results_dir = os.path.join(base_results_path, dataset_name)
                    print(f"Processing requested dataset: {results_dir}")
                else:
                    print(f"Dataset {dataset_name} not found in {base_results_path}")
                    sys.exit(1)
            else:
                datasets.sort(key=lambda x: int(x.split('_')[-1]))
                latest_dataset = datasets[-1]
                results_dir = os.path.join(base_results_path, latest_dataset)
                print(f"Processing latest dataset: {results_dir}")
                
            # Default options (you can change them here)
            plot_3d_energy(
                results_dir, 
                band_idx=2, 
                cloud_opacity=0.2, 
                cloud_stride=2,  # Increase to render faster
                slice_stride=2,  # Increase to render faster
                levels=[0]       # Energy levels to plot isosurfaces for
            )
        else:
            print(f"No datasets found in {base_results_path}")
    else:
        print(f"Base results path does not exist: {base_results_path}")