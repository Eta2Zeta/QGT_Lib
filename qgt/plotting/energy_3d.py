#!/usr/import/env python3
import os
import pickle
import numpy as np

# Adjust the path to include the QGT_Lib directory

from qgt.plotting.volume_3d import plot_volumetric_cloud, plot_arbitrary_slice_no_interp

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
