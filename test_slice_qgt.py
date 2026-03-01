import numpy as np
import sys
from Plot_QGT_3D import plot_3d_qgt_slices

# Test dataset 1 first
results_dir = "/Users/hongyuzhang/Documents/Quantum_Geometric_Tensor/QGT_Lib/results/3D_QGT_results/gWaveAltermagnetHamiltonian/data_set_1"
print("Testing dataset 1")

meta_file = results_dir + "/qgt_meta_info.pkl"
import pickle
with open(meta_file, "rb") as f:
    meta_info = pickle.load(f)

print(meta_info)
import os
def load_arr(name):
    path = os.path.join(results_dir, f"{name}.npy")
    return np.load(path) if os.path.exists(path) else None

gij_im = load_arr("g_xy_imag")
print("gij_im dtype:", gij_im.dtype)
print("max:", np.nanmax(gij_im))
print("min:", np.nanmin(gij_im))
print("has nan?", np.isnan(gij_im).any())

data_3d = -2.0 * gij_im
sl = data_3d[:, :, 0].T
print(sl.dtype, np.nanmax(sl), np.nanmin(sl))

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    im = ax.pcolormesh(np.arange(sl.shape[1]), np.arange(sl.shape[0]), sl, shading="auto", cmap="bwr")
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.savefig('test_slice_stack.png')
    print("Saved plot!")
except Exception as e:
    import traceback
    traceback.print_exc()

