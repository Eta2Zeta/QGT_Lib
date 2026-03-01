import numpy as np
from Library.plotting_lib_3d import plot_slice_stack

data_3d = np.random.rand(10, 10, 10)
kx = ky = kz = np.linspace(0, 1, 10)
plot_slice_stack(data_3d, kx, ky, kz, plane='xy', n_slices=3)
