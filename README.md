# QGT_Lib
 This library does the essential Caulations for the Quantum Geometric Tensor


## How to use it

1. Define your Hamiltonian in Hamiltonian_v2.py in Library
2. Run Eigenvalues_Calc.py with your Hamiltonian as the Hamiltonian Obj
3. Run QGT_Calc.py

## Naming conventions for the QGT calculations on the kx ky grid

square_lattice_90deg_0x_npio2y_300p_min5e0_max1e2_g_results_linear

square_lattice is the Hamiltonion we are doing the calculation on

90deg is the slide of the line on which we are calculating the QGT starting from 0 deg along the positive x axis

0x is the shift from the x=0

npio2 means negative pi over 2 shift from the y=0

300 points means we are plotting 300 points on the line

min5e0 is the minimum omega value

max1e2 is the maximum omega value

linear means we are plotting the omega values seperated linearly



## To Do Next
Berry Curvature Calculation -> Chern Number
Polarization changing Asinwt to -Asinwt