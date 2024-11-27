from Library.eigenvalue_calc_lib import *    

# Numerical derivative w.r.t. kx
def dpsi_dx_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction):
    eigenvector = Eigenvector(len(eigenfunction))
    eigenvector.set_eigenvectors(eigenfunction)

    Hamiltonian_plus = Hamiltonian(kx + delta_k, ky)
    eigenvalues_plus, psi_plus = get_eigenvalues_and_eigenvectors(Hamiltonian_plus)
    i_eigenvalue = min(range(len(eigenvalues_plus)), key=lambda i: abs(eigenvalues_plus[i] - eigenvalue))
    psi_plus = eigenvector.set_eigenvectors(psi_plus[i_eigenvalue,:])



    Hamiltonian_minus = Hamiltonian(kx - delta_k, ky)
    eigenvalues_minus, psi_minus = get_eigenvalues_and_eigenvectors(Hamiltonian_minus)
    eigenvalues_minus, psi_minus = get_eigenvalues_and_eigenvectors(Hamiltonian_minus)
    i_eigenvalue = min(range(len(eigenvalues_minus)), key=lambda i: abs(eigenvalues_minus[i] - eigenvalue))
    psi_minus = eigenvector.set_eigenvectors(psi_minus[i_eigenvalue,:])

    return (psi_plus - psi_minus) / (2 * delta_k)

# Numerical derivative w.r.t. ky
def dpsi_dy_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction):
    eigenvector = Eigenvector(len(eigenfunction))
    eigenvector.set_eigenvectors(eigenfunction)

    Hamiltonian_plus = Hamiltonian(kx, ky + delta_k)
    eigenvalues_plus, psi_plus = get_eigenvalues_and_eigenvectors(Hamiltonian_plus)
    i_eigenvalue = min(range(len(eigenvalues_plus)), key=lambda i: abs(eigenvalues_plus[i] - eigenvalue))
    psi_plus = eigenvector.set_eigenvectors(psi_plus[i_eigenvalue,:])


    Hamiltonian_minus = Hamiltonian(kx, ky - delta_k)
    eigenvalues_minus, psi_minus = get_eigenvalues_and_eigenvectors(Hamiltonian_minus)
    eigenvalues_minus, psi_minus = get_eigenvalues_and_eigenvectors(Hamiltonian_minus)
    i_eigenvalue = min(range(len(eigenvalues_minus)), key=lambda i: abs(eigenvalues_minus[i] - eigenvalue))
    psi_minus = eigenvector.set_eigenvectors(psi_minus[i_eigenvalue,:])
    

    return (psi_plus - psi_minus) / (2 * delta_k)

# Quantum geometric tensor components calculation using numerically obtained eigenfunctions
def quantum_geometric_tensor_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction, dim):
    dpsi_dx_val = dpsi_dx_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction)
    dpsi_dy_val = dpsi_dy_num(Hamiltonian, kx, ky, delta_k, eigenvalue, eigenfunction)
    psi_val = eigenfunction

    I = np.eye(dim)
    P = projection_operator(psi_val)
    
    g_xx = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dx_val).real
    g_xy_real = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).real
    g_xy_imag = np.vdot(dpsi_dx_val, (I - P) @ dpsi_dy_val).imag
    g_yy = np.vdot(dpsi_dy_val, (I - P) @ dpsi_dy_val).real
    
    return g_xx, g_xy_real, g_xy_imag, g_yy