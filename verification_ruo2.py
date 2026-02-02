import numpy as np
import sys
import os

# Add the parent directory to sys.path to ensure we can import the library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from QGT_Lib.Library.Hamiltonian.RuO2Hamiltonian import RuO2Hamiltonian

def check_hermitian(H):
    return np.allclose(H, H.conj().T)

def test_ruo2():
    print("Testing RuO2Hamiltonian...")
    
    # Initialize
    h = RuO2Hamiltonian()
    print("Initialized successfully.")
    
    # Test point (0,0,0)
    kx, ky, kz = 0, 0, 0
    H0 = h.compute_static(kx, ky, kz)
    print(f"H(0,0,0) shape: {H0.shape}")
    print(f"H(0,0,0) Hermitian: {check_hermitian(H0)}")
    print("Eigenvalues at (0,0,0):", np.linalg.eigvalsh(H0))
    
    # Test random point
    kx, ky, kz = np.random.rand(3)
    H_rand = h.compute_static(kx, ky, kz)
    print(f"H(random) Hermitian: {check_hermitian(H_rand)}")
    print("Eigenvalues at random point:", np.linalg.eigvalsh(H_rand))
    
    # Test vectorized
    kx_arr = np.random.rand(10)
    ky_arr = np.random.rand(10)
    kz_arr = np.random.rand(10)
    H_vec = h.compute_static_vectorized(kx_arr, ky_arr, kz_arr)
    print(f"Vectorized shape: {H_vec.shape}")
    
    all_hermitian = True
    for i in range(10):
        if not check_hermitian(H_vec[i]):
            all_hermitian = False
            break
    print(f"Vectorized all Hermitian: {all_hermitian}")
    
    if all_hermitian:
        print("VERIFICATION SUCCESSFUL")
    else:
        print("VERIFICATION FAILED")

if __name__ == "__main__":
    test_ruo2()
