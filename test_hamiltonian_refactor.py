
import numpy as np
from Library.Hamiltonian.gWaveAltermagnetHamiltonian import gWaveAltermagnetHamiltonian

def test_hamiltonian():
    print("Testing gWaveAltermagnetHamiltonian...")
    try:
        h = gWaveAltermagnetHamiltonian()
        print("Initialization successful.")
        
        kx, ky = 0.1, 0.2
        H = h.compute_static(kx, ky)
        print(f"compute_static({kx}, {ky}) shape: {H.shape}")
        
        kx_arr = np.linspace(0, 1, 10)
        ky_arr = np.linspace(0, 1, 10)
        H_vec = h.compute_static_vectorized(kx_arr, ky_arr)
        print(f"compute_static_vectorized shape: {H_vec.shape}")
        
        print("gWaveAltermagnetHamiltonian test PASSED.")
    except Exception as e:
        print(f"gWaveAltermagnetHamiltonian test FAILED: {e}")
        exit(1)

if __name__ == "__main__":
    test_hamiltonian()
