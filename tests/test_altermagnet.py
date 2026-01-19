import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the library path to sys.path to ensure we can import the module locally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock qutip if not installed
import sys
from unittest.mock import MagicMock
import numpy as np

class MockQobj:
    def __init__(self, arr):
        self.arr = arr
    def full(self):
        return self.arr

def mock_sigmax(): return MockQobj(np.array([[0, 1], [1, 0]], dtype=complex))
def mock_sigmay(): return MockQobj(np.array([[0, -1j], [1j, 0]], dtype=complex))
def mock_sigmaz(): return MockQobj(np.array([[1, 0], [0, -1]], dtype=complex))

mock_qutip = MagicMock()
mock_qutip.sigmax = mock_sigmax
mock_qutip.sigmay = mock_sigmay
mock_qutip.sigmaz = mock_sigmaz
sys.modules['qutip'] = mock_qutip
from Library.Hamiltonian.Altermagnet_Hamiltonian import AltermagnetHamiltonian

def test_hamiltonian_structure():
    """Check dimensions and hermiticity."""
    ham = AltermagnetHamiltonian()
    kx, ky = 0.5, 0.5
    H = ham.compute_static(kx, ky)
    
    print("Testing Hamiltonian Structure...")
    print(f"Shape: {H.shape}")
    assert H.shape == (4, 4), "Hamiltonian should be 4x4"
    
    # Check Hermiticity
    assert np.allclose(H, H.conj().T), "Hamiltonian is not Hermitian"
    print("Hermiticity check passed.")

def plot_dispersion():
    """Plot dispersion along high symmetry lines to compare with Fig 2."""
    print("Plotting dispersion...")
    
    # Parameters from the paper description (using default or typical values if not specified)
    # Fig 2 caption says "Nc = |4 td / J|".
    # Let's try to qualitatively reproduce the features: spin splitting except along diagonals.
    
    t1 = 1.0
    t2 = 0.0 # Simplify first? Paper has t2
    td = 0.2
    lamb = 0.1
    J = 1.0
    Nz = 0.5 # < Nc = |4*0.2/1| = 0.8
    
    ham = AltermagnetHamiltonian(t1=t1, t2=t2, td=td, lamb=lamb, J=J, Nz=Nz)
    
    # Path: Gamma(0,0) -> X(pi, 0) -> M(pi, pi) -> Gamma(0,0) -> Y(0, pi) -> M(pi, pi)
    # But text says "No spin splitting occurs along the high-symmetry line Gamma-M".
    # Let's check Gamma -> M
    
    # Gamma to M
    k_vals = np.linspace(0, np.pi, 100)
    kx = k_vals
    ky = k_vals
    
    H_vec = ham.compute_static_vectorized(kx, ky)
    evals = np.linalg.eigvalsh(H_vec)
    
    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(k_vals, evals[:, i], label=f'Band {i}')
        
    plt.title("Dispersion along Gamma-M (k_x = k_y)")
    plt.xlabel("k (along diagonal)")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.savefig("altermagnet_dispersion_Gamma_M.png")
    print("Saved dispersion plot to altermagnet_dispersion_Gamma_M.png")
    
    # Check for spin splitting
    # Along Gamma-M, bands should be degenerate (doubly degenerate if no other lifting)
    # The text says "No spin splitting occurs along... Gamma-M". 
    # This means we expect 2 distinct energy curves, each doubly degenerate.
    
    diff_1 = np.abs(evals[:, 0] - evals[:, 1])
    diff_2 = np.abs(evals[:, 2] - evals[:, 3])
    
    max_diff = np.max(np.concatenate([diff_1, diff_2]))
    print(f"Max splitting along Gamma-M: {max_diff}")
    
    if max_diff < 1e-10:
        print("Confirmed: No spin splitting along Gamma-M.")
    else:
        print("WARNING: Spin splitting observed along Gamma-M!")

    # Check X-M for splitting (should be split)
    # X(pi, 0) -> M(pi, pi) => kx=pi, ky goes 0 to pi
    ky_vals = np.linspace(0, np.pi, 100)
    kx_vals = np.full_like(ky_vals, np.pi)
    
    H_XM = ham.compute_static_vectorized(kx_vals, ky_vals)
    evals_XM = np.linalg.eigvalsh(H_XM)
    
    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(ky_vals, evals_XM[:, i], label=f'Band {i}')
    plt.title("Dispersion along X-M (kx=pi, ky=0->pi)")
    plt.savefig("altermagnet_dispersion_X_M.png")

if __name__ == "__main__":
    test_hamiltonian_structure()
    plot_dispersion()
