import numpy as np

from qgt.hamiltonians.base import hamiltonian


class GrapheneHamiltonian(hamiltonian):
    def __init__(self, omega = np.pi, A0 = 0):
        super().__init__(dim=2, omega=omega, A0=A0)  

    def compute_static(self, kx, ky, kz=0):
        H_k = np.array([
            [0, kx - 1j*ky],
            [kx + 1j*ky, 0]
        ])
        
        return H_k
    def Hp1(self, kx, ky, kz=0):
        # Define the expression
        H21 = self.A0
        return H21 # Temporary, change it to full matrix later
