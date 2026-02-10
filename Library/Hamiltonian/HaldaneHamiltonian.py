import numpy as np
from .Hamiltonian_v2 import hamiltonian

class HaldaneHamiltonian(hamiltonian):
    """
    Haldane model Hamiltonian on a honeycomb lattice, using full expressions.
    """
    def __init__(self, t1=-1.0, t2=1.0/3.0, M=0.5, psi=np.pi/2, a=1.0, omega=2*np.pi, A0=0.0):
        super().__init__(dim=2, omega=omega, A0=A0)
        self.t1 = t1       # Nearest-neighbor hopping
        self.t2 = t2       # Next-nearest-neighbor hopping
        self.M = M         # Sublattice mass term
        self.psi = psi     # TRS-breaking phase
        self.a = a         # Lattice constant

        # Nearest-neighbor vectors δ_i
        self.delta = np.array([
            [a, 0],
            [-0.5 * a, np.sqrt(3) * a / 2],
            [-0.5 * a, -np.sqrt(3) * a / 2]
        ])


        # Next-nearest-neighbor vectors v_i
        sqrt3 = np.sqrt(3)
        self.v = np.array([
            [0, a * sqrt3],
            [-1.5 * a, -0.5 * a * sqrt3],
            [1.5 * a, -0.5 * a * sqrt3]
        ])

        # reciprocal lattice vectors
        self.b1 = (2*np.pi/(3*a)) * np.array([1.0,  np.sqrt(3.0)])
        self.b2 = (2*np.pi/(3*a)) * np.array([1.0, -np.sqrt(3.0)])

    def h_x(self, kx, ky):
        return np.sum(np.cos(kx * self.delta[:, 0] + ky * self.delta[:, 1]))

    def h_y(self, kx, ky):
        return np.sum(np.sin(kx * self.delta[:, 0] + ky * self.delta[:, 1]))

    def h_z(self, kx, ky):
        return np.sum(np.sin(kx * self.v[:, 0] + ky * self.v[:, 1]))

    def compute_static(self, kx, ky, kz=0):
        hx = self.h_x(kx, ky)
        hy = self.h_y(kx, ky)
        hz = self.h_z(kx, ky)
        
        d_x = self.t1 * hx
        d_y = -self.t1 * hy
        d_z = self.M - 2 * self.t2 * np.sin(self.psi) * hz

        H = d_x * sigma_x + d_y * sigma_y + d_z * sigma_z
        return H
