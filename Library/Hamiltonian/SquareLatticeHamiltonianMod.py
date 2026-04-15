import numpy as np
from .Hamiltonian import hamiltonian


class SquareLatticeHamiltonianMod(hamiltonian):
    """
    Hamiltonian for the modified square lattice model (Equation 3 from PhysRevLett.106.236804).
    """
    def __init__(self, t1=1, t2=1/np.sqrt(2), omega=2 * np.pi, A0=0):
        super().__init__(dim=2, omega=omega, A0=A0)
        self.t1 = t1
        self.t2 = t2

    def compute_static(self, kx, ky, kz=0):
        M11 = 2 * self.t2 * (np.cos(kx) - np.cos(ky))
        M12 = (self.t1 * np.exp(1j * np.pi / 4) * (1 + np.exp(-1j * (ky - kx)))
               + self.t1 * np.exp(-1j * np.pi / 4) * (np.exp(1j * kx) + np.exp(-1j * ky)))
        M21 = np.conj(M12)  # M21 is the Hermitian conjugate of M12
        M22 = -2 * self.t2 * (np.cos(kx) - np.cos(ky))

        H_k = np.array([
            [M11, M12],
            [M21, M22]
        ], dtype=complex)

        return H_k
