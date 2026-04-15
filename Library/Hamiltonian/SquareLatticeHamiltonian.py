import numpy as np
from .Hamiltonian import hamiltonian


class SquareLatticeHamiltonian(hamiltonian):
    """
    Hamiltonian for a square lattice model.
    """
    def __init__(self, t1=1, t2=1/np.sqrt(2), t5=0, omega=2 * np.pi, A0=0):
        super().__init__(dim=2, omega=omega, A0=A0)  # Pass omega and A0 to the base class
        self.t1 = t1
        self.t2 = t2
        self.t5 = t5

    def compute_static(self, kx, ky, kz=0):
        """
        Compute the static Hamiltonian for the square lattice.
        """
        H11 = -2 * self.t5 * (np.cos(2 * (kx + ky)) + np.cos(2 * (kx - ky))) - 2 * self.t2 * (np.cos(kx + ky) - np.cos(kx - ky))
        H12 = -2 * self.t1 * np.exp(1j * np.pi / 4) * np.exp(1j * ky) * np.cos(ky) - 2 * self.t1 * np.exp(-1j * np.pi / 4) * np.exp(1j * ky) * np.cos(kx)
        H21 = np.conj(H12)  # H21 is the Hermitian conjugate of H12
        H22 = -2 * self.t5 * (np.cos(2 * (kx + ky)) + np.cos(2 * (kx - ky))) + 2 * self.t2 * (np.cos(kx + ky) - np.cos(kx - ky))

        H_k = np.array([
            [H11, H12],
            [H21, H22]
        ], dtype=complex)

        return H_k

    def Hp1(self, kx, ky, kz=0):
        # Define the expression
        H12 = (self.A0 / 2) * (
            2 * self.t1 * np.exp(-1j * np.pi / 4) * np.exp(1j * ky) * np.sin(kx)
            - 2 * self.t1 * np.exp(1j * np.pi / 4) * np.exp(1j * ky) * np.cos(ky)
            - 1j * 2 * self.t1 * np.exp(1j * np.pi / 4) * np.sin(ky) * np.exp(1j * ky)
            - 2 * self.t1 * np.cos(kx) * np.exp(-1j * np.pi / 4) * np.exp(1j * ky)
        )
        H21 = self.A0 * self.t1 * (-1j * np.sin(ky) + np.sqrt(2) * 1j *
                                   np.sin(kx + np.pi / 4) + np.cos(ky)) * np.exp(-1j * (ky + np.pi / 4))
        return H21  # Temporary, change it to full matrix later
