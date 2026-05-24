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

        # Reciprocal lattice vectors for the square lattice (a=1)
        self.b1 = np.pi * np.array([1.0,  1.0])
        self.b2 = np.pi * np.array([1.0, -1.0])

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

    def get_sym_path(self):
        """
        High-symmetry k-path for the square-lattice Brillouin zone.

        All points are derived from the reciprocal lattice vectors b1 and b2:
            b1 = π * [1,  1]
            b2 = π * [1, -1]

        High-symmetry points (fractional reciprocal-lattice coordinates):
            Γ  = 0·b1 + 0·b2          = (0,    0   )  – zone centre
            M  = ½·b1 + ½·b2          = (π,    0   )  – zone corner
            X  = ½·b1 + 0·b2          = (π/2,  π/2 )  – zone-edge midpoint

        Path:  Γ → M → X → Γ

        Returns
        -------
        sym_points : dict
            Mapping of label -> np.ndarray of shape (2,).
        path : list of str
            Ordered labels defining the path (with repeated endpoints).
        """
        G = np.zeros(2)                        # 0·b1 + 0·b2
        M = 0.5 * self.b1 + 0.5 * self.b2     # zone corner
        X = 0.5 * self.b1                      # zone-edge midpoint

        sym_points = {"G": G, "M": M, "X": X}
        path = ["G", "M", "X", "G"]
        return sym_points, path

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
