import numpy as np
from .Hamiltonian import hamiltonian


class TwoOrbitalSpinfulHamiltonian(hamiltonian):
    """
    Hamiltonian for the two-orbital spinful model. From PRL 130, 226001 eq (1)
    """
    def __init__(self, t=1, mu=0, zeta=0, a=1, omega=np.pi/2, A0=0, magnus_order=1):
        """
        Initialize the two-orbital spinful Hamiltonian.
        Parameters:
        - t: Hopping parameter
        - mu: Chemical potential
        - zeta: Parameter for alpha_k
        - a: Lattice spacing
        - omega: Driving frequency
        - A0: Driving amplitude
        """
        super().__init__(dim=4, omega=omega, A0=A0, magnus_order=magnus_order)
        self.t = t
        self.mu = mu
        self.zeta = zeta
        self.a = a

    def compute_static(self, kx, ky, kz=0):
        """
        Compute the static Hamiltonian for the two-orbital spinful model.
        """
        # Compute alpha_k
        alpha_k = self.zeta * (np.cos(kx * self.a) + np.cos(ky * self.a))

        # Define matrix elements
        H = np.zeros((4, 4), dtype=complex)
        sin_alpha = np.sin(alpha_k)
        cos_alpha = np.cos(alpha_k)

        # Fill the Hamiltonian matrix
        H[0, 0] = -self.t * self.mu
        H[1, 1] = -self.t * self.mu
        H[2, 2] = -self.t * self.mu
        H[3, 3] = -self.t * self.mu

        H[0, 2] = -self.t * (sin_alpha - 1j * cos_alpha)
        H[2, 0] = -self.t * (sin_alpha + 1j * cos_alpha)

        H[1, 3] = -self.t * (sin_alpha + 1j * cos_alpha)
        H[3, 1] = -self.t * (sin_alpha - 1j * cos_alpha)

        return H
