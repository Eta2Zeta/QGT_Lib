import numpy as np
from .Hamiltonian_v2 import hamiltonian, sigma_x, sigma_y, sigma_z



class ChiralHamiltonian(hamiltonian):
    """
    Full chiral (v_F, t1-only) Hamiltonian + linear displacement field H_D.
    Basis: (layer 0: A,B; layer 1: A,B; ...; layer n-1: A,B).
    """
    def __init__(self, n=2, a=1.0, vF=542.1, t1=355.16, V=0.0,
                 valley='K', omega=2*np.pi, A0=0, **kwargs):
        super().__init__(dim=2*n, omega=omega, A0=A0, **kwargs)
        self.n = int(n)
        self.vF = float(vF)
        self.a = a                      # Lattice constant
        self.t1 = float(t1)

        v = valley.lower()
        if v in ('k',):
            self.eta = 1.0
        elif v in ('kp', 'kprime', "k'"):
            self.eta = -1.0
        else:
            raise ValueError("valley must be 'K' or 'Kp'/'Kprime'")

        # 2x2 sublattice operators
        self._sigma_plus  = 0.5 * (sigma_x + 1j * sigma_y)
        self._sigma_minus = 0.5 * (sigma_x - 1j * sigma_y)
        self._I2 = np.eye(2, dtype=complex)

        # set V via the property to populate _V_layer
        self.V = float(V)
        
        # reciprocal lattice vectors that is the same as the single layer graphene lattice vectors
        self.b1 = (2*np.pi/(3*a)) * np.array([1.0,  np.sqrt(3.0)])
        self.b2 = (2*np.pi/(3*a)) * np.array([1.0, -np.sqrt(3.0)])

    @property
    def V(self) -> float:
        return self._V

    @V.setter
    def V(self, val: float):
        self._V = float(val)
        center = (self.n - 1) / 2.0
        # refresh cached per-layer onsite potentials
        self._V_layer = np.array([self._V * (l - center) for l in range(self.n)], dtype=float)


    def _D_block(self, kx, ky):
        # D = vF * ( kx σ_x + η ky σ_y )
        return self.vF * (kx * sigma_x + self.eta * ky * sigma_y)

    def _L_block(self):
        # L = t1 σ^+
        return self.t1 * self._sigma_plus

    def _U_block(self):
        # U = t1 σ^-
        return self.t1 * self._sigma_minus

    def compute_static(self, kx, ky, kz=0):
        """
        H(k) = h_n(k) + H_D
        h_n(k): block-tridiagonal chiral Hamiltonian
        H_D: diagonal layer-linear onsite potentials (same for A/B on a layer)
        """
        n = self.n
        H = np.zeros((2*n, 2*n), dtype=complex)

        D = self._D_block(kx, ky)
        L = self._L_block()
        U = self._U_block()

        for l in range(n):
            # chiral intralayer block
            H[2*l:2*l+2, 2*l:2*l+2] = D
            # add displacement-field onsite to this layer (A & B equally)
            Vl = self._V_layer[l]
            H[2*l:2*l+2, 2*l:2*l+2] += Vl * self._I2

            # interlayer couplings
            if l < n - 1:
                H[2*(l+1):2*(l+1)+2, 2*l:2*l+2] = L   # sub-diagonal
                H[2*l:2*l+2, 2*(l+1):2*(l+1)+2] = U   # super-diagonal

        return H
    # ---- PSEUDO-EIGENVECTORS (holomorphic chiral basis) ----

    def _kpm(self, kx: float, ky: float):
        """Return k_plus, k_minus with the valley sign convention."""
        kp = kx + 1j * self.eta * ky
        km = kx - 1j * self.eta * ky
        return kp, km

    def _N(self, kx: float, ky: float) -> float:
        """
        Normalization N(|β|) with robust limits:
            N^2 = (1 - |β|^{2n}) / (1 - |β|^2),
            |β|^2 = (vF^2 * (kx^2 + ky^2)) / t1^2
        """
        beta2 = (self.vF * self.vF) * (kx * kx + ky * ky) / (self.t1 * self.t1)
        if beta2 < 1e-20:
            return 1.0  # at k=0, only the l=0 term survives
        if abs(1.0 - beta2) < 1e-10:
            return float(np.sqrt(self.n))  # limit |β|->1: sum -> n
        # general case
        N2 = (1.0 - beta2 ** self.n) / (1.0 - beta2)
        # guard tiny negatives from roundoff
        N2 = float(np.maximum(N2, 0.0))
        return float(np.sqrt(N2))

    @staticmethod
    def _phase_fix(psi: np.ndarray, prev_psi: np.ndarray | None):
        """
        Global phase (gauge) fixing: make <prev|psi> real positive
        to reduce numerical discontinuities for finite differencing.
        """
        if prev_psi is None:
            return psi
        ov = np.vdot(prev_psi, psi)
        if ov == 0:
            return psi
        return psi * np.exp(-1j * np.angle(ov))

    def psiA(self, kx: float, ky: float, prev_psi: np.ndarray | None = None) -> np.ndarray:
        """
        A-chiral (holomorphic) state:
            (A_l, B_l) = ((-β_+)^l / N, 0),   l = 0..n-1
        Returns a (2n,)-shaped complex vector ordered as (A0,B0|A1,B1|...).
        """
        kp, _ = self._kpm(kx, ky)
        beta_plus = (self.vF / self.t1) * kp
        N = self._N(kx, ky)

        psi = np.zeros(self.dim, dtype=complex)
        for l in range(self.n):
            psi[2*l] = (-beta_plus) ** l / N        # A_l
            psi[2*l + 1] = 0.0                      # B_l = 0

        # tiny renormalization (safety) and smooth phase vs prev_psi
        norm = np.linalg.norm(psi)
        if norm > 0:
            psi = psi / norm
        return self._phase_fix(psi, prev_psi)

    def psiB(self, kx: float, ky: float, prev_psi: np.ndarray | None = None) -> np.ndarray:
        """
        B-chiral (anti-holomorphic) state:
            (A_l, B_l) = (0, (-β_-)^{n-1-l} / N),   l = 0..n-1
        Returns a (2n,)-shaped complex vector ordered as (A0,B0|A1,B1|...).
        """
        _, km = self._kpm(kx, ky)
        beta_minus = (self.vF / self.t1) * km
        N = self._N(kx, ky)

        psi = np.zeros(self.dim, dtype=complex)
        for l in range(self.n):
            psi[2*l] = 0.0                          # A_l = 0
            psi[2*l + 1] = (-beta_minus) ** (self.n - 1 - l) / N  # B_l

        # tiny renormalization (safety) and smooth phase vs prev_psi
        norm = np.linalg.norm(psi)
        if norm > 0:
            psi = psi / norm
        return self._phase_fix(psi, prev_psi)

    def pseudo_eigenvector(self, band_index: int):
        """
        Return a callable psi(kx, ky, prev_psi=None) that evaluates the
        requested pseudo-eigenvector at (kx, ky).

        band_index = 0 -> psiA (A-chiral, holomorphic)
        band_index = 1 -> psiB (B-chiral, anti-holomorphic)
        """
        if band_index == 0:
            return lambda kx, ky, prev_psi=None: self.psiA(kx, ky, prev_psi=prev_psi)
        elif band_index == 1:
            return lambda kx, ky, prev_psi=None: self.psiB(kx, ky, prev_psi=prev_psi)
        else:
            raise ValueError("band_index must be 0 (psiA) or 1 (psiB)")
