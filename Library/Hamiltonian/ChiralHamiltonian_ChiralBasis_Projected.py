import numpy as np
from .Hamiltonian import hamiltonian
sigma_z = np.array([[1, 0],
                    [0, -1]], dtype=complex)
sigma_p = np.array([[0, 1],
                    [0, 0]], dtype=complex)  # σ_+
sigma_m = np.array([[0, 0],
                    [1, 0]], dtype=complex)  # σ_-

class ChiralHamiltonianChiralBasisProjected(hamiltonian):
    """
    Minimal chiral effective Hamiltonian for n-layer rhombohedral graphene in the chiral basis,
    projected onto low-energy subspace, with a displacement field included.
    """
    _MAGNUS_FIRST_TERM_METHODS = {
        "direct_drive": "_analytic_magnus_first_term_direct_drive",
        "projected_full_drive": "_analytic_magnus_first_term_projected_full_drive",
    }

    def __init__(
        self,
        vF=542.1,
        t1=355.16,
        V=30.0,
        n=2,
        omega=2 * np.pi,
        A0=0,
        polarization='left',
        magnus_first_term_mode="direct_drive",
        **kwargs,
    ):
        super().__init__(dim=2, omega=omega, A0=A0, polarization=polarization, **kwargs)
        self.vF = vF
        self.t1 = t1
        self.V = V
        self.n = n
        self.magnus_first_term_mode = self._validate_magnus_first_term_mode(magnus_first_term_mode)

        self.b1 = (2 * np.pi / 3) * np.array([1.0,  np.sqrt(3.0)])
        self.b2 = (2 * np.pi / 3) * np.array([1.0, -np.sqrt(3.0)])

    def get_sym_path(self):
        G = np.zeros(2)
        K = (2.0 * self.b1 + self.b2) / 3.0
        M = 0.5 * self.b1

        sym_points = {"G": G, "K": K, "M": M}
        path = ["G", "K", "M", "G"]
        return sym_points, path

    def analytic_band_from_index(self, band_index):
        if band_index == 0:
            return -1
        if band_index == 1:
            return +1
        raise ValueError(
            f"{self.__class__.__name__} analytic QGT only supports band indices 0 and 1; got {band_index}."
        )

    def _validate_magnus_first_term_mode(self, mode):
        if mode not in self._MAGNUS_FIRST_TERM_METHODS:
            valid_modes = ", ".join(sorted(self._MAGNUS_FIRST_TERM_METHODS))
            raise ValueError(
                f"Unknown magnus_first_term_mode={mode!r}. Valid modes are: {valid_modes}."
            )
        return mode

    def valid_k_point(self, kx, ky):
        k = np.sqrt(kx**2 + ky**2)
        # return True
        return (self.vF * k / self.t1) < 1

    def N_k(self, k):
        """
        Vectorized, numerically stable N(k).
        N^2 = sum_{m=0}^{n-1} (beta^2)^m,  beta^2 = (vF k / t1)^2
        Works on scalars or arrays.
        """
        beta2 = (self.vF * k / self.t1)**2
        # Horner: S = 1 + r + ... + r^{n-1}  ->  accumulate: S <- S*r + 1
        N2 = np.zeros_like(beta2, dtype=float)
        for _ in range(self.n):
            N2 = N2 * beta2 + 1.0
        # Guard tiny negative due to roundoff
        return np.sqrt(np.maximum(N2, 0.0))

    
    def V_k(self, k):
        """
        Vectorized, smooth V(k) with soft reciprocal to avoid blowups
        near (1 - a^2 k^2) * (1 - (a k)^{2n}) ~ 0.
        """
        a = self.vF / self.t1
        n = self.n
        ak = a * k
        ak2   = ak * ak
        ak2n  = ak2**n

        num = (n - 1) * ak**(2*n + 2) + ak2 - n * ak**(2*n)
        den = (1.0 - ak2) * (1.0 - ak2n)

        # smooth reciprocal: 1/x ≈ x/(x^2 + eps^2) — C^1 at x=0
        eps = 1e-10
        inv_den = den / (den*den + eps*eps)

        return self.V * (-0.5 * (n - 1) + num * inv_den)

#@ Essentials 
    def compute_static(self, kx, ky, kz=0):
        k = np.sqrt(kx**2 + ky**2)
        k_plus = kx + 1j * ky
        N = self.N_k(k)
        V_k = self.V_k(k)
        off_diag = -self.t1 * (self.vF * k_plus / self.t1) ** self.n / N**2

        return np.array([
            [V_k, np.conj(off_diag)],
            [off_diag, -V_k]
        ], dtype=complex)
        
    def compute_static_vectorized(self, kx_arr, ky_arr, kz_arr=0):
        kx_arr = np.asarray(kx_arr, dtype=float)
        ky_arr = np.asarray(ky_arr, dtype=float)
        M = kx_arr.shape[0]

        k = np.hypot(kx_arr, ky_arr)                      # (M,)
        N = self.N_k(k)                                    # (M,)
        Vk = self.V_k(k)                                   # (M,)

        k_plus = kx_arr + 1j * ky_arr                      # (M,)
        # off-diagonal: -t1 * (vF k_+ / t1)^n / N^2
        off = -self.t1 * (self.vF * k_plus / self.t1)**self.n / (N**2)

        H = np.zeros((M, 2, 2), dtype=complex)
        H[:, 0, 0] = Vk
        H[:, 1, 1] = -Vk
        H[:, 1, 0] = off
        H[:, 0, 1] = np.conj(off)
        return H
        
    def analytic_eigenvalues(self, kx, ky):
        """
        Return the correct analytical eigenvalues of the projected 2x2 rhombohedral graphene Hamiltonian
        with a displacement field, based on Appendix B of the paper.

        Parameters:
            kx, ky (float): Momentum components.

        Returns:
            np.ndarray: Two eigenvalues (E_minus, E_plus)
        """
        k = np.sqrt(kx**2 + ky**2)
        N = self.N_k(k)
        V_k = self.V_k(k)

        # |off-diagonal|^2 = (vF^n * k^n)^2 / (t1^{2n-2} * N^4)
        h_off_sq = (self.vF**self.n * k**self.n / (self.t1**(self.n - 1) * N**2))**2

        E = np.sqrt(V_k**2 + h_off_sq)
        return np.array([-E, E])
    
#@ Derivative helper used by direct-drive and embedding terms
    def _dN_dk(self, k):
        """
        Analytical derivative ∂N/∂k.
        """
        a = self.vF / self.t1
        n = self.n
        ak = a * k
        ak2 = ak**2
        ak2n = ak2**n

        numerator = a**2 * k**2 * (n * ak2n - ak2n + 1) - n * ak2n
        denominator = (
            k * (1 - a**2 * k**2) ** 2 *
            np.sqrt((ak2n - 1) / (a**2 * k**2 - 1))
        )
        return numerator / denominator

#@ Analytic Magnus first terms
    def _C_orbital(self, kx, ky):
        """
        C(k) = -t1 * (vF/t1)^n * n * k^{n-1} * A0 * e^{i (n-1) θ} / N(k)^2
        This is the coefficient of σ_- in H_{+1}^{(orb)} for A_+ drive.
        """
        k = np.hypot(kx, ky)
        if k == 0.0:
            return 0.0 + 0.0j

        theta = np.arctan2(ky, kx)
        N = self.N_k(k)
        pref = (self.vF / self.t1) ** self.n
        C = -self.t1 * pref * self.n * (k ** (self.n - 1)) * self.A0 * np.exp(1j * (self.n - 1) * theta) / (N ** 2)
        return C

    def _dlnN_chiral(self, kx, ky):
        """
        Return (∂_- ln N, ∂_+ ln N) for a radial N(k) = N(|k|).
        """
        k = np.hypot(kx, ky)
        if k == 0.0:
            return 0.0 + 0.0j, 0.0 + 0.0j

        N = self.N_k(k)
        dNdk = self._dN_dk(k)
        dlnNdk = dNdk / N

        dlnN_dx = dlnNdk * (kx / k)
        dlnN_dy = dlnNdk * (ky / k)

        dlnN_minus = 0.5 * (dlnN_dx + 1j * dlnN_dy)  # ∂_-
        dlnN_plus  = 0.5 * (dlnN_dx - 1j * dlnN_dy)  # ∂_+

        return dlnN_minus, dlnN_plus

    def analytic_magnus_first_term(self, kx, ky, kz=0, return_parts=False):
        """
        Dispatch to the selected analytic first-order Magnus correction.

        Available modes:
        - direct_drive: drive the projected 2x2 Hamiltonian directly.
        - projected_full_drive: drive the full Hamiltonian and project to the chiral basis.
        """
        method_name = self._MAGNUS_FIRST_TERM_METHODS[self.magnus_first_term_mode]
        method = getattr(self, method_name)
        return method(kx, ky, kz=kz, return_parts=return_parts)

    def _analytic_magnus_first_term_direct_drive(self, kx, ky, kz=0, return_parts=False):
        """
        Magnus first term when driving the 2x2 projected Hamiltonian directly.
        """
        k = np.hypot(kx, ky)
        if k == 0.0:
            H_zero = np.zeros((2, 2), dtype=complex)
            return (H_zero, H_zero) if return_parts else H_zero

        C = self._C_orbital(kx, ky)
        absC2 = (np.abs(C) ** 2)

        dlnN_minus, dlnN_plus = self._dlnN_chiral(kx, ky)

        H_orb_orb = -(absC2 / self.omega) * sigma_z
        if self.polarization == 'right':
            H_orb_orb = -H_orb_orb

        H_cross = 2.0 * self.A0 * (
            dlnN_plus * np.conj(C) * sigma_p +
            dlnN_minus  * C         * sigma_m
        )

        if return_parts:
            return H_orb_orb, H_cross
        return H_orb_orb + H_cross

    def _analytic_magnus_first_term_projected_full_drive(self, kx, ky, kz=0, return_parts=False):
        """
        Projected first-order Floquet correction from full Hamiltonian projection:
        H_F^(1), 2x2 = -((vF * A0)^2 / omega) * sigma_z
        (Simple constant term, valid for A_+ helicity).
        
        If right polarization (A_- helicity), sign is reversed.

        The difference from the one above is that we are driving the full Hamiltonian and then project it down here. 
        But the previous one is when we drive the 2x2 Hamiltonian directly.
        """
        # Constant term independent of k
        val = -((self.vF * self.A0) ** 2) / self.omega
        
        if self.polarization == 'right':
            val = -val  # Flip sign for right polarization
            
        H_projected = val * sigma_z
        if return_parts:
            return H_projected, np.zeros_like(H_projected)
        return H_projected

# @ Analytic Berry curvature corrected with projection terms
    def _dlnN_dk(self, k):
        """
        d/dk ln N(k), using the closed form from the derivation.
        """
        k = np.asarray(k, dtype=float)
        a = self.vF / self.t1
        n = self.n

        ak = a * k
        ak2 = ak * ak
        ak2n = ak2 ** n

        num = ak2 * (n * ak2n - ak2n + 1.0) - n * ak2n
        denom = k * (a**2 * k**2 - 1.0) * (ak2n - 1.0)

        eps = 1e-14
        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / (denom + eps)

        return np.where(np.abs(k) < eps, 0.0, out)

    def _d_k_dlnN_dk(self, k):
        """
        Analytic d/dk [ k d/dk ln N(k) ].
        """
        k = np.asarray(k, dtype=float)
        a = self.vF / self.t1
        n = self.n

        ak = a * k
        ak2 = ak * ak
        ak2n = ak2 ** n

        num = (
            -2.0 * n**2 * ak2 * ak2 * ak2n
            + 2.0 * ak2 * (2.0 * n**2 * ak2n + (ak2n - 1.0)**2)
            - 2.0 * n**2 * ak2n
        )

        denom = k * (ak2 - 1.0)**2 * (ak2n - 1.0)**2

        eps = 1e-14
        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / (denom + eps)

        return np.where(np.abs(k) < eps, 0.0, out)

    def _Omega_emb_B(self, k):
        """
        Embedding Berry curvature for the B component.
        """
        k = np.asarray(k, dtype=float)
        eps = 1e-14
        d_k_term = self._d_k_dlnN_dk(k)

        with np.errstate(divide="ignore", invalid="ignore"):
            out = d_k_term / (k + eps)

        return np.where(np.abs(k) < eps, 0.0, out)

    def _Omega_emb_A(self, k):
        """
        Ω_emb^(A)(k) = -Ω_emb^(B)(k).
        """
        return -self._Omega_emb_B(k)

    def _F_radial(self, k):
        """
        F(k) = -t1 * (-a k)^n / N(k)^2, with a = vF/t1.
        """
        k = np.asarray(k, dtype=float)
        a = self.vF / self.t1
        N = self.N_k(k)
        return -self.t1 * (-a * k) ** self.n / (N ** 2)

    def _radial_derivative(self, func, k, rel_step=1e-5):
        """
        Central finite-difference derivative of a radial function.
        """
        k = np.asarray(k, dtype=float)
        h = rel_step * (1.0 + np.abs(k))

        k_plus = k + h
        k_minus = k - h

        f_plus = func(k_plus)
        f_minus = func(k_minus)

        with np.errstate(divide="ignore", invalid="ignore"):
            deriv = (f_plus - f_minus) / (2.0 * h)

        return np.where(h == 0.0, 0.0, deriv)

    def _beta_prime(self, k):
        """
        β'(k) = [V F' - F V'] / E^2.
        """
        k = np.asarray(k, dtype=float)

        F = self._F_radial(k)
        V = self.V_k(k)
        E2 = F * F + V * V

        Fp = self._radial_derivative(self._F_radial, k)
        Vp = self._radial_derivative(self.V_k, k)

        eps = 1e-14
        with np.errstate(divide="ignore", invalid="ignore"):
            beta_p = (V * Fp - F * Vp) / (E2 + eps)

        return np.where(E2 < eps, 0.0, beta_p)

    def _dcosbeta_dk(self, k):
        """
        d/dk cos β(k) using d/dk cosβ = -sinβ β'.
        """
        k = np.asarray(k, dtype=float)
        F = self._F_radial(k)
        V = self.V_k(k)
        E = np.sqrt(F * F + V * V)

        eps = 1e-14
        sinb = F / (E + eps)
        beta_p = self._beta_prime(k)

        return -sinb * beta_p

    def _berry_curvature_full_radial(self, k, band=+1):
        """
        Ω_full,±(k) using the compact radial formula with embedding terms.
        """
        k = np.asarray(k, dtype=float)
        eps = 1e-14
        k_safe = np.where(np.abs(k) < eps, eps, k)

        F = self._F_radial(k)
        V = self.V_k(k)
        E = np.sqrt(F * F + V * V)
        E_safe = E + eps

        sinb = F / E_safe
        cosb = V / E_safe

        beta_p = self._beta_prime(k)
        dcosb_dk = self._dcosbeta_dk(k)

        dlnN = self._dlnN_dk(k)
        d_kdlnN = self._d_k_dlnN_dk(k)

        s = -band

        term1 = (self.n / (2.0 * k_safe)) * sinb * beta_p
        term2 = (cosb / k_safe) * d_kdlnN
        term3 = dcosb_dk * dlnN

        Omega = s * (term1 + term2 + term3)
        small = np.abs(k) < 1e-8
        a = self.vF / self.t1

        # band here is already ±1 inside _berry_curvature_full_radial
        V0 = self.V_k(0.0)
        cosb0 = np.sign(V0)

        Omega0 = (-band) * cosb0 * 2.0 * a**2

        return np.where(small, Omega0, Omega)

    def berry_curvature_full(self, kx, ky, band=+1):
        """
        Wrapper: Ω_full(kx, ky) = Ω_full(|k|).
        """
        k = np.hypot(kx, ky)
        band = self.analytic_band_from_index(band)
        return self._berry_curvature_full_radial(k, band=band)

    def g_xy_imag(self, kx, ky, kz=0, band=+1):
        """
        Imaginary QGT component from the analytic Berry-curvature formula.

        The library convention is Omega_xy = -2 Im(Q_xy), so
        Im(Q_xy) = -Omega_xy / 2.
        """
        return -0.5 * self.berry_curvature_full(kx, ky, band=band)
