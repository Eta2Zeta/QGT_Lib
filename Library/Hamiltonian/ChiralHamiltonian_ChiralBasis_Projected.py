import numpy as np
from .Hamiltonian import hamiltonian
sigma_x = np.array([[0, 1],
                    [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j],
                    [1j, 0]], dtype=complex)
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
    def __init__(self, vF=542.1, t1=355.16, V=30.0, n=2, omega=2 * np.pi, A0=0, polarization='left', **kwargs):
        super().__init__(dim=2, omega=omega, A0=A0, polarization=polarization, **kwargs)
        self.vF = vF
        self.t1 = t1
        self.V = V
        self.n = n

        self.b1 = (2 * np.pi / 3) * np.array([1.0,  np.sqrt(3.0)])
        self.b2 = (2 * np.pi / 3) * np.array([1.0, -np.sqrt(3.0)])

    def get_sym_path(self):
        G = np.zeros(2)
        K = (2.0 * self.b1 + self.b2) / 3.0
        M = 0.5 * self.b1

        sym_points = {"G": G, "K": K, "M": M}
        path = ["G", "K", "M", "G"]
        return sym_points, path

    def k(self, kx, ky):
        return np.sqrt(kx**2 + ky**2)
    
    def valid_k_point(self, kx, ky):
        k = np.sqrt(kx**2 + ky**2)
        # return True
        return (self.vF * k / self.t1) < 1
    
    def theta(self, kx, ky):
        return np.arctan2(ky, kx)  # angle of k-vector

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
        k_minus = kx - 1j * ky
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
        vk_t1 = self.vF * k / self.t1
        N = self.N_k(k)
        V_k = self.V_k(k)

        # |off-diagonal|^2 = (vF^n * k^n)^2 / (t1^{2n-2} * N^4)
        h_off_sq = (self.vF**self.n * k**self.n / (self.t1**(self.n - 1) * N**2))**2

        E = np.sqrt(V_k**2 + h_off_sq)
        return np.array([-E, E])
    
#@ Derivatives
    def dN_dk(self, k):
        """
        This is a helper for the derivatives. 
        Analytical derivative ∂N/∂k as defined by:
        ∂N/∂k = [a^2 k^2 (n (a k)^{2n} - (a k)^{2n} + 1) - n (a k)^{2n}]
                / [k (1 - a^2 k^2)^2 * sqrt(((a k)^{2n} - 1) / (a^2 k^2 - 1))]
        """
        a = self.vF / self.t1
        n = self.n
        ak = a * k
        ak2 = ak**2
        ak2n = ak2**n
        ak2n_plus = ak**(2 * n)

        numerator = a**2 * k**2 * (n * ak2n - ak2n + 1) - n * ak2n
        denominator = (
            k * (1 - a**2 * k**2) ** 2 *
            np.sqrt((ak2n - 1) / (a**2 * k**2 - 1))
        )
        return numerator / denominator

    def d_x(self, kx, ky):
        k = self.k(kx, ky)
        theta = self.theta(kx, ky)
        N = self.N_k(k)
        return - (self.vF ** self.n / self.t1 ** (self.n - 1)) * k ** self.n * np.cos(self.n * theta) / N**2

    def d_y(self, kx, ky):
        k = self.k(kx, ky)
        theta = self.theta(kx, ky)
        N = self.N_k(k)
        return - (self.vF ** self.n / self.t1 ** (self.n - 1)) * k ** self.n * np.sin(self.n * theta) / N**2

    def d_z(self, kx, ky):
        k = self.k(kx, ky)
        a = self.vF / self.t1
        ak = a * k
        numerator = (self.n - 1) * ak ** (2 * self.n + 2) + ak ** 2 - self.n * ak ** (2 * self.n)
        denominator = (1 - ak**2) * (1 - ak ** (2 * self.n))
        return self.V * (-0.5 * (self.n - 1) + numerator / denominator)
    
    def d_squared(self, kx, ky):
        """
        Computes the squared magnitude |d|^2 = d_x^2 + d_y^2 + d_z^2.
        Useful for expressions requiring the norm squared of the d vector.
        """
        dx = self.d_x(kx, ky)
        dy = self.d_y(kx, ky)
        dz = self.d_z(kx, ky)
        return dx**2 + dy**2 + dz**2

    def ddx_dx(self, kx, ky):
        k = self.k(kx, ky)
        theta = self.theta(kx, ky)
        dNk = self.dN_dk(k)
        a = self.vF / self.t1
        N = self.N_k(k)
        n = self.n

        term1 = (2 * self.vF**n) / (self.t1**(n - 1) * N**3) * dNk * (kx / k) * k**n * np.cos(n * theta)
        term2 = -(self.vF**n * n * k**(n - 1)) / (self.t1**(n - 1) * N**2) * np.cos(n * theta - theta)
        return term1 + term2

    def ddy_dx(self, kx, ky):
        k = self.k(kx, ky)
        theta = self.theta(kx, ky)
        dNk = self.dN_dk(k)
        a = self.vF / self.t1
        N = self.N_k(k)
        n = self.n

        term1 = (2 * self.vF**n) / (self.t1**(n - 1) * N**3) * dNk * (kx / k) * k**n * np.sin(n * theta)
        term2 = -(self.vF**n * n * k**(n - 1)) / (self.t1**(n - 1) * N**2) * np.sin(n * theta - theta)
        return term1 + term2

    def ddz_dx(self, kx, ky):
        k = self.k(kx, ky)
        a = self.vF / self.t1
        n = self.n
        vkx = kx / k

        num = (-2 * a**4 * k**4 * n**2 * (a * k)**(2 * n)
               + 2 * a**2 * k**2 * (2 * n**2 * (a * k)**(2 * n) + ((a * k)**(2 * n) - 1)**2)
               - 2 * n**2 * (a * k)**(2 * n))

        denom = k * (1 - a**2 * k**2)**2 * ((a * k)**(2 * n) - 1)**2

        return self.V * vkx * num / denom


    def ddx_dy(self, kx, ky):
        k = self.k(kx, ky)
        theta = self.theta(kx, ky)
        dNk = self.dN_dk(k)
        a = self.vF / self.t1
        N = self.N_k(k)
        n = self.n

        term1 = (2 * self.vF**n) / (self.t1**(n - 1) * N**3) * dNk * (ky / k) * k**n * np.cos(n * theta)
        term2 = (self.vF**n * n * k**(n - 1)) / (self.t1**(n - 1) * N**2) * np.sin(n * theta - theta)
        return term1 + term2

    def ddy_dy(self, kx, ky):
        k = self.k(kx, ky)
        theta = self.theta(kx, ky)
        dNk = self.dN_dk(k)
        a = self.vF / self.t1
        N = self.N_k(k)
        n = self.n

        term1 = (2 * self.vF**n) / (self.t1**(n - 1) * N**3) * dNk * (ky / k) * k**n * np.sin(n * theta)
        term2 = -(self.vF**n * n * k**(n - 1)) / (self.t1**(n - 1) * N**2) * np.cos(n * theta - theta)
        return term1 + term2

    def ddz_dy(self, kx, ky):
        k = self.k(kx, ky)
        a = self.vF / self.t1
        n = self.n
        vky = ky / k

        num = (-2 * a**4 * k**4 * n**2 * (a * k)**(2 * n)
               + 2 * a**2 * k**2 * (2 * n**2 * (a * k)**(2 * n) + ((a * k)**(2 * n) - 1)**2)
               - 2 * n**2 * (a * k)**(2 * n))

        denom = k * (1 - a**2 * k**2)**2 * ((a * k)**(2 * n) - 1)**2

        return self.V * vky * num / denom

#@ Analytic Static
    def d_magnitude(self, kx, ky):
        """
        Computes the norm |d| = sqrt(d_x^2 + d_y^2 + d_z^2)
        using the analytical expression:
        |d| = sqrt(V(k)^2 + [v_F^n k^n]^2 / N^4)
        """
        k = self.k(kx, ky)
        V_k = self.d_z(kx, ky)  # this is the scalar V(k)
        N = self.N_k(k)
        off_diag_term = (self.vF ** self.n * k ** self.n / (self.t1 ** (self.n - 1))) / N**2
        return np.sqrt(V_k**2 + off_diag_term**2)
    
    def sum_x(self, kx, ky):
        """
        Computes the sum of d_x components.
        """

        # Individual components
        ddx_dx = self.ddx_dx(kx, ky)
        ddy_dx = self.ddy_dx(kx, ky)
        ddz_dx = self.ddz_dx(kx, ky)

        dx = self.d_x(kx, ky)
        dy = self.d_y(kx, ky)
        dz = self.d_z(kx, ky)

        return dx * ddx_dx + dy * ddy_dx + dz * ddz_dx
    
    def sum_y(self, kx, ky):
        """
        Computes the sum of d_y components.
        """

        # Individual components
        ddx_dy = self.ddx_dy(kx, ky)
        ddy_dy = self.ddy_dy(kx, ky)
        ddz_dy = self.ddz_dy(kx, ky)

        dx = self.d_x(kx, ky)
        dy = self.d_y(kx, ky)
        dz = self.d_z(kx, ky)

        return dx * ddx_dy + dy * ddy_dy + dz * ddz_dy

    def partial_x_dhat_dot_partial_x_dhat(self, kx, ky):
        """
        Computes ∂x d̂ ⋅ ∂x d̂ using the full expansion:
        (1/d^2)(∂x d_i)^2 - (2/d^4)(∑ d_i ∂x d_i) + (1/d^6)(∑ d_i^2)(∑ d_i ∂x d_i)^2
        """
        # Individual components
        ddx_dx = self.ddx_dx(kx, ky)
        ddy_dx = self.ddy_dx(kx, ky)
        ddz_dx = self.ddz_dx(kx, ky)

        d_norm = self.d_magnitude(kx, ky)
        d2 = d_norm**2
        d4 = d2**2

        dot_d_dxd = self.sum_x(kx, ky)

        term1 = (ddx_dx**2 + ddy_dx**2 + ddz_dx**2)/ d2
        term2 = - dot_d_dxd ** 2 / d4

        return term1 + term2 
    
    def partial_y_dhat_dot_partial_y_dhat(self, kx, ky):
        """
        Computes ∂y d̂ ⋅ ∂y d̂ using the full expansion:
        (1/d^2)(∂y d_i)^2 - (2/d^4)(∑ d_i ∂y d_i)^2 + (1/d^6)(∑ d_i^2)(∑ d_i ∂y d_i)^2
        """
        # ∂y d_i components
        ddx_dy = self.ddx_dy(kx, ky)
        ddy_dy = self.ddy_dy(kx, ky)
        ddz_dy = self.ddz_dy(kx, ky)

        # d_i values
        dx = self.d_x(kx, ky)
        dy = self.d_y(kx, ky)
        dz = self.d_z(kx, ky)

        d_norm = self.d_magnitude(kx, ky)
        d2 = d_norm**2
        d4 = d2**2
        d6 = d2**3

        # ∑ d_i ∂y d_i
        dot_d_dyd = dx * ddx_dy + dy * ddy_dy + dz * ddz_dy

        # (∂y d_x)^2 + (∂y d_y)^2 + (∂y d_z)^2
        term1 = (ddx_dy**2 + ddy_dy**2 + ddz_dy**2) / d2

        # -2 * (∑ d_i ∂y d_i)^2 / d^4
        term2 = -2 * dot_d_dyd**2 / d4

        # (∑ d_i^2) * (∑ d_i ∂y d_i)^2 / d^6
        d_squared_sum = dx**2 + dy**2 + dz**2
        term3 = dot_d_dyd**2 * d_squared_sum / d6

        return term1 + term2 + term3


    def g_xx(self, kx, ky):
        """
        Returns the quantum metric component g_xx(k) = 1/4 * (∂x d̂ ⋅ ∂x d̂)
        """
        return 0.25 * self.partial_x_dhat_dot_partial_x_dhat(kx, ky)
    
    def g_yy(self, kx, ky):
        """
        Returns the quantum metric component g_yy(k) = 1/4 * (∂y d̂ ⋅ ∂y d̂)
        """
        return 0.25 * self.partial_y_dhat_dot_partial_y_dhat(kx, ky)

#@ Hard Analytic Trace
    def H(self, k):
        """
        Computes the analytical expression H(k) used in trace formula.
        H = V * [−2 a⁴ k⁴ n² (a k)^{2n} + 2 a² k² (2 n² (a k)^{2n} + ((a k)^{2n} − 1)²) − 2 n² (a k)^{2n}]
                / [ (1 − a² k²)² ((a k)^{2n} − 1)² ]
        """
        a = self.vF / self.t1
        n = self.n
        ak = a * k
        ak2 = ak**2
        ak2n = ak2**n
        V = self.V

        num = (-2 * a**4 * k**4 * n**2 * ak2n +
            2 * a**2 * k**2 * (2 * n**2 * ak2n + (ak2n - 1)**2) -
            2 * n**2 * ak2n)

        denom = (1 - a**2 * k**2)**2 * (ak2n - 1)**2
        return V * num / denom
        
    def M(self, k):
        """
        Computes the analytical expression M(k) used in trace formula.
        M = [a^2 k^2 (n (a k)^{2n} - (a k)^{2n} + 1) - n (a k)^{2n}] / (1 - a^2 k^2)^2
        """
        a = self.vF / self.t1
        n = self.n
        ak = a * k
        ak2n = (ak ** (2 * n))
        
        numerator = a**2 * k**2 * (n * ak2n - ak2n + 1) - n * ak2n
        denominator = (1 - a**2 * k**2)**2
        return numerator / denominator
    

    def trace(self, kx, ky):
        """
        Computes the analytical expression for the trace of the quantum metric tensor Tr[g],
        based on Appendix B formula using M(k), H(k), and V_k(k).
        """
        k = self.k(kx, ky)
        n = self.n
        vF = self.vF
        t1 = self.t1
        d = self.d_magnitude(kx, ky)

        k_pow_4n_2 = k ** (4 * n - 2)
        k_pow_2n_2 = k ** (2 * n - 2)

        N = self.N_k(k)
        N2 = N**2
        N4 = N2**2
        N6 = N2**3
        N8 = N4**2

        M_val = self.M(k)
        H_val = self.H(k)
        V_k_val = self.V_k(k)

        term1 = (vF ** (4 * n) * n**2 * k_pow_4n_2) / (t1 ** (4 * n - 4) * N8)
        term2 = (vF ** (2 * n) * H_val**2 * k_pow_2n_2) / (t1 ** (2 * n - 2) * N4)
        term3 = (4 * vF ** (2 * n) * M_val**2 * V_k_val**2 * k_pow_2n_2) / (t1 ** (2 * n - 2) * N8)
        term4 = (2 * vF ** (2 * n) * n**2 * V_k_val**2 * k_pow_2n_2) / (t1 ** (2 * n - 2) * N4)
        term5 = (-4 * vF ** (2 * n) * M_val * n * V_k_val**2 * k_pow_2n_2) / (t1 ** (2 * n - 2) * N6)
        term6 = (4 * vF ** (2 * n) * M_val * H_val * V_k_val * k_pow_2n_2) / (t1 ** (2 * n - 2) * N6)
        term7 = (-2 * vF ** (2 * n) * n * V_k_val * H_val * k_pow_2n_2) / (t1 ** (2 * n - 2) * N4)

        return 1/4 * (term1 + term2 + term3 + term4 + term5 + term6 + term7)/d**4

#@ Analytic Manugs Frist term
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
        # (vF/t1)^n
        pref = (self.vF / self.t1) ** self.n
        C = -self.t1 * pref * self.n * (k ** (self.n - 1)) * self.A0 * np.exp(1j * (self.n - 1) * theta) / (N ** 2)
        return C

    def _dlnN_chiral(self, kx, ky):
        """
        Return (∂_- ln N, ∂_+ ln N) for a radial N(k) = N(|k|).

        For a radial function f(k), ∂x f = f'(k) * kx/k, ∂y f = f'(k) * ky/k.
        Then
            ∂_- = 1/2 (∂x + i ∂y),
            ∂_+ = 1/2 (∂x - i ∂y).
        """
        k = np.hypot(kx, ky)
        if k == 0.0:
            # at k=0 the drive is anyway regularized; just return 0
            return 0.0 + 0.0j, 0.0 + 0.0j

        N = self.N_k(k)
        dNdk = self.dN_dk(k)  # you already implemented this
        # gradients of ln N
        dlnNdk = dNdk / N

        # ∂x ln N, ∂y ln N
        dlnN_dx = dlnNdk * (kx / k)
        dlnN_dy = dlnNdk * (ky / k)

        # chiral derivatives
        dlnN_minus = 0.5 * (dlnN_dx + 1j * dlnN_dy)  # ∂_-
        dlnN_plus  = 0.5 * (dlnN_dx - 1j * dlnN_dy)  # ∂_+

        return dlnN_minus, dlnN_plus

    def analytic_magnus_first_term(self, kx, ky, kz=0, return_parts=False):
        """
        This is the magnus first term when we drive the 2x2 Hamiltonian directly, not the full one.
        """
        # ----- shared k-geometry -----
        k = np.hypot(kx, ky)
        if k == 0.0:
            # safest: no correction right at the origin
            H_zero = np.zeros((2, 2), dtype=complex)
            return (H_zero, H_zero) if return_parts else H_zero

        # orbital coefficient C(k)
        C = self._C_orbital(kx, ky)
        absC2 = (np.abs(C) ** 2)

        # chiral derivatives of ln N
        dlnN_minus, dlnN_plus = self._dlnN_chiral(kx, ky)

        # ----- Part 1: orb–orb → σ_z -----
        # [H_{+1}^{orb}, H_{-1}^{orb}] = -|C|^2 σ_z
        # divide by ω for the Magnus term
        H_orb_orb = -(absC2 / self.omega) * sigma_z

        # ----- Part 2: cross terms → σ_+, σ_- -----
        # (1/ω) 2 B_- C^* σ_+  - (1/ω) 2 C B_+ σ_-
        # with B_- = -ω A0 ∂_- ln N,  B_+ = +ω A0 ∂_+ ln N
        # ⇒ H_cross = -2 A0 [ (∂_- ln N) C^* σ_+ + (∂_+ ln N) C σ_- ]
        H_cross = 2.0 * self.A0 * (
            dlnN_plus * np.conj(C) * sigma_p +
            dlnN_minus  * C         * sigma_m
        )

        if return_parts:
            return H_orb_orb, H_cross
        else:
            return H_orb_orb + H_cross

    def analytic_magnus_first_term_projected(self, kx, ky, kz=0):
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
            
        return val * sigma_z

# @ Analytic Berry Curvature corrected with projection terms
    # Helper functions
    # --- Analytic log-derivatives of N(k) ---

    def dlnN_dk_analytic(self, k):
        """
        Analytic d/dk ln N(k), using the closed form from the derivation:

        d/dk ln N =
            [ (ak)^2 ( n (ak)^{2n} - (ak)^{2n} + 1 ) - n (ak)^{2n} ]
            / [ k (a^2 k^2 - 1) ((ak)^{2n} - 1) ]

        with a = vF/t1,  ak = a*k.
        """
        k = np.asarray(k, dtype=float)
        a = self.vF / self.t1
        n = self.n

        ak = a * k
        ak2 = ak * ak
        ak2n = ak2 ** n   # (ak)^{2n}

        num = ak2 * (n * ak2n - ak2n + 1.0) - n * ak2n
        denom = k * (a**2 * k**2 - 1.0) * (ak2n - 1.0)

        eps = 1e-14
        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / (denom + eps)

        # regularize exactly at k=0
        out = np.where(np.abs(k) < eps, 0.0, out)
        return out

    def d_k_dlnN_dk(self, k):
        """
        Analytic d/dk [ k d/dk ln N(k) ], using the closed form:

        d/dk [ k d/dk ln N ] =
            [-2 n^2 (ak)^4 (ak)^{2n}
             + 2 (ak)^2 ( 2 n^2 (ak)^{2n} + ((ak)^{2n}-1)^2 )
             - 2 n^2 (ak)^{2n}] /
            [ k ((ak)^2 - 1)^2 ((ak)^{2n}-1)^2 ]
        """
        k = np.asarray(k, dtype=float)
        a = self.vF / self.t1
        n = self.n

        ak = a * k
        ak2 = ak * ak          # (ak)^2
        ak2n = ak2 ** n        # (ak)^{2n}

        num = (
            -2.0 * n**2 * ak2 * ak2 * ak2n
            + 2.0 * ak2 * (2.0 * n**2 * ak2n + (ak2n - 1.0)**2)
            - 2.0 * n**2 * ak2n
        )

        denom = k * (ak2 - 1.0)**2 * (ak2n - 1.0)**2

        eps = 1e-14
        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / (denom + eps)

        out = np.where(np.abs(k) < eps, 0.0, out)
        return out
        # --- Embedding Berry curvatures Ω_emb^(A/B)(k) ---

    def Omega_emb_B(self, k):
        """
        Embedding Berry curvature for the B component:

        Ω_emb^(B)(k) = (1/k) d/dk [ k d/dk ln N(k) ].
        """
        k = np.asarray(k, dtype=float)
        eps = 1e-14
        d_k_term = self.d_k_dlnN_dk(k)

        with np.errstate(divide="ignore", invalid="ignore"):
            out = d_k_term / (k + eps)

        out = np.where(np.abs(k) < eps, 0.0, out)
        return out

    def Omega_emb_A(self, k):
        """
        Ω_emb^(A)(k) = -Ω_emb^(B)(k)
        """
        return -self.Omega_emb_B(k)
    
        # === Radial helpers: F(k), E(k), beta(k) ===

    def F_radial(self, k):
        """
        F(k) = -t1 * (-a k)^n / N(k)^2, with a = vF/t1.

        This matches your definition
            F(k) = - t1 (-a k)^n / N(k)^2
        and works on scalars or arrays.
        """
        k = np.asarray(k, dtype=float)
        a = self.vF / self.t1
        N = self.N_k(k)  # already vectorized
        return -self.t1 * (-a * k) ** self.n / (N ** 2)

    def E_radial(self, k):
        """
        E(k) = sqrt(F(k)^2 + V(k)^2) = |d(k)|
        """
        k = np.asarray(k, dtype=float)
        F = self.F_radial(k)
        V = self.V_k(k)
        return np.sqrt(F * F + V * V)

    def beta_radial(self, k):
        """
        β(k) such that sinβ = F/E, cosβ = V/E.
        We can take β = arctan2(F, V).
        """
        k = np.asarray(k, dtype=float)
        F = self.F_radial(k)
        V = self.V_k(k)
        return np.arctan2(F, V)

        # === Generic radial derivative helper ===

    def _radial_derivative(self, func, k, rel_step=1e-5):
        """
        Central finite-difference derivative of a radial function func(k).

        func must accept numpy arrays and return arrays of same shape.
        """
        k = np.asarray(k, dtype=float)
        h = rel_step * (1.0 + np.abs(k))

        k_plus  = k + h
        k_minus = k - h

        f_plus  = func(k_plus)
        f_minus = func(k_minus)

        with np.errstate(divide="ignore", invalid="ignore"):
            deriv = (f_plus - f_minus) / (2.0 * h)

        deriv = np.where(h == 0.0, 0.0, deriv)
        return deriv


        # === β'(k) and d/dk cosβ(k) ===

    def beta_prime(self, k):
        """
        β'(k) via the identity:

            β'(k) = [V F' - F V'] / E^2,

        where F = F_radial(k), V = V_k(k), E^2 = F^2 + V^2.
        F' and V' are computed by a small radial finite difference.
        """
        k = np.asarray(k, dtype=float)

        F = self.F_radial(k)
        V = self.V_k(k)
        E2 = F * F + V * V

        # numeric derivatives of F(k) and V(k)
        Fp = self._radial_derivative(self.F_radial, k)
        Vp = self._radial_derivative(self.V_k,       k)

        eps = 1e-14
        with np.errstate(divide="ignore", invalid="ignore"):
            beta_p = (V * Fp - F * Vp) / (E2 + eps)

        beta_p = np.where(E2 < eps, 0.0, beta_p)
        return beta_p

    def dcosbeta_dk(self, k):
        """
        d/dk cos β(k) using d/dk cosβ = -sinβ β'.

        sinβ = F/E, with F = F_radial(k), E = E_radial(k).
        """
        k = np.asarray(k, dtype=float)
        F = self.F_radial(k)
        V = self.V_k(k)
        E = np.sqrt(F * F + V * V)

        eps = 1e-14
        E_safe = E + eps

        sinb = F / E_safe
        beta_p = self.beta_prime(k)

        return -sinb * beta_p

    # === Full projected Berry curvature in radial form ===

    def berry_curvature_full_radial(self, k, band=+1):
        """
        Ω_full,±(k) using your compact radial formula:

          Ω_±(k) =
            ∓ (n / (2k)) sinβ β'
            ∓ (cosβ / k) d/dk [ k d/dk ln N ]
            ∓ (d/dk cosβ) d/dk ln N.

        band = +1 → "+" band (conduction)
        band = -1 → "−" band (valence)
        """
        k = np.asarray(k, dtype=float)
        eps = 1e-14
        k_safe = np.where(np.abs(k) < eps, eps, k)

        # F, V, E, sinβ, cosβ
        F = self.F_radial(k)
        V = self.V_k(k)
        E = np.sqrt(F * F + V * V)
        E_safe = E + eps

        sinb = F / E_safe
        cosb = V / E_safe

        # derivatives
        beta_p   = self.beta_prime(k)
        dcosb_dk = self.dcosbeta_dk(k)

        dlnN      = self.dlnN_dk_analytic(k)
        d_kdlnN   = self.d_k_dlnN_dk(k)

        # overall sign ∓ → -band
        s = -band

        term1 = (self.n / (2.0 * k_safe)) * sinb * beta_p
        term2 = (cosb / k_safe) * d_kdlnN
        term3 = dcosb_dk * dlnN

        Omega = s * (term1 + term2 + term3)

        # regularize exactly at k=0
        Omega = np.where(np.abs(k) < eps, 0.0, Omega)
        return Omega

    def berry_curvature_full(self, kx, ky, band=+1):
        """
        Wrapper: Ω_full(kx, ky) = Ω_full(|k|) with |k| = sqrt(kx^2 + ky^2).
        """
        k = np.hypot(kx, ky)
        return self.berry_curvature_full_radial(k, band=band)
