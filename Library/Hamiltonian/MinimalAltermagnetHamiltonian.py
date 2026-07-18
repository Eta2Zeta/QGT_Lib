import numpy as np
from types import SimpleNamespace
from .Hamiltonian import hamiltonian


sigma_0 = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

tau_x = np.kron(sigma_x, sigma_0)
tau_y = np.kron(sigma_y, sigma_0)
tau_z = np.kron(sigma_z, sigma_0)
tau_0 = np.kron(sigma_0, sigma_0)

tau_y_sigma_x = np.kron(sigma_y, sigma_x)
tau_y_sigma_y = np.kron(sigma_y, sigma_y)
tau_y_sigma_z = np.kron(sigma_y, sigma_z)

tau_z_sigma_x = np.kron(sigma_z, sigma_x)
tau_z_sigma_y = np.kron(sigma_z, sigma_y)
tau_z_sigma_z = np.kron(sigma_z, sigma_z)


class MinimalAltermagnetHamiltonian(hamiltonian):
    """
    Shared two-site, two-spin minimal model for altermagnets.

    The model has the paper's common form

        H(k) = eps0(k) tau_0 + tx(k) tau_x + tz(k) tau_z
             + tau_y [lambda(k) . sigma] + tau_z [J . sigma].

    Subclasses specify only the material/symmetry-dependent coefficient
    functions by implementing _model_terms(kx, ky, kz).
    """
    def __init__(
        self,
        *,
        mu=0.0,
        Jx=0.0,
        Jy=0.0,
        Jz=0.0,
        omega=0.0,
        A0=0.0,
        magnus_order=1,
        polarization='left',
        analytic_magnus=False,
    ):
        super().__init__(
            dim=4,
            omega=omega,
            A0=A0,
            polarization=polarization,
            magnus_order=magnus_order,
            analytic_magnus=analytic_magnus,
        )
        self.mu = mu
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz

    def _model_terms(self, kx, ky, kz):
        """
        Return eps0, tx, tz, lambda_x, lambda_y, lambda_z.
        """
        raise NotImplementedError("Subclasses must implement _model_terms().")

    def _trig_terms(self, kx, ky, kz):
        """
        Common trig and form-factor terms used by minimal models.
        """
        sqrt3 = np.sqrt(3.0)
        cx = np.cos(kx)
        cy = np.cos(ky)
        cz = np.cos(kz)
        sx = np.sin(kx)
        sy = np.sin(ky)
        sz = np.sin(kz)
        cx_2 = np.cos(kx / 2.0)
        cy_2 = np.cos(ky / 2.0)
        cz_2 = np.cos(kz / 2.0)
        sx_2 = np.sin(kx / 2.0)
        sy_2 = np.sin(ky / 2.0)
        sz_2 = np.sin(kz / 2.0)
        csq3ky_2 = np.cos(sqrt3 * ky / 2.0)
        ssq3ky_2 = np.sin(sqrt3 * ky / 2.0)
        fx = sx + sx_2 * csq3ky_2
        fy = sqrt3 * cx_2 * ssq3ky_2

        return SimpleNamespace(
            sqrt3=sqrt3,
            cx=cx,
            cy=cy,
            cz=cz,
            sx=sx,
            sy=sy,
            sz=sz,
            cx_2=cx_2,
            cy_2=cy_2,
            cz_2=cz_2,
            sx_2=sx_2,
            sy_2=sy_2,
            sz_2=sz_2,
            csq3ky_2=csq3ky_2,
            ssq3ky_2=ssq3ky_2,
            fx=fx,
            fy=fy,
            fz=sz,
        )

    def compute_static(self, kx, ky, kz=0):
        eps0, tx, tz, lam_x, lam_y, lam_z = self._model_terms(kx, ky, kz)

        H = eps0 * tau_0
        H += tx * tau_x
        H += tz * tau_z
        H += lam_x * tau_y_sigma_x
        H += lam_y * tau_y_sigma_y
        H += lam_z * tau_y_sigma_z
        H += self.Jx * tau_z_sigma_x
        H += self.Jy * tau_z_sigma_y
        H += self.Jz * tau_z_sigma_z

        return H

    def compute_static_vectorized(self, kx_arr, ky_arr, kz_arr=0):
        kx = np.asarray(kx_arr)
        ky = np.asarray(ky_arr)
        if np.ndim(kz_arr) == 0:
            kz = np.full_like(kx, kz_arr, dtype=float)
        else:
            kz = np.asarray(kz_arr)

        shape = kx.shape
        kx = kx.flatten()
        ky = ky.flatten()
        kz = kz.flatten()
        M = kx.size

        eps0, tx, tz, lam_x, lam_y, lam_z = self._model_terms(kx, ky, kz)

        H = np.zeros((M, 4, 4), dtype=complex)
        H += eps0[:, None, None] * tau_0[None, :, :]
        H += tx[:, None, None] * tau_x[None, :, :]
        H += tz[:, None, None] * tau_z[None, :, :]
        H += lam_x[:, None, None] * tau_y_sigma_x[None, :, :]
        H += lam_y[:, None, None] * tau_y_sigma_y[None, :, :]
        H += lam_z[:, None, None] * tau_y_sigma_z[None, :, :]
        H += self.Jx * tau_z_sigma_x[None, :, :]
        H += self.Jy * tau_z_sigma_y[None, :, :]
        H += self.Jz * tau_z_sigma_z[None, :, :]

        if len(shape) > 1:
            H = H.reshape(shape + (4, 4))

        return H

    def get_analytical_eigenvalues(self, kx_arr, ky_arr, kz_arr=0):
        kx = np.asarray(kx_arr)
        ky = np.asarray(ky_arr)
        if np.ndim(kz_arr) == 0:
            kz = np.full_like(kx, kz_arr, dtype=float)
        else:
            kz = np.asarray(kz_arr)

        shape = kx.shape
        kx = kx.flatten()
        ky = ky.flatten()
        kz = kz.flatten()
        eps0, tx, tz, lam_x, lam_y, lam_z = self._model_terms(kx, ky, kz)

        J2 = self.Jx**2 + self.Jy**2 + self.Jz**2
        lam2 = lam_x**2 + lam_y**2 + lam_z**2

        cross_x = lam_y * self.Jz - lam_z * self.Jy
        cross_y = lam_z * self.Jx - lam_x * self.Jz
        cross_z = lam_x * self.Jy - lam_y * self.Jx
        cross2 = cross_x**2 + cross_y**2 + cross_z**2

        X = tx**2 + tz**2 + lam2 + J2
        Y = np.sqrt(tz**2 * J2 + cross2)

        E1 = eps0 - np.sqrt(np.maximum(X + 2 * Y, 0))
        E2 = eps0 - np.sqrt(np.maximum(X - 2 * Y, 0))
        E3 = eps0 + np.sqrt(np.maximum(X - 2 * Y, 0))
        E4 = eps0 + np.sqrt(np.maximum(X + 2 * Y, 0))

        E = np.stack([E1, E2, E3, E4], axis=-1)

        if len(shape) > 1:
            E = E.reshape(shape + (4,))

        return E

    def get_spin_operator(self, component='z'):
        """
        Return tau_0 x sigma_i for component 'x', 'y', or 'z'.
        """
        if component == 'x':
            return np.kron(sigma_0, sigma_x)
        if component == 'y':
            return np.kron(sigma_0, sigma_y)
        if component == 'z':
            return np.kron(sigma_0, sigma_z)
        raise ValueError(f"Unknown spin component: {component}")
