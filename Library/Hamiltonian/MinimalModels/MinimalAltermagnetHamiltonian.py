import numpy as np
from types import SimpleNamespace
from ..Hamiltonian import hamiltonian


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

    t1 and t2 set an illustrative symmetry-invariant scalar dispersion;
    t3 and t4 multiply the first intersublattice and sublattice-odd form
    factors. lamb and lamb_z set SOC amplitudes. Concrete models document
    any extra independent amplitudes. All parameters remain mutable.
    """
    def __init__(
        self,
        t1=0.8,
        t2=0.2,
        t3=0.3,
        t4=0.3,
        mu=0.0,
        Jx=0.0,
        Jy=0.0,
        Jz=0.2,
        lamb=0.1,
        lamb_z=0.1,
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
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.mu = mu
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.lamb = lamb
        self.lamb_z = lamb_z

    def _eps0(self, trig):
        """Scalar dispersion in the model's lattice coordinates."""
        if self.crystal_system in ("trigonal", "hexagonal"):
            return (self.t1 * (trig.cx + 2 * trig.cx_2 * trig.csq3ky_2)
                    + self.t2 * trig.cz - self.mu)
        if self.crystal_system == "cubic":
            return (self.t1 * (trig.cx + trig.cy + trig.cz)
                    + self.t2 * (trig.cx * trig.cy + trig.cy * trig.cz + trig.cz * trig.cx)
                    - self.mu)
        return self.t1 * (trig.cx + trig.cy) + self.t2 * trig.cz - self.mu

    def get_sym_path(self, path=None):
        """Return a conventional path in dimensionless momentum coordinates.

        Orthorhombic/monoclinic points use reciprocal phase coordinates with
        boundaries at pi; monoclinic metric-dependent BZ vertices require
        lattice information beyond the minimal model. Trigonal models use
        the Cartesian hexagonal setting of the paper's fx and fy.
        """
        pi = np.pi
        if self.crystal_system in ("trigonal", "hexagonal"):
            y = 2 * pi / np.sqrt(3.0)
            points = {
                "G": (0, 0, 0), "M": (0, y, 0), "K": (2 * pi / 3, y, 0),
                "A": (0, 0, pi), "L": (0, y, pi), "H": (2 * pi / 3, y, pi),
            }
            default = "GMKGALHA"
        elif self.crystal_system == "cubic":
            points = {"G": (0, 0, 0), "X": (pi, 0, 0), "M": (pi, pi, 0), "R": (pi, pi, pi)}
            default = "GXMGRXMR"
        elif self.crystal_system == "tetragonal":
            points = {
                "G": (0, 0, 0), "X": (pi, 0, 0), "M": (pi, pi, 0),
                "Z": (0, 0, pi), "R": (pi, 0, pi), "A": (pi, pi, pi),
            }
            default = "GXMGZRAZ"
        else:
            points = {
                "G": (0, 0, 0), "X": (pi, 0, 0), "Y": (0, pi, 0),
                "Z": (0, 0, pi), "S": (pi, pi, 0), "T": (0, pi, pi),
                "U": (pi, 0, pi), "R": (pi, pi, pi),
            }
            default = "GXSYGZURTZ"
        labels = list(default if path is None else path)
        unknown = [label for label in labels if label not in points]
        if unknown:
            raise ValueError(
                f"Unknown SG{self.space_group} symmetry labels {unknown}. "
                f"Valid labels are: {''.join(sorted(points))}."
            )
        return {label: np.array(point, dtype=float) for label, point in points.items()}, labels

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

    def _broadcast_model_terms(self, kx_arr, ky_arr, kz_arr):
        """Broadcast momentum coordinates and constant coefficient functions.

        Scalar calls return a one-point batch.
        """
        kx, ky, kz = np.broadcast_arrays(kx_arr, ky_arr, kz_arr)
        shape = kx.shape or (1,)
        terms = self._model_terms(kx, ky, kz)
        return tuple(np.broadcast_to(term, shape) for term in terms)

    def compute_static_vectorized(self, kx_arr, ky_arr, kz_arr=0):
        """Return H with broadcast momentum shape + (4, 4).

        Scalar coordinates return a one-point batch of shape (1, 4, 4).
        """
        terms = self._broadcast_model_terms(kx_arr, ky_arr, kz_arr)
        basis = (tau_0, tau_x, tau_z, tau_y_sigma_x,
                 tau_y_sigma_y, tau_y_sigma_z)
        H = np.zeros(terms[0].shape + (4, 4), dtype=complex)
        for coefficient, matrix in zip(terms, basis):
            H += coefficient[..., None, None] * matrix
        H += self.Jx * tau_z_sigma_x
        H += self.Jy * tau_z_sigma_y
        H += self.Jz * tau_z_sigma_z
        return H

    def get_analytical_eigenvalues(self, kx_arr, ky_arr, kz_arr=0):
        eps0, tx, tz, lam_x, lam_y, lam_z = self._broadcast_model_terms(
            kx_arr, ky_arr, kz_arr
        )

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
