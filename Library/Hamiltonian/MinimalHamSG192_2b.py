import numpy as np
from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class MinimalHamSG192_2b(MinimalAltermagnetHamiltonian):
    """
    Minimal altermagnet model for SG 192 with Wyckoff site 2b.

    Table III coefficients:
        tau_x         : cos(kz/2)
        tau_z         : fx fy (fx^2 - 3 fy^2) (3 fx^2 - fy^2)
        tau_y sigma_x : lambda sin(kz/2) fx
        tau_y sigma_y : lambda sin(kz/2) fy
        tau_y sigma_z : cos(kz/2)

    where
        fx = sin(kx) + sin(kx/2) cos(sqrt(3) ky/2)
        fy = sqrt(3) cos(kx/2) sin(sqrt(3) ky/2).
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
            mu=mu,
            Jx=Jx,
            Jy=Jy,
            Jz=Jz,
            omega=omega,
            A0=A0,
            magnus_order=magnus_order,
            polarization=polarization,
            analytic_magnus=analytic_magnus,
        )
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.lamb = lamb
        self.lamb_z = lamb_z

    def get_sym_path(self, path="GMKGALHA"):
        """
        Standard hexagonal/D6h high-symmetry path.

        Default path: G -> M -> K -> G -> A -> L -> H -> A.
        """
        kM_y = 2.0 * np.pi / np.sqrt(3.0)
        kK_x = 2.0 * np.pi / 3.0
        kz_A = np.pi

        sym_points = {
            "G": np.array([0.0, 0.0, 0.0]),
            "M": np.array([0.0, kM_y, 0.0]),
            "K": np.array([kK_x, kM_y, 0.0]),
            "A": np.array([0.0, 0.0, kz_A]),
            "L": np.array([0.0, kM_y, kz_A]),
            "H": np.array([kK_x, kM_y, kz_A]),
        }

        path_labels = list(path)
        unknown_labels = [label for label in path_labels if label not in sym_points]
        if unknown_labels:
            valid_labels = "".join(sorted(sym_points))
            raise ValueError(
                f"Unknown SG192 symmetry labels {unknown_labels}. Valid labels are: {valid_labels}."
            )

        return sym_points, path_labels

    def _model_terms(self, kx, ky, kz):
        trig = self._trig_terms(kx, ky, kz)

        eps0 = self.t1 * (trig.cx + 2 * trig.cx_2 * trig.csq3ky_2) + self.t2 * trig.cz - self.mu
        tx = self.t3 * trig.cz_2
        tz = self.t4 * trig.fx * trig.fy * (trig.fx**2 - 3 * trig.fy**2) * (3 * trig.fx**2 - trig.fy**2)

        lam_x = self.lamb * trig.sz_2 * trig.fx
        lam_y = self.lamb * trig.sz_2 * trig.fy
        lam_z = self.lamb_z * trig.cz_2

        return eps0, tx, tz, lam_x, lam_y, lam_z
