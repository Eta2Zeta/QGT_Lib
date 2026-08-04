import numpy as np

from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class MinimalHamSG124_2b2d(MinimalAltermagnetHamiltonian):
    """
    Minimal altermagnet model for SG 124 at Wyckoff position 2b or 2d.

    Table IV coefficients:
        tau_x         : cos(kz/2)
        tau_z         : sin(kx) sin(ky) [cos(kx) - cos(ky)]
        tau_y sigma_x : lambda sin(kx) sin(kz/2)
        tau_y sigma_y : lambda sin(ky) sin(kz/2)
        tau_y sigma_z : cos(kz/2)
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
        polarization="left",
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

    def get_sym_path(self, path="GXMGZRAZ"):
        """
        Standard primitive-tetragonal/D4h high-symmetry path.

        Default path: G -> X -> M -> G -> Z -> R -> A -> Z.
        """
        sym_points = {
            "G": np.array([0.0, 0.0, 0.0]),
            "X": np.array([np.pi, 0.0, 0.0]),
            "M": np.array([np.pi, np.pi, 0.0]),
            "Z": np.array([0.0, 0.0, np.pi]),
            "R": np.array([np.pi, 0.0, np.pi]),
            "A": np.array([np.pi, np.pi, np.pi]),
        }

        path_labels = list(path)
        unknown_labels = [label for label in path_labels if label not in sym_points]
        if unknown_labels:
            valid_labels = "".join(sorted(sym_points))
            raise ValueError(
                f"Unknown SG124 symmetry labels {unknown_labels}. "
                f"Valid labels are: {valid_labels}."
            )

        return sym_points, path_labels

    def _model_terms(self, kx, ky, kz):
        trig = self._trig_terms(kx, ky, kz)

        eps0 = self.t1 * (trig.cx + trig.cy) + self.t2 * trig.cz - self.mu
        tx = self.t3 * trig.cz_2
        tz = self.t4 * trig.sx * trig.sy * (trig.cx - trig.cy)

        lam_x = self.lamb * trig.sx * trig.sz_2
        lam_y = self.lamb * trig.sy * trig.sz_2
        lam_z = self.lamb_z * trig.cz_2

        return eps0, tx, tz, lam_x, lam_y, lam_z
