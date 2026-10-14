from qgt.hamiltonians.minimal_models.MinimalAltermagnetHamiltonian import (
    MinimalAltermagnetHamiltonian,
)


class MinimalHamSG49_2a2b2c2d(MinimalAltermagnetHamiltonian):
    """SG 49, Wyckoff 2a/2b/2c/2d, from Roig et al., PRB 110, 144412, Table IV.

    ``tx = t3*cos(kz/2)``, ``tz = t4*sin(kx)*sin(ky)``. SOC is
    ``(lamb*sin(kx)*sin(kz/2), lamb_y*sin(ky)*sin(kz/2),
    lamb_z*cos(kz/2))``; the three orthorhombic amplitudes are independent.
    """

    space_group = 49
    wyckoff_positions = ("2a", "2b", "2c", "2d")
    source_table = "IV"
    crystal_system = "orthorhombic"

    def __init__(self, *args, lamb_y=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lamb_y = lamb_y

    def _model_terms(self, kx, ky, kz):
        q = self._trig_terms(kx, ky, kz)
        tx = self.t3 * q.cz_2
        tz = self.t4 * q.sx * q.sy
        lam_x = self.lamb * q.sx * q.sz_2
        lam_y = self.lamb_y * q.sy * q.sz_2
        lam_z = self.lamb_z * q.cz_2
        return self._eps0(q), tx, tz, lam_x, lam_y, lam_z
