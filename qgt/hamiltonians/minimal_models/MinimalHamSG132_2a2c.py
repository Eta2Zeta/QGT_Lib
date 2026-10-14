from qgt.hamiltonians.minimal_models.MinimalAltermagnetHamiltonian import (
    MinimalAltermagnetHamiltonian,
)


class MinimalHamSG132_2a2c(MinimalAltermagnetHamiltonian):
    """SG 132, Wyckoff 2a/2c, from Roig et al., PRB 110, 144412, Table IV.

    For ``s_i=sin(k_i)``, ``c_i=cos(k_i)`` and half-argument subscripts::

        tx = t3*c_z/2
        tz = t4*s_x*s_y
        lambda_x = lamb*s_x*s_z/2
        lambda_y = -lamb*s_y*s_z/2
        lambda_z = lamb_z*c_z/2*(c_x-c_y)
    """

    space_group = 132
    wyckoff_positions = ("2a", "2c")
    source_table = "IV"
    crystal_system = "tetragonal"

    def _model_terms(self, kx, ky, kz):
        q = self._trig_terms(kx, ky, kz)
        tx = self.t3 * q.cz_2
        tz = self.t4 * q.sx * q.sy
        lam_x = self.lamb * q.sx * q.sz_2
        lam_y = -self.lamb * q.sy * q.sz_2
        lam_z = self.lamb_z * q.cz_2 * (q.cx - q.cy)
        return self._eps0(q), tx, tz, lam_x, lam_y, lam_z
