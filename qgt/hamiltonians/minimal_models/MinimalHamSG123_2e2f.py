from qgt.hamiltonians.minimal_models.MinimalAltermagnetHamiltonian import (
    MinimalAltermagnetHamiltonian,
)


class MinimalHamSG123_2e2f(MinimalAltermagnetHamiltonian):
    """SG 123, Wyckoff 2e/2f, from Roig et al., PRB 110, 144412, Table IV.

    For ``s_i=sin(k_i)``, ``c_i=cos(k_i)`` and half-argument subscripts::

        tx = t3*c_x/2*c_y/2
        tz = t4*(c_x-c_y)
        lambda_x = lamb*c_x/2*s_y/2*s_z
        lambda_y = lamb*s_x/2*c_y/2*s_z
        lambda_z = lamb_z*s_x/2*s_y/2
    """

    space_group = 123
    wyckoff_positions = ("2e", "2f")
    source_table = "IV"
    crystal_system = "tetragonal"

    def _model_terms(self, kx, ky, kz):
        q = self._trig_terms(kx, ky, kz)
        tx = self.t3 * q.cx_2 * q.cy_2
        tz = self.t4 * (q.cx - q.cy)
        lam_x = self.lamb * q.cx_2 * q.sy_2 * q.sz
        lam_y = self.lamb * q.sx_2 * q.cy_2 * q.sz
        lam_z = self.lamb_z * q.sx_2 * q.sy_2
        return self._eps0(q), tx, tz, lam_x, lam_y, lam_z
