from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class MinimalHamSG131_2c2d(MinimalAltermagnetHamiltonian):
    """SG 131, Wyckoff 2c/2d, from Roig et al., PRB 110, 144412, Table IV.

    For ``s_i=sin(k_i)``, ``c_i=cos(k_i)`` and half-argument subscripts::

        tx = t3*c_x/2*c_y/2*c_z/2
        tz = t4*(c_x-c_y)
        lambda_x = lamb*c_x/2*s_y/2*s_z/2
        lambda_y = lamb*s_x/2*c_y/2*s_z/2
        lambda_z = lamb_z*s_x/2*s_y/2*c_z/2
    """

    space_group = 131
    wyckoff_positions = ("2c", "2d")
    source_table = "IV"
    crystal_system = "tetragonal"

    def _model_terms(self, kx, ky, kz):
        q = self._trig_terms(kx, ky, kz)
        tx = self.t3 * q.cx_2 * q.cy_2 * q.cz_2
        tz = self.t4 * (q.cx - q.cy)
        lam_x = self.lamb * q.cx_2 * q.sy_2 * q.sz_2
        lam_y = self.lamb * q.sx_2 * q.cy_2 * q.sz_2
        lam_z = self.lamb_z * q.sx_2 * q.sy_2 * q.cz_2
        return self._eps0(q), tx, tz, lam_x, lam_y, lam_z
