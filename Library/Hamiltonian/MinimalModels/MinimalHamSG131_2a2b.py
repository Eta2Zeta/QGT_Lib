from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class MinimalHamSG131_2a2b(MinimalAltermagnetHamiltonian):
    """SG 131, Wyckoff 2a/2b, from Roig et al., PRB 110, 144412, Table IV.

    For ``s_i=sin(k_i)``, ``c_i=cos(k_i)`` and half-argument subscripts::

        tx = t3*c_z/2
        tz = t4*(c_x-c_y)
        lambda_x = lamb*s_y*s_z/2
        lambda_y = lamb*s_x*s_z/2
        lambda_z = lamb_z*s_x*s_y*c_z/2
    """

    space_group = 131
    wyckoff_positions = ("2a", "2b")
    source_table = "IV"
    crystal_system = "tetragonal"

    def _model_terms(self, kx, ky, kz):
        q = self._trig_terms(kx, ky, kz)
        tx = self.t3 * q.cz_2
        tz = self.t4 * (q.cx - q.cy)
        lam_x = self.lamb * q.sy * q.sz_2
        lam_y = self.lamb * q.sx * q.sz_2
        lam_z = self.lamb_z * q.sx * q.sy * q.cz_2
        return self._eps0(q), tx, tz, lam_x, lam_y, lam_z
