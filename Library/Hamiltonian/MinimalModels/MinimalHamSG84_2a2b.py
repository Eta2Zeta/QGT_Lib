from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class MinimalHamSG84_2a2b(MinimalAltermagnetHamiltonian):
    """SG 84, Wyckoff 2a/2b, from Roig et al., PRB 110, 144412, Table IV.

    For ``s_i=sin(k_i)``, ``c_i=cos(k_i)`` and half-argument subscripts::

        tx = t3*c_z/2
        tz = t4*(c_x-c_y) + t4_alt*s_x*s_y
        lambda_x = (lamb*s_x + lamb2*s_y)*s_z/2
        lambda_y = (-lamb*s_y + lamb2*s_x)*s_z/2
        lambda_z = lamb_z*c_z/2*(c_x-c_y)

    ``lamb``/``lamb2`` are the table's lambda_1/lambda_2.
    """

    space_group = 84
    wyckoff_positions = ("2a", "2b")
    source_table = "IV"
    crystal_system = "tetragonal"

    def __init__(self, *args, t4_alt=0.2, lamb2=0.07, **kwargs):
        super().__init__(*args, **kwargs)
        self.t4_alt = t4_alt
        self.lamb2 = lamb2

    def _model_terms(self, kx, ky, kz):
        q = self._trig_terms(kx, ky, kz)
        tx = self.t3 * q.cz_2
        tz = self.t4 * (q.cx - q.cy) + self.t4_alt * q.sx * q.sy
        lam_x = (self.lamb * q.sx + self.lamb2 * q.sy) * q.sz_2
        lam_y = (-self.lamb * q.sy + self.lamb2 * q.sx) * q.sz_2
        lam_z = self.lamb_z * q.cz_2 * (q.cx - q.cy)
        return self._eps0(q), tx, tz, lam_x, lam_y, lam_z
