from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class MinimalHamSG83_2e2f(MinimalAltermagnetHamiltonian):
    """SG 83, Wyckoff 2e/2f, from Roig et al., PRB 110, 144412, Table IV.

    Let ``s_i=sin(k_i)``, ``c_i=cos(k_i)`` and ``d=c_x-c_y``;
    subscripts ``i/2`` denote half arguments. The table gives::

        tx = t3*c_x/2*c_y/2 + t3_alt*s_x/2*s_y/2*d
        tz = t4*d + t4_alt*s_x*s_y
        lambda_x = (lamb*s_x/2*c_y/2 + lamb2*c_x/2*s_y/2)*s_z
        lambda_y = (-lamb*c_x/2*s_y/2 + lamb2*s_x/2*c_y/2)*s_z
        lambda_z = lamb_z*s_x/2*s_y/2 + lamb_z_alt*c_x/2*c_y/2*d

    ``lamb`` and ``lamb2`` map to the printed lambda_1 and lambda_2.
    """

    space_group = 83
    wyckoff_positions = ("2e", "2f")
    source_table = "IV"
    crystal_system = "tetragonal"

    def __init__(self, *args, t3_alt=0.15, t4_alt=0.2, lamb2=0.07, lamb_z_alt=0.08, **kwargs):
        super().__init__(*args, **kwargs)
        self.t3_alt = t3_alt
        self.t4_alt = t4_alt
        self.lamb2 = lamb2
        self.lamb_z_alt = lamb_z_alt

    def _model_terms(self, kx, ky, kz):
        q = self._trig_terms(kx, ky, kz)
        difference = q.cx - q.cy
        tx = self.t3 * q.cx_2 * q.cy_2 + self.t3_alt * q.sx_2 * q.sy_2 * difference
        tz = self.t4 * difference + self.t4_alt * q.sx * q.sy
        lam_x = (self.lamb * q.sx_2 * q.cy_2 + self.lamb2 * q.cx_2 * q.sy_2) * q.sz
        lam_y = (-self.lamb * q.cx_2 * q.sy_2 + self.lamb2 * q.sx_2 * q.cy_2) * q.sz
        lam_z = self.lamb_z * q.sx_2 * q.sy_2 + self.lamb_z_alt * q.cx_2 * q.cy_2 * difference
        return self._eps0(q), tx, tz, lam_x, lam_y, lam_z
