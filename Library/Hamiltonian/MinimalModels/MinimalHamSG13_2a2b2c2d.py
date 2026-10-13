from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class MinimalHamSG13_2a2b2c2d(MinimalAltermagnetHamiltonian):
    """SG 13, Wyckoff 2a/2b/2c/2d, from Roig et al., PRB 110, 144412, Table IV.

    With ``s_i = sin(k_i)`` and ``c_i = cos(k_i)`` (``i/2`` means
    half the argument), the coefficients are::

        tx = t3*c_z/2 + t3_alt*s_x*s_z/2
        tz = t4*s_y*s_x + t4_alt*s_y*s_z
        lambda_x = lamb*c_z/2 + lamb_x_alt*s_x*s_z/2
        lambda_y = lamb_y*s_y*s_z/2 + lamb_y_alt*s_y*s_x*c_z/2
        lambda_z = lamb_z*c_z/2 + lamb_z_alt*s_x*s_z/2

    Every comma-separated basis in the table has its own amplitude.
    """

    space_group = 13
    wyckoff_positions = ("2a", "2b", "2c", "2d")
    source_table = "IV"
    crystal_system = "monoclinic"

    def __init__(
        self, *args, t3_alt=0.15, t4_alt=0.2, lamb_y=0.1,
        lamb_x_alt=0.04, lamb_y_alt=0.06, lamb_z_alt=0.08, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.t3_alt = t3_alt
        self.t4_alt = t4_alt
        self.lamb_y = lamb_y
        self.lamb_x_alt = lamb_x_alt
        self.lamb_y_alt = lamb_y_alt
        self.lamb_z_alt = lamb_z_alt

    def _model_terms(self, kx, ky, kz):
        q = self._trig_terms(kx, ky, kz)
        tx = self.t3 * q.cz_2 + self.t3_alt * q.sx * q.sz_2
        tz = self.t4 * q.sy * q.sx + self.t4_alt * q.sy * q.sz
        lam_x = self.lamb * q.cz_2 + self.lamb_x_alt * q.sx * q.sz_2
        lam_y = self.lamb_y * q.sy * q.sz_2 + self.lamb_y_alt * q.sy * q.sx * q.cz_2
        lam_z = self.lamb_z * q.cz_2 + self.lamb_z_alt * q.sx * q.sz_2
        return self._eps0(q), tx, tz, lam_x, lam_y, lam_z
