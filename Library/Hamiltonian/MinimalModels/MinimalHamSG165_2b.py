from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class MinimalHamSG165_2b(MinimalAltermagnetHamiltonian):
    """SG 165, Wyckoff 2b, from Roig et al., PRB 110, 144412, Table IV.

    Using the paper's periodic hexagonal ``fx, fy`` (see the shared
    ``_trig_terms``), and half-argument sine/cosine subscripts::

        tx = t3*c_z/2 + t3_alt*fy*(3*fx**2-fy**2)*s_z/2
        tz = t4*fx*(fx**2-3*fy**2)*s_z
             + t4_alt*fx*fy*(fx**2-3*fy**2)*(3*fx**2-fy**2)
        lambda_x = lamb*fx*s_z/2 + 2*lamb2*fx*fy*c_z/2
        lambda_y = lamb*fy*s_z/2 + lamb2*(fx**2-fy**2)*c_z/2
        lambda_z = lamb_z*c_z/2 + lamb_z_alt*fy*(3*fx**2-fy**2)*s_z/2

    ``lamb``/``lamb2`` are the table's lambda_1/lambda_2. The additional
    hopping and z-SOC amplitudes independently select the second bases.
    """

    space_group = 165
    wyckoff_positions = ("2b",)
    source_table = "IV"
    crystal_system = "trigonal"

    def __init__(self, *args, t3_alt=0.15, t4_alt=0.2, lamb2=0.07, lamb_z_alt=0.08, **kwargs):
        super().__init__(*args, **kwargs)
        self.t3_alt = t3_alt
        self.t4_alt = t4_alt
        self.lamb2 = lamb2
        self.lamb_z_alt = lamb_z_alt

    def _model_terms(self, kx, ky, kz):
        q = self._trig_terms(kx, ky, kz)
        second_x = q.fy * (3 * q.fx**2 - q.fy**2) * q.sz_2
        second_z = q.fx * q.fy * (q.fx**2 - 3 * q.fy**2) * (3 * q.fx**2 - q.fy**2)
        tx = self.t3 * q.cz_2 + self.t3_alt * second_x
        tz = self.t4 * q.fx * (q.fx**2 - 3 * q.fy**2) * q.sz + self.t4_alt * second_z
        lam_x = self.lamb * q.fx * q.sz_2 + 2 * self.lamb2 * q.fx * q.fy * q.cz_2
        lam_y = self.lamb * q.fy * q.sz_2 + self.lamb2 * (q.fx**2 - q.fy**2) * q.cz_2
        lam_z = self.lamb_z * q.cz_2 + self.lamb_z_alt * second_x
        return self._eps0(q), tx, tz, lam_x, lam_y, lam_z
