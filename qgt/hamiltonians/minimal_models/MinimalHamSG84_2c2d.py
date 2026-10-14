from qgt.hamiltonians.minimal_models.MinimalAltermagnetHamiltonian import (
    MinimalAltermagnetHamiltonian,
)


class MinimalHamSG84_2c2d(MinimalAltermagnetHamiltonian):
    """SG 84, Wyckoff 2c/2d, from Roig et al., PRB 110, 144412, Table IV.

    For ``s_i=sin(k_i)``, ``c_i=cos(k_i)``, ``d=c_x-c_y``, and
    half-argument subscripts::

        tx = (t3*c_x/2*c_y/2 + t3_alt*s_x/2*s_y/2)*c_z/2
        tz = t4*d + t4_alt*s_x*s_y
        lambda_x = (lamb*s_x/2*c_y/2 + lamb2*c_x/2*s_y/2)*s_z/2
        lambda_y = (-lamb*c_x/2*s_y/2 + lamb2*s_x/2*c_y/2)*s_z/2
        lambda_z = (lamb_z*c_x/2*c_y/2 + lamb_z_alt*s_x/2*s_y/2)*c_z/2*d

    ``lamb``/``lamb2`` are the table's lambda_1/lambda_2. Both printed
    z-SOC bases include ``d``; they are reproduced literally, without an
    inferred correction to the second term.
    """

    space_group = 84
    wyckoff_positions = ("2c", "2d")
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
        tx = (self.t3 * q.cx_2 * q.cy_2 + self.t3_alt * q.sx_2 * q.sy_2) * q.cz_2
        tz = self.t4 * difference + self.t4_alt * q.sx * q.sy
        lam_x = (self.lamb * q.sx_2 * q.cy_2 + self.lamb2 * q.cx_2 * q.sy_2) * q.sz_2
        lam_y = (-self.lamb * q.cx_2 * q.sy_2 + self.lamb2 * q.sx_2 * q.cy_2) * q.sz_2
        lam_z = (self.lamb_z * q.cx_2 * q.cy_2 + self.lamb_z_alt * q.sx_2 * q.sy_2) * q.cz_2 * difference
        return self._eps0(q), tx, tz, lam_x, lam_y, lam_z
