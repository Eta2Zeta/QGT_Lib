from qgt.hamiltonians.minimal_models.MinimalAltermagnetHamiltonian import (
    MinimalAltermagnetHamiltonian,
)


class MinimalHamSG176_2b(MinimalAltermagnetHamiltonian):
    """SG 176 at Wyckoff 2b; Roig et al., Table III.

    With ci = cos(ki), si = sin(ki), and ci_2/si_2 at ki/2,
    fx = sx + sx_2 cos(sqrt(3) ky/2),
    fy = sqrt(3) cx_2 sin(sqrt(3) ky/2),
    gx = fx (fx**2 - 3 fy**2), and gy = fy (fy**2 - 3 fx**2):
        tx = t3 cz_2
        tz = sz (t4 gx + t4_alt gy)
        lambda_x = cz_2 [lamb (fx**2 - fy**2) + 2 lamb2 fx fy]
        lambda_y = cz_2 [-2 lamb fx fy + lamb2 (fx**2 - fy**2)]
        lambda_z = sz_2 (lamb_z gx + lamb_z_alt gy).

    ``lamb`` and ``lamb2`` are the table's lambda_1 and lambda_2.
    Source: Physical Review B 110, 144412 (2024), Table III.
    """

    space_group = 176
    wyckoff_positions = ("2b",)
    source_table = "III"
    crystal_system = "hexagonal"

    def __init__(self, *args, t4_alt=0.2, lamb2=0.07, lamb_z_alt=0.08, **kwargs):
        super().__init__(*args, **kwargs)
        self.t4_alt = t4_alt
        self.lamb2 = lamb2
        self.lamb_z_alt = lamb_z_alt

    def _model_terms(self, kx, ky, kz):
        trig = self._trig_terms(kx, ky, kz)
        gx = trig.fx * (trig.fx**2 - 3 * trig.fy**2)
        gy = trig.fy * (trig.fy**2 - 3 * trig.fx**2)
        quadrupole = trig.fx**2 - trig.fy**2
        product = trig.fx * trig.fy
        tx = self.t3 * trig.cz_2
        tz = trig.sz * (self.t4 * gx + self.t4_alt * gy)
        lam_x = trig.cz_2 * (self.lamb * quadrupole + 2 * self.lamb2 * product)
        lam_y = trig.cz_2 * (-2 * self.lamb * product + self.lamb2 * quadrupole)
        lam_z = trig.sz_2 * (self.lamb_z * gx + self.lamb_z_alt * gy)
        return self._eps0(trig), tx, tz, lam_x, lam_y, lam_z
