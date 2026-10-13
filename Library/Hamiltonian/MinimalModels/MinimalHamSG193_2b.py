from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class MinimalHamSG193_2b(MinimalAltermagnetHamiltonian):
    """SG 193 at Wyckoff 2b; Roig et al., Table III.

    With ci = cos(ki), si = sin(ki), and ci_2/si_2 at ki/2,
    fx = sx + sx_2 cos(sqrt(3) ky/2),
    and fy = sqrt(3) cx_2 sin(sqrt(3) ky/2):
        tx = t3 cz_2
        tz = t4 sz fx (fx**2 - 3 fy**2)
        lambda_x = 2 lamb cz_2 fx fy
        lambda_y = lamb cz_2 (fx**2 - fy**2)
        lambda_z = lamb_z sz_2 fy (fy**2 - 3 fx**2).

    Source: Physical Review B 110, 144412 (2024), Table III.
    """

    space_group = 193
    wyckoff_positions = ("2b",)
    source_table = "III"
    crystal_system = "hexagonal"

    def _model_terms(self, kx, ky, kz):
        trig = self._trig_terms(kx, ky, kz)
        tx = self.t3 * trig.cz_2
        tz = self.t4 * trig.sz * trig.fx * (trig.fx**2 - 3 * trig.fy**2)
        lam_x = 2 * self.lamb * trig.cz_2 * trig.fx * trig.fy
        lam_y = self.lamb * trig.cz_2 * (trig.fx**2 - trig.fy**2)
        lam_z = self.lamb_z * trig.sz_2 * trig.fy * (trig.fy**2 - 3 * trig.fx**2)
        return self._eps0(trig), tx, tz, lam_x, lam_y, lam_z
