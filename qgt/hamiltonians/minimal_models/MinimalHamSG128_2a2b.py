from qgt.hamiltonians.minimal_models.MinimalAltermagnetHamiltonian import (
    MinimalAltermagnetHamiltonian,
)


class MinimalHamSG128_2a2b(MinimalAltermagnetHamiltonian):
    """SG 128 at Wyckoff 2a or 2b; Roig et al., Table III.

    With ci = cos(ki), si = sin(ki), and ci_2/si_2 at ki/2:
        tx = t3 cx_2 cy_2 cz_2; tz = t4 sx sy (cx - cy)
        lambda_x = lamb sx_2 cy_2 sz_2
        lambda_y = lamb cx_2 sy_2 sz_2
        lambda_z = lamb_z cx_2 cy_2 cz_2.

    Source: Physical Review B 110, 144412 (2024), Table III.
    """

    space_group = 128
    wyckoff_positions = ("2a", "2b")
    source_table = "III"
    crystal_system = "tetragonal"

    def _model_terms(self, kx, ky, kz):
        trig = self._trig_terms(kx, ky, kz)
        tx = self.t3 * trig.cx_2 * trig.cy_2 * trig.cz_2
        tz = self.t4 * trig.sx * trig.sy * (trig.cx - trig.cy)
        lam_x = self.lamb * trig.sx_2 * trig.cy_2 * trig.sz_2
        lam_y = self.lamb * trig.cx_2 * trig.sy_2 * trig.sz_2
        lam_z = self.lamb_z * trig.cx_2 * trig.cy_2 * trig.cz_2
        return self._eps0(trig), tx, tz, lam_x, lam_y, lam_z
