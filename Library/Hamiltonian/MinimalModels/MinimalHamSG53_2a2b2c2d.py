from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class MinimalHamSG53_2a2b2c2d(MinimalAltermagnetHamiltonian):
    """SG 53 at Wyckoff 2a, 2b, 2c, or 2d; Roig et al., Table III.

    With ci = cos(ki), si = sin(ki), and ci_2/si_2 at ki/2:
        tx = t3 cx_2 cz_2; tz = t4 sy sz
        lambda_x = lamb cx_2 cz_2
        lambda_y = lamb_y sx_2 sy cz_2
        lambda_z = lamb_z sx_2 sz_2.

    The three SOC amplitudes are independent.
    Source: Physical Review B 110, 144412 (2024), Table III.
    """

    space_group = 53
    wyckoff_positions = ("2a", "2b", "2c", "2d")
    source_table = "III"
    crystal_system = "orthorhombic"

    def __init__(self, *args, lamb_y=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lamb_y = lamb_y

    def _model_terms(self, kx, ky, kz):
        trig = self._trig_terms(kx, ky, kz)
        tx = self.t3 * trig.cx_2 * trig.cz_2
        tz = self.t4 * trig.sy * trig.sz
        lam_x = self.lamb * trig.cx_2 * trig.cz_2
        lam_y = self.lamb_y * trig.sx_2 * trig.sy * trig.cz_2
        lam_z = self.lamb_z * trig.sx_2 * trig.sz_2
        return self._eps0(trig), tx, tz, lam_x, lam_y, lam_z
