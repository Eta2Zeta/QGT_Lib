from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class MinimalHamSG14_2a2b2c2d(MinimalAltermagnetHamiltonian):
    """SG 14 at Wyckoff 2a, 2b, 2c, or 2d; Roig et al., Table III.

    With ci = cos(ki), si = sin(ki), and ci_2/si_2 at ki/2:
        tx = cy_2 (t3 sx sz_2 + t3_alt cz_2)
        tz = sy (t4 sx + t4_alt sz)
        lambda_x = cy_2 (lamb sx sz_2 + lamb_x_alt cz_2)
        lambda_y = sy_2 (lamb_y sx cz_2 + lamb_y_alt sz_2)
        lambda_z = cy_2 (lamb_z sx sz_2 + lamb_z_alt cz_2).

    Each comma/parenthesis-separated basis in the source has its own
    coefficient. Source: Physical Review B 110, 144412 (2024), Table III.
    """

    space_group = 14
    wyckoff_positions = ("2a", "2b", "2c", "2d")
    source_table = "III"
    crystal_system = "monoclinic"

    def __init__(
        self,
        *args,
        t3_alt=0.15,
        t4_alt=0.2,
        lamb_y=0.1,
        lamb_x_alt=0.04,
        lamb_y_alt=0.06,
        lamb_z_alt=0.08,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.t3_alt = t3_alt
        self.t4_alt = t4_alt
        self.lamb_y = lamb_y
        self.lamb_x_alt = lamb_x_alt
        self.lamb_y_alt = lamb_y_alt
        self.lamb_z_alt = lamb_z_alt

    def _model_terms(self, kx, ky, kz):
        trig = self._trig_terms(kx, ky, kz)
        tx = trig.cy_2 * (self.t3 * trig.sx * trig.sz_2 + self.t3_alt * trig.cz_2)
        tz = trig.sy * (self.t4 * trig.sx + self.t4_alt * trig.sz)
        lam_x = trig.cy_2 * (
            self.lamb * trig.sx * trig.sz_2 + self.lamb_x_alt * trig.cz_2
        )
        lam_y = trig.sy_2 * (
            self.lamb_y * trig.sx * trig.cz_2 + self.lamb_y_alt * trig.sz_2
        )
        lam_z = trig.cy_2 * (
            self.lamb_z * trig.sx * trig.sz_2 + self.lamb_z_alt * trig.cz_2
        )
        return self._eps0(trig), tx, tz, lam_x, lam_y, lam_z
