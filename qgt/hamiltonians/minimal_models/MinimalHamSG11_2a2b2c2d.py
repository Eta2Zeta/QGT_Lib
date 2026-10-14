from qgt.hamiltonians.minimal_models.MinimalAltermagnetHamiltonian import (
    MinimalAltermagnetHamiltonian,
)


class MinimalHamSG11_2a2b2c2d(MinimalAltermagnetHamiltonian):
    """SG 11 at Wyckoff 2a, 2b, 2c, or 2d; Table III plus Note 3-3.

    With ci = cos(ki), si = sin(ki), and ci_2/si_2 at ki/2:
        tx = cy_2 (t3 + t3_add cx)
        tz = sy (t4 sx + t4_alt sz)
        lambda_x = lamb cy_2
        lambda_y = sy_2 (lamb_y sx + lamb_y_alt sz)
        lambda_z = cy_2 (lamb_z + lamb_z_add cz + lamb_z_xz_add sx sz).

    The *_alt amplitudes distinguish the paper's listed independent bases.
    The *_add amplitudes are additional symmetry-allowed hoppings from
    research Note 3-3, beyond the explicit Table III model: t3_add is the
    spin-independent (x, y/2) bond contribution; lamb_z_add is the z-spin SOC
    (y/2, z) bond contribution; lamb_z_xz_add is the z-spin SOC contribution
    from (x, y/2, z) bonds with opposite signs for opposite xz orientations.
    All three are active by default; setting all three to zero recovers the
    paper's explicit coefficient functions. The xz contribution vanishes on
    kz=0 but its z derivative lifts the accidental Omega_y line at ky=kz=0.
    Source: Physical Review B 110, 144412 (2024), Table III.
    """

    space_group = 11
    wyckoff_positions = ("2a", "2b", "2c", "2d")
    source_table = "III"
    crystal_system = "monoclinic"

    def __init__(
        self, *args, t4_alt=0.2, lamb_y=0.1, lamb_y_alt=0.06,
        t3_add=0.8, lamb_z_add=0.3, lamb_z_xz_add=0.3, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.t4_alt = t4_alt
        self.lamb_y = lamb_y
        self.lamb_y_alt = lamb_y_alt
        self.t3_add = t3_add
        self.lamb_z_add = lamb_z_add
        self.lamb_z_xz_add = lamb_z_xz_add

    def _model_terms(self, kx, ky, kz):
        trig = self._trig_terms(kx, ky, kz)
        tx = trig.cy_2 * (self.t3 + self.t3_add * trig.cx)
        tz = trig.sy * (self.t4 * trig.sx + self.t4_alt * trig.sz)
        lam_x = self.lamb * trig.cy_2
        lam_y = trig.sy_2 * (self.lamb_y * trig.sx + self.lamb_y_alt * trig.sz)
        lam_z = trig.cy_2 * (
            self.lamb_z + self.lamb_z_add * trig.cz
            + self.lamb_z_xz_add * trig.sx * trig.sz
        )
        return self._eps0(trig), tx, tz, lam_x, lam_y, lam_z
