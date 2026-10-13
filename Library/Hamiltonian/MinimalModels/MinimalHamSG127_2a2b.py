from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class MinimalHamSG127_2a2b(MinimalAltermagnetHamiltonian):
    """
    Minimal altermagnet model for SG 127 with Wyckoff sites 2a,2b.

    Table III coefficients:
        tau_x         : cos(kx/2) cos(ky/2)
        tau_z         : sin(kx) sin(ky) [cos(kx) - cos(ky)]
        tau_y sigma_x : lambda sin(kx/2) cos(ky/2) sin(kz)
        tau_y sigma_y : lambda cos(kx/2) sin(ky/2) sin(kz)
        tau_y sigma_z : cos(kx/2) cos(ky/2)
    """
    space_group = 127
    wyckoff_positions = ('2a', '2b')
    source_table = 'III'
    crystal_system = 'tetragonal'

    def _model_terms(self, kx, ky, kz):
        trig = self._trig_terms(kx, ky, kz)

        eps0 = self._eps0(trig)
        tx = self.t3 * trig.cx_2 * trig.cy_2
        tz = self.t4 * trig.sx * trig.sy * (trig.cx - trig.cy)

        lam_x = self.lamb * trig.sx_2 * trig.cy_2 * trig.sz
        lam_y = self.lamb * trig.cx_2 * trig.sy_2 * trig.sz
        lam_z = self.lamb_z * trig.cx_2 * trig.cy_2

        return eps0, tx, tz, lam_x, lam_y, lam_z
