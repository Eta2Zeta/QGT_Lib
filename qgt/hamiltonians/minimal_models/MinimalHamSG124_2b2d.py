from qgt.hamiltonians.minimal_models.MinimalAltermagnetHamiltonian import (
    MinimalAltermagnetHamiltonian,
)


class MinimalHamSG124_2b2d(MinimalAltermagnetHamiltonian):
    """
    Minimal altermagnet model for SG 124 at Wyckoff position 2b or 2d.

    Table IV coefficients:
        tau_x         : cos(kz/2)
        tau_z         : sin(kx) sin(ky) [cos(kx) - cos(ky)]
        tau_y sigma_x : lambda sin(kx) sin(kz/2)
        tau_y sigma_y : lambda sin(ky) sin(kz/2)
        tau_y sigma_z : cos(kz/2)
    """

    space_group = 124
    wyckoff_positions = ('2b', '2d')
    source_table = 'IV'
    crystal_system = 'tetragonal'

    def _model_terms(self, kx, ky, kz):
        trig = self._trig_terms(kx, ky, kz)

        eps0 = self._eps0(trig)
        tx = self.t3 * trig.cz_2
        tz = self.t4 * trig.sx * trig.sy * (trig.cx - trig.cy)

        lam_x = self.lamb * trig.sx * trig.sz_2
        lam_y = self.lamb * trig.sy * trig.sz_2
        lam_z = self.lamb_z * trig.cz_2

        return eps0, tx, tz, lam_x, lam_y, lam_z
