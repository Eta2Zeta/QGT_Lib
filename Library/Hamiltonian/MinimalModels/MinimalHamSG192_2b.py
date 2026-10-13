from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class MinimalHamSG192_2b(MinimalAltermagnetHamiltonian):
    """
    Minimal altermagnet model for SG 192 with Wyckoff site 2b.

    Table IV coefficients:
        tau_x         : cos(kz/2)
        tau_z         : fx fy (fx^2 - 3 fy^2) (3 fx^2 - fy^2)
        tau_y sigma_x : lambda sin(kz/2) fx
        tau_y sigma_y : lambda sin(kz/2) fy
        tau_y sigma_z : cos(kz/2)

    where
        fx = sin(kx) + sin(kx/2) cos(sqrt(3) ky/2)
        fy = sqrt(3) cos(kx/2) sin(sqrt(3) ky/2).
    """
    space_group = 192
    wyckoff_positions = ('2b',)
    source_table = 'IV'
    crystal_system = 'hexagonal'

    def _model_terms(self, kx, ky, kz):
        trig = self._trig_terms(kx, ky, kz)

        eps0 = self._eps0(trig)
        tx = self.t3 * trig.cz_2
        tz = self.t4 * trig.fx * trig.fy * (trig.fx**2 - 3 * trig.fy**2) * (3 * trig.fx**2 - trig.fy**2)

        lam_x = self.lamb * trig.sz_2 * trig.fx
        lam_y = self.lamb * trig.sz_2 * trig.fy
        lam_z = self.lamb_z * trig.cz_2

        return eps0, tx, tz, lam_x, lam_y, lam_z
