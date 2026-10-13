from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class MinimalHamSG223_2a(MinimalAltermagnetHamiltonian):
    """SG 223, Wyckoff 2a, from Roig et al., PRB 110, 144412, Table IV.

    For ``s_i=sin(k_i)``, ``c_i=cos(k_i)`` and half-argument subscripts::

        tx = t3*c_x/2*c_y/2*c_z/2
        tz = t4*(c_x-c_y)*(c_y-c_z)*(c_z-c_x)
        lambda_x = lamb*c_x/2*s_y/2*s_z/2
        lambda_y = lamb*s_x/2*c_y/2*s_z/2
        lambda_z = lamb*s_x/2*s_y/2*c_z/2

    Cubic symmetry requires one shared SOC amplitude ``lamb``. The
    optional ``lamb_z`` constructor argument must equal ``lamb`` when
    supplied; afterward it is an alias of ``lamb``, including in sweeps.
    """

    space_group = 223
    wyckoff_positions = ("2a",)
    source_table = "IV"
    crystal_system = "cubic"

    def __init__(
        self, t1=0.8, t2=0.2, t3=0.3, t4=0.3, mu=0.0,
        Jx=0.0, Jy=0.0, Jz=0.2, lamb=0.1, lamb_z=None,
        omega=0.0, A0=0.0, magnus_order=1, polarization="left",
        analytic_magnus=False,
    ):
        if lamb_z is not None and lamb_z != lamb:
            raise ValueError("SG223 cubic symmetry requires lamb_z to equal lamb.")
        super().__init__(
            t1=t1, t2=t2, t3=t3, t4=t4, mu=mu,
            Jx=Jx, Jy=Jy, Jz=Jz, lamb=lamb, lamb_z=lamb,
            omega=omega, A0=A0, magnus_order=magnus_order,
            polarization=polarization, analytic_magnus=analytic_magnus,
        )

    @property
    def lamb_z(self):
        return self.lamb

    @lamb_z.setter
    def lamb_z(self, value):
        self.lamb = value

    def _model_terms(self, kx, ky, kz):
        q = self._trig_terms(kx, ky, kz)
        tx = self.t3 * q.cx_2 * q.cy_2 * q.cz_2
        tz = self.t4 * (q.cx - q.cy) * (q.cy - q.cz) * (q.cz - q.cx)
        lam_x = self.lamb * q.cx_2 * q.sy_2 * q.sz_2
        lam_y = self.lamb * q.sx_2 * q.cy_2 * q.sz_2
        lam_z = self.lamb * q.sx_2 * q.sy_2 * q.cz_2
        return self._eps0(q), tx, tz, lam_x, lam_y, lam_z
