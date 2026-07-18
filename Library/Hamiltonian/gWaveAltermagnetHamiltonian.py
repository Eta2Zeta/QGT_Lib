import numpy as np
from .MinimalAltermagnetHamiltonian import MinimalAltermagnetHamiltonian


class gWaveAltermagnetHamiltonian(MinimalAltermagnetHamiltonian):
    """
    Hamiltonian for g-wave altermagnet (CrSb/MnTe).
    
    H = epsilon_{0, k} + t_{x, k} tau_x + t_{z, k} tau_z + tau_y (vec{lambda}_k . vec{sigma}) + tau_z (vec{J} . vec{sigma})
    
    Parameters per user request:
    t1 = 0.8 (Main Hopping)
    t2 = 0.0 (Interlayer? Not specified, default 0)
    t3 = 0.3 (Inter-layer Hopping)
    t4 = 0.3 (Altermagnetism)
    mu = 0.0
    J = [0.2, 0, 0] (Jx=0.2)
    lambda = 0.1
    lamb_z = 0.1 (Assume same as lambda if not specified, or use lambda for all)
    """
    def __init__(self, t1=0.8, t2=0.2, t3=0.3, t4=0.3,
                 mu=0.0, Jx=0.2, Jy=0.0, Jz=0.0, lamb=0.1, lamb_z=0.1, 
                 omega=0.0, A0=0.0, magnus_order=1,
                 polarization='left', analytic_magnus=False):
        super().__init__(
            mu=mu,
            Jx=Jx,
            Jy=Jy,
            Jz=Jz,
            omega=omega,
            A0=A0,
            magnus_order=magnus_order,
            polarization=polarization,
            analytic_magnus=analytic_magnus,
        )
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.lamb = lamb
        self.lamb_z = lamb_z

    def get_sym_path(self, path="ALHABHPLUADUPDCPBCAPKDGMKGEKUMDECK"):
        """
        High-symmetry path for the hexagonal g-wave altermagnet Brillouin zone.

        The path is supplied as a string of single-character point labels. Useful
        historical choices:

            Standard hexagonal path:
                "GMKGALHA"          # G -> M -> K -> G -> A -> L -> H -> A

            Old requested path:
                "LHALMHKMGKAGL"     # L -> H -> A -> L -> M -> H -> K -> M -> G -> K -> A -> G -> L

            Old requested path:
                "LAHLUHPUDPKUMGKMADG"

            Newest requested path, used by default:
                "ALHABHPLUADUPDCPBCAPKDGMKGEKUMDECK"

        Returns
        -------
        sym_points : dict
            Mapping of label -> np.ndarray of shape (3,).
        path_labels : list of str
            Ordered labels defining the path.
        """
        kM_y = 2.0 * np.pi / np.sqrt(3.0)
        kK_x = 2.0 * np.pi / 3.0
        kz_A = np.pi

        kB_x = np.pi
        kB_y = np.pi / np.sqrt(3.0)

        sym_points = {
            "G": np.array([0.0, 0.0, 0.0]),
            "M": np.array([0.0, kM_y, 0.0]),
            "K": np.array([kK_x, kM_y, 0.0]),
            "A": np.array([0.0, 0.0, kz_A]),
            "L": np.array([0.0, kM_y, kz_A]),
            "H": np.array([kK_x, kM_y, kz_A]),
            "U": np.array([0.0, kM_y, kz_A / 2.0]),
            "D": np.array([0.0, 0.0, kz_A / 2.0]),
            "P": np.array([kK_x, kM_y, kz_A / 2.0]),
            "B": np.array([kB_x, kB_y, kz_A]),
            "C": np.array([kB_x, kB_y, kz_A / 2.0]),
            "E": np.array([kB_x, kB_y, 0.0]),
        }

        path_labels = list(path)
        unknown_labels = [label for label in path_labels if label not in sym_points]
        if unknown_labels:
            valid_labels = "".join(sorted(sym_points))
            raise ValueError(
                f"Unknown g-wave symmetry labels {unknown_labels}. Valid labels are: {valid_labels}."
            )

        return sym_points, path_labels

    def _model_terms(self, kx, ky, kz):
        trig = self._trig_terms(kx, ky, kz)

        eps0 = self.t1 * (trig.cx + 2 * trig.cx_2 * trig.csq3ky_2) + self.t2 * trig.cz - self.mu
        tx = self.t3 * trig.cz_2
        tz = self.t4 * trig.sz * trig.fy * (trig.fy**2 - 3 * trig.fx**2)

        lam_xk = self.lamb * trig.cz_2 * (trig.fx**2 - trig.fy**2)
        lam_yk = -2 * self.lamb * trig.cz_2 * trig.fx * trig.fy
        lam_zk = self.lamb_z * trig.sz_2 * trig.fx * (trig.fx**2 - 3 * trig.fy**2)

        return eps0, tx, tz, lam_xk, lam_yk, lam_zk
