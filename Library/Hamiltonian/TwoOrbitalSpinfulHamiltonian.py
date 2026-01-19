import numpy as np
from .Hamiltonian_v2 import hamiltonian


class TwoOrbitalSpinfulHamiltonian(hamiltonian):
    """
    Hamiltonian for the two-orbital unspinful model. Modified from PRL 130, 226001 eq (1) ignoring the spin DOF. 
    """
    def __init__(self, t=1, mu=0, zeta=0, a=1, omega=np.pi/2, A0 = 0, magnus_order = 1):
        """
        Initialize the two-orbital unspinful Hamiltonian.
        Parameters:
        - t: Hopping parameter
        - mu: Chemical potential
        - zeta: Parameter for alpha_k
        - a: Lattice spacing
        - omega: Driving frequency
        - A0: Driving amplitude
        """
        super().__init__(dim=2, omega=omega, A0 = A0, magnus_order=magnus_order)
        self.t = t
        self.mu = mu
        self.zeta = zeta
        self.a = a

        # Compute the symbolic derivatives and lambdify them.
        self.setup_symbolic_derivatives()


    def alpha_k(self, kx, ky):
        return self.zeta * (np.cos(kx * self.a) + np.cos(ky * self.a))

    def compute_static(self, kx, ky):
        """
        Compute the static Hamiltonian for the two-orbital unspinful model.
        """
        # Compute alpha_k
        alpha_k = self.alpha_k(kx, ky)
        
        # Compute trigonometric terms
        sin_alpha = np.sin(alpha_k)
        cos_alpha = np.cos(alpha_k)
        
        # Define Hamiltonian matrix
        H11 = -self.t * self.mu
        H22 = -self.t * self.mu
        H12 = -self.t * (sin_alpha - 1j * cos_alpha)
        H21 = -self.t * (sin_alpha + 1j * cos_alpha)

        H_k = np.array([
            [H11, H12],
            [H21, H22]
        ], dtype=complex)
        
        return H_k
    
    # # First-order derivatives NOTE: We are not considering the lattice constant a here: 
    # def dfxdx(self, kx, ky):
    #     return -self.zeta * np.sin(kx) * np.cos(self.alpha_k(kx, ky))
    
    # def dfxdy(self, kx, ky):
    #     return -self.zeta * np.sin(ky) * np.cos(self.alpha_k(kx, ky))
    
    # def dfydx(self, kx, ky):
    #     return self.zeta * np.sin(kx) * np.sin(self.alpha_k(kx, ky))
    
    # def dfydy(self, kx, ky):
    #     return self.zeta * np.sin(ky) * np.sin(self.alpha_k(kx, ky))
    
    # # Second-order derivatives
    # def dfxdxx(self, kx, ky):
    #     return -self.zeta * (np.cos(kx) * np.cos(self.alpha_k(kx,ky)) + self.zeta * np.sin(kx)**2 * np.sin(self.alpha_k(kx, ky)))
    
    # def dfxdyy(self, kx, ky):
    #     return -self.zeta * (np.cos(ky) * np.cos(self.alpha_k(kx,ky)) + self.zeta * np.sin(ky)**2 * np.sin(self.alpha_k(kx, ky)))
    
    # def dfxdxy(self, kx, ky):
    #     return -self.zeta ** 2 * np.sin(kx) * np.sin(ky) * np.sin(self.alpha_k(kx, ky))
    
    # def dfydxx(self, kx, ky):
    #     return self.zeta * (np.cos(kx) * np.sin(self.alpha_k(kx,ky)) - self.zeta * np.sin(kx)**2 * np.cos(self.alpha_k(kx, ky)))
    
    # def dfydyy(self, kx, ky):
    #     return self.zeta * (np.cos(ky) * np.sin(self.alpha_k(kx,ky)) - self.zeta * np.sin(ky)**2 * np.cos(self.alpha_k(kx, ky)))
    
    # def dfydxy(self, kx, ky):
    #     return -self.zeta ** 2 * np.sin(kx) * np.sin(ky) * np.cos(self.alpha_k(kx, ky))

    # # Third-order derivatives
    # def dfxdxxx(self, kx, ky):
    #     return self.zeta * np.sin(kx) * ((self.zeta**2 * np.sin(kx)**2 + 1) * np.cos(self.alpha_k(kx, ky)) - 3 * self.zeta * np.cos(kx) * np.sin(self.alpha_k(kx, ky)))
    
    # def dfxdxxy(self, kx, ky):
    #     return self.zeta**2 * np.sin(ky) * (self.zeta * np.sin(kx)**2 * np.cos(self.alpha_k(kx, ky)) - np.cos(kx) * np.sin(self.alpha_k(kx, ky)))
    
    # def dfxdxyy(self, kx, ky):
    #     return self.zeta**2 * np.sin(kx) * (self.zeta * np.sin(ky)**2 * np.cos(self.alpha_k(kx, ky)) - np.cos(ky) * np.sin(self.alpha_k(kx, ky)))
    
    # def dfxdyyy(self, kx, ky):
    #     return self.zeta * np.sin(ky) * ((self.zeta**2 * np.sin(ky)**2 + 1) * np.cos(self.alpha_k(kx, ky)) - 3 * self.zeta * np.cos(ky) * np.sin(self.alpha_k(kx, ky)))

    # def dfydxxx(self, kx, ky):
    #     return - self.zeta * np.sin(kx) * ((self.zeta**2 * np.sin(kx)**2 + 1) * np.sin(self.alpha_k(kx, ky)) + 3 * self.zeta * np.cos(kx) * np.cos(self.alpha_k(kx, ky)))
    
    # def dfydxxy(self, kx, ky):
    #     return - self.zeta**2 * np.sin(ky) * (np.cos(kx) * np.cos(self.alpha_k(kx, ky)) + self.zeta * np.sin(kx)**2 * np.sin(self.alpha_k(kx, ky)))
    
    # def dfydxyy(self, kx, ky):
    #     return - self.zeta**2 * np.sin(kx) * (np.cos(ky) * np.cos(self.alpha_k(kx, ky)) + self.zeta * np.sin(ky)**2 * np.sin(self.alpha_k(kx, ky)))
    
    # def dfydyyy(self, kx, ky):
    #     return - self.zeta * np.sin(ky) * ((self.zeta**2 * np.sin(ky)**2 + 1) * np.sin(self.alpha_k(kx, ky)) + 3 * self.zeta * np.cos(ky) * np.cos(self.alpha_k(kx, ky)))
    
    def setup_symbolic_derivatives(self):
        """
        Compute the symbolic expressions for fₓ and f_y and all of their derivatives 
        (from first up to fifth order) with respect to kx and ky. Then lambdify these 
        expressions so they can be evaluated numerically.
        """
        import sympy as sp

        # Define sympy symbols for kx and ky.
        kx_sym, ky_sym = sp.symbols('kx ky', real=True)

        # Define the symbolic expression for alpha_k:
        #   alpha_k = ζ * (cos(a * kx) + cos(a * ky))
        alpha_sym = self.zeta * (sp.cos(kx_sym * self.a) + sp.cos(ky_sym * self.a))

        # Define the symbolic functions for fₓ and f_y.
        fx_sym = -self.t * sp.sin(alpha_sym)
        fy_sym = -self.t * sp.cos(alpha_sym)

        # ---------------------------
        # First-order derivatives
        # ---------------------------
        dfxdx_sym = sp.diff(fx_sym, kx_sym)
        dfxdy_sym = sp.diff(fx_sym, ky_sym)
        dfydx_sym = sp.diff(fy_sym, kx_sym)
        dfydy_sym = sp.diff(fy_sym, ky_sym)

        # ---------------------------
        # Second-order derivatives
        # ---------------------------
        dfxdxx_sym = sp.diff(fx_sym, kx_sym, 2)
        dfxdyy_sym = sp.diff(fx_sym, ky_sym, 2)
        dfxdxy_sym = sp.diff(fx_sym, kx_sym, 1, ky_sym, 1)
        dfydxx_sym = sp.diff(fy_sym, kx_sym, 2)
        dfydyy_sym = sp.diff(fy_sym, ky_sym, 2)
        dfydxy_sym = sp.diff(fy_sym, kx_sym, 1, ky_sym, 1)

        # ---------------------------
        # Third-order derivatives
        # ---------------------------
        dfxdxxx_sym = sp.diff(fx_sym, kx_sym, 3)
        dfxdxxy_sym = sp.diff(fx_sym, kx_sym, 2, ky_sym, 1)
        dfxdxyy_sym = sp.diff(fx_sym, kx_sym, 1, ky_sym, 2)
        dfxdyyy_sym = sp.diff(fx_sym, ky_sym, 3)
        dfydxxx_sym = sp.diff(fy_sym, kx_sym, 3)
        dfydxxy_sym = sp.diff(fy_sym, kx_sym, 2, ky_sym, 1)
        dfydxyy_sym = sp.diff(fy_sym, kx_sym, 1, ky_sym, 2)
        dfydyyy_sym = sp.diff(fy_sym, ky_sym, 3)

        # ---------------------------
        # Fourth-order derivatives
        # ---------------------------
        dfxdxxxx_sym = sp.diff(fx_sym, kx_sym, 4)
        dfxdxxyy_sym = sp.diff(fx_sym, kx_sym, 2, ky_sym, 2)
        dfxdxyyy_sym = sp.diff(fx_sym, kx_sym, 1, ky_sym, 3)
        dfxdyyyy_sym = sp.diff(fx_sym, ky_sym, 4)
        dfydxxxx_sym = sp.diff(fy_sym, kx_sym, 4)
        dfydxxyy_sym = sp.diff(fy_sym, kx_sym, 2, ky_sym, 2)
        dfydxyyy_sym = sp.diff(fy_sym, kx_sym, 1, ky_sym, 3)
        dfydyyyy_sym = sp.diff(fy_sym, ky_sym, 4)

        # ---------------------------
        # Fifth-order derivatives
        # ---------------------------
        dfxdxxxxx_sym = sp.diff(fx_sym, kx_sym, 5)
        dfxdxxyyy_sym = sp.diff(fx_sym, kx_sym, 2, ky_sym, 3)
        dfxdxyyyy_sym = sp.diff(fx_sym, kx_sym, 1, ky_sym, 4)
        dfxdyyyyy_sym = sp.diff(fx_sym, ky_sym, 5)
        dfydxxxxx_sym = sp.diff(fy_sym, kx_sym, 5)
        dfydxxyyy_sym = sp.diff(fy_sym, kx_sym, 2, ky_sym, 3)
        dfydxyyyy_sym = sp.diff(fy_sym, kx_sym, 1, ky_sym, 4)
        dfydyyyyy_sym = sp.diff(fy_sym, ky_sym, 5)

        # ---------------------------
        # Lambdify all symbolic expressions into numerical functions.
        # These functions depend only on (kx, ky) since self.zeta and self.a are fixed.
        # ---------------------------
        self.fx         = sp.lambdify((kx_sym, ky_sym), fx_sym, 'numpy')
        self.fy         = sp.lambdify((kx_sym, ky_sym), fy_sym, 'numpy')
        
        self.dfxdx      = sp.lambdify((kx_sym, ky_sym), dfxdx_sym, 'numpy')
        self.dfxdy      = sp.lambdify((kx_sym, ky_sym), dfxdy_sym, 'numpy')
        self.dfydx      = sp.lambdify((kx_sym, ky_sym), dfydx_sym, 'numpy')
        self.dfydy      = sp.lambdify((kx_sym, ky_sym), dfydy_sym, 'numpy')
        
        self.dfxdxx     = sp.lambdify((kx_sym, ky_sym), dfxdxx_sym, 'numpy')
        self.dfxdyy     = sp.lambdify((kx_sym, ky_sym), dfxdyy_sym, 'numpy')
        self.dfxdxy     = sp.lambdify((kx_sym, ky_sym), dfxdxy_sym, 'numpy')
        self.dfydxx     = sp.lambdify((kx_sym, ky_sym), dfydxx_sym, 'numpy')
        self.dfydyy     = sp.lambdify((kx_sym, ky_sym), dfydyy_sym, 'numpy')
        self.dfydxy     = sp.lambdify((kx_sym, ky_sym), dfydxy_sym, 'numpy')
        
        self.dfxdxxx    = sp.lambdify((kx_sym, ky_sym), dfxdxxx_sym, 'numpy')
        self.dfxdxxy    = sp.lambdify((kx_sym, ky_sym), dfxdxxy_sym, 'numpy')
        self.dfxdxyy    = sp.lambdify((kx_sym, ky_sym), dfxdxyy_sym, 'numpy')
        self.dfxdyyy    = sp.lambdify((kx_sym, ky_sym), dfxdyyy_sym, 'numpy')
        self.dfydxxx    = sp.lambdify((kx_sym, ky_sym), dfydxxx_sym, 'numpy')
        self.dfydxxy    = sp.lambdify((kx_sym, ky_sym), dfydxxy_sym, 'numpy')
        self.dfydxyy    = sp.lambdify((kx_sym, ky_sym), dfydxyy_sym, 'numpy')
        self.dfydyyy    = sp.lambdify((kx_sym, ky_sym), dfydyyy_sym, 'numpy')
        
        self.dfxdxxxx   = sp.lambdify((kx_sym, ky_sym), dfxdxxxx_sym, 'numpy')
        self.dfxdxxyy   = sp.lambdify((kx_sym, ky_sym), dfxdxxyy_sym, 'numpy')
        self.dfxdxyyy   = sp.lambdify((kx_sym, ky_sym), dfxdxyyy_sym, 'numpy')
        self.dfxdyyyy   = sp.lambdify((kx_sym, ky_sym), dfxdyyyy_sym, 'numpy')
        self.dfydxxxx   = sp.lambdify((kx_sym, ky_sym), dfydxxxx_sym, 'numpy')
        self.dfydxxyy   = sp.lambdify((kx_sym, ky_sym), dfydxxyy_sym, 'numpy')
        self.dfydxyyy   = sp.lambdify((kx_sym, ky_sym), dfydxyyy_sym, 'numpy')
        self.dfydyyyy   = sp.lambdify((kx_sym, ky_sym), dfydyyyy_sym, 'numpy')
        
        self.dfxdxxxxx  = sp.lambdify((kx_sym, ky_sym), dfxdxxxxx_sym, 'numpy')
        self.dfxdxxyyy  = sp.lambdify((kx_sym, ky_sym), dfxdxxyyy_sym, 'numpy')
        self.dfxdxyyyy  = sp.lambdify((kx_sym, ky_sym), dfxdxyyyy_sym, 'numpy')
        self.dfxdyyyyy  = sp.lambdify((kx_sym, ky_sym), dfxdyyyyy_sym, 'numpy')
        self.dfydxxxxx  = sp.lambdify((kx_sym, ky_sym), dfydxxxxx_sym, 'numpy')
        self.dfydxxyyy  = sp.lambdify((kx_sym, ky_sym), dfydxxyyy_sym, 'numpy')
        self.dfydxyyyy  = sp.lambdify((kx_sym, ky_sym), dfydxyyyy_sym, 'numpy')
        self.dfydyyyyy  = sp.lambdify((kx_sym, ky_sym), dfydyyyyy_sym, 'numpy')
