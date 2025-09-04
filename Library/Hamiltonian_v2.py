import numpy as np
from scipy.integrate import quad_vec, quad
from qutip import sigmax, sigmay, sigmaz
from .basic_math import commutator_static


sigma_x = sigmax().full()  # Converts QuTiP object to NumPy array
sigma_y = sigmay().full()
sigma_z = sigmaz().full()



class hamiltonian:
    """
    Base class for defining time-dependent Hamiltonians with driven terms.
    """
    def __init__(self, dim, omega, A0=0, polarization='left', magnus_order=1, analytic_magnus=False):
        """
        Initialize the Hamiltonian with its dimension, driving frequency (omega), and driving amplitude (A0).
        
        Parameters:
            dim (int): Dimension of the Hamiltonian matrix.
            omega (float): Driving frequency.
            A0 (float): Driving amplitude (default 0).
            polarization (str): Polarization type ('left', 'right', or 'custom').
            magnus_order (int): Order of Magnus expansion to include (default 1).
        """
        self.name = self.__class__.__name__  # Automatically stores the subclass name
        self.dim = dim  # Dimension of the Hamiltonian matrix
        self.omega = omega  # Driving frequency
        self.A0 = A0  # Driving amplitude
        self.polarization = polarization.lower()  # Polarization type ('left', 'right', or 'custom')
        self.magnus_order = magnus_order  # Order of Magnus expansion
        self.analytic_magnus = analytic_magnus
    
    def get_filename(self, parameter='2D'):
        """
        Generate a filename-style string encoding key parameter values.
        
        Parameters:
            papermeter (str): '1D' or '2D'. In '1D', omega is excluded. Default is '2D'.
        
        Returns:
            str: Filename string.
        """
        # Create a dictionary of relevant attributes
        params = {
            key: value for key, value in vars(self).items()
            if not callable(value) and not key.startswith('_')
            and key not in ('name', 'dim')  # Always exclude 'name' and 'dim'
        }

        # Exclude 'omega' if 1D
        if parameter == '1D':
            params.pop('omega', None)

        # Format parameters into a filename string
        param_str = "_".join(f"{key}{value}" for key, value in params.items())
        return param_str
    
    def compute_static(self, kx, ky):
        """
        Compute the static Hamiltonian matrix for a given (kx, ky).
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'compute_static' method.")

    def compute_driven(self, t, kx, ky):
        """
        Compute the time-dependent Hamiltonian matrix for a given time t and (kx, ky),
        based on the polarization type.
        """
        if self.polarization == 'left':
            # Apply left-hand polarized driving
            kx_t = kx + self.A0 * np.cos(self.omega * t)
            ky_t = ky + self.A0 * np.sin(self.omega * t)
        elif self.polarization == 'right':
            # Apply right-hand polarized driving
            kx_t = kx + self.A0 * np.cos(self.omega * t)
            ky_t = ky - self.A0 * np.sin(self.omega * t)
        elif self.polarization == 'linear_x':
            # Linearly polarized along x
            kx_t = kx + self.A0 * np.cos(self.omega * t)
            ky_t = ky
        elif self.polarization == 'linear_y':
            # Linearly polarized along y
            kx_t = kx
            ky_t = ky + self.A0 * np.cos(self.omega * t)
        else:
            raise ValueError("Invalid polarization type. Choose 'left', 'right', 'linear_x', or 'linear_y'.")

        # Return the static Hamiltonian evaluated at the transformed kx, ky
        return self.compute_static(kx_t, ky_t)

    def numerical_fourier_component(self, n, kx, ky):
        """
        Compute the nth Fourier component of the time-dependent Hamiltonian.

        Parameters:
            n (int): Fourier component index.
            kx, ky (float): Parameters for the Hamiltonian.
        Returns:
            complex: The nth Fourier component.
        """
        baseline_epsrel = 1e-7
        adjusted_epsrel = baseline_epsrel / (1 + n * self.omega)

        # Define the integrand function
        def integrand(t):
            return self.compute_driven(t, kx, ky) * np.exp(-1j * n * self.omega * t)

        # Perform numerical integration
        result, error = quad_vec(
            integrand, 
            0, 
            2 * np.pi / self.omega, 
            epsrel=adjusted_epsrel
        )

        # Scale the result by the pre-factor
        integral = (self.omega / (2 * np.pi)) * result

        # Estimate how many decimal digits we can trust
        if error == 0 or not np.isfinite(error):
            return integral  # nothing to round to

        # Determine number of decimal digits from error magnitude
        digits = max(0, -int(np.floor(np.log10(error))))


        return integral
    
    def numerical_fourier_component_rounding(self, n, kx, ky):
        """
        Compute the nth Fourier component of the time-dependent Hamiltonian,
        integrating each matrix element independently with proper error-aware rounding
        applied before scaling. Uses adaptive epsrel based on n to improve accuracy.
        """
        T = 2 * np.pi / self.omega
        scale = self.omega / (2 * np.pi)

        # Adjust error tolerance based on Fourier index
        baseline_epsrel = 1e-7
        adjusted_epsrel = baseline_epsrel / (1 + n * self.omega)

        # Determine matrix shape
        sample = self.compute_driven(0, kx, ky)
        shape = sample.shape

        # Output arrays
        rounded_real = np.zeros(shape)
        rounded_imag = np.zeros(shape)

        # Rounding helper
        def round_to_error(val, err):
            if err == 0 or not np.isfinite(err):
                return val
            digits = max(0, -int(np.floor(np.log10(err))))
            return np.round(val, digits)

        # Loop over matrix elements
        for i in range(shape[0]):
            for j in range(shape[1]):
                def integrand_real(t):
                    val = self.compute_driven(t, kx, ky)[i, j]
                    return np.real(val * np.exp(-1j * n * self.omega * t))

                def integrand_imag(t):
                    val = self.compute_driven(t, kx, ky)[i, j]
                    return np.imag(val * np.exp(-1j * n * self.omega * t))

                real_val, real_err = quad(integrand_real, 0, T, epsrel=adjusted_epsrel)
                imag_val, imag_err = quad(integrand_imag, 0, T, epsrel=adjusted_epsrel)

                # Round before scaling
                rounded_real[i, j] = round_to_error(real_val, real_err)
                rounded_imag[i, j] = round_to_error(imag_val, imag_err)

        # Apply scaling
        result = scale * (rounded_real + 1j * rounded_imag)
        return result

    def magnus_first_term(self, kx, ky):
        """
        Compute the first term of the Magnus expansion:
        (1/omega) * [H1, H-1], rounded to 1e-16 precision.
        """
        # Compute H1 and H-1 Fourier components
        H1 = self.numerical_fourier_component(1, kx, ky)
        Hm1 = self.numerical_fourier_component(-1, kx, ky)

        # Compute the commutator [H1, H-1]
        comm = commutator_static(H1, Hm1)

        # Compute Magnus first term
        magnus_term = (1 / self.omega) * comm

        # Round to 1e-16 precision
        epsilon = 1e-16
        rounded_magnus = np.round(magnus_term / epsilon) * epsilon

        return rounded_magnus

    def magnus_second_term(self, kx, ky):
        """
        Compute the second Magnus term:
        (1/omega) * (1/2) * [H2, H-2]
        """
        # Compute H2 and H-2 Fourier components
        H2 = self.numerical_fourier_component(2, kx, ky)
        Hm2 = self.numerical_fourier_component(-2, kx, ky)

        # Compute the commutator [H2, H-2]
        comm = commutator_static(H2, Hm2)

        # Return the second Magnus term
        return (1 / (2 * self.omega)) * comm

    def effective_hamiltonian(self, kx, ky):
        """
        Compute the total effective Hamiltonian and its perturbation:
        H_eff = H_0 + sum of Magnus terms up to the specified order.
        H_prime = sum of Magnus terms (perturbation from static H_0).

        Parameters:
            kx, ky (float): Parameters for the Hamiltonian.

        Returns:
            H_eff (ndarray): Effective Hamiltonian.
            H_prime (ndarray): Perturbation Hamiltonian (sum of Magnus terms).
        """
        # Compute the original static Hamiltonian (H_0)
        H_0 = self.compute_static(kx, ky)

        # Initialize perturbation Hamiltonian (H_prime) as zero matrix of same shape as H_0
        H_prime = np.zeros_like(H_0)

        # If A0 is 0, there are no driving terms; return static Hamiltonian directly
        if self.A0 == 0:
            return H_0, H_prime

        # Add Magnus terms to H_prime based on specified order
        if self.magnus_order >= 1:
            if self.analytic_magnus and hasattr(self, "analytic_magnus_first_term"):
                # Expect analytic_magnus_first_term to ALREADY include the 1/omega factor
                H_prime += self.analytic_magnus_first_term(kx, ky)
            else:
                H_prime += self.magnus_first_term(kx, ky)
        if self.magnus_order >= 2:
            H_prime += self.magnus_second_term(kx, ky)
        
        # Compute effective Hamiltonian (H_eff = H_0 + H_prime)
        H_eff = H_0 + H_prime

        return H_eff, H_prime
    
    # ____________________________________________________________________________________________________________
    # Below are the method to compute the Fourier Harmonics of the Hamiltonian by the Taylor Expansion Method
    def get_derivative(self, func_name, kx, ky):
        """Helper function to check and call derivative functions dynamically."""
        if hasattr(self, func_name):
            return getattr(self, func_name)(kx, ky)
        return 0  # Return 0 if the function does not exist

    def fx11p(self, kx, ky):
        return self.A0 * (0.5 * self.get_derivative("dfxdx", kx, ky) + (0.5 / 1j) * self.get_derivative("dfxdy", kx, ky))
    
    def fy11p(self, kx, ky):
        return self.A0 * (0.5 * self.get_derivative("dfydx", kx, ky) + (0.5 / 1j) * self.get_derivative("dfydy", kx, ky))

    def fx22p(self, kx, ky):
        return (self.A0**2 / 4) * (0.5 * self.get_derivative("dfxdxx", kx, ky) - 0.5 * self.get_derivative("dfxdyy", kx, ky) - 1j * self.get_derivative("dfxdxy", kx, ky))

    def fy22p(self, kx, ky):
        return (self.A0**2 / 4) * (0.5 * self.get_derivative("dfydxx", kx, ky) - 0.5 * self.get_derivative("dfydyy", kx, ky) - 1j * self.get_derivative("dfydxy", kx, ky))

    def fx31p(self, kx, ky):
        return (self.A0**3 / 16) * (self.get_derivative("dfxdxxx", kx, ky) + 3 * self.get_derivative("dfxdxyy", kx, ky)) + \
               (self.A0**3 / (16j)) * (3 * self.get_derivative("dfxdxxy", kx, ky) + self.get_derivative("dfxdyyy", kx, ky))

    def fy31p(self, kx, ky):
        return (self.A0**3 / 16) * (self.get_derivative("dfydxxx", kx, ky) + 3 * self.get_derivative("dfydxyy", kx, ky)) + \
               (self.A0**3 / (16j)) * (3 * self.get_derivative("dfydxxy", kx, ky) + self.get_derivative("dfydyyy", kx, ky))

    def fx33p(self, kx, ky):
        return (self.A0**3 / 48) * (self.get_derivative("dfxdxxx", kx, ky) + 3 * self.get_derivative("dfxdxyy", kx, ky)) + \
               (self.A0**3 / (48j)) * (3 * self.get_derivative("dfxdxxy", kx, ky) + self.get_derivative("dfxdyyy", kx, ky))

    def fy33p(self, kx, ky):
        return (self.A0**3 / 48) * (self.get_derivative("dfydxxx", kx, ky) + 3 * self.get_derivative("dfydxyy", kx, ky)) + \
               (self.A0**3 / (48j)) * (3 * self.get_derivative("dfydxxy", kx, ky) + self.get_derivative("dfydyyy", kx, ky))
    
    def fx51p(self, kx, ky):
        """
        Compute the positive fifth-order Fourier component for f_x:
        
        f₅,₊₁ = (A₀⁵/120) * { ½[ (5/8)∂ₖₓ⁵ f + (5/4)∂ₖₓ³∂ₖ_y² f + (5/8)∂ₖₓ∂ₖ_y⁴ f ]
                + (1/(2i))[ (5/8)∂ₖₓ⁴∂ₖ_y f + (5/4)∂ₖₓ²∂ₖ_y³ f + (5/8)∂ₖ_y⁵ f ] }.
        """
        prefactor = self.A0**5 / 120
        real_part = 0.5 * (
            (5/8) * self.get_derivative("dfxdxxxxx", kx, ky) +
            (5/4) * self.get_derivative("dfxdxxxyy", kx, ky) +
            (5/8) * self.get_derivative("dfxdxyyyy", kx, ky)
        )
        imag_part = (1/(2j)) * (
            (5/8) * self.get_derivative("dfxdxxxxy", kx, ky) +
            (5/4) * self.get_derivative("dfxdxxyy", kx, ky) +
            (5/8) * self.get_derivative("dfxdyyyyy", kx, ky)
        )
        return prefactor * (real_part + imag_part)

    def fy51p(self, kx, ky):
        """
        Compute the positive fifth-order Fourier component for f_y:
        
        f₅,₊₁ = (A₀⁵/120) * { ½[ (5/8)∂ₖₓ⁵ f + (5/4)∂ₖₓ³∂ₖ_y² f + (5/8)∂ₖₓ∂ₖ_y⁴ f ]
                + (1/(2i))[ (5/8)∂ₖₓ⁴∂ₖ_y f + (5/4)∂ₖₓ²∂ₖ_y³ f + (5/8)∂ₖ_y⁵ f ] }.
        """
        prefactor = self.A0**5 / 120
        real_part = 0.5 * (
            (5/8) * self.get_derivative("dfydxxxxx", kx, ky) +
            (5/4) * self.get_derivative("dfydxxxyy", kx, ky) +
            (5/8) * self.get_derivative("dfydxyyyy", kx, ky)
        )
        imag_part = (1/(2j)) * (
            (5/8) * self.get_derivative("dfydxxxxy", kx, ky) +
            (5/4) * self.get_derivative("dfydxxyy", kx, ky) +
            (5/8) * self.get_derivative("dfydyyyyy", kx, ky)
        )
        return prefactor * (real_part + imag_part)
    
    # The first Harmonic calculated from the first three orders of Taylor Expansions
    def Hp1(self, kx, ky):
        return (self.fx11p(kx, ky) + self.fx31p(kx, ky) + self.fx51p(kx, ky)) * sigma_x + (self.fy11p(kx, ky) + self.fy31p(kx, ky) + self.fy51p(kx, ky)) * sigma_y

    def Hp113(self, kx, ky):
        return (self.fx11p(kx, ky) + self.fx31p(kx, ky)) * sigma_x + (self.fy11p(kx, ky) + self.fy31p(kx, ky)) * sigma_y
    
    def Hp11(self, kx, ky):
        return (self.fx11p(kx, ky)) * sigma_x + (self.fy11p(kx, ky)) * sigma_y

    def Hp13(self, kx, ky):
        return (self.fx31p(kx, ky)) * sigma_x + (self.fy31p(kx, ky)) * sigma_y
        
    def Hp15(self, kx, ky):
        return (self.fx51p(kx, ky)) * sigma_x + (self.fy51p(kx, ky)) * sigma_y

    def Hp2(self, kx, ky):
        return self.fx22p(kx, ky) * sigma_x + self.fy22p(kx, ky) * sigma_y

    def Hp3(self, kx, ky):
        return self.fx33p(kx, ky) * sigma_x + self.fy33p(kx, ky) * sigma_y
    
    # Removed lambdified function so the Hamiltonian can be pickled
    def __getstate__(self):
        # Make a copy of the instance's state.
        state = self.__dict__.copy()
        # Remove entries that are not picklable (lambdified functions).
        # You can use a list of keys to remove, for example, all that start with 'dfx' or 'dfyd'
        keys_to_remove = [key for key in state if key.startswith("df") or key in ["fx", "fy"]]
        for key in keys_to_remove:
            del state[key]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Only call setup_symbolic_derivatives if the method exists
        if hasattr(self, "setup_symbolic_derivatives"):
            self.setup_symbolic_derivatives()




class TestHamiltonian(hamiltonian):
    def __init__(self, a=1, b=1, c=1, omega=2 * np.pi, A0=1):
        super().__init__(dim=2, omega=omega, A0=A0)  # Pass omega and A0 to the base class
        self.a = a
        self.b = b
        self.c = c
    
    def fx(self, kx, ky):
        return self.a * kx**2 + self.b * ky**2 + self.c * kx * ky
    
    def compute_static(self, kx, ky):
        return self.fx(kx, ky) * sigma_x
    
    # First derivatives
    def dfxdx(self, kx, ky):
        return 2 * self.a * kx + self.c * ky
    
    def dfxdy(self, kx, ky):
        return 2 * self.b * ky + self.c * kx
    
    # Second derivatives
    def dfxdxx(self, kx, ky):
        return 2 * self.a
    
    def dfxdyy(self, kx, ky):
        return 2 * self.b
    
    def dfxdxy(self, kx, ky):
        return self.c
    
    # Third derivatives (All Zero)
    def dfxdxxx(self, kx, ky):
        return 0
    
    def dfxdxxy(self, kx, ky):
        return 0
    
    def dfxdxyy(self, kx, ky):
        return 0
    
    def dfxdyyy(self, kx, ky):
        return 0




class GrapheneHamiltonian(hamiltonian):
    def __init__(self, omega = np.pi, A0 = 0):
        super().__init__(dim=2, omega=omega, A0=A0)  # THF model has a 6x6 matrix
    
    def compute_static(self, kx, ky):
        H_k = np.array([
            [0, kx - 1j*ky],
            [kx + 1j*ky, 0]
        ])
        
        return H_k
    def Hp1(self, kx, ky):
        # Define the expression
        H21 = self.A0
        return H21 # Temporary, change it to full matrix later
    


class RashbaHamiltonian(hamiltonian):
    """
    Rashba Hamiltonian for a 2D system with spin-orbit coupling.
    """
    def __init__(self, m=1, alpha=1, omega=2 * np.pi, A0=0.0):
        """
        Initialize the Rashba Hamiltonian.
        
        Parameters:
        - m: Effective mass of the particle
        - alpha: Rashba spin-orbit coupling strength
        - omega: Driving frequency
        - A0: Driving amplitude
        """
        super().__init__(dim=2, omega=omega, A0=A0)
        self.m = m
        self.alpha = alpha

    def compute_static(self, kx, ky):
        """
        Compute the static Rashba Hamiltonian for a given (kx, ky).
        """
        k_squared = kx**2 + ky**2
        H11 = k_squared / (2 * self.m)
        H22 = k_squared / (2 * self.m)
        H12 = self.alpha * (ky + 1j * kx)
        H21 = self.alpha * (ky - 1j * kx)

        H_k = np.array([
            [H11, H12],
            [H21, H22]
        ], dtype=complex)
        
        return H_k
    
    def Hp1(self, kx, ky):
        H11 = (self.A0 / (2 * self.m)) * (kx - 1j*ky)
        H12 = 0
        H21 = -1j * self.alpha * self.A0
        H22 = H11


        H_k = np.array([
            [H11, H12],
            [H21, H22]
        ], dtype=complex)
        
        return H_k
    
    def Hp2(self, kx, ky):
        H11 = 0
        H12 = 0
        H21 = 0
        H22 = 0


        H_k = np.array([
            [H11, H12],
            [H21, H22]
        ], dtype=complex)
        
        return H_k




class TwoOrbitalSpinfulHamiltonian(hamiltonian):
    """
    Hamiltonian for the two-orbital spinful model. From PRL 130, 226001 eq (1)
    """
    def __init__(self, t=1, mu=0, zeta=0, a=1, omega = np.pi/2, A0 = 0, magnus_order = 1):
        """
        Initialize the two-orbital spinful Hamiltonian.
        Parameters:
        - t: Hopping parameter
        - mu: Chemical potential
        - zeta: Parameter for alpha_k
        - a: Lattice spacing
        - omega: Driving frequency
        - A0: Driving amplitude
        """
        super().__init__(dim=4, omega=omega, A0=A0, magnus_order=magnus_order)
        self.t = t
        self.mu = mu
        self.zeta = zeta
        self.a = a

    def compute_static(self, kx, ky):
        """
        Compute the static Hamiltonian for the two-orbital spinful model.
        """
        # Compute alpha_k
        alpha_k = self.zeta * (np.cos(kx * self.a) + np.cos(ky * self.a))
        
        # Define matrix elements
        H = np.zeros((4, 4), dtype=complex)
        sin_alpha = np.sin(alpha_k)
        cos_alpha = np.cos(alpha_k)
        
        # Fill the Hamiltonian matrix
        H[0, 0] = -self.t * self.mu
        H[1, 1] = -self.t * self.mu
        H[2, 2] = -self.t * self.mu
        H[3, 3] = -self.t * self.mu
        
        H[0, 2] = -self.t * (sin_alpha - 1j * cos_alpha)
        H[2, 0] = -self.t * (sin_alpha + 1j * cos_alpha)
        
        H[1, 3] = -self.t * (sin_alpha + 1j * cos_alpha)
        H[3, 1] = -self.t * (sin_alpha - 1j * cos_alpha)
        
        return H
    


class TwoOrbitalUnspinfulHamiltonian(hamiltonian):
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


class SquareLatticeHamiltonian(hamiltonian):
    """
    Hamiltonian for a square lattice model.
    """
    def __init__(self, t1=1, t2=1/np.sqrt(2), t5=0, omega=2 * np.pi, A0=0):
        super().__init__(dim=2, omega=omega, A0=A0)  # Pass omega and A0 to the base class
        self.t1 = t1
        self.t2 = t2
        self.t5 = t5
    
    def compute_static(self, kx, ky):
        """
        Compute the static Hamiltonian for the square lattice.
        """
        H11 = -2 * self.t5 * (np.cos(2 * (kx + ky)) + np.cos(2 * (kx - ky))) - 2 * self.t2 * (np.cos(kx + ky) - np.cos(kx - ky))
        H12 = -2 * self.t1 * np.exp(1j * np.pi / 4) * np.exp(1j * ky) * np.cos(ky) - 2 * self.t1 * np.exp(-1j * np.pi / 4) * np.exp(1j * ky) * np.cos(kx)
        H21 = np.conj(H12)  # H21 is the Hermitian conjugate of H12
        H22 = -2 * self.t5 * (np.cos(2 * (kx + ky)) + np.cos(2 * (kx - ky))) + 2 * self.t2 * (np.cos(kx + ky) - np.cos(kx - ky))
        
        H_k = np.array([
            [H11, H12],
            [H21, H22]
        ], dtype=complex)
        
        return H_k
    
    def Hp1(self, kx, ky):
        # Define the expression
        H12 = (self.A0 / 2) * (
            2 * self.t1 * np.exp(-1j * np.pi / 4) * np.exp(1j * ky) * np.sin(kx)
            - 2 * self.t1 * np.exp(1j * np.pi / 4) * np.exp(1j * ky) * np.cos(ky)
            - 1j * 2 * self.t1 * np.exp(1j * np.pi / 4) * np.sin(ky) * np.exp(1j * ky)
            - 2 * self.t1 * np.cos(kx) * np.exp(-1j * np.pi / 4) * np.exp(1j * ky)
        )
        H21 = self.A0 * self.t1 * (-1j * np.sin(ky) + np.sqrt(2) * 1j * 
                                   np.sin(kx + np.pi / 4) + np.cos(ky)) * np.exp(-1j * (ky + np.pi / 4))
        return H21 # Temporary, change it to full matrix later


class SquareLatticeHamiltonianMod(hamiltonian):
    """
    Hamiltonian for the modified square lattice model (Equation 3 from PhysRevLett.106.236804). Not used anymore
    """
    def __init__(self, t1=1, t2=1/np.sqrt(2)):
        super().__init__(dim=2)  # Square lattice model has a 2x2 matrix
        self.t1 = t1
        self.t2 = t2
    
    def compute(self, kx, ky):
        M11 = 2 * self.t2 * (np.cos(kx) - np.cos(ky))
        M12 = self.t1 * np.exp(1j * np.pi / 4) * (1 + np.exp(-1j * (ky - kx))) + self.t1 * np.exp(-1j * np.pi / 4) * (np.exp(1j * kx) + np.exp(-1j * ky))
        M21 = np.conj(M12)  # M21 is the Hermitian conjugate of M12
        M22 = -2 * self.t2 * (np.cos(kx) - np.cos(ky))
        
        H_k = np.array([
            [M11, M12],
            [M21, M22]
        ], dtype=complex)
        
        return H_k


class THF_Hamiltonian(hamiltonian):
    """
    Hamiltonian for the THF model. This is not optimized for the Magnus expansion yet and it is 
    with the frequency term included as G = A0^2/omega. 
    """
    def __init__(self, nu_star=-50, nu_star_prime=13.0, gamma=-25.0, M=5, G=0.001, omega = np.pi, A0 = 0):
        super().__init__(dim=6, omega=omega, A0=A0)  # THF model has a 6x6 matrix
        self.nu_star = nu_star
        self.nu_star_prime = nu_star_prime
        self.gamma = gamma
        self.M = M
        self.G = G
    
    def compute_static(self, kx, ky):
        k = np.sqrt(kx**2 + ky**2)
        theta = np.arctan2(ky, kx)
        
        H_k = np.array([
            [self.G * (self.nu_star**2 - self.nu_star_prime**2), 0, self.nu_star * k * np.exp(1j * theta), 0, self.gamma, self.nu_star_prime * k * np.exp(-1j * theta)],
            [0, -self.G * (self.nu_star**2 - self.nu_star_prime**2), 0, self.nu_star * k * np.exp(-1j * theta), self.nu_star_prime * k * np.exp(1j * theta), self.gamma],
            [self.nu_star * k * np.exp(-1j * theta), 0, -self.G * self.nu_star**2, self.M, 0, 0],
            [0, self.nu_star * k * np.exp(1j * theta), self.M, self.G * self.nu_star**2, 0, 0],
            [self.gamma, self.nu_star_prime * k * np.exp(-1j * theta), 0, 0, -self.G * self.nu_star_prime**2, 0],
            [self.nu_star_prime * k * np.exp(1j * theta), self.gamma, 0, 0, 0, self.G * self.nu_star_prime**2]
        ])
        
        return H_k


class RhombohedralGrapheneHamiltonian(hamiltonian):
    """
    Minimal chiral effective Hamiltonian for n-layer rhombohedral graphene in the chiral basis,
    projected onto low-energy subspace, with a displacement field included.
    """
    def __init__(self, vF=542.1, t1=355.16, V=30.0, n=2, omega=2 * np.pi, A0=0):
        super().__init__(dim=2, omega=omega, A0=A0)
        self.vF = vF
        self.t1 = t1
        self.V = V
        self.n = n

    def k(self, kx, ky):
        return np.sqrt(kx**2 + ky**2)

    def theta(self, kx, ky):
        return np.arctan2(ky, kx)  # angle of k-vector


    def N_k(self, k):
        vk_t1 = (self.vF * k / self.t1)
        numerator = 1 - vk_t1 ** (2 * self.n)
        denominator = 1 - vk_t1 ** 2
        return np.sqrt(numerator / denominator)

    def V_k(self, k):
        vk_t1 = (self.vF * k / self.t1)
        num = (self.n - 1) * vk_t1 ** (2 * self.n + 2) + vk_t1 ** 2 - self.n * vk_t1 ** (2 * self.n)
        denom = (1 - vk_t1 ** 2) * (1 - vk_t1 ** (2 * self.n))
        return self.V * (-0.5 * (self.n - 1) + num / denom)
    
    def dN_dk(self, k):
        """
        Analytical derivative ∂N/∂k as defined by:
        ∂N/∂k = [a^2 k^2 (n (a k)^{2n} - (a k)^{2n} + 1) - n (a k)^{2n}]
                / [k (1 - a^2 k^2)^2 * sqrt(((a k)^{2n} - 1) / (a^2 k^2 - 1))]
        """
        a = self.vF / self.t1
        n = self.n
        ak = a * k
        ak2 = ak**2
        ak2n = ak2**n
        ak2n_plus = ak**(2 * n)

        numerator = a**2 * k**2 * (n * ak2n - ak2n + 1) - n * ak2n
        denominator = (
            k * (1 - a**2 * k**2) ** 2 *
            np.sqrt((ak2n - 1) / (a**2 * k**2 - 1))
        )
        return numerator / denominator

    def G_k(self, k):
        return -self.vF**self.n * k**self.n / (self.t1**(self.n - 1) * self.N_k(k)**2)

    def A_k(self, k):
        a = self.vF / self.t1
        return 1.0 - (a * k) ** 2

    def B_k(self, k):
        a = self.vF / self.t1
        n = self.n
        return (a * k) ** (2 * n) - 1.0
    
    def U_k(self, k):
        return 2*self.vF**self.n * self.M(k) * k**(self.n - 2) / (self.t1**(self.n - 1) * self.N_k(k)**4)

    def gamma_k(self, k):
        return self.vF**self.n * self.n * k**(self.n - 1) / (self.t1**(self.n - 1) * self.N_k(k)**2)
    
    def P_k(self, k):
        a = self.vF / self.t1
        n = self.n
        ak = a * k
        ak2 = ak ** 2
        ak2n = ak2 ** n
        return (-2 * a**4 * k**4 * n**2 * ak2n
                + 2 * a**2 * k**2 * (2 * n**2 * ak2n + (ak2n - 1)**2)
                - 2 * n**2 * ak2n)

    def Q_k(self, k):
        a = self.vF / self.t1
        n = self.n
        ak = a * k
        ak2 = ak ** 2
        ak2n = ak2 ** n
        return a**2 * k**2 * (n * ak2n - ak2n + 1) - n * ak2n

    
    def M(self, k):
        """
        Computes the analytical expression M(k) used in trace formula.
        M = [a^2 k^2 (n (a k)^{2n} - (a k)^{2n} + 1) - n (a k)^{2n}] / (1 - a^2 k^2)^2
        """
        a = self.vF / self.t1
        n = self.n
        ak = a * k
        ak2n = (ak ** (2 * n))
        
        numerator = a**2 * k**2 * (n * ak2n - ak2n + 1) - n * ak2n
        denominator = (1 - a**2 * k**2)**2
        return numerator / denominator



    def d_x(self, kx, ky):
        k = self.k(kx, ky)
        theta = self.theta(kx, ky)
        N = self.N_k(k)
        return - (self.vF ** self.n / self.t1 ** (self.n - 1)) * k ** self.n * np.cos(self.n * theta) / N**2

    def d_y(self, kx, ky):
        k = self.k(kx, ky)
        theta = self.theta(kx, ky)
        N = self.N_k(k)
        return - (self.vF ** self.n / self.t1 ** (self.n - 1)) * k ** self.n * np.sin(self.n * theta) / N**2

    def d_z(self, kx, ky):
        k = self.k(kx, ky)
        a = self.vF / self.t1
        ak = a * k
        numerator = (self.n - 1) * ak ** (2 * self.n + 2) + ak ** 2 - self.n * ak ** (2 * self.n)
        denominator = (1 - ak**2) * (1 - ak ** (2 * self.n))
        return self.V * (-0.5 * (self.n - 1) + numerator / denominator)
    
    def d_squared(self, kx, ky):
        """
        Computes the squared magnitude |d|^2 = d_x^2 + d_y^2 + d_z^2.
        Useful for expressions requiring the norm squared of the d vector.
        """
        dx = self.d_x(kx, ky)
        dy = self.d_y(kx, ky)
        dz = self.d_z(kx, ky)
        return dx**2 + dy**2 + dz**2

    def ddx_dx(self, kx, ky):
        k = self.k(kx, ky)
        theta = self.theta(kx, ky)
        dNk = self.dN_dk(k)
        a = self.vF / self.t1
        N = self.N_k(k)
        n = self.n

        term1 = (2 * self.vF**n) / (self.t1**(n - 1) * N**3) * dNk * (kx / k) * k**n * np.cos(n * theta)
        term2 = -(self.vF**n * n * k**(n - 1)) / (self.t1**(n - 1) * N**2) * np.cos(n * theta - theta)
        return term1 + term2

    def ddy_dx(self, kx, ky):
        k = self.k(kx, ky)
        theta = self.theta(kx, ky)
        dNk = self.dN_dk(k)
        a = self.vF / self.t1
        N = self.N_k(k)
        n = self.n

        term1 = (2 * self.vF**n) / (self.t1**(n - 1) * N**3) * dNk * (kx / k) * k**n * np.sin(n * theta)
        term2 = -(self.vF**n * n * k**(n - 1)) / (self.t1**(n - 1) * N**2) * np.sin(n * theta - theta)
        return term1 + term2

    def ddz_dx(self, kx, ky):
        k = self.k(kx, ky)
        a = self.vF / self.t1
        n = self.n
        vkx = kx / k

        num = (-2 * a**4 * k**4 * n**2 * (a * k)**(2 * n)
               + 2 * a**2 * k**2 * (2 * n**2 * (a * k)**(2 * n) + ((a * k)**(2 * n) - 1)**2)
               - 2 * n**2 * (a * k)**(2 * n))

        denom = k * (1 - a**2 * k**2)**2 * ((a * k)**(2 * n) - 1)**2

        return self.V * vkx * num / denom


    def ddx_dy(self, kx, ky):
        k = self.k(kx, ky)
        theta = self.theta(kx, ky)
        dNk = self.dN_dk(k)
        a = self.vF / self.t1
        N = self.N_k(k)
        n = self.n

        term1 = (2 * self.vF**n) / (self.t1**(n - 1) * N**3) * dNk * (ky / k) * k**n * np.cos(n * theta)
        term2 = (self.vF**n * n * k**(n - 1)) / (self.t1**(n - 1) * N**2) * np.sin(n * theta - theta)
        return term1 + term2

    def ddy_dy(self, kx, ky):
        k = self.k(kx, ky)
        theta = self.theta(kx, ky)
        dNk = self.dN_dk(k)
        a = self.vF / self.t1
        N = self.N_k(k)
        n = self.n

        term1 = (2 * self.vF**n) / (self.t1**(n - 1) * N**3) * dNk * (ky / k) * k**n * np.sin(n * theta)
        term2 = -(self.vF**n * n * k**(n - 1)) / (self.t1**(n - 1) * N**2) * np.cos(n * theta - theta)
        return term1 + term2

    def ddz_dy(self, kx, ky):
        k = self.k(kx, ky)
        a = self.vF / self.t1
        n = self.n
        vky = ky / k

        num = (-2 * a**4 * k**4 * n**2 * (a * k)**(2 * n)
               + 2 * a**2 * k**2 * (2 * n**2 * (a * k)**(2 * n) + ((a * k)**(2 * n) - 1)**2)
               - 2 * n**2 * (a * k)**(2 * n))

        denom = k * (1 - a**2 * k**2)**2 * ((a * k)**(2 * n) - 1)**2

        return self.V * vky * num / denom
    
    def d_magnitude(self, kx, ky):
        """
        Computes the norm |d| = sqrt(d_x^2 + d_y^2 + d_z^2)
        using the analytical expression:
        |d| = sqrt(V(k)^2 + [v_F^n k^n]^2 / N^4)
        """
        k = self.k(kx, ky)
        V_k = self.d_z(kx, ky)  # this is the scalar V(k)
        N = self.N_k(k)
        off_diag_term = (self.vF ** self.n * k ** self.n / (self.t1 ** (self.n - 1))) / N**2
        return np.sqrt(V_k**2 + off_diag_term**2)
    
    def sum_x(self, kx, ky):
        """
        Computes the sum of d_x components.
        """

        # Individual components
        ddx_dx = self.ddx_dx(kx, ky)
        ddy_dx = self.ddy_dx(kx, ky)
        ddz_dx = self.ddz_dx(kx, ky)

        dx = self.d_x(kx, ky)
        dy = self.d_y(kx, ky)
        dz = self.d_z(kx, ky)

        return dx * ddx_dx + dy * ddy_dx + dz * ddz_dx
    
    def sum_y(self, kx, ky):
        """
        Computes the sum of d_y components.
        """

        # Individual components
        ddx_dy = self.ddx_dy(kx, ky)
        ddy_dy = self.ddy_dy(kx, ky)
        ddz_dy = self.ddz_dy(kx, ky)

        dx = self.d_x(kx, ky)
        dy = self.d_y(kx, ky)
        dz = self.d_z(kx, ky)

        return dx * ddx_dy + dy * ddy_dy + dz * ddz_dy

    def partial_x_dhat_dot_partial_x_dhat(self, kx, ky):
        """
        Computes ∂x d̂ ⋅ ∂x d̂ using the full expansion:
        (1/d^2)(∂x d_i)^2 - (2/d^4)(∑ d_i ∂x d_i) + (1/d^6)(∑ d_i^2)(∑ d_i ∂x d_i)^2
        """
        # Individual components
        ddx_dx = self.ddx_dx(kx, ky)
        ddy_dx = self.ddy_dx(kx, ky)
        ddz_dx = self.ddz_dx(kx, ky)

        d_norm = self.d_magnitude(kx, ky)
        d2 = d_norm**2
        d4 = d2**2

        dot_d_dxd = self.sum_x(kx, ky)

        term1 = (ddx_dx**2 + ddy_dx**2 + ddz_dx**2)/ d2
        term2 = - dot_d_dxd ** 2 / d4

        return term1 + term2 
    
    def partial_y_dhat_dot_partial_y_dhat(self, kx, ky):
        """
        Computes ∂y d̂ ⋅ ∂y d̂ using the full expansion:
        (1/d^2)(∂y d_i)^2 - (2/d^4)(∑ d_i ∂y d_i)^2 + (1/d^6)(∑ d_i^2)(∑ d_i ∂y d_i)^2
        """
        # ∂y d_i components
        ddx_dy = self.ddx_dy(kx, ky)
        ddy_dy = self.ddy_dy(kx, ky)
        ddz_dy = self.ddz_dy(kx, ky)

        # d_i values
        dx = self.d_x(kx, ky)
        dy = self.d_y(kx, ky)
        dz = self.d_z(kx, ky)

        d_norm = self.d_magnitude(kx, ky)
        d2 = d_norm**2
        d4 = d2**2
        d6 = d2**3

        # ∑ d_i ∂y d_i
        dot_d_dyd = dx * ddx_dy + dy * ddy_dy + dz * ddz_dy

        # (∂y d_x)^2 + (∂y d_y)^2 + (∂y d_z)^2
        term1 = (ddx_dy**2 + ddy_dy**2 + ddz_dy**2) / d2

        # -2 * (∑ d_i ∂y d_i)^2 / d^4
        term2 = -2 * dot_d_dyd**2 / d4

        # (∑ d_i^2) * (∑ d_i ∂y d_i)^2 / d^6
        d_squared_sum = dx**2 + dy**2 + dz**2
        term3 = dot_d_dyd**2 * d_squared_sum / d6

        return term1 + term2 + term3


    def compute_static(self, kx, ky):
        k = np.sqrt(kx**2 + ky**2)
        k_plus = kx + 1j * ky
        k_minus = kx - 1j * ky
        N = self.N_k(k)
        V_k = self.V_k(k)
        off_diag = -self.t1 * (self.vF * k_plus / self.t1) ** self.n / N**2

        return np.array([
            [V_k, np.conj(off_diag)],
            [off_diag, -V_k]
        ], dtype=complex)
        
    
    def analytic_eigenvalues(self, kx, ky):
        """
        Return the correct analytical eigenvalues of the projected 2x2 rhombohedral graphene Hamiltonian
        with a displacement field, based on Appendix B of the paper.

        Parameters:
            kx, ky (float): Momentum components.

        Returns:
            np.ndarray: Two eigenvalues (E_minus, E_plus)
        """
        k = np.sqrt(kx**2 + ky**2)
        vk_t1 = self.vF * k / self.t1
        N = self.N_k(k)
        V_k = self.V_k(k)

        # |off-diagonal|^2 = (vF^n * k^n)^2 / (t1^{2n-2} * N^4)
        h_off_sq = (self.vF**self.n * k**self.n / (self.t1**(self.n - 1) * N**2))**2

        E = np.sqrt(V_k**2 + h_off_sq)
        return np.array([-E, E])

    def g_xx(self, kx, ky):
        """
        Returns the quantum metric component g_xx(k) = 1/4 * (∂x d̂ ⋅ ∂x d̂)
        """
        return 0.25 * self.partial_x_dhat_dot_partial_x_dhat(kx, ky)
    
    def g_yy(self, kx, ky):
        """
        Returns the quantum metric component g_yy(k) = 1/4 * (∂y d̂ ⋅ ∂y d̂)
        """
        return 0.25 * self.partial_y_dhat_dot_partial_y_dhat(kx, ky)


    def trace(self, kx, ky):
        """
        Computes the analytical expression for the trace of the quantum metric tensor Tr[g],
        based on Appendix B formula using M(k), H(k), and V_k(k).
        """
        k = self.k(kx, ky)
        n = self.n
        vF = self.vF
        t1 = self.t1
        d = self.d_magnitude(kx, ky)

        k_pow_4n_2 = k ** (4 * n - 2)
        k_pow_2n_2 = k ** (2 * n - 2)

        N = self.N_k(k)
        N2 = N**2
        N4 = N2**2
        N6 = N2**3
        N8 = N4**2

        M_val = self.M(k)
        H_val = self.H(k)
        V_k_val = self.V_k(k)

        term1 = (vF ** (4 * n) * n**2 * k_pow_4n_2) / (t1 ** (4 * n - 4) * N8)
        term2 = (vF ** (2 * n) * H_val**2 * k_pow_2n_2) / (t1 ** (2 * n - 2) * N4)
        term3 = (4 * vF ** (2 * n) * M_val**2 * V_k_val**2 * k_pow_2n_2) / (t1 ** (2 * n - 2) * N8)
        term4 = (2 * vF ** (2 * n) * n**2 * V_k_val**2 * k_pow_2n_2) / (t1 ** (2 * n - 2) * N4)
        term5 = (-4 * vF ** (2 * n) * M_val * n * V_k_val**2 * k_pow_2n_2) / (t1 ** (2 * n - 2) * N6)
        term6 = (4 * vF ** (2 * n) * M_val * H_val * V_k_val * k_pow_2n_2) / (t1 ** (2 * n - 2) * N6)
        term7 = (-2 * vF ** (2 * n) * n * V_k_val * H_val * k_pow_2n_2) / (t1 ** (2 * n - 2) * N4)

        return 1/4 * (term1 + term2 + term3 + term4 + term5 + term6 + term7)/d**4

    def valid_k_point(self, kx, ky):
        k = np.sqrt(kx**2 + ky**2)
        # return True
        return (self.vF * k / self.t1) < 1
    
    def H(self, k):
        """
        Computes the analytical expression H(k) used in trace formula.
        H = V * [−2 a⁴ k⁴ n² (a k)^{2n} + 2 a² k² (2 n² (a k)^{2n} + ((a k)^{2n} − 1)²) − 2 n² (a k)^{2n}]
                / [ (1 − a² k²)² ((a k)^{2n} − 1)² ]
        """
        a = self.vF / self.t1
        n = self.n
        ak = a * k
        ak2 = ak**2
        ak2n = ak2**n
        V = self.V

        num = (-2 * a**4 * k**4 * n**2 * ak2n +
            2 * a**2 * k**2 * (2 * n**2 * ak2n + (ak2n - 1)**2) -
            2 * n**2 * ak2n)

        denom = (1 - a**2 * k**2)**2 * (ak2n - 1)**2
        return V * num / denom
    
    def analytic_magnus_first_term(self, kx, ky):
        """
        Analytic first-order Magnus correction H^(1)/omega (2×2), returned as a matrix.
        This ALREADY includes the 1/omega factor.
        """
        k = np.hypot(kx, ky)
        if k == 0:
            # Singular prefactors like k^{n-2}; take zero correction at exactly k=0 by default
            return np.zeros((2, 2), dtype=complex)

        theta = np.arctan2(ky, kx)
        n = self.n

        A = self.A_k(k)
        B = self.B_k(k)
        P = self.P_k(k)
        Q = self.Q_k(k)

        eps = 1e-15
        if abs(A) < eps or abs(B) < eps:
            return np.zeros((2, 2), dtype=complex)

        # ---- off-diagonal term ----
        # using constant V (self.V). If you meant V_k(k), replace self.V with self.V_k(k).
        pref_off = (self.vF ** n) * n * self.V * P
        pref_off /= (self.t1 ** (n - 1)) * A * (B ** 3)
        pref_off *= k ** (n - 2)

        phase = np.exp(1j * n * theta)
        off = pref_off * phase
        H_off = np.array([[0.0, np.conj(off)],
                          [off, 0.0]], dtype=complex)

        # ---- σ_z term ----
        coeff_z = (self.vF ** (2 * n)) / (self.t1 ** (2 * n - 2))
        coeff_z *= (n * A) / (B ** 2)
        coeff_z *= k ** (2 * n - 2)
        coeff_z *= (n * A - 2 * Q / B)

        H_z = np.array([[coeff_z, 0.0],
                        [0.0, -coeff_z]], dtype=complex)

        # include division by omega here
        return - (H_off + H_z) * self.A0**2 / self.omega
    
class ChiralHamiltonian(hamiltonian):
    """
    Full chiral (v_F, t1-only) Hamiltonian + linear displacement field H_D.
    Basis: (layer 0: A,B; layer 1: A,B; ...; layer n-1: A,B).
    """
    def __init__(self, n=2, vF=542.1, t1=355.16, V=0.0,
                 valley='K', omega=2*np.pi, A0=0):
        super().__init__(dim=2*n, omega=omega, A0=A0)
        self.n = int(n)
        self.vF = float(vF)
        self.t1 = float(t1)
        self.V  = float(V)           # displacement-field strength (same units as energies)

        v = valley.lower()
        if v in ('k',):
            self.eta = 1.0
        elif v in ('kp', 'kprime', "k'"):
            self.eta = -1.0
        else:
            raise ValueError("valley must be 'K' or 'Kp'/'Kprime'")

        # 2x2 sublattice operators
        self._sigma_plus  = 0.5 * (sigma_x + 1j * sigma_y)
        self._sigma_minus = 0.5 * (sigma_x - 1j * sigma_y)
        self._I2 = np.eye(2, dtype=complex)

        # Precompute the layer potentials V_l = V * (l - (n-1)/2)
        center = (self.n - 1) / 2.0
        self._V_layer = np.array([self.V * (l - center) for l in range(self.n)], dtype=float)

    def _D_block(self, kx, ky):
        # D = vF * ( kx σ_x + η ky σ_y )
        return self.vF * (kx * sigma_x + self.eta * ky * sigma_y)

    def _L_block(self):
        # L = t1 σ^+
        return self.t1 * self._sigma_plus

    def _U_block(self):
        # U = t1 σ^-
        return self.t1 * self._sigma_minus

    def compute_static(self, kx, ky):
        """
        H(k) = h_n(k) + H_D
        h_n(k): block-tridiagonal chiral Hamiltonian
        H_D: diagonal layer-linear onsite potentials (same for A/B on a layer)
        """
        n = self.n
        H = np.zeros((2*n, 2*n), dtype=complex)

        D = self._D_block(kx, ky)
        L = self._L_block()
        U = self._U_block()

        for l in range(n):
            # chiral intralayer block
            H[2*l:2*l+2, 2*l:2*l+2] = D
            # add displacement-field onsite to this layer (A & B equally)
            Vl = self._V_layer[l]
            H[2*l:2*l+2, 2*l:2*l+2] += Vl * self._I2

            # interlayer couplings
            if l < n - 1:
                H[2*(l+1):2*(l+1)+2, 2*l:2*l+2] = L   # sub-diagonal
                H[2*l:2*l+2, 2*(l+1):2*(l+1)+2] = U   # super-diagonal

        return H
