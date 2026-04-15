import numpy as np
from scipy.integrate import quad
from qutip import sigmax, sigmay, sigmaz
from ..basic_math import commutator_static
from scipy.fft import fft


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
    
    def get_parameters_dict(self, parameter='2D'):
        """
        Get all the simple types of the properties of the Hamiltonian as a dictionary.
        
        Parameters:
            parameter (str): '1D', '2D', or '3D'. If '1D', exclude omega (default '2D').
        """
        def is_simple_type(val):
            return isinstance(val, (int, float, str, bool))

        # Collect simple, public attributes (exclude name, dim)
        params = {
            k: v for k, v in vars(self).items()
            if not callable(v)
            and not k.startswith('_')
            and k not in ('name', 'dim')
            and is_simple_type(v)
        }

        # Exclude omega for 1D
        if parameter == '1D':
            params.pop('omega', None)

        return params

    def get_filename(self, parameter='2D', decimals=2):
        """
        Generate a compact, stable filename-style string of parameters.
        Example: t1_-1.00-t2_0.33-A0_0.10-omega_6.28-polarization_left
        
        Parameters:
            parameter (str): '1D' or '2D'. If '1D', exclude omega (default '2D').
            decimals (int): float precision for formatting.
        """
        params = self.get_parameters_dict(parameter=parameter)

        def fmt_val(v):
            if isinstance(v, float):
                return f"{v:.{decimals}f}"
            return str(v)

        # Build parts in a stable order
        parts = [f"{k}_{fmt_val(params[k])}" for k in sorted(params.keys())]

        # Join with '-' between parameters
        return "-".join(parts)

    
    def compute_static(self, kx, ky, kz=0):
        """
        Compute the static Hamiltonian matrix for a given (kx, ky, kz).
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'compute_static' method.")

    def compute_driven(self, t, kx, ky, kz=0):
        """
        Compute the time-dependent Hamiltonian matrix for a given time t and (kx, ky, kz),
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
        return self.compute_static(kx_t, ky_t, kz)

    # def numerical_fourier_component(self, n, kx, ky):
    #     """
    #     Compute the nth Fourier component of the time-dependent Hamiltonian.

    #     Parameters:
    #         n (int): Fourier component index.
    #         kx, ky (float): Parameters for the Hamiltonian.
    #     Returns:
    #         complex: The nth Fourier component.
    #     """
    #     baseline_epsrel = 1e-7
    #     adjusted_epsrel = baseline_epsrel / (1 + n * self.omega)

    #     # Define the integrand function
    #     def integrand(t):
    #         return self.compute_driven(t, kx, ky) * np.exp(-1j * n * self.omega * t)

    #     # Perform numerical integration
    #     result, error = quad_vec(
    #         integrand, 
    #         0, 
    #         2 * np.pi / self.omega, 
    #         epsrel=adjusted_epsrel
    #     )

    #     # Scale the result by the pre-factor
    #     integral = (self.omega / (2 * np.pi)) * result

    #     # Estimate how many decimal digits we can trust
    #     if error == 0 or not np.isfinite(error):
    #         return integral  # nothing to round to

    #     # Determine number of decimal digits from error magnitude
    #     digits = max(0, -int(np.floor(np.log10(error))))
    #     return integral


    # ---- Vectorized sampler over one period (M time samples) ----
    def _kt_over_period(self, kx, ky, M, kz=0):
        """Return kx_t, ky_t arrays of shape (M,) over one period."""
        T = 2 * np.pi / self.omega
        t = np.linspace(0.0, T, M, endpoint=False)
        if self.polarization == 'left':
            kx_t = kx + self.A0 * np.cos(self.omega * t)
            ky_t = ky + self.A0 * np.sin(self.omega * t)
        elif self.polarization == 'right':
            kx_t = kx + self.A0 * np.cos(self.omega * t)
            ky_t = ky - self.A0 * np.sin(self.omega * t)
        elif self.polarization == 'linear_x':
            kx_t = kx + self.A0 * np.cos(self.omega * t)
            ky_t = np.full_like(t, ky, dtype=float)
        elif self.polarization == 'linear_y':
            kx_t = np.full_like(t, kx, dtype=float)
            ky_t = ky + self.A0 * np.cos(self.omega * t)
        else:
            raise ValueError("Invalid polarization type.")
        return kx_t, ky_t

    def compute_static_vectorized(self, kx_arr, ky_arr, kz_arr=0):
        """
        Default vectorized fallback: loops. Subclasses should override with
        a fully vectorized implementation (no Python loop) for speed.
        """
        kx_arr = np.asarray(kx_arr)
        ky_arr = np.asarray(ky_arr)
        M = kx_arr.shape[0]
        H = np.empty((M, self.dim, self.dim), dtype=complex)
        for m in range(M):
            kz_val = kz_arr[m] if isinstance(kz_arr, (np.ndarray, list)) else kz_arr
            H[m] = self.compute_static(kx_arr[m], ky_arr[m], kz_val)
        return H

    def _H_stack_over_period(self, kx, ky, M, kz=0):
        """Build H(t_m) stack with shape (M, dim, dim), preferably vectorized."""
        kx_t, ky_t = self._kt_over_period(kx, ky, M, kz)
        return self.compute_static_vectorized(kx_t, ky_t, kz)

    def fourier_components_fft(self, ns, kx, ky, kz=0, M=512):
        """
        Compute multiple harmonics H_n in one shot via FFT.
        Args:
            ns: iterable of ints (can include negatives)
            kx, ky: center point
            M: # of uniform samples over one period (power of two is good)
        Returns:
            dict {n: (dim,dim) complex ndarray}
        """
        # Sample H(t) over one period (vectorized)
        Ht = self._H_stack_over_period(kx, ky, M, kz)  # (M, dim, dim)

        # FFT along time axis. Convention of np/scipy FFT matches
        # sum_{m=0}^{M-1} H(t_m) * exp(-i 2π m k / M).
        F = fft(Ht, axis=0, workers=-1, overwrite_x=True)

        # Our coefficient is (1/M) Σ H(t_m) e^{-i n ω t_m};
        # with t_m = m*T/M and ωT=2π, index = n mod M.
        out = {}
        for n in ns:
            idx = n % M
            out[n] = F[idx] / M
        return out

    # Backward-compatible single-harmonic API
    def numerical_fourier_component(self, n, kx, ky, kz=0, M=512):
        return self.fourier_components_fft([n], kx, ky, kz, M=M)[n]


    def numerical_fourier_component_rounding(self, n, kx, ky, kz=0):
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
        sample = self.compute_driven(0, kx, ky, kz)
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
                    val = self.compute_driven(t, kx, ky, kz)[i, j]
                    return np.real(val * np.exp(-1j * n * self.omega * t))

                def integrand_imag(t):
                    val = self.compute_driven(t, kx, ky, kz)[i, j]
                    return np.imag(val * np.exp(-1j * n * self.omega * t))

                real_val, real_err = quad(integrand_real, 0, T, epsrel=adjusted_epsrel)
                imag_val, imag_err = quad(integrand_imag, 0, T, epsrel=adjusted_epsrel)

                # Round before scaling
                rounded_real[i, j] = round_to_error(real_val, real_err)
                rounded_imag[i, j] = round_to_error(imag_val, imag_err)

        # Apply scaling
        result = scale * (rounded_real + 1j * rounded_imag)
        return result

    def magnus_first_term(self, kx, ky, kz=0):
        """
        Compute the first term of the Magnus expansion:
        (1/omega) * [H1, H-1], rounded to 1e-16 precision.
        """
        # Compute H1 and H-1 Fourier components
        H1 = self.numerical_fourier_component(1, kx, ky, kz)
        Hm1 = self.numerical_fourier_component(-1, kx, ky, kz)

        # Compute the commutator [H1, H-1]
        comm = commutator_static(H1, Hm1)

        # Compute Magnus first term
        magnus_term = (1 / self.omega) * comm

        # Round to 1e-16 precision
        epsilon = 1e-16
        rounded_magnus = np.round(magnus_term / epsilon) * epsilon

        return rounded_magnus

    def magnus_second_term(self, kx, ky, kz=0):
        """
        Compute the second Magnus term:
        (1/omega) * (1/2) * [H2, H-2]
        """
        # Compute H2 and H-2 Fourier components
        H2 = self.numerical_fourier_component(2, kx, ky, kz)
        Hm2 = self.numerical_fourier_component(-2, kx, ky, kz)

        # Compute the commutator [H2, H-2]
        comm = commutator_static(H2, Hm2)

        # Return the second Magnus term
        return (1 / (2 * self.omega)) * comm

    def effective_hamiltonian(self, kx, ky, kz=0):
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
        H_0 = self.compute_static(kx, ky, kz)

        # Initialize perturbation Hamiltonian (H_prime) as zero matrix of same shape as H_0
        H_prime = np.zeros_like(H_0)

        # If A0 is 0, there are no driving terms; return static Hamiltonian directly
        if self.A0 == 0:
            return H_0, H_prime

        # Add Magnus terms to H_prime based on specified order
        if self.magnus_order >= 1:
            if self.analytic_magnus and hasattr(self, "analytic_magnus_first_term"):
                # Expect analytic_magnus_first_term to ALREADY include the 1/omega factor
                H_prime += self.analytic_magnus_first_term(kx, ky, kz)
            else:
                H_prime += self.magnus_first_term(kx, ky, kz)
        if self.magnus_order >= 2:
            H_prime += self.magnus_second_term(kx, ky, kz)
        
        # Compute effective Hamiltonian (H_eff = H_0 + H_prime)
        H_eff = H_0 + H_prime

        return H_eff, H_prime
    
    # ____________________________________________________________________________________________________________
    # Below are the method to compute the Fourier Harmonics of the Hamiltonian by the Taylor Expansion Method
    def get_derivative(self, func_name, kx, ky, kz=0):
        """Helper function to check and call derivative functions dynamically."""
        if hasattr(self, func_name):
            return getattr(self, func_name)(kx, ky, kz)
        return 0  # Return 0 if the function does not exist

    def fx11p(self, kx, ky, kz=0):
        return self.A0 * (0.5 * self.get_derivative("dfxdx", kx, ky, kz) + (0.5 / 1j) * self.get_derivative("dfxdy", kx, ky, kz))
    
    def fy11p(self, kx, ky, kz=0):
        return self.A0 * (0.5 * self.get_derivative("dfydx", kx, ky, kz) + (0.5 / 1j) * self.get_derivative("dfydy", kx, ky, kz))

    def fx22p(self, kx, ky, kz=0):
        return (self.A0**2 / 4) * (0.5 * self.get_derivative("dfxdxx", kx, ky, kz) - 0.5 * self.get_derivative("dfxdyy", kx, ky, kz) - 1j * self.get_derivative("dfxdxy", kx, ky, kz))

    def fy22p(self, kx, ky, kz=0):
        return (self.A0**2 / 4) * (0.5 * self.get_derivative("dfydxx", kx, ky, kz) - 0.5 * self.get_derivative("dfydyy", kx, ky, kz) - 1j * self.get_derivative("dfydxy", kx, ky, kz))

    def fx31p(self, kx, ky, kz=0):
        return (self.A0**3 / 16) * (self.get_derivative("dfxdxxx", kx, ky, kz) + 3 * self.get_derivative("dfxdxyy", kx, ky, kz)) + \
               (self.A0**3 / (16j)) * (3 * self.get_derivative("dfxdxxy", kx, ky, kz) + self.get_derivative("dfxdyyy", kx, ky, kz))

    def fy31p(self, kx, ky, kz=0):
        return (self.A0**3 / 16) * (self.get_derivative("dfydxxx", kx, ky, kz) + 3 * self.get_derivative("dfydxyy", kx, ky, kz)) + \
               (self.A0**3 / (16j)) * (3 * self.get_derivative("dfydxxy", kx, ky, kz) + self.get_derivative("dfydyyy", kx, ky, kz))

    def fx33p(self, kx, ky, kz=0):
        return (self.A0**3 / 48) * (self.get_derivative("dfxdxxx", kx, ky, kz) + 3 * self.get_derivative("dfxdxyy", kx, ky, kz)) + \
               (self.A0**3 / (48j)) * (3 * self.get_derivative("dfxdxxy", kx, ky, kz) + self.get_derivative("dfxdyyy", kx, ky, kz))

    def fy33p(self, kx, ky, kz=0):
        return (self.A0**3 / 48) * (self.get_derivative("dfydxxx", kx, ky, kz) + 3 * self.get_derivative("dfydxyy", kx, ky, kz)) + \
               (self.A0**3 / (48j)) * (3 * self.get_derivative("dfydxxy", kx, ky, kz) + self.get_derivative("dfydyyy", kx, ky, kz))
    
    def fx51p(self, kx, ky, kz=0):
        """
        Compute the positive fifth-order Fourier component for f_x:
        
        f₅,₊₁ = (A₀⁵/120) * { ½[ (5/8)∂ₖₓ⁵ f + (5/4)∂ₖₓ³∂ₖ_y² f + (5/8)∂ₖₓ∂ₖ_y⁴ f ]
                + (1/(2i))[ (5/8)∂ₖₓ⁴∂ₖ_y f + (5/4)∂ₖₓ²∂ₖ_y³ f + (5/8)∂ₖ_y⁵ f ] }.
        """
        prefactor = self.A0**5 / 120
        real_part = 0.5 * (
            (5/8) * self.get_derivative("dfxdxxxxx", kx, ky, kz) +
            (5/4) * self.get_derivative("dfxdxxxyy", kx, ky, kz) +
            (5/8) * self.get_derivative("dfxdxyyyy", kx, ky, kz)
        )
        imag_part = (1/(2j)) * (
            (5/8) * self.get_derivative("dfxdxxxxy", kx, ky, kz) +
            (5/4) * self.get_derivative("dfxdxxyy", kx, ky, kz) +
            (5/8) * self.get_derivative("dfxdyyyyy", kx, ky, kz)
        )
        return prefactor * (real_part + imag_part)

    def fy51p(self, kx, ky, kz=0):
        """
        Compute the positive fifth-order Fourier component for f_y:
        
        f₅,₊₁ = (A₀⁵/120) * { ½[ (5/8)∂ₖₓ⁵ f + (5/4)∂ₖₓ³∂ₖ_y² f + (5/8)∂ₖₓ∂ₖ_y⁴ f ]
                + (1/(2i))[ (5/8)∂ₖₓ⁴∂ₖ_y f + (5/4)∂ₖₓ²∂ₖ_y³ f + (5/8)∂ₖ_y⁵ f ] }.
        """
        prefactor = self.A0**5 / 120
        real_part = 0.5 * (
            (5/8) * self.get_derivative("dfydxxxxx", kx, ky, kz) +
            (5/4) * self.get_derivative("dfydxxxyy", kx, ky, kz) +
            (5/8) * self.get_derivative("dfydxyyyy", kx, ky, kz)
        )
        imag_part = (1/(2j)) * (
            (5/8) * self.get_derivative("dfydxxxxy", kx, ky, kz) +
            (5/4) * self.get_derivative("dfydxxyy", kx, ky, kz) +
            (5/8) * self.get_derivative("dfydyyyyy", kx, ky, kz)
        )
        return prefactor * (real_part + imag_part)
    
    # The first Harmonic calculated from the first three orders of Taylor Expansions
    def Hp1(self, kx, ky, kz=0):
        return (self.fx11p(kx, ky, kz) + self.fx31p(kx, ky, kz) + self.fx51p(kx, ky, kz)) * sigma_x + (self.fy11p(kx, ky, kz) + self.fy31p(kx, ky, kz) + self.fy51p(kx, ky, kz)) * sigma_y

    def Hp113(self, kx, ky, kz=0):
        return (self.fx11p(kx, ky, kz) + self.fx31p(kx, ky, kz)) * sigma_x + (self.fy11p(kx, ky, kz) + self.fy31p(kx, ky, kz)) * sigma_y
    
    def Hp11(self, kx, ky, kz=0):
        return (self.fx11p(kx, ky, kz)) * sigma_x + (self.fy11p(kx, ky, kz)) * sigma_y

    def Hp13(self, kx, ky, kz=0):
        return (self.fx31p(kx, ky, kz)) * sigma_x + (self.fy31p(kx, ky, kz)) * sigma_y
        
    def Hp15(self, kx, ky, kz=0):
        return (self.fx51p(kx, ky, kz)) * sigma_x + (self.fy51p(kx, ky, kz)) * sigma_y

    def Hp2(self, kx, ky, kz=0):
        return self.fx22p(kx, ky, kz) * sigma_x + self.fy22p(kx, ky, kz) * sigma_y

    def Hp3(self, kx, ky, kz=0):
        return self.fx33p(kx, ky, kz) * sigma_x + self.fy33p(kx, ky, kz) * sigma_y
    
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
    
    def fx(self, kx, ky, kz=0):
        return self.a * kx**2 + self.b * ky**2 + self.c * kx * ky
    
    def compute_static(self, kx, ky, kz=0):
        return self.fx(kx, ky, kz) * sigma_x
    
    # First derivatives
    def dfxdx(self, kx, ky, kz=0):
        return 2 * self.a * kx + self.c * ky
    
    def dfxdy(self, kx, ky, kz=0):
        return 2 * self.b * ky + self.c * kx
    
    # Second derivatives
    def dfxdxx(self, kx, ky, kz=0):
        return 2 * self.a
    
    def dfxdyy(self, kx, ky, kz=0):
        return 2 * self.b
    
    def dfxdxy(self, kx, ky, kz=0):
        return self.c
    
    # Third derivatives (All Zero)
    def dfxdxxx(self, kx, ky, kz=0):
        return 0
    
    def dfxdxxy(self, kx, ky, kz=0):
        return 0
    
    def dfxdxyy(self, kx, ky, kz=0):
        return 0
    
    def dfxdyyy(self, kx, ky, kz=0):
        return 0




class GrapheneHamiltonian(hamiltonian):
    def __init__(self, omega = np.pi, A0 = 0):
        super().__init__(dim=2, omega=omega, A0=A0)  

    def compute_static(self, kx, ky, kz=0):
        H_k = np.array([
            [0, kx - 1j*ky],
            [kx + 1j*ky, 0]
        ])
        
        return H_k
    def Hp1(self, kx, ky, kz=0):
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

    def compute_static(self, kx, ky, kz=0):
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
    
    def Hp1(self, kx, ky, kz=0):
        H11 = (self.A0 / (2 * self.m)) * (kx - 1j*ky)
        H12 = 0
        H21 = -1j * self.alpha * self.A0
        H22 = H11


        H_k = np.array([
            [H11, H12],
            [H21, H22]
        ], dtype=complex)
        
        return H_k
    
    def Hp2(self, kx, ky, kz=0):
        H11 = 0
        H12 = 0
        H21 = 0
        H22 = 0

        H_k = np.array([
            [H11, H12],
            [H21, H22]
        ], dtype=complex)
        
        return H_k

