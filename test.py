import numpy as np
import matplotlib.pyplot as plt

# Constants and Component Values
Z0 = 50  # Characteristic Impedance in Ohms
R = 50e3  # Resistance in Ohms
L = 7.9577e-9  # Inductance in Henries
C = 3.0407e-12  # Capacitance in Farads
Cc = 0.1424e-12  # Coupling Capacitance in Farads

# Frequency Range: 0.98 GHz to 1.02 GHz
f_start = 0.98e9  # 0.98 GHz
f_end = 1.02e9    # 1.02 GHz
num_points = 1000  # Number of frequency points
frequencies = np.linspace(f_start, f_end, num_points)
omega = 2 * np.pi * frequencies  # Angular frequency

# Calculate Impedance Z for each frequency
# Z = Zc + Z_parallel
# Zc = 1 / (jωCc)
# Z_parallel = 1 / (1/R + 1/(jωL) + jωC)

# Calculate Z_parallel
Y_R = 1 / R  # Admittance of Resistor
Y_L = 1 / (1j * omega * L)  # Admittance of Inductor
Y_C = 1j * omega * C  # Admittance of Capacitor

Y_parallel = Y_R + Y_L + Y_C  # Total Admittance of Parallel Branch
Z_parallel = 1 / Y_parallel  # Impedance of Parallel Branch

# Calculate Zc
Zc = 1 / (1j * omega * Cc)  # Impedance of Coupling Capacitor

# Total Shunt Impedance Z
Z_total = Zc + Z_parallel

# Calculate Scattering Matrix S
# From part (d): S = 1 / (2 + Z0 / Z) * [ -Z0 / Z, 2; 2, -Z0 / Z ]
# However, to generalize for complex Z, it's better to compute as follows:

# Calculate denominator for S matrix
denominator = Z0 + 2*Z_total

# S11 and S21
S11 = -Z0 / denominator
S21 = 2*Z_total / denominator

# Calculate |S11| and |S21|
abs_S11 = np.abs(S11)
abs_S21 = np.abs(S21)

# Calculate Power Absorption Coefficient eta_abs
eta_abs = 1 - abs_S11**2 - abs_S21**2

# Plotting the Results
plt.figure(figsize=(12, 8))

# Plot |S11| and |S21|
plt.subplot(2, 1, 1)
plt.plot(frequencies/1e9, abs_S11, label=r'$|S_{11}|$ (Reflection)')
plt.plot(frequencies/1e9, abs_S21, label=r'$|S_{21}|$ (Transmission)')
plt.title('Reflection and Transmission Coefficients vs Frequency')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

# Plot eta_abs
plt.subplot(2, 1, 2)
plt.plot(frequencies/1e9, eta_abs, color='purple', label=r'$\eta_{\mathrm{abs}}$ (Power Absorption)')
plt.title('Power Absorption Coefficient vs Frequency')
plt.xlabel('Frequency (GHz)')
plt.ylabel(r'$\eta_{\mathrm{abs}}$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
