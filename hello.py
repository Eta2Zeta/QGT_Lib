import numpy as np
import matplotlib.pyplot as plt

# Parameters
omega0 = 5.0     # natural frequency
zeta = 0.2       # damping ratio

# Time domain (start from 1 to avoid log(0))
t = np.linspace(1, 10, 1000)

# Damped frequency
omega_d = omega0 * np.sqrt(1 - zeta**2)

# Original underdamped step response
f = 1 - np.exp(-zeta * omega0 * t) * (
    np.cos(omega_d * t) + (zeta / np.sqrt(1 - zeta**2)) * np.sin(omega_d * t)
)

# Shift so that f(t=1) = 0
time_working = f - f[0]

# Work efficiency
work_efficiency = np.log(t)

# Work done = time_working × work_efficiency
work_done = time_working * work_efficiency

# Plot
plt.figure(figsize=(8, 5))
plt.plot(t, time_working, label="Time Working (shifted)", linewidth=2)
plt.plot(t, work_efficiency, label="Work Efficiency (log t)", linewidth=2)
plt.plot(t, work_done, label="Work Done", linewidth=3, linestyle='--')

plt.title("Work Done = Time Working × Work Efficiency", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.axhline(0, color='k', linestyle='--', alpha=0.4)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
