import numpy as np
import matplotlib.pyplot as plt

# ----- Parameters -----
a = 1.5  # choose any a so that 1/(2a) < 1

# ----- Domain and piecewise definition -----
A0 = np.linspace(0.0, 1.0, 1001)
k_line1 = A0
k_line2 = 1.0 / a - A0
k_star = np.maximum(k_line1, k_line2)

# ----- Key points -----
A0_cross = 1.0 / (2.0 * a)
k_cross = A0_cross
y_max = 1.0 / a

# ----- Plot -----
plt.figure(figsize=(9, 9))  # square figure

# main curve (bold)
plt.plot(A0, k_star, linewidth=2.2, color="black", label=r"$k_\star$")
# helper lines
plt.plot(A0, k_line1, linewidth=1.1, color="black", linestyle="--", label=r"$k=A_0$")
plt.plot(A0, k_line2, linewidth=1.1, color="black", linestyle=":", label=rf"$k=\frac{{1}}{{a}} - A_0$")

# mark intersection and boundaries
plt.axhline(y=y_max, linestyle=(0, (1, 3)), linewidth=1.0, color="black")  # y = 1/a
plt.axvline(x=A0_cross, linestyle="-.", linewidth=1.0, color="black")      # A0 = 1/(2a)
plt.scatter([A0_cross], [k_cross], s=24, color="black", zorder=5)

# labels and limits
plt.xlabel(r"$A_0$")
plt.ylabel(r"$k_\star$")
plt.xlim(0.0, 1.0)
plt.ylim(0.0, max(1.05, y_max * 1.05))
plt.gca().set_aspect('equal', adjustable='box')  # make the plot square

# text annotations
plt.text(A0_cross, 0.02 * y_max, rf"$A_0=\frac{{1}}{{2a}}$", ha="center", va="bottom", fontsize=10)
plt.text(0.02, y_max + 0.02, rf"$y=\frac{{1}}{{a}}$", ha="left", va="bottom", fontsize=10)

# legend on top-left
plt.legend(frameon=False, fontsize=9, loc="upper left")

# layout and render
plt.tight_layout()
plt.show()
