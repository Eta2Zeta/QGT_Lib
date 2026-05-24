import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
theta_deg = 10          # use 10 deg so the rotation is visually obvious
KD = 1.0                # magnitude of Brillouin-zone corner vector |K|
theta = np.deg2rad(theta_deg)

# BM convention: rotate the two layers by +theta/2 and -theta/2
theta_top = +theta / 2
theta_bottom = -theta / 2


# -----------------------------
# Basic geometry helpers
# -----------------------------
def R(angle):
    """2D rotation matrix."""
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])


def hexagon_vertices(KD=1.0, rotation=0.0):
    """
    Regular hexagon vertices representing the graphene BZ.
    KD is the distance from Gamma to a corner, i.e. |K|.
    """
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + rotation
    return KD * np.column_stack([np.cos(angles), np.sin(angles)])


def draw_hex(ax, verts, color, label):
    """Draw closed hexagon."""
    closed = np.vstack([verts, verts[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, lw=2, label=label)


# -----------------------------
# Layer BZs
# -----------------------------
bz_top = hexagon_vertices(KD, theta_top)
bz_bottom = hexagon_vertices(KD, theta_bottom)

# Pick the Dirac point that originally sat at +x direction.
# After rotation, these are the two layer Dirac points.
K_top = R(theta_top) @ np.array([KD, 0.0])
K_bottom = R(theta_bottom) @ np.array([KD, 0.0])

# Their separation
q_b = K_bottom - K_top
k_theta = np.linalg.norm(q_b)

print("theta =", theta_deg, "deg")
print("k_theta numeric =", k_theta)
print("2 KD sin(theta/2) =", 2 * KD * np.sin(theta / 2))


# -----------------------------
# The three BM q vectors
# -----------------------------
# BM's three q vectors have equal length k_theta and are separated by 120 degrees.
# In the paper caption, directions are:
# q_b  = (0, -1)
# q_tr = (sqrt(3)/2, 1/2)
# q_tl = (-sqrt(3)/2, 1/2)
#
# Our q_b above points roughly downward if K_top and K_bottom are near +x.
# We generate the other two by rotating q_b by +/- 120 degrees.
q1 = q_b
q2 = R(+2*np.pi/3) @ q1
q3 = R(-2*np.pi/3) @ q1

q_vectors = [q1, q2, q3]
q_names = [r"$q_b$", r"$q_{tl}$", r"$q_{tr}$"]


# -----------------------------
# Plot 1: two rotated Brillouin zones
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 8))

draw_hex(ax, bz_top, "red", r"Layer 1 BZ, rotated $+\theta/2$")
draw_hex(ax, bz_bottom, "black", r"Layer 2 BZ, rotated $-\theta/2$")

# Mark Gamma
ax.scatter([0], [0], s=60, color="gray", zorder=5)
ax.text(0.03, 0.03, r"$\Gamma$", fontsize=14)

# Thin guide lines from Gamma to the selected K points
ax.plot([0, K_top[0]], [0, K_top[1]], color="red", lw=0.8, alpha=0.65, linestyle="--")
ax.plot([0, K_bottom[0]], [0, K_bottom[1]], color="black", lw=0.8, alpha=0.65, linestyle="--")

# Mark selected K points
ax.scatter([K_top[0]], [K_top[1]], s=80, color="red", zorder=5)
ax.text(K_top[0] + 0.03, K_top[1] + 0.03, r"$K_1$", color="red", fontsize=14)

ax.scatter([K_bottom[0]], [K_bottom[1]], s=80, color="black", zorder=5)
ax.text(K_bottom[0] + 0.03, K_bottom[1] - 0.08, r"$K_2$", color="black", fontsize=14)

# Draw k_theta arrow from top-layer K to bottom-layer K
ax.arrow(
    K_top[0], K_top[1],
    q_b[0], q_b[1],
    length_includes_head=True,
    head_width=0.025,
    head_length=0.04,
    color="blue",
    lw=2,
)
mid = 0.5 * (K_top + K_bottom)
ax.text(mid[0] + 0.04, mid[1], r"$k_\theta$", color="blue", fontsize=16)

# Draw arcs from x-axis showing +/- theta/2
arc_r = 0.35
angles_top = np.linspace(0, theta_top, 40)
angles_bottom = np.linspace(0, theta_bottom, 40)

ax.plot(arc_r*np.cos(angles_top), arc_r*np.sin(angles_top), color="red", lw=1.5)
ax.plot(arc_r*np.cos(angles_bottom), arc_r*np.sin(angles_bottom), color="black", lw=1.5)

ax.text(0.38, 0.05, r"$+\theta/2$", color="red", fontsize=12)
ax.text(0.38, -0.08, r"$-\theta/2$", color="black", fontsize=12)

ax.set_aspect("equal")
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_title("Two rotated graphene Brillouin zones")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
plt.show()


# -----------------------------
# Plot 2: local BM q-vector geometry
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 8))

# Put one layer-1 Dirac point at the origin of this local plot
origin = np.array([0.0, 0.0])

ax.scatter([0], [0], s=100, color="red", zorder=5)
ax.text(0.03, 0.03, r"Layer 1 Dirac point", color="red", fontsize=13)

# Draw the three q vectors to the neighboring layer-2 Dirac points
for q, name in zip(q_vectors, q_names):
    ax.arrow(
        0, 0,
        q[0], q[1],
        length_includes_head=True,
        head_width=0.015,
        head_length=0.025,
        lw=2,
        color="blue"
    )
    end = q
    ax.scatter([end[0]], [end[1]], s=80, color="black", zorder=5)
    ax.text(end[0] + 0.02, end[1] + 0.02, name, color="blue", fontsize=15)

# Circle of radius k_theta
circle = plt.Circle((0, 0), k_theta, fill=False, linestyle="--", alpha=0.5)
ax.add_patch(circle)
ax.text(k_theta + 0.02, 0, r"$|q_j|=k_\theta$", fontsize=14)

ax.set_aspect("equal")
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_title("BM three-q geometry: one Dirac point couples to three nearby copies")
ax.grid(True, alpha=0.3)

lim = 1.4 * k_theta
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
plt.show()
