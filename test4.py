import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ------------------ constants you can tweak ------------------
t1 = 355.16
V  = 60.0
# -------------------------------------------------------------

def N_of_x(x, a, n, eps=1e-8):
    ax = a * x
    ax_abs = np.abs(ax)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        log_pow = 2.0 * n * np.log(ax_abs)
        pow_term = np.exp(log_pow)        # (|ax|)^(2n)
        num = pow_term - 1.0
        den = (a**2) * (x**2) - 1.0
        N2 = num / den
    # removable limit at ax -> 1: N^2 -> n
    mask = np.isfinite(ax) & (np.abs(ax - 1.0) < eps)
    N2 = np.where(mask, float(n), N2)
    with np.errstate(invalid='ignore'):
        N = np.sqrt(N2)
    return N

def F_of_k(k, a, n):
    N = N_of_x(k, a, n)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return - t1 * (a * k)**n / (N**2)

def M_of_k(k, a, n, eps=1e-8):
    ak = a * k
    ak2 = ak**2
    ak2n = ak**(2*n)
    num = (n - 1) * ak**(2*n + 2) + ak2 - n * ak2n
    den = (1.0 - ak2) * (1.0 - ak2n)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        frac = num / den
    # guard near ak≈1
    mask = np.isfinite(ak) & (np.abs(ak - 1.0) < eps)
    if np.any(mask):
        delta = 1e-6
        akp = ak + np.sign(ak - 1.0) * delta
        num_p = (n - 1) * akp**(2*n + 2) + akp**2 - n * akp**(2*n)
        den_p = (1.0 - akp**2) * (1.0 - akp**(2*n))
        frac_p = num_p / den_p
        frac = np.where(mask, frac_p, frac)
    return V * ( -(n - 1)/2.0 + frac )

def finite_diff(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    dy = np.empty_like(y)
    dy[0]  = (y[1]  - y[0])  / (x[1]  - x[0])
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    dy[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    return dy

def trace_g(k, a, n):
    F = F_of_k(k, a, n)
    M = M_of_k(k, a, n)
    D2 = F*F + M*M
    Fp = finite_diff(F, k)
    Mp = finite_diff(M, k)
    Q  = M*Fp - F*Mp
    with np.errstate(divide='ignore', invalid='ignore'):
        Tr = (Q**2) / (D2**2) + (n**2 / (k**2)) * (F**2 / D2)
    Tr[~np.isfinite(Tr)] = np.nan
    return Tr, F, Q, np.sqrt(D2)

def berry_conduction(k, F, Q, D, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        Om = - (n / (2.0 * k)) * (F * Q) / (D**3)
    Om[~np.isfinite(Om)] = np.nan
    return Om

def G_of_k(k, a, n, A0):
    x = k + A0
    N = N_of_x(x, a, n)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        G = (t1**2) * (a**(2*n)) * (n**2) * (A0**2) * (k**(2*(n-1))) / (N**4)
    G[~np.isfinite(G)] = np.nan
    return G

def safe_argmax_k(k, y):
    y = np.asarray(y, float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return np.nan, np.nan
    kf, yf = k[finite], y[finite]
    idx = np.nanargmax(yf)
    return kf[idx], yf[idx]

def normalize_by(value, y):
    if not np.isfinite(value) or value == 0.0:
        return np.zeros_like(y, dtype=float)
    return y / value

def finite_max(y):
    finite = np.isfinite(y)
    return np.nanmax(y[finite]) if np.any(finite) else np.nan

def make_k_grid(a, A0, num=2500):
    """Force k to [0.4, 4]; return grid and asymptote locations."""
    k_min, k_max = 0.0, 4.0
    k = np.linspace(k_min, k_max, num)
    inv_a = 1.0 / max(a, 1e-12)
    return k, inv_a, (inv_a - A0), A0

# -------- initial slider values --------
a0, n0, A00 = 0.8, 3, 0.10
# ---------------------------------------

# Initial data
k, k_eq0, k_bd0, k_A00 = make_k_grid(a0, A00)
Tr0, F0, Q0, D0 = trace_g(k, a0, n0)
Om0 = berry_conduction(k, F0, Q0, D0, n0)
Df0 = Tr0 - Om0
G0  = G_of_k(k, a0, n0, A00)

Tr0_max = finite_max(Tr0)
G0_max  = finite_max(G0)

# Normalize (G uses its own max; the other three use Tr max)
Trn0 = normalize_by(Tr0_max, Tr0)
Omn0 = normalize_by(Tr0_max, Om0)
Dfn0 = normalize_by(Tr0_max, Df0)
Gn0  = normalize_by(G0_max,  G0)

fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.25)

(line_Tr,) = ax.plot(k, Trn0, label="Tr g  (norm by max Tr g)")
(line_Om,) = ax.plot(k, Omn0, label="Ω_cond  (norm by max Tr g)")
(line_Df,) = ax.plot(k, Dfn0, label="Tr g − Ω_cond  (norm by max Tr g)")
(line_G, ) = ax.plot(k, Gn0,  label="G  (norm by max G)")

ax.set_xlabel("k")
ax.set_ylabel("normalized value")
ax.set_title("Normalized G, Tr g, Ω_cond, and (Tr g − Ω_cond)  — sliders: a(=vF/t1), n, A0")
ax.legend(loc="best")

# Fixed x-range
ax.set_xlim(0.0, 4.0)

# Asymptote lines (only shown if they lie in [0.4, 4.0])
def in_range(x): 
    return np.isfinite(x) and (0.0 <= x <= 4.0)

v_A0 = ax.axvline(k_A00, linestyle="--", alpha=0.8, color="tab:green")  if in_range(k_A00) else ax.axvline(0, visible=False)
v_bd = ax.axvline(k_bd0, linestyle="--", alpha=0.8, color="tab:orange") if in_range(k_bd0) else ax.axvline(0, visible=False)
v_eq = ax.axvline(k_eq0, linestyle="--", alpha=0.8, color="tab:purple") if in_range(k_eq0) else ax.axvline(0, visible=False)

# Dot markers at maxima (finite)
dot_Tr,  = ax.plot([], [], 'o', ms=6, alpha=0.9, color=line_Tr.get_color())
dot_Om,  = ax.plot([], [], 'o', ms=6, alpha=0.9, color=line_Om.get_color())
dot_Df,  = ax.plot([], [], 'o', ms=6, alpha=0.9, color=line_Df.get_color())
dot_G,   = ax.plot([], [], 'o', ms=6, alpha=0.9, color=line_G.get_color())

# Textbox centered in the plot
textbox = ax.text(
    0.5, 0.5, "", transform=ax.transAxes,
    ha="center", va="center", fontsize=9, family="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85)
)

def update_text_and_dots(k_vals, y_vals, y_norms):
    (k_Tr, k_Om, k_Df, k_G) = k_vals
    (y_Tr, y_Om, y_Df, y_G) = y_vals      # raw (unnormalized) maxima values
    (yn_Tr, yn_Om, yn_Df, yn_G) = y_norms # what’s plotted

    def put(dot, k0, y0):
        if np.isfinite(k0) and np.isfinite(y0):
            dot.set_data([k0], [y0])
            dot.set_visible(True)
        else:
            dot.set_visible(False)

    put(dot_Tr, k_Tr, yn_Tr)
    put(dot_Om, k_Om, yn_Om)
    put(dot_Df, k_Df, yn_Df)
    put(dot_G,  k_G,  yn_G)

    def fmt(x): return "N/A" if (x is None or not np.isfinite(x)) else f"{x:.6g}"
    textbox.set_text(
        "k* (finite argmax on current grid)\n"
        f"  Tr g:          {fmt(k_Tr)}\n"
        f"  Ω_cond:        {fmt(k_Om)}\n"
        f"  Tr g − Ω_cond: {fmt(k_Df)}\n"
        f"  G:             {fmt(k_G)}"
    )

# Initial maxima display
k_Tr0, y_Tr0 = safe_argmax_k(k, Tr0)
k_Om0, y_Om0 = safe_argmax_k(k, Om0)
k_Df0, y_Df0 = safe_argmax_k(k, Df0)
k_G0,  y_G0  = safe_argmax_k(k, G0)
update_text_and_dots(
    (k_Tr0, k_Om0, k_Df0, k_G0),
    (y_Tr0, y_Om0, y_Df0, y_G0),
    (normalize_by(Tr0_max, y_Tr0),
     normalize_by(Tr0_max, y_Om0),
     normalize_by(Tr0_max, y_Df0),
     normalize_by(G0_max,  y_G0))
)

# Sliders
ax_a  = plt.axes([0.10, 0.16, 0.70, 0.03])
ax_n  = plt.axes([0.10, 0.11, 0.70, 0.03])
ax_A0 = plt.axes([0.10, 0.06, 0.70, 0.03])

s_a  = Slider(ax_a,  "a = vF/t1", valmin=0.2, valmax=2.0,  valinit=a0,  valstep=0.01)
s_n  = Slider(ax_n,  "n",         valmin=1,   valmax=200,   valinit=n0,  valstep=1)
s_A0 = Slider(ax_A0, "A0",        valmin=0.0, valmax=1.00, valinit=A00, valstep=0.01)

def update(_):
    a  = float(s_a.val)
    n  = int(s_n.val)
    A0 = float(s_A0.val)

    # Fixed grid in [0.4, 4]
    k_new, k_eq, k_bd, k_A0 = make_k_grid(a, A0)

    # Curves
    Tr, F, Q, D = trace_g(k_new, a, n)
    Om = berry_conduction(k_new, F, Q, D, n)
    Df = Tr - Om
    G  = G_of_k(k_new, a, n, A0)

    # Normalization bases
    Tr_max = finite_max(Tr)
    G_max  = finite_max(G)

    # Normalize
    Trn = normalize_by(Tr_max, Tr)
    Omn = normalize_by(Tr_max, Om)
    Dfn = normalize_by(Tr_max, Df)
    Gn  = normalize_by(G_max,  G)

    # Update lines
    line_Tr.set_xdata(k_new); line_Tr.set_ydata(Trn)
    line_Om.set_xdata(k_new); line_Om.set_ydata(Omn)
    line_Df.set_xdata(k_new); line_Df.set_ydata(Dfn)
    line_G.set_xdata(k_new);  line_G.set_ydata(Gn)

    # Update asymptotes (only show if within [0.4, 4])
    def set_vline(vline, x):
        if np.isfinite(x) and (0.0 <= x <= 4.0):
            vline.set_xdata([x, x]); vline.set_visible(True)
        else:
            vline.set_visible(False)
    set_vline(v_A0, k_A0)
    set_vline(v_bd, k_bd)
    set_vline(v_eq, k_eq)

    # Keep fixed x-range; autoscale y
    # !
    ax.set_xlim(0.0, 4.0)
    ax.relim(); ax.autoscale_view()

    # Maxima (k*)
    k_Tr, y_Tr = safe_argmax_k(k_new, Tr)
    k_Om, y_Om = safe_argmax_k(k_new, Om)
    k_Df, y_Df = safe_argmax_k(k_new, Df)
    k_G,  y_G  = safe_argmax_k(k_new, G)

    update_text_and_dots(
        (k_Tr, k_Om, k_Df, k_G),
        (y_Tr, y_Om, y_Df, y_G),
        (normalize_by(Tr_max, y_Tr),
         normalize_by(Tr_max, y_Om),
         normalize_by(Tr_max, y_Df),
         normalize_by(G_max,  y_G))
    )

    print(f"[a={a:.3g}, n={n}, A0={A0:.3g}] "
          f"k* Tr g={k_Tr:.6g} | k* Ω={k_Om:.6g} | k* (Tr−Ω)={k_Df:.6g} | k* G={k_G:.6g}")

    fig.canvas.draw_idle()

s_a.on_changed(update)
s_n.on_changed(update)
s_A0.on_changed(update)

plt.show()
