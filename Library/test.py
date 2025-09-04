from sympy import symbols, sqrt, cos, sin, simplify, diff

# Declare all symbolic variables
k, theta, n, vF, t1, V = symbols('k theta n vF t1 V', real=True)
a = vF / t1

# Define N(k) and its derivative
N_expr = sqrt((1 - (vF * k / t1)**(2 * n)) / (1 - (vF * k / t1)**2))
dNdk = diff(N_expr, k)

# Define dx, dy, dz
dx = -(vF**n / t1**(n - 1)) * k**n * cos(n * theta) / N_expr**2
dy = -(vF**n / t1**(n - 1)) * k**n * sin(n * theta) / N_expr**2
dz = V * (-((n - 1)/2) + ((n - 1)*(a * k)**(2 * n + 2) + (a * k)**2 - n * (a * k)**(2 * n)) /
         ((1 - (a * k)**2)*(1 - (a * k)**(2 * n))))

# Derivatives w.r.t x
ddx_dx = (2 * vF**n / (t1**(n - 1) * N_expr**3)) * dNdk * cos(theta) * k**n * cos(n * theta) - \
         (vF**n * n * k**(n - 1)) / (t1**(n - 1) * N_expr**2) * cos(n * theta - theta)

ddy_dx = (2 * vF**n / (t1**(n - 1) * N_expr**3)) * dNdk * cos(theta) * k**n * sin(n * theta) - \
         (vF**n * n * k**(n - 1)) / (t1**(n - 1) * N_expr**2) * sin(n * theta - theta)

ddz_dx = V * cos(theta) * (
    (-2 * a**4 * k**4 * n**2 * (a * k)**(2 * n) +
     2 * a**2 * k**2 * (2 * n**2 * (a * k)**(2 * n) + ((a * k)**(2 * n) - 1)**2) -
     2 * n**2 * (a * k)**(2 * n)) /
    (k * (1 - a**2 * k**2)**2 * ((a * k)**(2 * n) - 1)**2))

# Derivatives w.r.t y
ddx_dy = (2 * vF**n / (t1**(n - 1) * N_expr**3)) * dNdk * sin(theta) * k**n * cos(n * theta) + \
         (vF**n * n * k**(n - 1)) / (t1**(n - 1) * N_expr**2) * sin(n * theta - theta)

ddy_dy = (2 * vF**n / (t1**(n - 1) * N_expr**3)) * dNdk * sin(theta) * k**n * sin(n * theta) - \
         (vF**n * n * k**(n - 1)) / (t1**(n - 1) * N_expr**2) * cos(n * theta - theta)

ddz_dy = V * sin(theta) * (
    (-2 * a**4 * k**4 * n**2 * (a * k)**(2 * n) +
     2 * a**2 * k**2 * (2 * n**2 * (a * k)**(2 * n) + ((a * k)**(2 * n) - 1)**2) -
     2 * n**2 * (a * k)**(2 * n)) /
    (k * (1 - a**2 * k**2)**2 * ((a * k)**(2 * n) - 1)**2))

# Magnitude squared and dot products
d2 = dx**2 + dy**2 + dz**2
dot_x = dx * ddx_dx + dy * ddy_dx + dz * ddz_dx
dot_y = dx * ddx_dy + dy * ddy_dy + dz * ddz_dy

# Trace numerator
numerator = d2 * (ddx_dx**2 + ddy_dx**2 + ddz_dx**2 + ddx_dy**2 + ddy_dy**2 + ddz_dy**2) - dot_x**2 - dot_y**2

# Optionally simplify
numerator_simplified = simplify(numerator)
