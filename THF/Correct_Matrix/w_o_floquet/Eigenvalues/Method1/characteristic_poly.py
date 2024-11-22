import sympy as sp

# Define the symbols
A, C, D, M, a = sp.symbols('A C D M a', real=True)
I = sp.I  # Imaginary unit

# Define the matrix with sympy's exponential notation
matrix = sp.Matrix([
    [0, 0, A * sp.exp(I * a), 0, C, D * sp.exp(-I * a)],
    [0, 0, 0, A * sp.exp(-I * a), D * sp.exp(I * a), C],
    [A * sp.exp(-I * a), 0, 0, M, 0, 0],
    [0, A * sp.exp(I * a), M, 0, 0, 0],
    [C, D * sp.exp(-I * a), 0, 0, 0, 0],
    [D * sp.exp(I * a), C, 0, 0, 0, 0]
])

# Calculate the characteristic polynomial
char_poly = matrix.charpoly()

# Simplify the characteristic polynomial for better readability
char_poly_expr = char_poly.as_expr().simplify()

# Display the characteristic polynomial
print("Characteristic Polynomial:")
sp.pprint(char_poly_expr)
