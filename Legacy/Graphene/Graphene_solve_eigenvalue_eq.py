import sympy as sp

def solve_eigenvalue_problem_symbolic(matrix):
    """
    Solves the eigenvalue problem symbolically for a given matrix.

    Parameters:
    matrix (sympy.Matrix): The input matrix.

    Returns:
    tuple: A tuple containing the eigenvalues and eigenvectors.
    """
    # Define the symbol for the eigenvalue
    lambda_symbol = sp.symbols('lambda')
    
    # Calculate the characteristic polynomial
    char_poly = matrix.charpoly(lambda_symbol)
    
    # Solve for the eigenvalues
    eigenvalues = sp.solve(char_poly, lambda_symbol)
    
    # Calculate the eigenvectors
    eigenvectors = [matrix.eigenvects()[i][2][0] for i in range(len(eigenvalues))]
    
    return eigenvalues, eigenvectors

def main():
    # Define the symbolic variables
    kx, ky, M, v = sp.symbols('kx ky M v', real=True)
    i = sp.I  # imaginary unit

    # Define the graphene Hamiltonian matrix
    H = sp.Matrix([[(M), v * (kx - i * ky)],
                   [v * (kx + i * ky), (-M)]])

    # Solve the eigenvalue problem symbolically
    eigenvalues, eigenvectors = solve_eigenvalue_problem_symbolic(H)

    # Print the results
    print("Graphene Hamiltonian:")
    sp.pprint(H)
    print("\nEigenvalues:")
    for ev in eigenvalues:
        sp.pprint(ev)
    print("\nEigenvectors:")
    for vec in eigenvectors:
        sp.pprint(vec)

if __name__ == "__main__":
    main()
