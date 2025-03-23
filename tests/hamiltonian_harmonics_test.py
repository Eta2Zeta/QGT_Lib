import sys
import os
import numpy as np
import pickle
from tqdm import tqdm  # Import tqdm for progress bar


# from Library import * 
from Library.plotting_lib import *
from Library.Hamiltonian_v1 import *
from Library.Hamiltonian_v2 import * 
from Library.eigenvalue_calc_lib import *
from Library.QGT_lib import *
from Library.topology import *


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec



# Define the test function
def test_test_lattice_harmonics():
    # Define parameters
    A0 = 10.0  # Driving amplitude
    omega = 5 * np.pi  # Driving frequency

    # Initialize the Hamiltonian
    Test_Hamiltonian = TestHamiltonian( a=1, b=1, c=1, omega=omega, A0=A0)

    # Create a grid of kx and ky values
    k_max = 2*np.pi
    mesh_spacing = 50
    kx = np.linspace(-k_max, k_max, mesh_spacing)
    ky = np.linspace(-k_max, k_max, mesh_spacing)
    kx, ky = np.meshgrid(kx, ky)

    # Calculate the analytical H12 component
    analytical_H12 = np.zeros_like(kx, dtype=complex)
    testing_H1 = False
    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            if testing_H1:
                H1 = Test_Hamiltonian.Hp1(kx[i, j], ky[i, j])
                analytical_H12[i, j] = H1[0,1]
            else: 
                H2 = Test_Hamiltonian.Hp2(kx[i, j], ky[i, j])
                analytical_H12[i, j] = H2[0,1]

    # Calculate the numerical H12 component
    numerical_H12 = np.zeros_like(kx, dtype=complex)
    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            if testing_H1:
                H1 = Test_Hamiltonian.numerical_fourier_component(1, kx[i, j], ky[i, j])
                numerical_H12[i, j] = H1[0, 1]  # Extract the H12 component
            else: 
                H2 = Test_Hamiltonian.numerical_fourier_component(2, kx[i, j], ky[i, j])
                numerical_H12[i, j] = H2[0, 1]  # Extract the H12 component

    # Compute absolute values
    abs_analytical = np.abs(analytical_H12)
    abs_numerical = np.abs(numerical_H12)
    abs_difference = np.abs(abs_analytical - abs_numerical)  # Absolute difference for better comparison

    # Determine global min and max for z-axis limits
    z_min = min(np.min(abs_analytical), np.min(abs_numerical), np.min(abs_difference))
    z_max = max(np.max(abs_analytical), np.max(abs_numerical), np.max(abs_difference))

    # Plot the results in 3D
    fig = plt.figure(figsize=(18, 6))

    # Analytical H12 magnitude
    ax0 = fig.add_subplot(131, projection='3d')
    ax0.plot_surface(kx, ky, abs_analytical, cmap='viridis')
    ax0.set_title('Analytical |H12|')
    ax0.set_xlabel('$k_x$')
    ax0.set_ylabel('$k_y$')
    ax0.set_zlabel('|H12|')
    ax0.set_zlim(z_min, z_max)  # Set the same limits for all plots

    # Numerical H12 magnitude
    ax1 = fig.add_subplot(132, projection='3d')
    ax1.plot_surface(kx, ky, abs_numerical, cmap='viridis')
    ax1.set_title('Numerical |H12|')
    ax1.set_xlabel('$k_x$')
    ax1.set_ylabel('$k_y$')
    ax1.set_zlabel('|H12|')
    ax1.set_zlim(z_min, z_max)  # Set the same limits for all plots

    # Difference in H12 magnitude
    ax2 = fig.add_subplot(133, projection='3d')
    ax2.plot_surface(kx, ky, abs_difference, cmap='viridis')
    ax2.set_title('Difference |H12|')
    ax2.set_xlabel('$k_x$')
    ax2.set_ylabel('$k_y$')
    ax2.set_zlabel('Difference')
    ax2.set_zlim(z_min, z_max)  # Set the same limits for all plots

    plt.tight_layout()
    plt.show()




# Define the test function
def test_square_lattice_harmonics():
    # Define parameters
    A0 = 2  # Driving amplitude
    omega = 2 * np.pi  # Driving frequency
    t1 = 5.0   
    t2 = 10
    t5 = 1000

    # Initialize the Hamiltonian
    square_lattice_hamiltonian = SquareLatticeHamiltonian(t1=t1, t2=t2, t5=t5, omega=omega, A0=A0)

    # Create a grid of kx and ky values
    k_max = 1*np.pi
    mesh_spacing = 50
    kx = np.linspace(-k_max, k_max, mesh_spacing)
    ky = np.linspace(-k_max, k_max, mesh_spacing)
    kx, ky = np.meshgrid(kx, ky)

    # Calculate the analytical H12 component
    analytical_H12 = np.zeros_like(kx, dtype=complex)
    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            analytical_H12[i, j] = square_lattice_hamiltonian.Hp1(kx[i, j], ky[i, j])

    # Calculate the numerical H12 component
    numerical_H12 = np.zeros_like(kx, dtype=complex)
    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            H1 = square_lattice_hamiltonian.numerical_fourier_component(1, kx[i, j], ky[i, j])
            numerical_H12[i, j] = H1[1, 0]  # Extract the H12 component

    # Plot the results in 3D
    fig = plt.figure(figsize=(18, 6))

    # Analytical H12 magnitude
    ax0 = fig.add_subplot(131, projection='3d')
    ax0.plot_surface(kx, ky, np.abs(analytical_H12), cmap='viridis')
    ax0.set_title('Analytical |H12|')
    ax0.set_xlabel('$k_x$')
    ax0.set_ylabel('$k_y$')
    ax0.set_zlabel('|H12|')

    # Numerical H12 magnitude
    ax1 = fig.add_subplot(132, projection='3d')
    ax1.plot_surface(kx, ky, np.abs(numerical_H12), cmap='viridis')
    ax1.set_title('Numerical |H12|')
    ax1.set_xlabel('$k_x$')
    ax1.set_ylabel('$k_y$')
    ax1.set_zlabel('|H12|')

    # Difference in H12 magnitude
    ax2 = fig.add_subplot(133, projection='3d')
    ax2.plot_surface(kx, ky, np.abs(np.abs(analytical_H12) / np.abs(numerical_H12)), cmap='viridis')
    ax2.set_title('Difference |H12|')
    ax2.set_xlabel('$k_x$')
    ax2.set_ylabel('$k_y$')
    ax2.set_zlabel('Difference')

    plt.tight_layout()
    plt.show()


# Define the test function
def test_graphene_lattice_harmonics():
    # Define parameters
    A0 = 4.0  # Driving amplitude
    omega = 67 * np.pi  # Driving frequency


    # Initialize the Hamiltonian
    Graphene_Hamiltonian = GrapheneHamiltonian(omega=omega, A0=A0)

    # Create a grid of kx and ky values
    k_max = np.pi
    mesh_spacing = 50
    kx = np.linspace(-k_max, k_max, mesh_spacing)
    ky = np.linspace(-k_max, k_max, mesh_spacing)
    kx, ky = np.meshgrid(kx, ky)

    # Calculate the analytical H12 component
    analytical_H12 = np.zeros_like(kx, dtype=complex)
    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            analytical_H12[i, j] = Graphene_Hamiltonian.Hp1(kx[i, j], ky[i, j])

    # Calculate the numerical H12 component
    numerical_H12 = np.zeros_like(kx, dtype=complex)
    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            H1 = Graphene_Hamiltonian.numerical_fourier_component(1, kx[i, j], ky[i, j])
            numerical_H12[i, j] = H1[1, 0]  # Extract the H12 component


    # Plot the results in 3D
    fig = plt.figure(figsize=(18, 6))

    # Analytical H12 magnitude
    ax0 = fig.add_subplot(131, projection='3d')
    ax0.plot_surface(kx, ky, np.real(analytical_H12), cmap='viridis')
    ax0.set_title('Analytical |H12|')
    ax0.set_xlabel('$k_x$')
    ax0.set_ylabel('$k_y$')
    ax0.set_zlabel('|H12|')

    # Numerical H12 magnitude
    ax1 = fig.add_subplot(132, projection='3d')
    ax1.plot_surface(kx, ky, np.real(numerical_H12), cmap='viridis')
    ax1.set_title('Numerical |H12|')
    ax1.set_xlabel('$k_x$')
    ax1.set_ylabel('$k_y$')
    ax1.set_zlabel('|H12|')

    # Difference in H12 magnitude
    ax2 = fig.add_subplot(133, projection='3d')
    ax2.plot_surface(kx, ky, np.real(np.abs(analytical_H12) / np.abs(numerical_H12)), cmap='viridis')
    ax2.set_title('Ratio |H12|')
    ax2.set_xlabel('$k_x$')
    ax2.set_ylabel('$k_y$')
    ax2.set_zlabel('Difference')

    plt.tight_layout()
    plt.show()


# Define the test function
def test_twoorbitalunspinful_lattice_harmonics():
    # Initialize the Hamiltonian
    TwoOrbitalUnspinful_Hamiltonian = TwoOrbitalUnspinfulHamiltonian(omega=1e2*np.pi/2, A0=0.1, zeta=1)

    # Create a grid of kx and ky values
    k_max = np.pi
    mesh_spacing = 50
    stride = 1  # Define stride setting
    kx = np.linspace(-k_max, k_max, mesh_spacing)
    ky = np.linspace(-k_max, k_max, mesh_spacing)
    kx, ky = np.meshgrid(kx, ky)

    # Calculate the analytical H12 component
    analytical_H12 = np.zeros_like(kx, dtype=complex)
    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            analytical_H12[i, j] = TwoOrbitalUnspinful_Hamiltonian.Hp1(kx[i, j], ky[i, j])[0, 1] 

    # Calculate the numerical H12 component
    numerical_H12 = np.zeros_like(kx, dtype=complex)
    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            H1 = TwoOrbitalUnspinful_Hamiltonian.numerical_fourier_component(1, kx[i, j], ky[i, j])
            numerical_H12[i, j] = H1[0, 1]  # Extract the H12 component

    # Plot the results in 3D
    fig = plt.figure(figsize=(18, 6))

    # Analytical H12 magnitude
    ax0 = fig.add_subplot(121, projection='3d')
    ax0.plot_surface(kx, ky, np.imag(analytical_H12), cmap='viridis', rstride=stride, cstride=stride)
    ax0.set_title('Analytical |H12|')
    ax0.set_xlabel('$k_x$')
    ax0.set_ylabel('$k_y$')
    ax0.set_zlabel('|H12|')

    # Numerical H12 magnitude
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.plot_surface(kx, ky, np.imag(numerical_H12), cmap='viridis', rstride=stride, cstride=stride)
    ax1.set_title('Numerical |H12|')
    ax1.set_xlabel('$k_x$')
    ax1.set_ylabel('$k_y$')
    ax1.set_zlabel('|H12|')

    # # Difference in H12 magnitude
    # ax2 = fig.add_subplot(133, projection='3d')
    # ax2.plot_surface(kx, ky, np.abs(np.abs(analytical_H12) / np.abs(numerical_H12)), cmap='viridis', rstride=stride, cstride=stride)
    # ax2.set_title('Ratio |H12|')
    # ax2.set_xlabel('$k_x$')
    # ax2.set_ylabel('$k_y$')
    # ax2.set_zlabel('Difference')

    plt.tight_layout()
    plt.show()


# Define the test function
def test_rashba_lattice_harmonics():
    # Define parameters
    A0 = 10.0  # Driving amplitude
    omega = 5 * np.pi  # Driving frequency

    # Initialize the Hamiltonian
    Rashba_Hamiltonian = RashbaHamiltonian(omega=omega, A0=A0)

    # Create a grid of kx and ky values
    k_max = 2*np.pi
    mesh_spacing = 50
    kx = np.linspace(-k_max, k_max, mesh_spacing)
    ky = np.linspace(-k_max, k_max, mesh_spacing)
    kx, ky = np.meshgrid(kx, ky)

    # Calculate the analytical H12 component
    analytical_H12 = np.zeros_like(kx, dtype=complex)
    testing_H1 = True
    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            if testing_H1:
                H1 = Rashba_Hamiltonian.Hp1(kx[i, j], ky[i, j])
                analytical_H12[i, j] = H1[1,1]
            else: 
                H2 = Rashba_Hamiltonian.Hp2(kx[i, j], ky[i, j])
                analytical_H12[i, j] = H2[0,0]

    # Calculate the numerical H12 component
    numerical_H12 = np.zeros_like(kx, dtype=complex)
    for i in range(mesh_spacing):
        for j in range(mesh_spacing):
            if testing_H1:
                H1 = Rashba_Hamiltonian.numerical_fourier_component(1, kx[i, j], ky[i, j])
                numerical_H12[i, j] = H1[1, 1]  # Extract the H12 component
            else: 
                H2 = Rashba_Hamiltonian.numerical_fourier_component(2, kx[i, j], ky[i, j])
                numerical_H12[i, j] = H2[0, 0]  # Extract the H12 component

    # Compute absolute values
    abs_analytical = np.imag(analytical_H12)
    abs_numerical = np.imag(numerical_H12)
    abs_difference = np.real(abs_analytical - abs_numerical)  # Absolute difference for better comparison

    # Determine global min and max for z-axis limits
    z_min = min(np.min(abs_analytical), np.min(abs_numerical), np.min(abs_difference))
    z_max = max(np.max(abs_analytical), np.max(abs_numerical), np.max(abs_difference))

    # Plot the results in 3D
    fig = plt.figure(figsize=(18, 6))

    # Analytical H12 magnitude
    ax0 = fig.add_subplot(131, projection='3d')
    ax0.plot_surface(kx, ky, abs_analytical, cmap='viridis')
    ax0.set_title('Analytical |H12|')
    ax0.set_xlabel('$k_x$')
    ax0.set_ylabel('$k_y$')
    ax0.set_zlabel('|H12|')
    ax0.set_zlim(z_min, z_max)  # Set the same limits for all plots

    # Numerical H12 magnitude
    ax1 = fig.add_subplot(132, projection='3d')
    ax1.plot_surface(kx, ky, abs_numerical, cmap='viridis')
    ax1.set_title('Numerical |H12|')
    ax1.set_xlabel('$k_x$')
    ax1.set_ylabel('$k_y$')
    ax1.set_zlabel('|H12|')
    ax1.set_zlim(z_min, z_max)  # Set the same limits for all plots

    # Difference in H12 magnitude
    ax2 = fig.add_subplot(133, projection='3d')
    ax2.plot_surface(kx, ky, abs_difference, cmap='viridis')
    ax2.set_title('Difference |H12|')
    ax2.set_xlabel('$k_x$')
    ax2.set_ylabel('$k_y$')
    ax2.set_zlabel('Difference')
    ax2.set_zlim(z_min, z_max)  # Set the same limits for all plots

    plt.tight_layout()
    plt.show()



# Run the test function
# test_test_lattice_harmonics()

# test_graphene_lattice_harmonics()

# test_rashba_lattice_harmonics()

test_twoorbitalunspinful_lattice_harmonics()

# test_square_lattice_harmonics()