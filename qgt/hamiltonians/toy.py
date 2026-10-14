"""Toy model used to test the Floquet harmonics."""

import numpy as np

from qgt.hamiltonians.base import hamiltonian, sigma_x


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
