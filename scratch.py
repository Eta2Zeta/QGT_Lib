import sys
import pickle

import Library.Hamiltonian.Hamiltonian
from Library.Hamiltonian.ChiralHamiltonian import ChiralHamiltonian
from Library.Hamiltonian.SquareLatticeHamiltonian import SquareLatticeHamiltonian
from Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected import ChiralHamiltonianChiralBasisProjected

sys.modules["Library.Hamiltonian_v2"] = Library.Hamiltonian.Hamiltonian
sys.modules["Library.Hamiltonian.Chiral_Hamiltonian_Projected"] = Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected

Library.Hamiltonian.Hamiltonian.ChiralHamiltonian      = ChiralHamiltonian
Library.Hamiltonian.Hamiltonian.SquareLatticeHamiltonian = SquareLatticeHamiltonian
Library.Hamiltonian.Hamiltonian.RhombohedralGrapheneHamiltonian = ChiralHamiltonianChiralBasisProjected
Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected.Chiral_Hamiltonian_Projected = ChiralHamiltonianChiralBasisProjected
Library.Hamiltonian.ChiralHamiltonian_ChiralBasis_Projected.RhombohedralGrapheneHamiltonian = ChiralHamiltonianChiralBasisProjected

with open("results/2D_QGT_omega_sweep/RhombohedralGrapheneHamiltonian/A0_0.10-V_60-analytic_magnus_False-magnus_order_1-n_4-omega_1000-polarization_right-t1_355.16-vF_542.10_kx-0.80_0.80_ky-0.80_0.80_mesh100_omega3.00e_01_5.00e_03_spacing_log_points32_band0_data_set1/meta_info.pkl", "rb") as f:
    meta_info = pickle.load(f)
print("Loaded successfully! Class:", type(meta_info.get("Hamiltonian_Obj", meta_info.get("Hamiltonian_Template"))))
