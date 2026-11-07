"""
Tests for Time-Evolving Block Decimation (TEBD) simulation engine.
"""

import numpy as np

from entangle_tensor.exact_diag import ExactDiagonalization
from entangle_tensor.mpo import MatrixProductOperator
from entangle_tensor.mps import MatrixProductState
from entangle_tensor.tebd import TEBDSimulator
from entangle_tensor.types import EvolutionType, TEBDConfig


def test_tebd_real_time_norm_and_energy_conservation() -> None:
    L = 6
    J = 1.0
    h = 0.5
    bonds = TEBDSimulator.ising_bond_hamiltonians(L=L, J=J, h=h)
    mpo = MatrixProductOperator.transverse_field_ising(L=L, J=J, h=h)

    # Start in a product state
    mps = MatrixProductState.ferromagnetic_state(L=L, d=2, state_idx=0)
    e_initial = float(np.real(mpo.expectation_value(mps)))

    config = TEBDConfig(dt=0.01, total_steps=20, chi_max=16)
    tebd = TEBDSimulator(mps, config)

    tebd.evolve(bonds, mode=EvolutionType.REAL)

    # Unitary evolution must strictly preserve norm
    norm_after = mps.norm()
    assert np.isclose(norm_after, 1.0, atol=1e-8)

    # Energy must be conserved under time-independent Hamiltonian (within second-order Trotter error O(dt^2))
    e_final = float(np.real(mpo.expectation_value(mps)))
    assert np.isclose(e_final, e_initial, atol=1e-2)


def test_tebd_imaginary_time_cooling() -> None:
    L = 4
    J = 1.0
    h = 1.0
    bonds = TEBDSimulator.ising_bond_hamiltonians(L=L, J=J, h=h)
    mpo = MatrixProductOperator.transverse_field_ising(L=L, J=J, h=h)

    # Exact ground state energy from ED
    h_sparse = ExactDiagonalization.ising_hamiltonian(L=L, J=J, h=h)
    e_exact, _ = ExactDiagonalization.ground_state(h_sparse)

    # Random initial trial state
    mps = MatrixProductState.random(L=L, d=2, chi=4, seed=42)
    e_initial = float(np.real(mpo.expectation_value(mps)))

    config = TEBDConfig(dt=0.05, total_steps=80, chi_max=16)
    tebd = TEBDSimulator(mps, config)

    tebd.evolve(bonds, mode=EvolutionType.IMAGINARY)
    e_cooled = float(np.real(mpo.expectation_value(mps)))

    # Energy must decrease and match exact ground state to high accuracy
    assert e_cooled < e_initial
    assert abs(e_cooled - e_exact) < 0.01
