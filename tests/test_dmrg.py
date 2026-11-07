"""
Tests for Two-Site Variational DMRG ground state solver.
"""

import numpy as np
import pytest

from entangle_tensor.dmrg import DMRGSolver
from entangle_tensor.exact_diag import ExactDiagonalization
from entangle_tensor.mpo import MatrixProductOperator
from entangle_tensor.mps import MatrixProductState
from entangle_tensor.types import DMRGConfig


def test_dmrg_tfim_vs_exact_diagonalization() -> None:
    L = 6
    J = 1.0
    h = 1.0  # Quantum critical point

    # 1. Exact Diagonalization ground truth
    h_sparse = ExactDiagonalization.ising_hamiltonian(L=L, J=J, h=h)
    e_exact, _ = ExactDiagonalization.ground_state(h_sparse)

    # 2. DMRG solver
    mpo = MatrixProductOperator.transverse_field_ising(L=L, J=J, h=h)
    mps = MatrixProductState.random(L=L, d=2, chi=2, seed=42)

    config = DMRGConfig(max_sweeps=8, chi_max=16, energy_tol=1e-10)
    solver = DMRGSolver(config)
    res = solver.run(mps, mpo)

    assert res.converged
    assert abs(res.ground_state_energy - e_exact) < 1e-7
    # Expectation value via MPO matches reported ground state energy
    assert np.isclose(mpo.expectation_value(mps), res.ground_state_energy, atol=1e-8)


def test_dmrg_heisenberg_vs_exact_diagonalization() -> None:
    L = 6
    J = 1.0
    Jz = 1.0

    # Exact Diagonalization
    h_sparse = ExactDiagonalization.heisenberg_hamiltonian(L=L, J=J, Jz=Jz)
    e_exact, _ = ExactDiagonalization.ground_state(h_sparse)

    # DMRG
    mpo = MatrixProductOperator.heisenberg_xxz(L=L, J=J, Jz=Jz)
    mps = MatrixProductState.neel_state(L=L, d=2)

    config = DMRGConfig(max_sweeps=10, chi_max=20, energy_tol=1e-10)
    solver = DMRGSolver(config)
    res = solver.run(mps, mpo)

    assert res.converged
    assert abs(res.ground_state_energy - e_exact) < 1e-7
    assert res.energy_per_site == pytest.approx(e_exact / L, abs=1e-7)


def test_dmrg_energy_monotonicity() -> None:
    L = 8
    mpo = MatrixProductOperator.transverse_field_ising(L=L, J=1.0, h=0.8)
    mps = MatrixProductState.random(L=L, d=2, chi=4, seed=101)

    config = DMRGConfig(max_sweeps=6, chi_max=16, energy_tol=1e-12)
    solver = DMRGSolver(config)
    res = solver.run(mps, mpo)

    # Energies across sweeps must be monotonically non-increasing (within numerical float precision)
    energies = res.sweep_energies
    for i in range(len(energies) - 1):
        assert energies[i + 1] <= energies[i] + 1e-10
