"""
Tests for Matrix Product Operator (MPO) constructions and expectation values.
"""

import numpy as np

from entangle_tensor.exact_diag import ExactDiagonalization
from entangle_tensor.mpo import MatrixProductOperator
from entangle_tensor.mps import MatrixProductState


def test_tfim_mpo_dense_equivalence() -> None:
    L = 4
    J = 1.0
    h = 0.5
    mpo = MatrixProductOperator.transverse_field_ising(L=L, J=J, h=h)

    dense_mpo = mpo.to_dense_matrix()
    ed_sparse = ExactDiagonalization.ising_hamiltonian(L=L, J=J, h=h)
    dense_ed = ed_sparse.toarray()

    # Check that dense MPO matches exact Hamiltonian matrix
    assert np.allclose(dense_mpo, dense_ed, atol=1e-12)
    # Check Hermiticity
    assert np.allclose(dense_mpo, dense_mpo.conj().T, atol=1e-12)


def test_heisenberg_mpo_dense_equivalence() -> None:
    L = 4
    J = 1.0
    Jz = 0.8
    hz = 0.2
    mpo = MatrixProductOperator.heisenberg_xxz(L=L, J=J, Jz=Jz, hz=hz)

    dense_mpo = mpo.to_dense_matrix()
    ed_sparse = ExactDiagonalization.heisenberg_hamiltonian(L=L, J=J, Jz=Jz, hz=hz)
    dense_ed = ed_sparse.toarray()

    assert np.allclose(dense_mpo, dense_ed, atol=1e-12)
    assert np.allclose(dense_mpo, dense_mpo.conj().T, atol=1e-12)


def test_haldane_mpo_dense_equivalence() -> None:
    L = 3  # Spin-1 dimension is 3^3 = 27
    J = 1.0
    D = 0.25
    mpo = MatrixProductOperator.haldane_spin_one(L=L, J=J, D=D)

    dense_mpo = mpo.to_dense_matrix()
    ed_sparse = ExactDiagonalization.haldane_spin_one_hamiltonian(L=L, J=J, D=D)
    dense_ed = ed_sparse.toarray()

    assert np.allclose(dense_mpo, dense_ed, atol=1e-12)
    assert np.allclose(dense_mpo, dense_mpo.conj().T, atol=1e-12)


def test_mpo_expectation_value_consistency() -> None:
    L = 6
    J = 1.0
    h = 0.7
    mpo = MatrixProductOperator.transverse_field_ising(L=L, J=J, h=h)

    # Test on random MPS
    mps = MatrixProductState.random(L=L, d=2, chi=4, seed=77)

    # Compute via fast MPO tensor contraction
    exp_mpo = mpo.expectation_value(mps)

    # Compute via dense matrix multiplication
    dense_h = mpo.to_dense_matrix()
    dense_psi = mps.to_dense_statevector()
    exp_dense = np.vdot(dense_psi, dense_h @ dense_psi)

    assert np.isclose(exp_mpo, exp_dense, atol=1e-10)
    assert np.isclose(np.imag(exp_mpo), 0.0, atol=1e-10)
