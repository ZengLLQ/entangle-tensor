"""
Tests for Exact Diagonalization sparse Hamiltonian constructions and spectrum.
"""

import numpy as np

from entangle_tensor.exact_diag import ExactDiagonalization


def test_ising_hamiltonian_properties() -> None:
    L = 4
    h_sp = ExactDiagonalization.ising_hamiltonian(L=L, J=1.0, h=0.5)

    # Dimension 2^4 = 16
    assert h_sp.shape == (16, 16)

    # Hermiticity
    dense = h_sp.toarray()
    assert np.allclose(dense, dense.conj().T, atol=1e-12)

    # Ground state eigenvalue is real
    e0, psi0 = ExactDiagonalization.ground_state(h_sp)
    assert isinstance(e0, float)
    assert np.isclose(np.linalg.norm(psi0), 1.0)


def test_heisenberg_hamiltonian_properties() -> None:
    L = 4
    h_sp = ExactDiagonalization.heisenberg_hamiltonian(L=L, J=1.0, Jz=1.0)
    assert h_sp.shape == (16, 16)

    dense = h_sp.toarray()
    assert np.allclose(dense, dense.conj().T, atol=1e-12)

    e0, psi0 = ExactDiagonalization.ground_state(h_sp)
    # For L=4 open boundary Heisenberg chain with J=1:
    # Analytical / numerical e0 is well-known
    assert e0 < 0.0
    assert np.isclose(np.linalg.norm(psi0), 1.0)


def test_haldane_hamiltonian_properties() -> None:
    L = 3
    h_sp = ExactDiagonalization.haldane_spin_one_hamiltonian(L=L, J=1.0, D=0.1)
    # Dimension 3^3 = 27
    assert h_sp.shape == (27, 27)

    dense = h_sp.toarray()
    assert np.allclose(dense, dense.conj().T, atol=1e-12)

    e0, psi0 = ExactDiagonalization.ground_state(h_sp)
    assert e0 < 0.0
