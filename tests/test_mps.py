"""
Tests for Matrix Product State (MPS) initialization, canonicalization, and operations.
"""

import numpy as np

from entangle_tensor.mps import MatrixProductState
from entangle_tensor.types import CanonicalForm


def test_product_state_properties() -> None:
    L = 6
    # Create Neel state |0 1 0 1 0 1>
    neel = MatrixProductState.neel_state(L=L, d=2)
    assert neel.L == L
    assert neel.d == 2
    assert neel.max_bond_dimension == 1
    assert np.isclose(neel.norm(), 1.0)

    # Statevector must have exactly 1.0 at index corresponding to binary 010101
    dense = neel.to_dense_statevector()
    assert dense.shape == (2**L,)
    idx = int("010101", 2)
    assert np.isclose(np.abs(dense[idx]), 1.0)
    assert np.isclose(np.sum(np.abs(dense) ** 2), 1.0)


def test_inner_product() -> None:
    psi1 = MatrixProductState.product_state([0, 0, 0, 0])
    psi2 = MatrixProductState.product_state([0, 0, 0, 1])
    psi3 = MatrixProductState.product_state([0, 0, 0, 0])

    # Orthogonal states
    assert np.isclose(psi1.inner(psi2), 0.0)
    # Identical normalized states
    assert np.isclose(psi1.inner(psi3), 1.0)


def test_left_and_right_canonicalization() -> None:
    L = 8
    mps = MatrixProductState.random(L=L, d=2, chi=6, seed=42)

    # Test Left Canonicalization
    mps_left = mps.copy()
    mps_left.left_canonicalize()
    assert mps_left.canonical_form == CanonicalForm.LEFT

    # Verify sum_s A(s)^dagger A(s) == Identity for each site
    for i in range(L - 1):
        t = mps_left.tensors[i]  # (Dl, d, Dr)
        # Contract over (Dl, d):
        gram = np.tensordot(t.conj(), t, axes=([0, 1], [0, 1]))
        dr = t.shape[2]
        assert np.allclose(gram, np.eye(dr), atol=1e-12)

    # Test Right Canonicalization
    mps_right = mps.copy()
    mps_right.right_canonicalize()
    assert mps_right.canonical_form == CanonicalForm.RIGHT

    # Verify sum_s B(s) B(s)^dagger == Identity for each site
    for i in range(1, L):
        t = mps_right.tensors[i]  # (Dl, d, Dr)
        # Contract over (d, Dr):
        gram = np.tensordot(t, t.conj(), axes=([1, 2], [1, 2]))
        dl = t.shape[0]
        assert np.allclose(gram, np.eye(dl), atol=1e-12)


def test_schmidt_decomposition_normalization() -> None:
    L = 8
    mps = MatrixProductState.random(L=L, d=2, chi=8, seed=123)

    # Singular values across all cuts must sum to 1 in squares
    for cut in range(1, L):
        s = mps.schmidt_decomposition(cut)
        assert len(s) > 0
        assert np.all(s >= 0.0)
        assert np.isclose(np.sum(s**2), 1.0, atol=1e-10)


def test_mps_compression() -> None:
    L = 8
    # Start with high bond dimension
    mps = MatrixProductState.random(L=L, d=2, chi=16, seed=99)
    initial_norm = mps.norm()
    assert np.isclose(initial_norm, 1.0)

    # Compress to chi = 4
    trunc_err = mps.compress(chi_max=4)
    assert mps.max_bond_dimension <= 4
    assert trunc_err >= 0.0
    assert np.isclose(mps.norm(), 1.0, atol=1e-8)
