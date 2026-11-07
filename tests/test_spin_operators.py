"""
Tests for spin operators and SU(2) Lie algebra commutation relations.
"""

import numpy as np

from entangle_tensor.spin_operators import (
    check_su2_commutation,
    pauli_matrices,
    spin_half_operators,
    spin_one_operators,
)


def test_pauli_algebra() -> None:
    s0, sx, sy, sz = pauli_matrices()
    # Squares to identity
    assert np.allclose(sx @ sx, s0)
    assert np.allclose(sy @ sy, s0)
    assert np.allclose(sz @ sz, s0)

    # Anticommutation {sigma_i, sigma_j} = 2 delta_ij I
    assert np.allclose(sx @ sy + sy @ sx, np.zeros((2, 2)))
    assert np.allclose(sy @ sz + sz @ sy, np.zeros((2, 2)))
    assert np.allclose(sz @ sx + sx @ sz, np.zeros((2, 2)))

    # Cross product sigma_x sigma_y = i sigma_z
    assert np.allclose(sx @ sy, 1.0j * sz)
    assert np.allclose(sy @ sz, 1.0j * sx)
    assert np.allclose(sz @ sx, 1.0j * sy)


def test_spin_half_commutation() -> None:
    ops = spin_half_operators()
    assert check_su2_commutation(ops)

    # Ladder operators S+ = Sx + i Sy, S- = Sx - i Sy
    # [Sz, S+] = + S+, [Sz, S-] = - S-
    sz = ops["Sz"]
    sp = ops["Sp"]
    sm = ops["Sm"]
    assert np.allclose(sz @ sp - sp @ sz, sp)
    assert np.allclose(sz @ sm - sm @ sz, -sm)

    # [S+, S-] = 2 Sz
    assert np.allclose(sp @ sm - sm @ sp, 2.0 * sz)


def test_spin_one_casimir_and_commutation() -> None:
    ops = spin_one_operators()
    assert check_su2_commutation(ops)

    # Total spin Casimir operator S^2 = Sx^2 + Sy^2 + Sz^2 = s(s+1) * I = 2 * I
    sx, sy, sz = ops["Sx"], ops["Sy"], ops["Sz"]
    s_squared = sx @ sx + sy @ sy + sz @ sz
    expected_casimir = 1.0 * (1.0 + 1.0) * ops["I"]
    assert np.allclose(s_squared, expected_casimir)

    # Raising / lowering relations
    sp, sm = ops["Sp"], ops["Sm"]
    assert np.allclose(sz @ sp - sp @ sz, sp)
    assert np.allclose(sz @ sm - sm @ sz, -sm)
    assert np.allclose(sp @ sm - sm @ sp, 2.0 * sz)
