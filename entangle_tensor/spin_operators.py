"""
Spin and Pauli operator algebra for quantum many-body lattices.
"""

from __future__ import annotations

import numpy as np


def identity(d: int = 2) -> np.ndarray:
    """Returns local identity matrix of dimension d."""
    return np.eye(d, dtype=np.complex128)


def pauli_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns standard 2x2 Pauli matrices (sigma_0/I, sigma_x, sigma_y, sigma_z).
    """
    s0 = np.eye(2, dtype=np.complex128)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return s0, sx, sy, sz


def spin_half_operators() -> dict[str, np.ndarray]:
    """
    Returns standard spin-1/2 operators (S = sigma / 2, hbar = 1):
    - 'I': 2x2 identity
    - 'Sx': sigma_x / 2
    - 'Sy': sigma_y / 2
    - 'Sz': sigma_z / 2
    - 'Sp': Raising operator S+ = Sx + i Sy
    - 'Sm': Lowering operator S- = Sx - i Sy
    """
    s0, sx, sy, sz = pauli_matrices()
    sx = 0.5 * sx
    sy = 0.5 * sy
    sz = 0.5 * sz
    sp = sx + 1.0j * sy  # [[0, 1], [0, 0]]
    sm = sx - 1.0j * sy  # [[0, 0], [1, 0]]
    return {
        "I": s0,
        "Sx": sx,
        "Sy": sy,
        "Sz": sz,
        "Sp": sp,
        "Sm": sm,
    }


def spin_one_operators() -> dict[str, np.ndarray]:
    """
    Returns standard spin-1 operators (d = 3, representation of SU(2)):
    - 'I': 3x3 identity
    - 'Sx', 'Sy', 'Sz': spin-1 angular momentum matrices satisfying [Si, Sj] = i eps_ijk Sk
    - 'Sp', 'Sm': raising and lowering operators S+ and S-
    - 'Sz2': (Sz)^2 for single-ion anisotropy terms
    """
    # Basis states: |+1>, |0>, |-1>
    s0 = np.eye(3, dtype=np.complex128)

    sz = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.complex128)

    # S+ = [[0, sqrt(2), 0], [0, 0, sqrt(2)], [0, 0, 0]]
    sp = np.array(
        [[0.0, np.sqrt(2.0), 0.0], [0.0, 0.0, np.sqrt(2.0)], [0.0, 0.0, 0.0]], dtype=np.complex128
    )

    sm = sp.conj().T  # Lowering operator S-

    sx = 0.5 * (sp + sm)
    sy = -0.5j * (sp - sm)
    sz2 = sz @ sz

    return {
        "I": s0,
        "Sx": sx,
        "Sy": sy,
        "Sz": sz,
        "Sp": sp,
        "Sm": sm,
        "Sz2": sz2,
    }


def check_su2_commutation(ops: dict[str, np.ndarray], atol: float = 1e-12) -> bool:
    """
    Verifies the SU(2) Lie algebra commutation relations:
    [Sx, Sy] = i Sz, [Sy, Sz] = i Sx, [Sz, Sx] = i Sy.
    """
    sx = ops["Sx"]
    sy = ops["Sy"]
    sz = ops["Sz"]
    comm_xy = sx @ sy - sy @ sx
    comm_yz = sy @ sz - sz @ sy
    comm_zx = sz @ sx - sx @ sz
    ok_xy = bool(np.allclose(comm_xy, 1.0j * sz, atol=atol))
    ok_yz = bool(np.allclose(comm_yz, 1.0j * sx, atol=atol))
    ok_zx = bool(np.allclose(comm_zx, 1.0j * sy, atol=atol))
    return ok_xy and ok_yz and ok_zx
