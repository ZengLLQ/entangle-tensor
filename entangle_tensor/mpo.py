"""
Matrix Product Operator (MPO) representation and quantum spin Hamiltonian generators.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from entangle_tensor.mps import MatrixProductState
from entangle_tensor.spin_operators import (
    pauli_matrices,
    spin_half_operators,
    spin_one_operators,
)


class MatrixProductOperator:
    """
    1D Open Boundary Matrix Product Operator (MPO).

    Each site tensor W^[i] has 4 legs with shape (w_{i-1}, w_i, d_out, d_in):
      - w_{i-1}: Left virtual operator bond dimension (w_0 = 1).
      - w_i: Right virtual operator bond dimension (w_L = 1).
      - d_out: Output physical index (bra / ket physical dimension).
      - d_in: Input physical index.
    """

    def __init__(self, tensors: Sequence[np.ndarray]) -> None:
        """
        Initializes an MPO from a list of 4-leg tensors.
        """
        if len(tensors) == 0:
            raise ValueError("MPO must contain at least one site tensor.")

        self.tensors: List[np.ndarray] = [np.array(t, dtype=np.complex128) for t in tensors]
        self.L: int = len(self.tensors)
        self.d: int = self.tensors[0].shape[2]

        # Validate boundary operator bonds
        if self.tensors[0].shape[0] != 1:
            raise ValueError(
                f"First MPO tensor left bond must be 1, got {self.tensors[0].shape[0]}"
            )
        if self.tensors[-1].shape[1] != 1:
            raise ValueError(
                f"Last MPO tensor right bond must be 1, got {self.tensors[-1].shape[1]}"
            )

        # Validate virtual bond matching
        for i in range(self.L - 1):
            if self.tensors[i].shape[1] != self.tensors[i + 1].shape[0]:
                raise ValueError(
                    f"MPO bond mismatch between site {i} (w_R={self.tensors[i].shape[1]}) "
                    f"and site {i + 1} (w_L={self.tensors[i + 1].shape[0]})"
                )

    @classmethod
    def transverse_field_ising(
        cls, L: int, J: float = 1.0, h: float = 1.0
    ) -> MatrixProductOperator:
        """
        Builds the 1D Transverse-Field Ising Model (TFIM) Hamiltonian:
          H = -J sum_{i=0}^{L-2} sigma^z_i sigma^z_{i+1} - h sum_{i=0}^{L-1} sigma^x_i

        Local virtual bond dimension w = 3.
        """
        _, sx, _, sz = pauli_matrices()
        eye = np.eye(2, dtype=np.complex128)

        # 3x3 block matrix of 2x2 operators
        w_bulk = np.zeros((3, 3, 2, 2), dtype=np.complex128)
        # Row 0: [I, sigma^z, -h * sigma^x]
        w_bulk[0, 0] = eye
        w_bulk[0, 1] = sz
        w_bulk[0, 2] = -h * sx
        # Row 1: [0, 0, -J * sigma^z]
        w_bulk[1, 2] = -J * sz
        # Row 2: [0, 0, I]
        w_bulk[2, 2] = eye

        tensors = []
        for i in range(L):
            if i == 0:
                # Left boundary: select row 0 -> shape (1, 3, 2, 2)
                t = w_bulk[0:1, :, :, :]
            elif i == L - 1:
                # Right boundary: select column 2 -> shape (3, 1, 2, 2)
                t = w_bulk[:, 2:3, :, :]
            else:
                t = w_bulk.copy()
            tensors.append(t)

        return cls(tensors)

    @classmethod
    def heisenberg_xxz(
        cls, L: int, J: float = 1.0, Jz: float = 1.0, hz: float = 0.0
    ) -> MatrixProductOperator:
        """
        Builds the 1D Spin-1/2 Heisenberg XXZ Hamiltonian:
          H = sum_{i=0}^{L-2} [ J/2 (S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) + Jz S^z_i S^z_{i+1} ] - hz sum_{i=0}^{L-1} S^z_i

        When J = Jz, corresponds to the isotropic XXX antiferromagnet (J > 0).
        Local virtual bond dimension w = 5.
        """
        ops = spin_half_operators()
        eye = ops["I"]
        ops["Sx"]
        ops["Sy"]
        sz = ops["Sz"]
        sp = ops["Sp"]
        sm = ops["Sm"]

        w_bulk = np.zeros((5, 5, 2, 2), dtype=np.complex128)
        # Row 0: [I, S+, S-, Sz, -hz * Sz]
        w_bulk[0, 0] = eye
        w_bulk[0, 1] = sp
        w_bulk[0, 2] = sm
        w_bulk[0, 3] = sz
        w_bulk[0, 4] = -hz * sz

        # Coupling rows into column 4:
        # Row 1: S+ couples to (J/2) S-
        w_bulk[1, 4] = 0.5 * J * sm
        # Row 2: S- couples to (J/2) S+
        w_bulk[2, 4] = 0.5 * J * sp
        # Row 3: Sz couples to Jz Sz
        w_bulk[3, 4] = Jz * sz
        # Row 4: Identity
        w_bulk[4, 4] = eye

        tensors = []
        for i in range(L):
            if i == 0:
                t = w_bulk[0:1, :, :, :]
            elif i == L - 1:
                t = w_bulk[:, 4:5, :, :]
            else:
                t = w_bulk.copy()
            tensors.append(t)

        return cls(tensors)

    @classmethod
    def haldane_spin_one(cls, L: int, J: float = 1.0, D: float = 0.0) -> MatrixProductOperator:
        """
        Builds the Spin-1 Heisenberg chain Hamiltonian with single-ion anisotropy D:
          H = J sum_{i=0}^{L-2} (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + S^z_i S^z_{i+1}) + D sum_{i=0}^{L-1} (S^z_i)^2

        Physical dimension d = 3, local operator bond dimension w = 5.
        """
        ops = spin_one_operators()
        eye = ops["I"]
        sz = ops["Sz"]
        sp = ops["Sp"]
        sm = ops["Sm"]
        sz2 = ops["Sz2"]

        w_bulk = np.zeros((5, 5, 3, 3), dtype=np.complex128)
        # Row 0: [I, S+, S-, Sz, D * (Sz)^2]
        w_bulk[0, 0] = eye
        w_bulk[0, 1] = sp
        w_bulk[0, 2] = sm
        w_bulk[0, 3] = sz
        w_bulk[0, 4] = D * sz2

        # Coupling terms into column 4
        w_bulk[1, 4] = 0.5 * J * sm
        w_bulk[2, 4] = 0.5 * J * sp
        w_bulk[3, 4] = J * sz
        w_bulk[4, 4] = eye

        tensors = []
        for i in range(L):
            if i == 0:
                t = w_bulk[0:1, :, :, :]
            elif i == L - 1:
                t = w_bulk[:, 4:5, :, :]
            else:
                t = w_bulk.copy()
            tensors.append(t)

        return cls(tensors)

    def expectation_value(self, mps: MatrixProductState) -> complex:
        """
        Contracts the sandwich <mps | MPO | mps> to compute the expectation value <O>.
        Computational complexity: O(L * D^3 * w * d + L * D^2 * w^2 * d^2).
        """
        if self.L != mps.L or self.d != mps.d:
            raise ValueError("MPO and MPS dimensions must match for expectation value.")

        # Environment tensor E with legs (D_bra, w, D_ket) initialized to (1, 1, 1)
        e: np.ndarray = np.ones((1, 1, 1), dtype=np.complex128)

        for i in range(self.L):
            a_bra = mps.tensors[i].conj()  # (D_bra_l, d, D_bra_r)
            w = self.tensors[i]  # (w_l, w_r, d_out, d_in)
            a_ket = mps.tensors[i]  # (D_ket_l, d, D_ket_r)

            # 1. Contract E with a_bra on D_bra_l:
            # E(D_bra_l, w_l, D_ket_l) * a_bra(D_bra_l, d_out, D_bra_r) -> tmp1(w_l, D_ket_l, d_out, D_bra_r)
            tmp1 = np.tensordot(e, a_bra, axes=(0, 0))

            # 2. Contract tmp1 with W on w_l and d_out:
            # tmp1(w_l, D_ket_l, d_out, D_bra_r) * W(w_l, w_r, d_out, d_in) -> tmp2(D_ket_l, D_bra_r, w_r, d_in)
            tmp2 = np.tensordot(tmp1, w, axes=([0, 2], [0, 2]))

            # 3. Contract tmp2 with a_ket on D_ket_l and d_in:
            # tmp2(D_ket_l, D_bra_r, w_r, d_in) * a_ket(D_ket_l, d_in, D_ket_r) -> e_next(D_bra_r, w_r, D_ket_r)
            e = np.tensordot(tmp2, a_ket, axes=([0, 3], [0, 1]))

        return complex(e[0, 0, 0])

    def to_dense_matrix(self) -> np.ndarray:
        """
        Constructs the full dense 2^L x 2^L (or d^L x d^L) matrix representation.
        Only suitable for small system sizes (L <= 10).
        """
        # Start with first tensor: (1, w_1, d_0, d'_0)
        # Transpose to (1, w_1, d_0, d'_0)
        curr = self.tensors[0]
        # Reshape to (w_1, d_0, d'_0)
        curr = curr.reshape(curr.shape[1], curr.shape[2], curr.shape[3])

        for i in range(1, self.L):
            # next: (w_i, w_{i+1}, d_i, d'_i)
            nxt = self.tensors[i]
            # Contract over w_i:
            # curr: (w_i, d_0..d_{i-1}, d'_0..d'_{i-1})
            # nxt: (w_i, w_{i+1}, d_i, d'_i)
            # einsum: w d_out d_in, w w_next d_i_out d_i_in -> w_next (d_out d_i_out) (d_in d_i_in)
            shape_curr_out = curr.shape[1]
            shape_curr_in = curr.shape[2]

            curr.shape[0]
            w_next = nxt.shape[1]
            d_out = nxt.shape[2]
            d_in = nxt.shape[3]

            # tmp(d_out_prev, d_in_prev, w_next, d_out_next, d_in_next)
            tmp = np.tensordot(curr, nxt, axes=(0, 0))
            # Permute legs to: (w_next, d_out_prev * d_out_next, d_in_prev * d_in_next)
            tmp = tmp.transpose(2, 0, 3, 1, 4)
            curr = tmp.reshape(w_next, shape_curr_out * d_out, shape_curr_in * d_in)

        # curr has shape (1, d^L, d^L)
        res: np.ndarray = np.asarray(curr[0])
        return res
