"""
Environment block tensor caching and local effective Hamiltonian contractions for DMRG.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.sparse.linalg import LinearOperator

from entangle_tensor.mpo import MatrixProductOperator
from entangle_tensor.mps import MatrixProductState


class EnvironmentCache:
    """
    Maintains the left and right contraction environment blocks:
      - L_env[i]: 3-leg tensor of shape (D_bra, w, D_ket) for the subsystem sites 0..i.
      - R_env[i]: 3-leg tensor of shape (D_bra, w, D_ket) for the subsystem sites i..L-1.
    """

    def __init__(self, mps: MatrixProductState, mpo: MatrixProductOperator) -> None:
        if mps.L != mpo.L:
            raise ValueError("MPS and MPO lengths must match.")

        self.L: int = mps.L
        self.mps: MatrixProductState = mps
        self.mpo: MatrixProductOperator = mpo

        # Left environment tensors L_env[i] for sites 0..L-1
        # L_env[-1] represents the empty left boundary (1, 1, 1)
        self.left_envs: Dict[int, np.ndarray] = {-1: np.ones((1, 1, 1), dtype=np.complex128)}

        # Right environment tensors R_env[i] for sites 0..L-1
        # R_env[L] represents the empty right boundary (1, 1, 1)
        self.right_envs: Dict[int, np.ndarray] = {self.L: np.ones((1, 1, 1), dtype=np.complex128)}

    def initialize_right_environments(self) -> None:
        """
        Builds all right environments R_env from site L-1 down to 0.
        Assumes MPS is right-canonical.
        """
        for i in range(self.L - 1, -1, -1):
            b_bra = self.mps.tensors[i].conj()  # (Dl, d, Dr)
            w = self.mpo.tensors[i]  # (wl, wr, dout, din)
            b_ket = self.mps.tensors[i]  # (Dl, d, Dr)
            r_next = self.right_envs[i + 1]  # (Dr_bra, wr, Dr_ket)

            self.right_envs[i] = self.contract_right_step(b_bra, w, b_ket, r_next)

    @staticmethod
    def contract_left_step(
        a_bra: np.ndarray, w: np.ndarray, a_ket: np.ndarray, l_prev: np.ndarray
    ) -> np.ndarray:
        """
        Contracts L_prev(Dl_bra, wl, Dl_ket) with a_bra, W, a_ket to produce L_next(Dr_bra, wr, Dr_ket).
        """
        # 1. Contract L_prev with a_bra on Dl_bra
        # L_prev(Dl_bra, wl, Dl_ket) * a_bra(Dl_bra, dout, Dr_bra) -> tmp1(wl, Dl_ket, dout, Dr_bra)
        tmp1 = np.tensordot(l_prev, a_bra, axes=(0, 0))

        # 2. Contract tmp1 with W on wl and dout
        # tmp1(wl, Dl_ket, dout, Dr_bra) * W(wl, wr, dout, din) -> tmp2(Dl_ket, Dr_bra, wr, din)
        tmp2 = np.tensordot(tmp1, w, axes=([0, 2], [0, 2]))

        # 3. Contract tmp2 with a_ket on Dl_ket and din
        # tmp2(Dl_ket, Dr_bra, wr, din) * a_ket(Dl_ket, din, Dr_ket) -> l_next(Dr_bra, wr, Dr_ket)
        l_next = np.tensordot(tmp2, a_ket, axes=([0, 3], [0, 1]))
        return l_next

    @staticmethod
    def contract_right_step(
        b_bra: np.ndarray, w: np.ndarray, b_ket: np.ndarray, r_next: np.ndarray
    ) -> np.ndarray:
        """
        Contracts R_next(Dr_bra, wr, Dr_ket) with b_bra, W, b_ket to produce R_prev(Dl_bra, wl, Dl_ket).
        """
        # 1. Contract R_next with b_bra on Dr_bra
        # R_next(Dr_bra, wr, Dr_ket) * b_bra(Dl_bra, dout, Dr_bra) -> tmp1(wr, Dr_ket, Dl_bra, dout)
        tmp1 = np.tensordot(r_next, b_bra, axes=(0, 2))

        # 2. Contract tmp1 with W on wr and dout
        # tmp1(wr, Dr_ket, Dl_bra, dout) * W(wl, wr, dout, din) -> tmp2(Dr_ket, Dl_bra, wl, din)
        tmp2 = np.tensordot(tmp1, w, axes=([0, 3], [1, 2]))

        # 3. Contract tmp2 with b_ket on Dr_ket and din
        # tmp2(Dr_ket, Dl_bra, wl, din) * b_ket(Dl_ket, din, Dr_ket) -> r_prev(Dl_bra, wl, Dl_ket)
        r_prev = np.tensordot(tmp2, b_ket, axes=([0, 3], [2, 1]))
        return r_prev

    def update_left(self, site: int) -> None:
        """
        Updates L_env[site] using MPS tensor at site and L_env[site - 1].
        """
        a_bra = self.mps.tensors[site].conj()
        w = self.mpo.tensors[site]
        a_ket = self.mps.tensors[site]
        l_prev = self.left_envs[site - 1]
        self.left_envs[site] = self.contract_left_step(a_bra, w, a_ket, l_prev)

    def update_right(self, site: int) -> None:
        """
        Updates R_env[site] using MPS tensor at site and R_env[site + 1].
        """
        b_bra = self.mps.tensors[site].conj()
        w = self.mpo.tensors[site]
        b_ket = self.mps.tensors[site]
        r_next = self.right_envs[site + 1]
        self.right_envs[site] = self.contract_right_step(b_bra, w, b_ket, r_next)

    def apply_two_site_hamiltonian(self, site: int, theta: np.ndarray) -> np.ndarray:
        """
        Applies the effective two-site Hamiltonian H_eff on two-site state tensor theta.

        Args:
            site: Left site index i of the pair (i, i+1).
            theta: 4-leg tensor of shape (Dl, d_i, d_{i+1}, Dr).

        Returns:
            H_eff * theta: 4-leg tensor of same shape (Dl, d_i, d_{i+1}, Dr).
        """
        l_env = self.left_envs[site - 1]  # (Dl_bra, wl, Dl_ket)
        w_i = self.mpo.tensors[site]  # (wl, wm, d_out_i, d_in_i)
        w_ip1 = self.mpo.tensors[site + 1]  # (wm, wr, d_out_ip1, d_in_ip1)
        r_env = self.right_envs[site + 2]  # (Dr_bra, wr, Dr_ket)

        # 1. Contract L_env with theta on Dl_ket:
        # L_env(Dl_bra, wl, Dl_ket) * theta(Dl_ket, d_in_i, d_in_ip1, Dr_ket)
        # -> T1(Dl_bra, wl, d_in_i, d_in_ip1, Dr_ket)
        t1 = np.tensordot(l_env, theta, axes=(2, 0))

        # 2. Contract T1 with w_i on wl and d_in_i:
        # T1(Dl_bra, wl, d_in_i, d_in_ip1, Dr_ket) * w_i(wl, wm, d_out_i, d_in_i)
        # -> T2(Dl_bra, d_in_ip1, Dr_ket, wm, d_out_i)
        t2 = np.tensordot(t1, w_i, axes=([1, 2], [0, 3]))

        # 3. Contract T2 with w_ip1 on wm and d_in_ip1:
        # T2(Dl_bra, d_in_ip1, Dr_ket, wm, d_out_i) * w_ip1(wm, wr, d_out_ip1, d_in_ip1)
        # -> T3(Dl_bra, Dr_ket, d_out_i, wr, d_out_ip1)
        t3 = np.tensordot(t2, w_ip1, axes=([1, 3], [3, 0]))

        # 4. Contract T3 with r_env on wr and Dr_ket:
        # T3(Dl_bra, Dr_ket, d_out_i, wr, d_out_ip1) * r_env(Dr_bra, wr, Dr_ket)
        # axes: T3's (Dr_ket at idx 1, wr at idx 3) with r_env's (Dr_ket at idx 2, wr at idx 1)
        # -> T4(Dl_bra, d_out_i, d_out_ip1, Dr_bra)
        t4 = np.tensordot(t3, r_env, axes=([1, 3], [2, 1]))

        return t4

    def get_two_site_linear_operator(
        self, site: int, shape: tuple[int, int, int, int]
    ) -> LinearOperator:
        """
        Constructs a scipy LinearOperator wrapping the two-site Hamiltonian action for eigsh.
        """
        dim = int(np.prod(shape))

        def matvec(v: np.ndarray) -> np.ndarray:
            theta = v.reshape(shape)
            res = self.apply_two_site_hamiltonian(site, theta)
            return res.reshape(dim)

        return LinearOperator(shape=(dim, dim), matvec=matvec, dtype=np.complex128)
