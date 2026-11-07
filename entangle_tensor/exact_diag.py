"""
Exact Diagonalization (ED) benchmark solver for verification of DMRG and TEBD.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from entangle_tensor.spin_operators import (
    pauli_matrices,
    spin_half_operators,
    spin_one_operators,
)


class ExactDiagonalization:
    """
    Exact diagonalization solver constructing sparse Hamiltonian matrices using Kronecker products.
    """

    @staticmethod
    def ising_hamiltonian(L: int, J: float = 1.0, h: float = 1.0) -> sp.csr_matrix:
        """
        Builds sparse Hamiltonian for 1D Transverse-Field Ising model:
          H = -J sum_{i=0}^{L-2} sigma^z_i sigma^z_{i+1} - h sum_{i=0}^{L-1} sigma^x_i
        """
        _, sx, _, sz = pauli_matrices()
        eye = sp.eye(2, dtype=np.complex128, format="csr")
        sx_sp = sp.csr_matrix(sx)
        sz_sp = sp.csr_matrix(sz)

        dim = 2**L
        h_tot = sp.csr_matrix((dim, dim), dtype=np.complex128)

        # Nearest neighbor -J sz_i sz_{i+1}
        for i in range(L - 1):
            op_list = [eye] * L
            op_list[i] = sz_sp
            op_list[i + 1] = sz_sp

            term = op_list[0]
            for op in op_list[1:]:
                term = sp.kron(term, op, format="csr")
            h_tot -= J * term

        # Transverse field -h sx_i
        for i in range(L):
            op_list = [eye] * L
            op_list[i] = sx_sp

            term = op_list[0]
            for op in op_list[1:]:
                term = sp.kron(term, op, format="csr")
            h_tot -= h * term

        return h_tot

    @staticmethod
    def heisenberg_hamiltonian(
        L: int, J: float = 1.0, Jz: float = 1.0, hz: float = 0.0
    ) -> sp.csr_matrix:
        """
        Builds sparse Hamiltonian for 1D Spin-1/2 Heisenberg XXZ chain:
          H = J sum_{i=0}^{L-2} (Sx_i Sx_{i+1} + Sy_i Sy_{i+1}) + Jz sum Sz_i Sz_{i+1} - hz sum Sz_i
        """
        ops = spin_half_operators()
        eye = sp.eye(2, dtype=np.complex128, format="csr")
        sx_sp = sp.csr_matrix(ops["Sx"])
        sy_sp = sp.csr_matrix(ops["Sy"])
        sz_sp = sp.csr_matrix(ops["Sz"])

        dim = 2**L
        h_tot = sp.csr_matrix((dim, dim), dtype=np.complex128)

        # Exchange couplings
        for i in range(L - 1):
            for op_sp, coeff in [(sx_sp, J), (sy_sp, J), (sz_sp, Jz)]:
                op_list = [eye] * L
                op_list[i] = op_sp
                op_list[i + 1] = op_sp

                term = op_list[0]
                for op in op_list[1:]:
                    term = sp.kron(term, op, format="csr")
                h_tot += coeff * term

        # Magnetic field -hz Sz_i
        if abs(hz) > 1e-12:
            for i in range(L):
                op_list = [eye] * L
                op_list[i] = sz_sp
                term = op_list[0]
                for op in op_list[1:]:
                    term = sp.kron(term, op, format="csr")
                h_tot -= hz * term

        return h_tot

    @staticmethod
    def haldane_spin_one_hamiltonian(L: int, J: float = 1.0, D: float = 0.0) -> sp.csr_matrix:
        """
        Builds sparse Hamiltonian for Spin-1 Haldane chain (d = 3).
        """
        ops = spin_one_operators()
        eye = sp.eye(3, dtype=np.complex128, format="csr")
        sx_sp = sp.csr_matrix(ops["Sx"])
        sy_sp = sp.csr_matrix(ops["Sy"])
        sz_sp = sp.csr_matrix(ops["Sz"])
        sz2_sp = sp.csr_matrix(ops["Sz2"])

        dim = 3**L
        h_tot = sp.csr_matrix((dim, dim), dtype=np.complex128)

        for i in range(L - 1):
            for op_sp in [sx_sp, sy_sp, sz_sp]:
                op_list = [eye] * L
                op_list[i] = op_sp
                op_list[i + 1] = op_sp
                term = op_list[0]
                for op in op_list[1:]:
                    term = sp.kron(term, op, format="csr")
                h_tot += J * term

        if abs(D) > 1e-12:
            for i in range(L):
                op_list = [eye] * L
                op_list[i] = sz2_sp
                term = op_list[0]
                for op in op_list[1:]:
                    term = sp.kron(term, op, format="csr")
                h_tot += D * term

        return h_tot

    @staticmethod
    def ground_state(h_sparse: sp.csr_matrix) -> Tuple[float, np.ndarray]:
        """
        Computes the lowest eigenvalue and eigenvector of the Hamiltonian.
        """
        dim = h_sparse.shape[0]
        if dim <= 16:
            dense_h = h_sparse.toarray()
            # Ensure Hermitian
            dense_h = 0.5 * (dense_h + dense_h.conj().T)
            vals, vecs = np.linalg.eigh(dense_h)
            min_idx = int(np.argmin(vals))
            return float(vals[min_idx]), vecs[:, min_idx]

        vals, vecs = eigsh(h_sparse, k=1, which="SA", tol=1e-12)
        return float(np.real(vals[0])), vecs[:, 0]
