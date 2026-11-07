"""
Time-Evolving Block Decimation (TEBD) for real and imaginary time quantum dynamics.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from scipy.linalg import expm

from entangle_tensor.mps import MatrixProductState
from entangle_tensor.spin_operators import pauli_matrices, spin_half_operators
from entangle_tensor.types import EvolutionType, TEBDConfig


class TEBDSimulator:
    """
    Time-Evolving Block Decimation (TEBD) simulation engine with 2nd-order Trotter-Suzuki decomposition.
    """

    def __init__(self, mps: MatrixProductState, config: Optional[TEBDConfig] = None) -> None:
        self.mps: MatrixProductState = mps
        self.config: TEBDConfig = config or TEBDConfig()
        self.L: int = mps.L
        self.d: int = mps.d

    @staticmethod
    def ising_bond_hamiltonians(L: int, J: float = 1.0, h: float = 1.0) -> List[np.ndarray]:
        """
        Constructs the exact list of L-1 local bond Hamiltonians for the Transverse-Field Ising Model:
          H = -J sum_{i=0}^{L-2} (sigma^z_i (x) sigma^z_{i+1}) - h sum_{i=0}^{L-1} sigma^x_i

        Properly allocates full boundary field on site 0 and site L-1, and split h/2 on bulk sites,
        such that sum_{i=0}^{L-2} h_{i, i+1} == H_tot exactly.
        """
        _, sx, _, sz = pauli_matrices()
        eye = np.eye(2, dtype=np.complex128)

        term_zz = np.kron(sz, sz)
        term_x1 = np.kron(sx, eye)
        term_x2 = np.kron(eye, sx)

        bonds = []
        for i in range(L - 1):
            coeff_1 = h if i == 0 else 0.5 * h
            coeff_2 = h if i == L - 2 else 0.5 * h
            h_mat = -J * term_zz - (coeff_1 * term_x1 + coeff_2 * term_x2)
            bonds.append(h_mat.reshape(2, 2, 2, 2))
        return bonds

    @staticmethod
    def heisenberg_bond_hamiltonians(
        L: int, J: float = 1.0, Jz: float = 1.0, hz: float = 0.0
    ) -> List[np.ndarray]:
        """
        Constructs the exact list of L-1 local bond Hamiltonians for the Spin-1/2 Heisenberg XXZ chain.
        """
        ops = spin_half_operators()
        sx, sy, sz, eye = ops["Sx"], ops["Sy"], ops["Sz"], ops["I"]

        term_xx = np.kron(sx, sx)
        term_yy = np.kron(sy, sy)
        term_zz = np.kron(sz, sz)
        term_z1 = np.kron(sz, eye)
        term_z2 = np.kron(eye, sz)

        bonds = []
        for i in range(L - 1):
            coeff_1 = hz if i == 0 else 0.5 * hz
            coeff_2 = hz if i == L - 2 else 0.5 * hz
            h_mat = J * (term_xx + term_yy) + Jz * term_zz - (coeff_1 * term_z1 + coeff_2 * term_z2)
            bonds.append(h_mat.reshape(2, 2, 2, 2))
        return bonds

    def build_evolution_gates(
        self,
        bond_hamiltonians: Union[np.ndarray, Sequence[np.ndarray]],
        dt: float,
        mode: EvolutionType = EvolutionType.REAL,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Computes the full step gates U(dt) and half step gates U(dt/2) for all bonds.
        """
        d = self.d
        if isinstance(bond_hamiltonians, np.ndarray) and bond_hamiltonians.ndim == 4:
            ham_list = [bond_hamiltonians] * (self.L - 1)
        else:
            ham_list = list(bond_hamiltonians)

        if len(ham_list) != self.L - 1:
            raise ValueError(f"Expected {self.L - 1} bond Hamiltonians, got {len(ham_list)}")

        u_full_list = []
        u_half_list = []

        for h in ham_list:
            h_mat = h.reshape(d * d, d * d)
            if mode == EvolutionType.IMAGINARY:
                u_full_mat = expm(-dt * h_mat)
                u_half_mat = expm(-0.5 * dt * h_mat)
            else:
                u_full_mat = expm(-1.0j * dt * h_mat)
                u_half_mat = expm(-0.5j * dt * h_mat)
            u_full_list.append(u_full_mat.reshape(d, d, d, d))
            u_half_list.append(u_half_mat.reshape(d, d, d, d))

        return u_full_list, u_half_list

    def apply_two_site_gate(self, site: int, gate: np.ndarray) -> float:
        """
        Applies a two-site unitary gate U(d_out_1, d_out_2, d_in_1, d_in_2) to sites (site, site+1).
        Performs SVD truncation up to chi_max and updates MPS tensors.
        Returns the discarded truncation weight.
        """
        t_i = self.mps.tensors[site]  # (Dl, d, Dm)
        t_ip1 = self.mps.tensors[site + 1]  # (Dm, d, Dr)

        # 1. Form two-site state theta: (Dl, d_in_i, d_in_ip1, Dr)
        theta = np.tensordot(t_i, t_ip1, axes=(2, 0))

        # 2. Contract with gate U:
        theta_prime = np.tensordot(gate, theta, axes=([2, 3], [1, 2]))

        # Permute legs to standard order: (Dl, d_out_i, d_out_ip1, Dr)
        theta_prime = theta_prime.transpose(2, 0, 1, 3)

        dl, d1, d2, dr = theta_prime.shape
        mat = theta_prime.reshape(dl * d1, d2 * dr)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)

        chi_max = self.config.chi_max
        svd_tol = self.config.svd_tol

        keep = min(len(s), chi_max)
        keep = max(1, int(np.sum(s[:keep] > svd_tol)))
        keep = min(keep, chi_max)

        trunc_err = float(np.sum(s[keep:] ** 2))

        s_keep = s[:keep]
        s_norm = np.linalg.norm(s_keep)
        if s_norm > 1e-15:
            s_keep = s_keep / s_norm

        # Update MPS tensors
        self.mps.tensors[site] = u[:, :keep].reshape(dl, d1, keep)
        sv = s_keep[:, None] * vh[:keep, :]
        self.mps.tensors[site + 1] = sv.reshape(keep, d2, dr)

        return trunc_err

    def step(
        self,
        u_full_list: List[np.ndarray],
        u_half_list: List[np.ndarray],
        normalize_step: bool = False,
    ) -> float:
        """
        Executes a single 2nd-order Trotter-Suzuki time step:
          U_even(dt/2) -> U_odd(dt) -> U_even(dt/2)
        """
        max_trunc_err = 0.0

        # 1. First half-step on even bonds (0, 2, 4, ...)
        for i in range(0, self.L - 1, 2):
            err = self.apply_two_site_gate(i, u_half_list[i])
            if err > max_trunc_err:
                max_trunc_err = err

        # 2. Full step on odd bonds (1, 3, 5, ...)
        for i in range(1, self.L - 1, 2):
            err = self.apply_two_site_gate(i, u_full_list[i])
            if err > max_trunc_err:
                max_trunc_err = err

        # 3. Second half-step on even bonds (0, 2, 4, ...)
        for i in range(0, self.L - 1, 2):
            err = self.apply_two_site_gate(i, u_half_list[i])
            if err > max_trunc_err:
                max_trunc_err = err

        if normalize_step:
            self.mps.normalize()

        return max_trunc_err

    def evolve(
        self,
        bond_hamiltonians: Union[np.ndarray, Sequence[np.ndarray]],
        mode: EvolutionType = EvolutionType.REAL,
        num_steps: Optional[int] = None,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> List[float]:
        """
        Evolves the state through num_steps Trotter time steps.
        """
        steps = num_steps or self.config.total_steps
        dt = self.config.dt
        u_full_list, u_half_list = self.build_evolution_gates(bond_hamiltonians, dt, mode)

        normalize_every_step = mode == EvolutionType.IMAGINARY
        trunc_errors = []

        for s in range(steps):
            err = self.step(u_full_list, u_half_list, normalize_step=normalize_every_step)
            trunc_errors.append(err)
            if callback is not None:
                callback(s, (s + 1) * dt)

        self.mps.normalize()
        return trunc_errors
