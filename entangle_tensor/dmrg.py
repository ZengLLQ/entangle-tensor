"""
Two-site and single-site Variational Density Matrix Renormalization Group (DMRG) algorithm.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse.linalg import eigsh

from entangle_tensor.environments import EnvironmentCache
from entangle_tensor.mpo import MatrixProductOperator
from entangle_tensor.mps import MatrixProductState
from entangle_tensor.types import DMRGConfig, DMRGResult


class DMRGSolver:
    """
    Two-site Variational Density Matrix Renormalization Group (DMRG) ground state solver.
    """

    def __init__(self, config: Optional[DMRGConfig] = None) -> None:
        self.config: DMRGConfig = config or DMRGConfig()

    def run(self, mps: MatrixProductState, mpo: MatrixProductOperator) -> DMRGResult:
        """
        Executes variational DMRG optimization to find the ground state of the given MPO.

        Args:
            mps: Initial trial MatrixProductState (will be mutated in-place).
            mpo: MatrixProductOperator describing the quantum Hamiltonian.

        Returns:
            DMRGResult containing optimized energy, convergence diagnostics, and bond dimensions.
        """
        if mps.L != mpo.L or mps.d != mpo.d:
            raise ValueError("MPS and MPO lengths and physical dimensions must match.")

        L = mps.L
        chi_max = self.config.chi_max
        svd_tol = self.config.svd_tol
        energy_tol = self.config.energy_tol

        # 1. Prepare initial state in right-canonical form
        mps.right_canonicalize(stop_site=0)
        mps.normalize()

        # 2. Build right environment cache
        env_cache = EnvironmentCache(mps, mpo)
        env_cache.initialize_right_environments()

        sweep_energies: List[float] = []
        max_trunc_error: float = 0.0
        converged = False
        prev_energy = float("inf")
        curr_energy = float("inf")

        for sweep in range(self.config.max_sweeps):
            sweep_start = time.perf_counter()

            # --- Forward sweep: Left to Right (sites 0 to L-2) ---
            for site in range(L - 1):
                # Form two-site tensor theta: (Dl, d_i, d_{i+1}, Dr)
                t_i = mps.tensors[site]  # (Dl, d_i, Dm)
                t_ip1 = mps.tensors[site + 1]  # (Dm, d_{i+1}, Dr)
                theta = np.tensordot(t_i, t_ip1, axes=(2, 0))  # (Dl, d_i, d_{i+1}, Dr)

                # Diagonalize local effective Hamiltonian
                energy, theta_opt = self._solve_local_ground_state(env_cache, site, theta)
                curr_energy = energy

                # SVD truncation
                dl, d1, d2, dr = theta_opt.shape
                mat = theta_opt.reshape(dl * d1, d2 * dr)
                u, s, vh = np.linalg.svd(mat, full_matrices=False)

                # Determine number of singular values to keep
                keep = min(len(s), chi_max)
                keep = max(1, int(np.sum(s[:keep] > svd_tol)))
                keep = min(keep, chi_max)

                # Track truncation error
                discarded_err = float(np.sum(s[keep:] ** 2))
                if discarded_err > max_trunc_error:
                    max_trunc_error = discarded_err

                s_keep = s[:keep]
                s_norm = np.linalg.norm(s_keep)
                if s_norm > 1e-15:
                    s_keep = s_keep / s_norm

                # Left tensor is left-orthogonal U
                mps.tensors[site] = u[:, :keep].reshape(dl, d1, keep)
                # Next tensor absorbs s and vh
                sv = s_keep[:, None] * vh[:keep, :]
                mps.tensors[site + 1] = sv.reshape(keep, d2, dr)

                # Update left environment at site
                env_cache.update_left(site)

            # --- Backward sweep: Right to Left (sites L-2 down to 0) ---
            for site in range(L - 2, -1, -1):
                t_i = mps.tensors[site]  # (Dl, d_i, Dm)
                t_ip1 = mps.tensors[site + 1]  # (Dm, d_{i+1}, Dr)
                theta = np.tensordot(t_i, t_ip1, axes=(2, 0))  # (Dl, d_i, d_{i+1}, Dr)

                energy, theta_opt = self._solve_local_ground_state(env_cache, site, theta)
                curr_energy = energy

                dl, d1, d2, dr = theta_opt.shape
                mat = theta_opt.reshape(dl * d1, d2 * dr)
                u, s, vh = np.linalg.svd(mat, full_matrices=False)

                keep = min(len(s), chi_max)
                keep = max(1, int(np.sum(s[:keep] > svd_tol)))
                keep = min(keep, chi_max)

                discarded_err = float(np.sum(s[keep:] ** 2))
                if discarded_err > max_trunc_error:
                    max_trunc_error = discarded_err

                s_keep = s[:keep]
                s_norm = np.linalg.norm(s_keep)
                if s_norm > 1e-15:
                    s_keep = s_keep / s_norm

                # Left tensor absorbs u and s
                us = u[:, :keep] * s_keep[None, :]
                mps.tensors[site] = us.reshape(dl, d1, keep)
                # Right tensor is right-orthogonal V^dagger
                mps.tensors[site + 1] = vh[:keep, :].reshape(keep, d2, dr)

                # Update right environment at site + 1
                env_cache.update_right(site + 1)

            sweep_energies.append(curr_energy)
            dE = abs(curr_energy - prev_energy)

            if self.config.verbose:
                elapsed = time.perf_counter() - sweep_start
                print(
                    f"DMRG Sweep {sweep + 1:2d}/{self.config.max_sweeps}: "
                    f"E = {curr_energy:14.10f}, dE = {dE:9.2e}, "
                    f"max_chi = {mps.max_bond_dimension:3d}, time = {elapsed:.2f}s"
                )

            if dE < energy_tol and sweep >= 1:
                converged = True
                break

            prev_energy = curr_energy

        mps.normalize()

        return DMRGResult(
            ground_state_energy=curr_energy,
            energy_per_site=curr_energy / L,
            converged=converged,
            num_sweeps=len(sweep_energies),
            sweep_energies=sweep_energies,
            max_truncation_error=max_trunc_error,
            final_bond_dimensions=mps.bond_dimensions,
        )

    def _solve_local_ground_state(
        self, env_cache: EnvironmentCache, site: int, theta: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Finds the minimum eigenvalue and eigenvector of the effective two-site Hamiltonian.
        """
        shape = theta.shape
        dim = int(np.prod(shape))
        lin_op = env_cache.get_two_site_linear_operator(site, shape)

        # Flatten initial vector
        v0 = theta.reshape(dim)
        v0_norm = np.linalg.norm(v0)
        if v0_norm > 1e-15:
            v0 = v0 / v0_norm

        if dim <= 8:
            # For tiny Hilbert spaces, construct explicit matrix for exact diagonalisation
            eye = np.eye(dim, dtype=np.complex128)
            h_mat: np.ndarray = np.zeros((dim, dim), dtype=np.complex128)
            for col in range(dim):
                h_mat[:, col] = lin_op.matvec(eye[:, col])
            # Hermitize to eliminate tiny numerical asymmetry
            h_mat = 0.5 * (h_mat + h_mat.conj().T)
            vals, vecs = np.linalg.eigh(h_mat)
            min_idx = int(np.argmin(vals))
            energy = float(vals[min_idx])
            theta_opt = vecs[:, min_idx].reshape(shape)
            return energy, theta_opt

        # Otherwise use sparse ARPACK Lanczos via scipy.sparse.linalg.eigsh
        try:
            vals, vecs = eigsh(
                lin_op,
                k=1,
                which="SA",
                v0=v0,
                tol=self.config.lanczos_tol,
                maxiter=self.config.lanczos_maxiter,
            )
            energy = float(np.real(vals[0]))
            theta_opt = vecs[:, 0].reshape(shape)
            return energy, theta_opt
        except Exception:
            # Fallback if ARPACK fails to converge on initial seed: full eigh or power iteration
            eye = np.eye(dim, dtype=np.complex128)
            h_mat_fb: np.ndarray = np.zeros((dim, dim), dtype=np.complex128)
            for col in range(dim):
                h_mat_fb[:, col] = lin_op.matvec(eye[:, col])
            h_mat_fb = 0.5 * (h_mat_fb + h_mat_fb.conj().T)
            vals, vecs = np.linalg.eigh(h_mat_fb)
            min_idx = int(np.argmin(vals))
            energy = float(vals[min_idx])
            theta_opt = vecs[:, min_idx].reshape(shape)
            return energy, theta_opt
