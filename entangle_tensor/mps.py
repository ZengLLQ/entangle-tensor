"""
Matrix Product State (MPS) representation, canonical forms, and tensor operations.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from entangle_tensor.types import CanonicalForm


class MatrixProductState:
    """
    1D Open Boundary Matrix Product State (MPS) represented as a chain of 3-leg tensors.

    Each tensor A^[i] at site i has shape (D_{i-1}, d_i, D_i):
      - D_{i-1}: Left virtual bond dimension (D_0 = 1).
      - d_i: Physical dimension at site i (e.g. 2 for spin-1/2, 3 for spin-1).
      - D_i: Right virtual bond dimension (D_L = 1).
    """

    def __init__(
        self,
        tensors: Sequence[np.ndarray],
        canonical_form: CanonicalForm = CanonicalForm.UNCONSTRAINED,
    ) -> None:
        """
        Initializes an MPS with a sequence of tensors.

        Args:
            tensors: Sequence of 3-leg numpy arrays A^[i] with shape (D_left, d_phys, D_right).
            canonical_form: Current canonical state of the MPS.
        """
        if len(tensors) == 0:
            raise ValueError("MPS must contain at least one site tensor.")

        self.tensors: List[np.ndarray] = [np.array(t, dtype=np.complex128) for t in tensors]
        self.L: int = len(self.tensors)
        self.d: int = self.tensors[0].shape[1]
        self.canonical_form: CanonicalForm = canonical_form
        self.gauge_center: Optional[int] = None

        # Validate boundary virtual dimensions
        if self.tensors[0].shape[0] != 1:
            raise ValueError(
                f"First tensor left bond dimension must be 1, got {self.tensors[0].shape[0]}"
            )
        if self.tensors[-1].shape[2] != 1:
            raise ValueError(
                f"Last tensor right bond dimension must be 1, got {self.tensors[-1].shape[2]}"
            )

        # Validate bond dimension consistency along the chain
        for i in range(self.L - 1):
            if self.tensors[i].shape[2] != self.tensors[i + 1].shape[0]:
                raise ValueError(
                    f"Bond dimension mismatch between site {i} (D_R={self.tensors[i].shape[2]}) "
                    f"and site {i + 1} (D_L={self.tensors[i + 1].shape[0]})"
                )

    @classmethod
    def product_state(cls, states: Sequence[int], d: int = 2) -> MatrixProductState:
        """
        Creates an unentangled product state |s_0 s_1 ... s_{L-1}>.

        Args:
            states: List of local basis state indices (0 <= s_i < d).
            d: Local physical dimension.
        """
        tensors = []
        for s in states:
            if not 0 <= s < d:
                raise ValueError(f"State index {s} out of bounds for physical dimension {d}")
            # Shape (1, d, 1)
            t = np.zeros((1, d, 1), dtype=np.complex128)
            t[0, s, 0] = 1.0
            tensors.append(t)
        mps = cls(tensors, canonical_form=CanonicalForm.MIXED)
        mps.gauge_center = 0
        return mps

    @classmethod
    def ferromagnetic_state(cls, L: int, d: int = 2, state_idx: int = 0) -> MatrixProductState:
        """Creates a ferromagnetic state |state_idx state_idx ... state_idx>."""
        return cls.product_state([state_idx] * L, d=d)

    @classmethod
    def neel_state(cls, L: int, d: int = 2) -> MatrixProductState:
        """Creates an alternating Néel state |0 1 0 1 ...>."""
        states = [i % d for i in range(L)]
        return cls.product_state(states, d=d)

    @classmethod
    def random(
        cls, L: int, d: int = 2, chi: int = 4, seed: Optional[int] = None
    ) -> MatrixProductState:
        """
        Creates a normalized random MPS with bond dimension bounded by chi.
        """
        rng = np.random.default_rng(seed)
        tensors = []

        # Calculate maximum possible bond dimensions based on open boundaries
        bonds = [1]
        for i in range(1, L):
            max_bond = min(chi, d**i, d ** (L - i))
            bonds.append(max_bond)
        bonds.append(1)

        for i in range(L):
            d_l, d_r = bonds[i], bonds[i + 1]
            real_part = rng.standard_normal((d_l, d, d_r))
            imag_part = rng.standard_normal((d_l, d, d_r))
            t = (real_part + 1.0j * imag_part) / np.sqrt(2.0)
            tensors.append(t)

        mps = cls(tensors, canonical_form=CanonicalForm.UNCONSTRAINED)
        mps.normalize()
        return mps

    @property
    def bond_dimensions(self) -> List[int]:
        """Returns the list of bond dimensions [D_0, D_1, ..., D_L]."""
        bonds = [self.tensors[0].shape[0]]
        for t in self.tensors:
            bonds.append(t.shape[2])
        return bonds

    @property
    def max_bond_dimension(self) -> int:
        """Returns the maximum bond dimension present in the chain."""
        return max(self.bond_dimensions)

    def copy(self) -> MatrixProductState:
        """Deep copy of the MPS."""
        new_mps = MatrixProductState(
            [t.copy() for t in self.tensors], canonical_form=self.canonical_form
        )
        new_mps.gauge_center = self.gauge_center
        return new_mps

    def left_canonicalize(self, stop_site: Optional[int] = None) -> None:
        """
        Transforms tensors from site 0 up to stop_site into left-orthogonal form via QR.
        sum_{s} (A^[i] s)^dagger A^[i] s = Identity.
        """
        limit = self.L - 1 if stop_site is None else min(stop_site, self.L - 1)
        for i in range(limit):
            d_l, d_phys, d_r = self.tensors[i].shape
            mat = self.tensors[i].reshape(d_l * d_phys, d_r)
            q, r = np.linalg.qr(mat)
            new_dr = q.shape[1]
            self.tensors[i] = q.reshape(d_l, d_phys, new_dr)
            # Contract r into next tensor
            self.tensors[i + 1] = np.tensordot(r, self.tensors[i + 1], axes=(1, 0))

        if stop_site is None or stop_site == self.L - 1:
            self.canonical_form = CanonicalForm.LEFT
            self.gauge_center = self.L - 1
        else:
            self.canonical_form = CanonicalForm.MIXED
            self.gauge_center = stop_site

    def right_canonicalize(self, stop_site: int = 0) -> None:
        """
        Transforms tensors from site L-1 down to stop_site into right-orthogonal form via LQ/QR.
        sum_{s} B^[i] s (B^[i] s)^dagger = Identity.
        """
        for i in range(self.L - 1, stop_site, -1):
            d_l, d_phys, d_r = self.tensors[i].shape
            # Reshape to (d_l, d_phys * d_r)
            mat = self.tensors[i].reshape(d_l, d_phys * d_r)
            # Transpose conjugate for QR
            q, r = np.linalg.qr(mat.T)
            # mat = r^T @ q^T
            new_dl = q.shape[1]
            self.tensors[i] = q.T.reshape(new_dl, d_phys, d_r)
            # Contract r^T into previous tensor
            r_t = r.T
            self.tensors[i - 1] = np.tensordot(self.tensors[i - 1], r_t, axes=(2, 0))

        if stop_site == 0:
            self.canonical_form = CanonicalForm.RIGHT
            self.gauge_center = 0
        else:
            self.canonical_form = CanonicalForm.MIXED
            self.gauge_center = stop_site

    def mixed_canonicalize(self, center: int) -> None:
        """
        Puts the MPS into mixed-canonical form with the gauge center at site `center`.
        Sites < center are left-orthogonal; sites > center are right-orthogonal.
        """
        if not 0 <= center < self.L:
            raise ValueError(f"Gauge center {center} must be in range [0, {self.L - 1}].")
        self.right_canonicalize(stop_site=center)
        self.left_canonicalize(stop_site=center)
        self.gauge_center = center
        self.canonical_form = CanonicalForm.MIXED

    def inner(self, other: MatrixProductState) -> complex:
        """
        Calculates the inner product <self | other> using sequential left-to-right contractions.
        Total computational complexity: O(L * D_self * D_other * d * max(D_self, D_other)).
        """
        if self.L != other.L or self.d != other.d:
            raise ValueError("MPS dimension mismatch for inner product.")

        # Transfer tensor environment E of shape (D_self_l, D_other_l) initialized to [[1.0]]
        e: np.ndarray = np.ones((1, 1), dtype=np.complex128)
        for i in range(self.L):
            a_bra = self.tensors[i].conj()  # (Dl_bra, d, Dr_bra)
            a_ket = other.tensors[i]  # (Dl_ket, d, Dr_ket)

            # Contract E with bra: E(Dl_bra, Dl_ket) * a_bra(Dl_bra, d, Dr_bra) -> tmp(Dl_ket, d, Dr_bra)
            tmp = np.tensordot(e, a_bra, axes=(0, 0))
            # Contract tmp with ket: tmp(Dl_ket, d, Dr_bra) * a_ket(Dl_ket, d, Dr_ket) -> E_new(Dr_bra, Dr_ket)
            e = np.tensordot(tmp, a_ket, axes=([0, 1], [0, 1]))

        return complex(e[0, 0])

    def norm(self) -> float:
        """Returns the L2 state norm sqrt(|<psi|psi>|)."""
        val = self.inner(self)
        return float(np.sqrt(np.abs(val)))

    def normalize(self) -> float:
        """Normalizes the MPS in-place and returns the pre-normalization norm."""
        n = self.norm()
        if n > 1e-15:
            scale = n ** (1.0 / self.L)
            for i in range(self.L):
                self.tensors[i] /= scale
        return n

    def compress(self, chi_max: int, svd_tol: float = 1e-12) -> float:
        """
        Compresses the MPS to a maximum bond dimension chi_max using SVD truncation.
        Returns the maximum truncation error (sum of discarded singular values squared).
        """
        self.right_canonicalize(stop_site=0)
        max_trunc_err = 0.0

        for i in range(self.L - 1):
            d_l, d_phys, d_r = self.tensors[i].shape
            mat = self.tensors[i].reshape(d_l * d_phys, d_r)
            u, s, vh = np.linalg.svd(mat, full_matrices=False)

            # Truncation logic
            keep = min(len(s), chi_max)
            # Further truncate below svd_tol
            keep = max(1, int(np.sum(s[:keep] > svd_tol)))
            keep = min(keep, chi_max)

            discarded = s[keep:]
            trunc_err = float(np.sum(discarded**2))
            if trunc_err > max_trunc_err:
                max_trunc_err = trunc_err

            u_keep = u[:, :keep]
            s_keep = s[:keep]
            vh_keep = vh[:keep, :]

            self.tensors[i] = u_keep.reshape(d_l, d_phys, keep)
            # Contract s * vh into next site
            sv = s_keep[:, None] * vh_keep
            self.tensors[i + 1] = np.tensordot(sv, self.tensors[i + 1], axes=(1, 0))

        # Normalize last site
        self.normalize()
        self.canonical_form = CanonicalForm.LEFT
        self.gauge_center = self.L - 1
        return max_trunc_err

    def schmidt_decomposition(self, cut: int) -> np.ndarray:
        """
        Computes the Schmidt values (singular values) across the bipartition cut between site
        cut-1 and site cut (1 <= cut <= L-1).
        Puts gauge center at cut-1 and performs SVD.
        """
        if not 1 <= cut < self.L:
            raise ValueError(f"Cut {cut} must be between 1 and {self.L - 1}.")

        self.mixed_canonicalize(cut - 1)
        # Tensor at site cut-1 is M, right of it are right-orthogonal
        # Perform SVD on site cut-1
        t = self.tensors[cut - 1]
        d_l, d_phys, d_r = t.shape
        mat = t.reshape(d_l * d_phys, d_r)
        _, s, _ = np.linalg.svd(mat, full_matrices=False)
        # Normalize singular values so sum s^2 = 1
        s_norm = np.linalg.norm(s)
        if s_norm > 1e-15:
            s = s / s_norm
        res: np.ndarray = np.asarray(s)
        return res

    def to_dense_statevector(self) -> np.ndarray:
        """
        Contracts all MPS tensors into a full dense Hilbert space state vector of length d^L.
        Only practical for small systems (L <= 14).
        """
        vec = self.tensors[0]  # (1, d, D1)
        for i in range(1, self.L):
            # Contract vec with self.tensors[i]
            # vec: (1, d^(i), D_i)
            # next: (D_i, d, D_{i+1})
            vec = np.tensordot(vec, self.tensors[i], axes=(-1, 0))
        # Final shape: (1, d, d, ..., d, 1) -> squeeze to (d^L,)
        res: np.ndarray = np.asarray(vec.reshape(self.d**self.L))
        return res
