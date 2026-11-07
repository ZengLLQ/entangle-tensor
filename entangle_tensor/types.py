"""
Type definitions, enumerations, and dataclasses for EntangleTensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy as np


class SpinType(str, Enum):
    """Supported physical spin degrees of freedom."""

    SPIN_HALF = "spin_1/2"
    SPIN_ONE = "spin_1"


class CanonicalForm(str, Enum):
    """Canonical representation state of an MPS."""

    UNCONSTRAINED = "unconstrained"
    LEFT = "left"
    RIGHT = "right"
    MIXED = "mixed"


class EvolutionType(str, Enum):
    """TEBD evolution mode."""

    IMAGINARY = "imaginary"
    REAL = "real"


@dataclass
class DMRGConfig:
    """
    Configuration options for variational DMRG ground state optimization.

    Attributes:
        max_sweeps: Maximum number of forward/backward sweeps.
        chi_max: Maximum bond dimension allowed during truncation.
        svd_tol: Truncation threshold for singular values.
        energy_tol: Energy difference convergence threshold between consecutive sweeps.
        lanczos_tol: Convergence tolerance for Lanczos / Davidson sparse eigensolver.
        lanczos_maxiter: Maximum iterations for local ground state search.
        two_site: Whether to use two-site DMRG (allowing bond dimension growth) or single-site.
        verbose: Verbose progress reporting flag.
    """

    max_sweeps: int = 20
    chi_max: int = 64
    svd_tol: float = 1e-12
    energy_tol: float = 1e-10
    lanczos_tol: float = 1e-10
    lanczos_maxiter: int = 200
    two_site: bool = True
    verbose: bool = False


@dataclass
class DMRGResult:
    """
    Results and diagnostics returned by the DMRG solver.

    Attributes:
        ground_state_energy: Optimized ground state energy <psi|H|psi>.
        energy_per_site: Ground state energy divided by system size L.
        converged: Whether the energy tolerance threshold was reached.
        num_sweeps: Total number of half-sweeps completed.
        sweep_energies: History of ground state energy estimates per sweep.
        max_truncation_error: Maximum cumulative discarded weight sum(s_k^2) during SVD.
        final_bond_dimensions: List of bond dimensions across all cuts [D_0, D_1, ..., D_L].
    """

    ground_state_energy: float
    energy_per_site: float
    converged: bool
    num_sweeps: int
    sweep_energies: List[float] = field(default_factory=list)
    max_truncation_error: float = 0.0
    final_bond_dimensions: List[int] = field(default_factory=list)


@dataclass
class TEBDConfig:
    """
    Configuration options for Time-Evolving Block Decimation (TEBD).

    Attributes:
        dt: Time step size (real or imaginary).
        total_steps: Number of Trotter time steps to execute.
        chi_max: Maximum bond dimension allowed during SVD truncation.
        svd_tol: Truncation threshold for discarded singular values.
        order: Trotter-Suzuki decomposition order (1 or 2).
    """

    dt: float = 0.05
    total_steps: int = 100
    chi_max: int = 64
    svd_tol: float = 1e-12
    order: int = 2


@dataclass
class EntanglementProfile:
    """
    Bipartite entanglement diagnostics across all lattice cuts.

    Attributes:
        cuts: Integer cut indices k in 1..L-1.
        von_neumann_entropy: S_vN = -sum lambda_i^2 ln(lambda_i^2) for each cut.
        renyi_entropy: S_alpha for alpha=2 (or specified parameter) for each cut.
        schmidt_values: Dictionary mapping cut index to array of singular values.
    """

    cuts: List[int]
    von_neumann_entropy: List[float]
    renyi_entropy: List[float]
    schmidt_values: dict[int, np.ndarray] = field(default_factory=dict)
