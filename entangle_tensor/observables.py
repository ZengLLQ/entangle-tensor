"""
Quantum observables, entanglement metrics, correlation functions, and CFT central charge analysis.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from entangle_tensor.mps import MatrixProductState
from entangle_tensor.types import EntanglementProfile


def single_site_expectation_value(mps: MatrixProductState, op: np.ndarray, site: int) -> complex:
    """
    Computes the expectation value <psi | O_site | psi> for a single-site operator O.

    Args:
        mps: MatrixProductState.
        op: Single-site operator matrix of shape (d, d).
        site: Site index (0 <= site < L).

    Returns:
        Expectation value <O_site>.
    """
    if not 0 <= site < mps.L:
        raise ValueError(f"Site {site} out of bounds for chain length {mps.L}.")

    # Put MPS into mixed canonical form centered at site
    mps_copy = mps.copy()
    mps_copy.mixed_canonicalize(site)

    # Tensor at site is M (Dl, d, Dr)
    t = mps_copy.tensors[site]

    # Contract: sum_{Dl, Dr, s, s'} t*(Dl, s, Dr) * op(s, s') * t(Dl, s', Dr)
    op_t = np.tensordot(op, t, axes=(1, 1))  # (d, Dl, Dr)
    val = np.tensordot(t.conj(), op_t, axes=([0, 1, 2], [1, 0, 2]))
    return complex(val)


def all_single_site_expectation_values(mps: MatrixProductState, op: np.ndarray) -> np.ndarray:
    """
    Computes expectation values <O_i> across all sites i in 0..L-1.
    """
    vals = np.zeros(mps.L, dtype=np.complex128)
    for i in range(mps.L):
        vals[i] = single_site_expectation_value(mps, op, i)
    return vals


def two_site_correlation_function(
    mps: MatrixProductState, op_a: np.ndarray, op_b: np.ndarray, site_i: int, site_j: int
) -> complex:
    """
    Computes the two-site expectation value <psi | O_A(i) O_B(j) | psi>.
    """
    if site_i == site_j:
        return single_site_expectation_value(mps, op_a @ op_b, site_i)

    if site_i > site_j:
        return two_site_correlation_function(mps, op_b, op_a, site_j, site_i)

    # site_i < site_j
    # Contract left-to-right
    e: np.ndarray = np.ones((1, 1), dtype=np.complex128)

    for k in range(mps.L):
        a_bra = mps.tensors[k].conj()  # (Dl, d, Dr)
        a_ket = mps.tensors[k]  # (Dl, d, Dr)

        if k == site_i:
            # Insert op_a into ket: sum_{s'} op_a(s, s') * a_ket(Dl, s', Dr)
            a_ket_mod = np.tensordot(op_a, a_ket, axes=(1, 1))  # (d, Dl, Dr)
            a_ket_mod = a_ket_mod.transpose(1, 0, 2)  # (Dl, d, Dr)
        elif k == site_j:
            # Insert op_b into ket
            a_ket_mod = np.tensordot(op_b, a_ket, axes=(1, 1))  # (d, Dl, Dr)
            a_ket_mod = a_ket_mod.transpose(1, 0, 2)  # (Dl, d, Dr)
        else:
            a_ket_mod = a_ket

        # Transfer matrix contraction
        tmp = np.tensordot(e, a_bra, axes=(0, 0))
        e = np.tensordot(tmp, a_ket_mod, axes=([0, 1], [0, 1]))

    return complex(e[0, 0])


def connected_two_site_correlation(
    mps: MatrixProductState, op_a: np.ndarray, op_b: np.ndarray, site_i: int, site_j: int
) -> complex:
    """
    Computes the connected correlation:
      C(i, j) = <O_A(i) O_B(j)> - <O_A(i)> <O_B(j)>.
    """
    val_ab = two_site_correlation_function(mps, op_a, op_b, site_i, site_j)
    val_a = single_site_expectation_value(mps, op_a, site_i)
    val_b = single_site_expectation_value(mps, op_b, site_j)
    return val_ab - val_a * val_b


def string_order_parameter(
    mps: MatrixProductState, op_end: np.ndarray, op_string: np.ndarray, site_i: int, site_j: int
) -> complex:
    """
    Computes non-local string order parameter for topological phases:
      O_string(i, j) = - < op_end(i) prod_{k=i+1}^{j-1} op_string(k) op_end(j) >

    For the Spin-1 Haldane chain, op_end = S^z, op_string = exp(i * pi * S^z).
    """
    if site_i >= site_j:
        raise ValueError("site_i must be strictly less than site_j for string order.")

    e: np.ndarray = np.ones((1, 1), dtype=np.complex128)

    for k in range(mps.L):
        a_bra = mps.tensors[k].conj()
        a_ket = mps.tensors[k]

        if k == site_i or k == site_j:
            op = op_end
            a_ket_mod = np.tensordot(op, a_ket, axes=(1, 1)).transpose(1, 0, 2)
        elif site_i < k < site_j:
            op = op_string
            a_ket_mod = np.tensordot(op, a_ket, axes=(1, 1)).transpose(1, 0, 2)
        else:
            a_ket_mod = a_ket

        tmp = np.tensordot(e, a_bra, axes=(0, 0))
        e = np.tensordot(tmp, a_ket_mod, axes=([0, 1], [0, 1]))

    return -complex(e[0, 0])


def compute_entanglement_profile(
    mps: MatrixProductState, alpha: float = 2.0
) -> EntanglementProfile:
    """
    Computes bipartite von Neumann and Renyi entanglement entropy across all cuts 1..L-1.

    Args:
        mps: MatrixProductState.
        alpha: Renyi index (alpha > 0, alpha != 1).

    Returns:
        EntanglementProfile dataclass.
    """
    cuts = list(range(1, mps.L))
    vn_entropy: List[float] = []
    renyi_entropy: List[float] = []
    schmidt_dict: Dict[int, np.ndarray] = {}

    for cut in cuts:
        s = mps.schmidt_decomposition(cut)
        schmidt_dict[cut] = s

        # Probabilities p_i = s_i^2
        p = s**2
        p = p[p > 1e-15]
        p = p / np.sum(p)

        # Von Neumann entropy: S_vN = -sum p ln(p)
        s_vn = float(-np.sum(p * np.log(p)))
        vn_entropy.append(s_vn)

        # Renyi entropy: S_alpha = (1 / (1 - alpha)) * ln(sum p^alpha)
        if abs(alpha - 1.0) < 1e-6:
            s_renyi = s_vn
        else:
            p_pow = np.sum(p**alpha)
            s_renyi = float((1.0 / (1.0 - alpha)) * np.log(max(p_pow, 1e-15)))
        renyi_entropy.append(s_renyi)

    return EntanglementProfile(
        cuts=cuts,
        von_neumann_entropy=vn_entropy,
        renyi_entropy=renyi_entropy,
        schmidt_values=schmidt_dict,
    )


def fit_cft_central_charge(
    entanglement: EntanglementProfile,
    L: int,
    cut_fraction: Tuple[float, float] = (0.2, 0.8),
    filter_dimerization: bool = False,
) -> Tuple[float, float, float]:
    """
    Fits the Conformal Field Theory (CFT) central charge c from the Calabrese-Cardy formula
    for critical 1D chains with open boundary conditions:
      S(x, L) = (c / 6) * ln[ (2L / pi) * sin(pi * x / L) ] + s_0

    Args:
        entanglement: EntanglementProfile containing von Neumann entropies.
        L: Lattice length.
        cut_fraction: Tuple (low, high) defining fraction of chain to include in linear fit,
                      avoiding non-universal boundary effects near edges.
        filter_dimerization: If True, averages adjacent bond entropies [S(x) + S(x+1)] / 2 to cancel
                             Friedel-like dimerization oscillations present in open Heisenberg chains.

    Returns:
        Tuple (central_charge_c, s_0, r_squared).
    """
    cuts = entanglement.cuts
    entropies = entanglement.von_neumann_entropy

    if filter_dimerization:
        # Filter oscillations by averaging adjacent cuts
        filtered_cuts = cuts[:-1]
        filtered_entropies = [
            0.5 * (entropies[i] + entropies[i + 1]) for i in range(len(filtered_cuts))
        ]
        active_cuts = filtered_cuts
        active_s = filtered_entropies
    else:
        active_cuts = cuts
        active_s = entropies

    x_vals = []
    s_vals = []

    for cut, s in zip(active_cuts, active_s):
        frac = cut / L
        if cut_fraction[0] <= frac <= cut_fraction[1]:
            # Conformal distance: d_chord = (2L / pi) * sin(pi * cut / L)
            d_chord = (2.0 * L / np.pi) * np.sin(np.pi * cut / L)
            log_dist = np.log(d_chord)
            x_vals.append(log_dist)
            s_vals.append(s)

    if len(x_vals) < 3:
        raise ValueError("Insufficient points for central charge linear regression.")

    x_arr = np.array(x_vals)
    y_arr = np.array(s_vals)

    # Linear fit: y = slope * x + intercept
    # slope = c / 6 -> c = 6 * slope
    cov = np.cov(x_arr, y_arr)
    slope = cov[0, 1] / cov[0, 0]
    intercept = float(np.mean(y_arr) - slope * np.mean(x_arr))

    c = float(6.0 * slope)

    # R^2 calculation
    y_pred = slope * x_arr + intercept
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    ss_res = float(np.sum((y_arr - y_pred) ** 2))
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-15 else 1.0

    return c, intercept, r_squared
