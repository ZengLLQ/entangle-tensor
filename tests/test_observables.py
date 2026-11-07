"""
Tests for quantum observables, entanglement entropy, and correlation functions.
"""

import numpy as np
import pytest

from entangle_tensor.dmrg import DMRGSolver
from entangle_tensor.mpo import MatrixProductOperator
from entangle_tensor.mps import MatrixProductState
from entangle_tensor.observables import (
    all_single_site_expectation_values,
    compute_entanglement_profile,
    connected_two_site_correlation,
    fit_cft_central_charge,
    string_order_parameter,
    two_site_correlation_function,
)
from entangle_tensor.spin_operators import spin_half_operators, spin_one_operators
from entangle_tensor.types import DMRGConfig


def test_single_site_magnetization() -> None:
    L = 4
    ops = spin_half_operators()
    sz = ops["Sz"]

    # State |up up up up> (state index 0)
    ferro_up = MatrixProductState.product_state([0, 0, 0, 0], d=2)
    mag_up = all_single_site_expectation_values(ferro_up, sz)
    assert np.allclose(mag_up, 0.5 * np.ones(L))

    # State |up down up down>
    neel = MatrixProductState.neel_state(L=L, d=2)
    mag_neel = all_single_site_expectation_values(neel, sz)
    expected_neel = np.array([0.5, -0.5, 0.5, -0.5])
    assert np.allclose(mag_neel, expected_neel)


def test_two_site_correlation_product_state() -> None:
    L = 4
    ops = spin_half_operators()
    sz = ops["Sz"]

    # In an unentangled product state, connected correlation <A B> - <A><B> must be zero
    neel = MatrixProductState.neel_state(L=L, d=2)
    corr = connected_two_site_correlation(neel, sz, sz, 0, 1)
    assert np.isclose(corr, 0.0, atol=1e-12)

    # Unconnected correlation <Sz(0) Sz(1)> = (0.5) * (-0.5) = -0.25
    raw_corr = two_site_correlation_function(neel, sz, sz, 0, 1)
    assert np.isclose(raw_corr, -0.25, atol=1e-12)


def test_entanglement_profile_product_vs_entangled() -> None:
    L = 6
    # Product state has zero entanglement entropy everywhere
    prod_state = MatrixProductState.product_state([0] * L, d=2)
    prof_prod = compute_entanglement_profile(prod_state)
    for s_vn in prof_prod.von_neumann_entropy:
        assert np.isclose(s_vn, 0.0, atol=1e-12)
    for s_ren in prof_prod.renyi_entropy:
        assert np.isclose(s_ren, 0.0, atol=1e-12)

    # Entangled state has strictly positive entanglement entropy
    entangled = MatrixProductState.random(L=L, d=2, chi=4, seed=42)
    prof_ent = compute_entanglement_profile(entangled)
    for s_vn in prof_ent.von_neumann_entropy:
        assert s_vn > 0.0


def test_haldane_topological_string_order() -> None:
    L = 6
    ops = spin_one_operators()
    sz = ops["Sz"]

    # For spin-1, exp(i * pi * Sz) has eigenvalues exp(i pi 1) = -1, exp(0) = 1, exp(-i pi) = -1
    # Matrix: diag(-1, 1, -1)
    exp_pi_sz = np.diag([-1.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j])

    # 1. Trivial product state |0 0 0 ... 0> (where 0 is Sz=0 state, index 1)
    # String order must vanish
    trivial_state = MatrixProductState.product_state([1] * L, d=3)
    val_trivial = string_order_parameter(trivial_state, sz, exp_pi_sz, 0, L - 1)
    assert np.isclose(val_trivial, 0.0, atol=1e-10)

    # 2. Spin-1 Haldane ground state
    mpo = MatrixProductOperator.haldane_spin_one(L=L, J=1.0, D=0.0)
    mps = MatrixProductState.random(L=L, d=3, chi=6, seed=88)
    config = DMRGConfig(max_sweeps=8, chi_max=16, energy_tol=1e-8)
    DMRGSolver(config).run(mps, mpo)

    # String order in Haldane phase is strictly positive (non-zero)
    val_haldane = string_order_parameter(mps, sz, exp_pi_sz, 1, 4)
    assert np.real(val_haldane) > 0.05


def test_cft_fit_formula() -> None:
    # Synthetic entanglement profile with known central charge c = 0.5 (Ising)
    L = 20
    c_true = 0.5
    s0_true = 0.2

    cuts = list(range(1, L))
    vn = []
    for x in cuts:
        d_chord = (2.0 * L / np.pi) * np.sin(np.pi * x / L)
        vn.append(float((c_true / 6.0) * np.log(d_chord) + s0_true))

    from entangle_tensor.types import EntanglementProfile

    synthetic_prof = EntanglementProfile(cuts=cuts, von_neumann_entropy=vn, renyi_entropy=vn)

    c_fit, s0_fit, r2 = fit_cft_central_charge(synthetic_prof, L=L, cut_fraction=(0.2, 0.8))
    assert c_fit == pytest.approx(c_true, abs=1e-5)
    assert s0_fit == pytest.approx(s0_true, abs=1e-5)
    assert r2 > 0.999
