"""
Benchmark tests for DMRG scaling, thermodynamic limits, and conformal field theory central charges.
"""

import numpy as np

from entangle_tensor.dmrg import DMRGSolver
from entangle_tensor.mpo import MatrixProductOperator
from entangle_tensor.mps import MatrixProductState
from entangle_tensor.observables import compute_entanglement_profile, fit_cft_central_charge
from entangle_tensor.types import DMRGConfig


def test_heisenberg_thermodynamic_limit_convergence() -> None:
    """
    For the 1D infinite spin-1/2 Heisenberg XXX antiferromagnet, the exact Bethe Ansatz
    ground state energy per site in the thermodynamic limit is:
      e_0 = 1/4 - ln(2) approx -0.44314718

    For finite open chains, e(L) approaches e_0 as 1/L.
    """
    L = 16
    mpo = MatrixProductOperator.heisenberg_xxz(L=L, J=1.0, Jz=1.0)
    mps = MatrixProductState.neel_state(L=L, d=2)

    config = DMRGConfig(max_sweeps=8, chi_max=24, energy_tol=1e-8)
    solver = DMRGSolver(config)
    res = solver.run(mps, mpo)

    e_per_site = res.energy_per_site
    e_bethe = 0.25 - np.log(2.0)  # -0.443147...

    # Finite open chains have boundary correction ~ -0.42.. for L=16
    assert abs(e_per_site - e_bethe) < 0.03
    assert res.converged


def test_cft_central_charge_ising_critical() -> None:
    """
    At the quantum critical point h = J = 1 of the 1D Transverse-Field Ising model,
    the low-energy effective field theory is a 2D conformal field theory with central charge c = 1/2 = 0.5.
    """
    L = 20
    J = 1.0
    h = 1.0
    mpo = MatrixProductOperator.transverse_field_ising(L=L, J=J, h=h)
    mps = MatrixProductState.random(L=L, d=2, chi=4, seed=42)

    config = DMRGConfig(max_sweeps=10, chi_max=24, energy_tol=1e-9)
    res = DMRGSolver(config).run(mps, mpo)
    assert res.converged

    profile = compute_entanglement_profile(mps)
    c_fit, s0, r2 = fit_cft_central_charge(profile, L=L, cut_fraction=(0.2, 0.8))

    # Must fit c approx 0.5 within 10%
    assert abs(c_fit - 0.5) < 0.08
    assert r2 > 0.95
