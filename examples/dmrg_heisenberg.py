"""
Demonstration: Variational DMRG Ground State Optimization of the 1D Heisenberg Spin Chain.
"""

import numpy as np

from entangle_tensor.dmrg import DMRGSolver
from entangle_tensor.mpo import MatrixProductOperator
from entangle_tensor.mps import MatrixProductState
from entangle_tensor.observables import (
    compute_entanglement_profile,
    connected_two_site_correlation,
)
from entangle_tensor.spin_operators import spin_half_operators
from entangle_tensor.types import DMRGConfig


def main() -> None:
    print("=" * 70)
    print("EntangleTensor: 1D Antiferromagnetic Heisenberg S=1/2 DMRG Ground State")
    print("=" * 70)

    L = 20
    J = 1.0
    Jz = 1.0
    print(f"Lattice Size: L = {L} sites (Open Boundary Conditions)")
    print(f"Couplings: J = {J:.2f}, Jz = {Jz:.2f}")

    # Build MPO
    mpo = MatrixProductOperator.heisenberg_xxz(L=L, J=J, Jz=Jz)

    # Initial state: alternating Neel state |0 1 0 1 ...>
    mps = MatrixProductState.neel_state(L=L, d=2)
    print("Initial Trial State: Neel state |u d u d ...>")

    # Configure DMRG solver
    config = DMRGConfig(max_sweeps=10, chi_max=32, svd_tol=1e-12, energy_tol=1e-9, verbose=True)
    solver = DMRGSolver(config)

    print("\nStarting Variational DMRG Sweeps...")
    res = solver.run(mps, mpo)

    print("\n" + "-" * 70)
    print(f"Optimization Status: {'CONVERGED' if res.converged else 'MAX SWEEPS REACHED'}")
    print(f"Completed Sweeps:    {res.num_sweeps}")
    print(f"Ground State Energy: {res.ground_state_energy:16.10f}")
    print(f"Energy per Site:     {res.energy_per_site:16.10f}")
    print(f"Bethe Limit (L->inf):{0.25 - np.log(2.0):16.10f}")
    print(f"Max Truncation Error:{res.max_truncation_error:12.4e}")
    print(f"Max Bond Dimension:  {mps.max_bond_dimension}")

    # Compute Entanglement Profile
    print("\n" + "-" * 70)
    print("Bipartite Entanglement Entropy Profile:")
    prof = compute_entanglement_profile(mps)
    print(f"{'Cut (k)':<10} {'S_vN (von Neumann)':<22} {'S_2 (Renyi-2)':<20} {'Bond Dim':<10}")
    print("-" * 62)
    for cut, s_vn, s_ren in zip(prof.cuts, prof.von_neumann_entropy, prof.renyi_entropy):
        bond_dim = len(prof.schmidt_values[cut])
        print(f"{cut:<10d} {s_vn:<22.6f} {s_ren:<20.6f} {bond_dim:<10d}")

    # Compute Spin-Spin Correlation Functions from Center Site
    center = L // 2
    ops = spin_half_operators()
    sz = ops["Sz"]
    print("\n" + "-" * 70)
    print(f"Two-Point Connected Spin Correlations <Sz({center}) Sz({center}+r)>_c:")
    print(f"{'Distance r':<12} {'Site j':<10} {'Correlation C(center, j)':<26}")
    print("-" * 50)
    for r in range(1, (L - center)):
        target = center + r
        corr = float(np.real(connected_two_site_correlation(mps, sz, sz, center, target)))
        print(f"{r:<12d} {target:<10d} {corr:<+26.8e}")

    print("=" * 70)


if __name__ == "__main__":
    main()
