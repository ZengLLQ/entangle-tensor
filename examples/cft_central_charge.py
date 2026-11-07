"""
Demonstration: Conformal Field Theory (CFT) Central Charge Extraction from Entanglement Scaling.
"""

from entangle_tensor.dmrg import DMRGSolver
from entangle_tensor.mpo import MatrixProductOperator
from entangle_tensor.mps import MatrixProductState
from entangle_tensor.observables import compute_entanglement_profile, fit_cft_central_charge
from entangle_tensor.types import DMRGConfig


def main() -> None:
    print("=" * 75)
    print("EntangleTensor: CFT Central Charge c Extraction via Calabrese-Cardy Scaling")
    print("=" * 75)

    L = 24
    print(f"Lattice Size: L = {L} sites (Open Boundary Conditions)\n")

    # 1. Critical 1D Transverse-Field Ising Model (h = J = 1.0) -> Expected c = 1/2 = 0.5
    print("1. Critical 1D Transverse-Field Ising Model (J = 1.0, h = 1.0):")
    mpo_ising = MatrixProductOperator.transverse_field_ising(L=L, J=1.0, h=1.0)
    mps_ising = MatrixProductState.random(L=L, d=2, chi=4, seed=42)
    config = DMRGConfig(max_sweeps=8, chi_max=24, energy_tol=1e-8)

    res_ising = DMRGSolver(config).run(mps_ising, mpo_ising)
    prof_ising = compute_entanglement_profile(mps_ising)
    c_ising, s0_ising, r2_ising = fit_cft_central_charge(prof_ising, L=L, cut_fraction=(0.2, 0.8))

    print(f"   Ground State Energy:   {res_ising.ground_state_energy:16.8f}")
    print(f"   Extracted c:           {c_ising:16.4f} (Theoretical: 0.5000)")
    print(f"   Non-universal s0:      {s0_ising:16.4f}")
    print(f"   Regression Quality R2: {r2_ising:16.4f}")

    # 2. Critical 1D Heisenberg XXX Model (J = 1.0, Jz = 1.0) -> Expected c = 1.0
    print("\n2. Critical 1D Spin-1/2 Heisenberg XXX Chain (J = 1.0, Jz = 1.0):")
    mpo_xxx = MatrixProductOperator.heisenberg_xxz(L=L, J=1.0, Jz=1.0)
    mps_xxx = MatrixProductState.neel_state(L=L, d=2)

    res_xxx = DMRGSolver(config).run(mps_xxx, mpo_xxx)
    prof_xxx = compute_entanglement_profile(mps_xxx)
    c_xxx, s0_xxx, r2_xxx = fit_cft_central_charge(
        prof_xxx, L=L, cut_fraction=(0.2, 0.8), filter_dimerization=True
    )

    print(f"   Ground State Energy:   {res_xxx.ground_state_energy:16.8f}")
    print(f"   Extracted c (filtered):{c_xxx:16.4f} (Theoretical: 1.0000)")
    print(f"   Non-universal s0:      {s0_xxx:16.4f}")
    print(f"   Regression Quality R2: {r2_xxx:16.4f}")

    print("=" * 75)


if __name__ == "__main__":
    main()
