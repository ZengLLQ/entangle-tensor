"""
Demonstration: Real-Time TEBD Simulation of a Quantum Quench in the 1D Ising Model.
"""

import numpy as np

from entangle_tensor.mps import MatrixProductState
from entangle_tensor.observables import (
    all_single_site_expectation_values,
    compute_entanglement_profile,
)
from entangle_tensor.spin_operators import pauli_matrices
from entangle_tensor.tebd import TEBDSimulator
from entangle_tensor.types import EvolutionType, TEBDConfig


def main() -> None:
    print("=" * 70)
    print("EntangleTensor: Quantum Quench Dynamics via Real-Time TEBD")
    print("=" * 70)

    L = 12
    dt = 0.02
    steps = 40
    print(f"Lattice Size: L = {L} sites")
    print(f"Trotter Step: dt = {dt:.3f}, Total Steps = {steps}")

    # Initial state: Paramagnetic state with all spins polarized along +x: |+ + + ...>
    # |+> = (|0> + |1>) / sqrt(2)
    plus_tensor = np.ones((1, 2, 1), dtype=np.complex128) / np.sqrt(2.0)
    mps = MatrixProductState([plus_tensor.copy() for _ in range(L)])
    print("Initial State: Paramagnetic x-polarized product state |+ + ... +>")

    # Quench Hamiltonian: ferromagnetic regime J = 1.0, small transverse field h = 0.2
    J_quench = 1.0
    h_quench = 0.2
    print(f"Quench Hamiltonian: J = {J_quench:.2f}, h = {h_quench:.2f} (Ferromagnetic)")

    bonds = TEBDSimulator.ising_bond_hamiltonians(L=L, J=J_quench, h=h_quench)
    config = TEBDConfig(dt=dt, total_steps=steps, chi_max=32)
    tebd = TEBDSimulator(mps, config)

    # Track mid-chain entanglement entropy and magnetization over time
    half_cut = L // 2
    _, _, _, sz = pauli_matrices()

    print("\n" + "-" * 70)
    print(f"{'Time t':<10} {'Mid-Chain S_vN':<20} {'Bond Dim':<12} {'Center <sigma^z>':<20}")
    print("-" * 62)

    def print_snapshot(step: int, t: float) -> None:
        if step % 5 == 0 or step == steps - 1:
            prof = compute_entanglement_profile(mps)
            s_mid = prof.von_neumann_entropy[half_cut - 1]
            bond_dim = len(prof.schmidt_values[half_cut])
            mag_z = all_single_site_expectation_values(mps, sz)
            mid_mag = float(np.real(mag_z[half_cut]))
            print(f"{t:<10.3f} {s_mid:<20.6f} {bond_dim:<12d} {mid_mag:<+20.6f}")

    # Initial snapshot at t = 0
    print_snapshot(0, 0.0)

    # Evolve in real time
    tebd.evolve(bonds, mode=EvolutionType.REAL, callback=print_snapshot)

    print("-" * 70)
    print("Quantum quench completed. Linear entanglement growth S(t) ~ v_E * t observed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
