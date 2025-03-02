#!/usr/bin/env python3
"""
Hybrid Quantum Circuit Simulator

This implements a general-purpose quantum circuit simulator that efficiently handles
all types of circuits using an adaptive hybrid simulation architecture that integrates:
1. Block Simulation (for structured circuits with localized operations)
2. Tensor Network Methods (for circuits with high entanglement)
3. Sparse Representation & Hash Maps (for tracking nonzero state amplitudes)
4. Parallelization and GPU Acceleration (for efficient handling of large systems)
"""

from qsim.execution.simulator import HybridQuantumSimulator
from qsim.core.circuit import QuantumCircuit
from qsim.gates.permutation import PermutationGate
from qsim.utils.circuit_builders import (
    create_bell_state_circuit,
    create_ghz_state_circuit,
    create_quantum_fourier_transform_circuit,
)


def main():
    """Main function demonstrating the use of the simulator."""
    # Create a simulator
    simulator = HybridQuantumSimulator()

    # Bell state example
    print("\n=== Bell State Example ===")
    bell_circuit = create_bell_state_circuit()
    bell_results = simulator.simulate_and_measure(bell_circuit, num_shots=1000)

    print("Bell state measurements:")
    for outcome, count in sorted(bell_results.items()):
        print(f"  |{outcome}⟩: {count} shots ({count/10:.1f}%)")

    # GHZ state example
    print("\n=== GHZ State Example ===")
    ghz_circuit = create_ghz_state_circuit(5)
    ghz_results = simulator.simulate_and_measure(ghz_circuit, num_shots=1000)

    print("GHZ state measurements (top 5):")
    for outcome, count in sorted(ghz_results.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  |{outcome}⟩: {count} shots ({count/10:.1f}%)")

    # QFT example
    print("\n=== Quantum Fourier Transform Example ===")

    # Create an input state with only |1⟩ amplitude
    qft_circuit = QuantumCircuit(4)

    # Add X gate to the first qubit to get |1000⟩
    x_gate = PermutationGate(qft_circuit.qudits[0], [1, 0])
    qft_circuit.add_gate(x_gate)

    # Add the QFT gates
    qft_gates = create_quantum_fourier_transform_circuit(4).gates
    for gate in qft_gates:
        qft_circuit.add_gate(gate)

    qft_results = simulator.simulate_and_measure(qft_circuit, num_shots=1000)

    print("QFT state measurements (top 5):")
    for outcome, count in sorted(qft_results.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  |{outcome}⟩: {count} shots ({count/10:.1f}%)")


if __name__ == "__main__":
    main()
