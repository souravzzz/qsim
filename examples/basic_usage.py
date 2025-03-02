#!/usr/bin/env python3
"""
Basic usage example for the QSim quantum circuit simulator.

This example demonstrates how to create a custom quantum circuit,
simulate it, and measure the results.
"""

from qsim.execution.simulator import HybridQuantumSimulator
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.phase import PhaseGate
from qsim.gates.controlled import ControlledGate
from qsim.gates.permutation import PermutationGate
import numpy as np


def main():
    """Run the basic usage example."""
    # Create a 3-qubit circuit
    circuit = QuantumCircuit(3)

    # Apply Hadamard gates to all qubits
    for qudit in circuit.qudits:
        circuit.add_gate(HadamardGate(qudit))

    # Apply a phase gate to the first qubit
    circuit.add_gate(PhaseGate(circuit.qudits[0], np.pi / 4))

    # Apply a CNOT gate from the first qubit to the second qubit
    circuit.add_gate(
        ControlledGate(
            target_gate=PermutationGate(
                circuit.qudits[1], [1, 0]
            ),  # X gate permutation: |0⟩ → |1⟩, |1⟩ → |0⟩
            control_qudit=circuit.qudits[0],
            control_value=1,
        )
    )

    # Apply a CNOT gate from the second qubit to the third qubit
    circuit.add_gate(
        ControlledGate(
            target_gate=PermutationGate(
                circuit.qudits[2], [1, 0]
            ),  # X gate permutation: |0⟩ → |1⟩, |1⟩ → |0⟩
            control_qudit=circuit.qudits[1],
            control_value=1,
        )
    )

    # Create a simulator
    simulator = HybridQuantumSimulator()

    # Simulate the circuit
    state = simulator.simulate(circuit)

    # Print the state vector
    print("State vector:")
    amplitudes = state.get_amplitudes()
    for i, amplitude in enumerate(amplitudes):
        if abs(amplitude) > 0.01:  # Only print non-zero amplitudes
            # Convert index to binary representation
            binary = format(i, f"0{circuit.num_qudits}b")
            print(f"|{binary}⟩: {amplitude:.4f}")

    # Simulate and measure the circuit
    print("\nMeasurement results (1000 shots):")
    results = simulator.simulate_and_measure(circuit, num_shots=1000)

    # Print the top 5 most frequent outcomes
    for outcome, count in sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"|{outcome}⟩: {count} shots ({count/10:.1f}%)")


if __name__ == "__main__":
    main()
