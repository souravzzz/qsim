#!/usr/bin/env python3
"""
Quantum Error Correction Example

This example demonstrates how to implement a simple quantum error correction code
using the QSim quantum circuit simulator. We implement the 3-qubit bit flip code,
which can detect and correct a single bit flip error.

In this example, we:
1. Encode a single logical qubit into three physical qubits
2. Simulate a bit flip error on one of the qubits
3. Detect and correct the error using syndrome measurements
4. Decode the logical qubit and verify the correction worked
"""

import numpy as np
from typing import List
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.controlled import ControlledGate
from qsim.gates.multi_controlled import MultiControlledGate
from qsim.gates.phase import PhaseGate
from qsim.gates.base import Gate
from qsim.core.qudit import Qudit
import itertools


def create_encoding_circuit() -> QuantumCircuit:
    """
    Create a circuit that encodes a single logical qubit into three physical qubits.

    The encoding is:
    |0⟩_L -> |000⟩
    |1⟩_L -> |111⟩

    Returns:
        A quantum circuit for encoding
    """
    # We need 3 qubits for the code
    circuit = QuantumCircuit(3)

    # Apply Hadamard to the first qubit to create a superposition
    # This will be our logical qubit in state (|0⟩ + |1⟩)/√2
    h_gate = HadamardGate(circuit.qudits[0])
    circuit.add_gate(h_gate)

    # Encode by applying CNOT gates from the first qubit to the other qubits
    # This spreads the superposition to all three qubits
    for i in range(1, 3):
        x_gate = PermutationGate(circuit.qudits[i], [1, 0])  # X gate
        cnot = ControlledGate(x_gate, circuit.qudits[0])
        circuit.add_gate(cnot)

    return circuit


def apply_error(circuit: QuantumCircuit, error_qubit: int) -> None:
    """
    Apply a bit flip error to a specific qubit.

    Args:
        circuit: The quantum circuit
        error_qubit: Index of the qubit to apply the error to
    """
    # Apply X gate to flip the qubit
    x_gate = PermutationGate(circuit.qudits[error_qubit], [1, 0])
    circuit.add_gate(x_gate)


def create_error_detection_circuit(circuit: QuantumCircuit) -> None:
    """
    Add error detection (syndrome measurement) gates to the circuit.

    Args:
        circuit: The quantum circuit to add error detection to
    """
    # We need two ancilla qubits for syndrome measurement
    # The circuit should already have 3 qubits, so we'll add 2 more
    num_existing_qubits = circuit.num_qudits

    # Create a new circuit with the existing qudits plus two ancilla qudits
    new_dimensions = circuit.dimensions + [2, 2]  # Add two qubits
    new_circuit = QuantumCircuit(num_existing_qubits + 2, new_dimensions)

    # Copy the existing gates to the new circuit
    for gate in circuit.gates:
        new_circuit.add_gate(gate)

    # Update the circuit reference
    circuit.qudits = new_circuit.qudits
    circuit.gates = new_circuit.gates
    circuit.num_qudits = new_circuit.num_qudits
    circuit.dimensions = new_circuit.dimensions

    # Get the ancilla qubits
    ancilla1 = circuit.qudits[num_existing_qubits]
    ancilla2 = circuit.qudits[num_existing_qubits + 1]

    # Measure the parity of qubits 0 and 1 using the first ancilla
    # First, apply CNOT from qubit 0 to ancilla1
    x_gate1 = PermutationGate(ancilla1, [1, 0])
    cnot1 = ControlledGate(x_gate1, circuit.qudits[0])
    circuit.add_gate(cnot1)

    # Then, apply CNOT from qubit 1 to ancilla1
    x_gate2 = PermutationGate(ancilla1, [1, 0])
    cnot2 = ControlledGate(x_gate2, circuit.qudits[1])
    circuit.add_gate(cnot2)

    # Measure the parity of qubits 1 and 2 using the second ancilla
    # First, apply CNOT from qubit 1 to ancilla2
    x_gate3 = PermutationGate(ancilla2, [1, 0])
    cnot3 = ControlledGate(x_gate3, circuit.qudits[1])
    circuit.add_gate(cnot3)

    # Then, apply CNOT from qubit 2 to ancilla2
    x_gate4 = PermutationGate(ancilla2, [1, 0])
    cnot4 = ControlledGate(x_gate4, circuit.qudits[2])
    circuit.add_gate(cnot4)


def create_error_correction_circuit(circuit: QuantumCircuit) -> None:
    """
    Add error correction gates to the circuit based on syndrome measurements.

    Args:
        circuit: The quantum circuit to add error correction to
    """
    # Get the ancilla qubits (they are the last two qubits)
    ancilla1 = circuit.qudits[-2]
    ancilla2 = circuit.qudits[-1]

    # Apply correction based on syndrome measurement
    # Syndrome 10: Error on qubit 0
    # Syndrome 01: Error on qubit 2
    # Syndrome 11: Error on qubit 1

    # Correct qubit 0 if syndrome is 10
    x_gate0 = PermutationGate(circuit.qudits[0], [1, 0])
    correction0 = MultiControlledGate(
        x_gate0,
        [ancilla1, ancilla2],
        [1, 0],  # ancilla1=1, ancilla2=0
    )
    circuit.add_gate(correction0)

    # Correct qubit 1 if syndrome is 11
    x_gate1 = PermutationGate(circuit.qudits[1], [1, 0])
    correction1 = MultiControlledGate(
        x_gate1,
        [ancilla1, ancilla2],
        [1, 1],  # ancilla1=1, ancilla2=1
    )
    circuit.add_gate(correction1)

    # Correct qubit 2 if syndrome is 01
    x_gate2 = PermutationGate(circuit.qudits[2], [1, 0])
    correction2 = MultiControlledGate(
        x_gate2,
        [ancilla1, ancilla2],
        [0, 1],  # ancilla1=0, ancilla2=1
    )
    circuit.add_gate(correction2)


def create_decoding_circuit(circuit: QuantumCircuit) -> None:
    """
    Add decoding gates to the circuit to convert back to a single logical qubit.

    Args:
        circuit: The quantum circuit to add decoding to
    """
    # Decoding is just the reverse of encoding
    # Apply CNOT gates from the first qubit to the other qubits
    for i in range(1, 3):
        x_gate = PermutationGate(circuit.qudits[i], [1, 0])
        cnot = ControlledGate(x_gate, circuit.qudits[0])
        circuit.add_gate(cnot)


def create_full_error_correction_circuit(error_qubit: int) -> QuantumCircuit:
    """
    Create a full quantum error correction circuit.

    Args:
        error_qubit: Index of the qubit to apply an error to (0, 1, or 2)

    Returns:
        A quantum circuit implementing the full error correction protocol
    """
    # Step 1: Create encoding circuit
    circuit = create_encoding_circuit()

    # Step 2: Apply an error
    apply_error(circuit, error_qubit)

    # Step 3: Add error detection
    create_error_detection_circuit(circuit)

    # Step 4: Add error correction
    create_error_correction_circuit(circuit)

    # Step 5: Add decoding
    create_decoding_circuit(circuit)

    return circuit


def main():
    """Run the quantum error correction example."""
    # Create a simulator
    simulator = HybridQuantumSimulator()

    # Test error correction for errors on each qubit
    for error_qubit in range(3):
        print(f"\n--- Testing error correction for error on qubit {error_qubit} ---")

        # Create the circuit with an error on the specified qubit
        circuit = create_full_error_correction_circuit(error_qubit)

        # Simulate and measure
        num_shots = 1000
        results = simulator.simulate_and_measure(circuit, num_shots=num_shots)

        # Process and display results
        # We're only interested in the first qubit (the logical qubit)
        # and the last two qubits (the syndrome qubits)
        print(f"Measurement results ({num_shots} shots):")

        # Sort by frequency
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        for outcome, count in sorted_results[:5]:  # Show top 5 results
            # Extract the logical qubit state (first qubit)
            logical_state = outcome[0]

            # Extract the syndrome (last two qubits)
            syndrome = outcome[-2:]

            # Calculate the expected syndrome based on the error qubit
            expected_syndrome = ""
            if error_qubit == 0:
                expected_syndrome = "10"
            elif error_qubit == 1:
                expected_syndrome = "11"
            elif error_qubit == 2:
                expected_syndrome = "01"

            # Check if the syndrome matches the expected value
            syndrome_correct = syndrome == expected_syndrome
            syndrome_marker = "✓" if syndrome_correct else "✗"

            # Check if the logical qubit is in the expected state (should be 0 or 1 with equal probability)
            print(
                f"|{outcome}⟩: {count} shots ({count/num_shots*100:.1f}%) - Logical: {logical_state}, Syndrome: {syndrome} {syndrome_marker}"
            )

        # Calculate the probability of measuring the correct syndrome
        correct_syndrome_prob = (
            sum(count for outcome, count in results.items() if outcome[-2:] == expected_syndrome)
            / num_shots
        )
        print(f"Probability of correct syndrome: {correct_syndrome_prob:.4f}")

        # Calculate the probability of the logical qubit being in state |+⟩ (equal superposition)
        # This is approximated by the probability of measuring 0 and 1 being roughly equal
        zero_prob = (
            sum(count for outcome, count in results.items() if outcome[0] == "0") / num_shots
        )
        one_prob = sum(count for outcome, count in results.items() if outcome[0] == "1") / num_shots
        print(f"Logical qubit state probabilities: |0⟩: {zero_prob:.4f}, |1⟩: {one_prob:.4f}")


if __name__ == "__main__":
    main()
