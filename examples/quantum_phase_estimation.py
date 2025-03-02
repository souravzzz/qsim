#!/usr/bin/env python3
"""
Quantum Phase Estimation Example

This example demonstrates how to implement the Quantum Phase Estimation (QPE) algorithm
using the QSim quantum circuit simulator. QPE is a fundamental quantum algorithm that
estimates the eigenvalue of a unitary operator.

In this example, we:
1. Create a circuit with a target qubit and several estimation qubits
2. Apply a phase to the target qubit
3. Use controlled operations to estimate the phase
4. Apply inverse QFT to read out the phase estimate
"""

import numpy as np
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.phase import PhaseGate
from qsim.gates.controlled import ControlledGate
from qsim.utils.circuit_builders import create_quantum_fourier_transform_circuit


def create_inverse_qft_circuit(num_qubits: int) -> QuantumCircuit:
    """Create an inverse QFT circuit."""
    # First create a regular QFT circuit
    qft_circuit = create_quantum_fourier_transform_circuit(num_qubits)

    # Reverse the gate order to create the inverse QFT
    inverse_circuit = QuantumCircuit(num_qubits)
    for gate in reversed(qft_circuit.gates):
        # For phase gates, we need to negate the phase
        if isinstance(gate, PhaseGate):
            inverse_gate = PhaseGate(gate.qudits[0], -gate.phase)
            inverse_circuit.add_gate(inverse_gate)
        elif isinstance(gate, ControlledGate) and isinstance(gate.target_gate, PhaseGate):
            inverse_target = PhaseGate(gate.target_gate.qudits[0], -gate.target_gate.phase)
            # Assuming the first control qudit is the one we need
            control_qudit = gate.control_qudits[0]
            # Assuming the first control value is the one we need
            control_value = gate.control_values[0]
            inverse_gate = ControlledGate(inverse_target, control_qudit, control_value)
            inverse_circuit.add_gate(inverse_gate)
        else:
            # Other gates like Hadamard are their own inverse
            inverse_circuit.add_gate(gate)

    return inverse_circuit


def create_phase_estimation_circuit(
    num_estimation_qubits: int, target_phase: float
) -> QuantumCircuit:
    """
    Create a quantum phase estimation circuit.

    Args:
        num_estimation_qubits: Number of qubits to use for phase estimation
        target_phase: The phase to estimate (in multiples of 2π)

    Returns:
        A quantum circuit implementing phase estimation
    """
    # Total qubits = estimation qubits + 1 target qubit
    total_qubits = num_estimation_qubits + 1
    circuit = QuantumCircuit(total_qubits)

    # The last qubit is our target qubit (eigenstate)
    target_qubit_index = num_estimation_qubits
    target_qubit = circuit.qudits[target_qubit_index]

    # Step 1: Initialize the target qubit to the eigenstate |1⟩
    # For this example, we'll use |1⟩ as our eigenstate
    # We can prepare |1⟩ by applying X gate to |0⟩
    from qsim.gates.permutation import PermutationGate

    x_gate = PermutationGate(target_qubit, [1, 0])  # X gate is a permutation that swaps |0⟩ and |1⟩
    circuit.add_gate(x_gate)

    # Step 2: Apply Hadamard gates to all estimation qubits
    for i in range(num_estimation_qubits):
        h_gate = HadamardGate(circuit.qudits[i])
        circuit.add_gate(h_gate)

    # Step 3: Apply controlled-U operations
    # U is a phase gate with our target phase
    for i in range(num_estimation_qubits):
        # U^(2^i) = phase gate with phase 2^i * target_phase
        power = 2**i
        phase_value = power * target_phase * 2 * np.pi
        phase_gate = PhaseGate(target_qubit, phase_value)

        # Create controlled version of this phase gate
        control_qubit = circuit.qudits[num_estimation_qubits - i - 1]
        controlled_phase = ControlledGate(phase_gate, control_qubit, 1)
        circuit.add_gate(controlled_phase)

    # Step 4: Apply inverse QFT to the estimation qubits
    inverse_qft = create_inverse_qft_circuit(num_estimation_qubits)

    # Add the inverse QFT gates to our circuit
    # We need to map the qubits from the inverse_qft circuit to our circuit
    for gate in inverse_qft.gates:
        # Map the qudits from the inverse_qft circuit to our circuit
        if isinstance(gate, HadamardGate):
            mapped_qudit = circuit.qudits[gate.qudits[0].index]
            mapped_gate = HadamardGate(mapped_qudit)
            circuit.add_gate(mapped_gate)
        elif isinstance(gate, PhaseGate):
            mapped_qudit = circuit.qudits[gate.qudits[0].index]
            mapped_gate = PhaseGate(mapped_qudit, gate.phase)
            circuit.add_gate(mapped_gate)
        elif isinstance(gate, ControlledGate):
            # Map control qudits
            mapped_controls = [circuit.qudits[q.index] for q in gate.control_qudits]
            # Map target qudits
            mapped_targets = [circuit.qudits[q.index] for q in gate.target_qudits]

            # Create mapped target gate
            if isinstance(gate.target_gate, PhaseGate):
                mapped_target_gate = PhaseGate(mapped_targets[0], gate.target_gate.phase)
            else:
                # Handle other gate types if needed
                mapped_target_gate = gate.target_gate

            # Assuming the first control qudit is the one we need
            mapped_control_qudit = mapped_controls[0]
            # Assuming the first control value is the one we need
            control_value = gate.control_values[0]

            mapped_gate = ControlledGate(mapped_target_gate, mapped_control_qudit, control_value)
            circuit.add_gate(mapped_gate)

    return circuit


def main():
    """Run the quantum phase estimation example."""
    # Parameters
    num_estimation_qubits = 5
    true_phase = 0.25  # The phase we want to estimate (1/4 in this case)

    # Create the circuit
    circuit = create_phase_estimation_circuit(num_estimation_qubits, true_phase)

    # Create a simulator
    simulator = HybridQuantumSimulator()

    # Simulate and measure
    print(f"Running Quantum Phase Estimation with {num_estimation_qubits} estimation qubits")
    print(f"True phase: {true_phase} (fraction of 2π)")

    # Simulate and measure multiple times
    num_shots = 1000
    results = simulator.simulate_and_measure(circuit, num_shots=num_shots)

    # Process and display results
    print(f"\nMeasurement results ({num_shots} shots):")

    # Sort by frequency
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for outcome, count in sorted_results[:10]:  # Show top 10 results
        # Extract just the estimation qubits (exclude the target qubit)
        estimation_bits = outcome[:-1]

        # Convert binary to decimal
        decimal_value = int(estimation_bits, 2)

        # Convert to phase estimate (divide by 2^n)
        phase_estimate = decimal_value / (2**num_estimation_qubits)

        # Calculate error
        error = abs(phase_estimate - true_phase)

        print(
            f"|{estimation_bits}⟩: {count} shots ({count/num_shots*100:.1f}%) → Phase estimate: {phase_estimate:.6f} (error: {error:.6f})"
        )


if __name__ == "__main__":
    main()
