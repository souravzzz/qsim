#!/usr/bin/env python3
"""
Grover's Search Algorithm Example

This example demonstrates how to implement Grover's search algorithm using the QSim
quantum circuit simulator. Grover's algorithm is a quantum algorithm for searching
an unstructured database with a quadratic speedup over classical algorithms.

In this example, we:
1. Create a circuit with a specified number of qubits
2. Define a "marked" state that we want to find
3. Implement the oracle that recognizes the marked state
4. Apply the diffusion operator (Grover's diffusion)
5. Iterate the oracle and diffusion steps for the optimal number of iterations
6. Measure the final state to find the marked item
"""

import numpy as np
import math
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.phase import PhaseGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.controlled import ControlledGate


def create_oracle(circuit: QuantumCircuit, marked_state: str) -> None:
    """
    Create an oracle that flips the phase of the marked state.

    Args:
        circuit: The quantum circuit to add the oracle to
        marked_state: Binary string representing the marked state
    """
    # Convert marked_state to a list of 0s and 1s
    marked_bits = [int(bit) for bit in marked_state]

    # For each qubit that should be 0 in the marked state, apply X before and after
    # This converts the problem to recognizing the |11...1⟩ state
    flip_qubits = []
    for i, bit in enumerate(marked_bits):
        if bit == 0:
            # Apply X gate to flip the qubit
            x_gate = PermutationGate(circuit.qudits[i], [1, 0])
            circuit.add_gate(x_gate)
            flip_qubits.append(i)

    # Now we need a gate that flips the phase if all qubits are 1
    # We'll use a multi-controlled Z gate

    # First, get the last qubit as the target
    target_qubit = circuit.qudits[-1]

    # All other qubits are controls
    control_qubits = circuit.qudits[:-1]

    # Apply Hadamard to the target qubit
    h_gate = HadamardGate(target_qubit)
    circuit.add_gate(h_gate)

    # Apply multi-controlled X gate (equivalent to multi-controlled NOT)
    x_gate = PermutationGate(target_qubit, [1, 0])

    # Create a multi-controlled X gate by chaining controlled gates
    if len(control_qubits) == 1:
        # Simple case: just one control qubit
        mcx = ControlledGate(x_gate, control_qubits[0], 1)
    else:
        # Multiple control qubits: create a chain of controlled gates
        # Start with the last control qubit
        current_gate = ControlledGate(x_gate, control_qubits[-1], 1)

        # Add each control qubit, working backwards
        for i in range(len(control_qubits) - 2, -1, -1):
            current_gate = ControlledGate(current_gate, control_qubits[i], 1)

        mcx = current_gate

    circuit.add_gate(mcx)

    # Apply Hadamard to the target qubit again
    circuit.add_gate(h_gate)

    # Flip the qubits back
    for i in flip_qubits:
        x_gate = PermutationGate(circuit.qudits[i], [1, 0])
        circuit.add_gate(x_gate)


def create_diffusion_operator(circuit: QuantumCircuit) -> None:
    """
    Create the diffusion operator (Grover's diffusion) that performs the reflection about the average.

    Args:
        circuit: The quantum circuit to add the diffusion operator to
    """
    # Apply Hadamard to all qubits
    for qubit in circuit.qudits:
        h_gate = HadamardGate(qubit)
        circuit.add_gate(h_gate)

    # Apply X to all qubits
    for qubit in circuit.qudits:
        x_gate = PermutationGate(qubit, [1, 0])
        circuit.add_gate(x_gate)

    # Apply multi-controlled Z gate
    # We'll implement it as: H on target, multi-controlled X, H on target

    # First, get the last qubit as the target
    target_qubit = circuit.qudits[-1]

    # All other qubits are controls
    control_qubits = circuit.qudits[:-1]

    # Apply Hadamard to the target qubit
    h_gate = HadamardGate(target_qubit)
    circuit.add_gate(h_gate)

    # Apply multi-controlled X gate
    x_gate = PermutationGate(target_qubit, [1, 0])

    # Create a multi-controlled X gate by chaining controlled gates
    if len(control_qubits) == 1:
        # Simple case: just one control qubit
        mcx = ControlledGate(x_gate, control_qubits[0], 1)
    else:
        # Multiple control qubits: create a chain of controlled gates
        # Start with the last control qubit
        current_gate = ControlledGate(x_gate, control_qubits[-1], 1)

        # Add each control qubit, working backwards
        for i in range(len(control_qubits) - 2, -1, -1):
            current_gate = ControlledGate(current_gate, control_qubits[i], 1)

        mcx = current_gate

    circuit.add_gate(mcx)

    # Apply Hadamard to the target qubit again
    circuit.add_gate(h_gate)

    # Apply X to all qubits again
    for qubit in circuit.qudits:
        x_gate = PermutationGate(qubit, [1, 0])
        circuit.add_gate(x_gate)

    # Apply Hadamard to all qubits again
    for qubit in circuit.qudits:
        h_gate = HadamardGate(qubit)
        circuit.add_gate(h_gate)


def create_grovers_circuit(
    num_qubits: int, marked_state: str, num_iterations: int
) -> QuantumCircuit:
    """
    Create a circuit implementing Grover's search algorithm.

    Args:
        num_qubits: Number of qubits in the circuit
        marked_state: Binary string representing the marked state to find
        num_iterations: Number of Grover iterations to perform

    Returns:
        A quantum circuit implementing Grover's algorithm
    """
    circuit = QuantumCircuit(num_qubits)

    # Step 1: Initialize with Hadamard gates
    for qubit in circuit.qudits:
        h_gate = HadamardGate(qubit)
        circuit.add_gate(h_gate)

    # Step 2: Apply Grover iterations
    for _ in range(num_iterations):
        # Apply oracle
        create_oracle(circuit, marked_state)

        # Apply diffusion operator
        create_diffusion_operator(circuit)

    return circuit


def calculate_optimal_iterations(num_qubits: int) -> int:
    """
    Calculate the optimal number of Grover iterations for a given number of qubits.

    Args:
        num_qubits: Number of qubits in the circuit

    Returns:
        The optimal number of iterations
    """
    N = 2**num_qubits  # Size of the search space
    return int(np.round(np.pi / 4 * np.sqrt(N)))


def main():
    """Run the Grover's search algorithm example."""
    # Parameters
    num_qubits = 4

    # Choose a random marked state
    np.random.seed(42)  # For reproducibility
    marked_state_int = np.random.randint(0, 2**num_qubits)
    marked_state = format(marked_state_int, f"0{num_qubits}b")

    # Calculate optimal number of iterations
    optimal_iterations = calculate_optimal_iterations(num_qubits)

    print(f"Running Grover's search algorithm with {num_qubits} qubits")
    print(f"Search space size: {2**num_qubits}")
    print(f"Marked state: |{marked_state}⟩")
    print(f"Optimal number of iterations: {optimal_iterations}")

    # Create the circuit
    circuit = create_grovers_circuit(num_qubits, marked_state, optimal_iterations)

    # Create a simulator
    simulator = HybridQuantumSimulator()

    # Simulate and measure
    num_shots = 1000
    results = simulator.simulate_and_measure(circuit, num_shots=num_shots)

    # Process and display results
    print(f"\nMeasurement results ({num_shots} shots):")

    # Sort by frequency
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for outcome, count in sorted_results[:5]:  # Show top 5 results
        is_marked = outcome == marked_state
        marker = "✓" if is_marked else " "
        print(f"|{outcome}⟩: {count} shots ({count/num_shots*100:.1f}%) {marker}")

    # Calculate success probability
    if marked_state in results:
        success_prob = results[marked_state] / num_shots
        print(f"\nSuccess probability: {success_prob:.4f}")
    else:
        print("\nMarked state was not found in any measurements!")


if __name__ == "__main__":
    main()
