#!/usr/bin/env python3
"""
Multi-Qudit Simulation Example

This example demonstrates the QSim quantum circuit simulator's ability to handle
quantum systems with qudits of different dimensions (not just qubits).
It shows how to create and simulate circuits with mixed qudit types.

In this example, we:
1. Create a circuit with qudits of different dimensions (2, 3, and 4)
2. Apply appropriate gates to these qudits
3. Simulate the circuit and analyze the results
4. Demonstrate how the simulator handles the larger state space
"""

import numpy as np
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.phase import PhaseGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.controlled import ControlledGate


def create_multi_qudit_circuit() -> QuantumCircuit:
    """
    Create a quantum circuit with qudits of different dimensions.

    Returns:
        A quantum circuit with mixed qudit types
    """
    # Create a circuit with qudits of dimensions 2 (qubit), 3 (qutrit), and 4 (ququart)
    dimensions = [2, 3, 4]
    circuit = QuantumCircuit(len(dimensions), dimensions)

    # Get the qudits
    qubit = circuit.qudits[0]  # dimension 2
    qutrit = circuit.qudits[1]  # dimension 3
    ququart = circuit.qudits[2]  # dimension 4

    # Apply Hadamard to the qubit
    h_gate = HadamardGate(qubit)
    circuit.add_gate(h_gate)

    # Apply a permutation gate to the qutrit (cycles through states 0->1->2->0)
    # Create a permutation list that shifts each state by 1 (modulo dimension)
    qutrit_perm_list = [(i + 1) % qutrit.dimension for i in range(qutrit.dimension)]
    qutrit_perm = PermutationGate(qutrit, qutrit_perm_list)  # Shift by 1
    circuit.add_gate(qutrit_perm)

    # Apply a permutation gate to the ququart (cycles through states 0->1->2->3->0)
    # Create a permutation list that shifts each state by 1 (modulo dimension)
    ququart_perm_list = [(i + 1) % ququart.dimension for i in range(ququart.dimension)]
    ququart_perm = PermutationGate(ququart, ququart_perm_list)  # Shift by 1
    circuit.add_gate(ququart_perm)

    # Apply a phase gate to the qutrit
    qutrit_phase = PhaseGate(qutrit, np.pi / 3)
    circuit.add_gate(qutrit_phase)

    # Apply a controlled permutation from qubit to qutrit
    # This will shift the qutrit state by 1 if the qubit is in state 1
    qutrit_perm_list = [(i + 1) % qutrit.dimension for i in range(qutrit.dimension)]
    controlled_perm = ControlledGate(
        PermutationGate(qutrit, qutrit_perm_list),  # Target gate
        qubit,  # Control qudit
        1,  # Control value
    )
    circuit.add_gate(controlled_perm)

    # Apply a controlled permutation from qutrit to ququart
    # This will shift the ququart state by 1 if the qutrit is in state 2
    ququart_perm_list = [(i + 1) % ququart.dimension for i in range(ququart.dimension)]
    controlled_perm2 = ControlledGate(
        PermutationGate(ququart, ququart_perm_list),  # Target gate
        qutrit,  # Control qudit
        2,  # Control value
    )
    circuit.add_gate(controlled_perm2)

    return circuit


def analyze_multi_qudit_state(state_vector: np.ndarray, dimensions: list) -> None:
    """
    Analyze and print information about a multi-qudit state.

    Args:
        state_vector: The quantum state vector
        dimensions: List of dimensions for each qudit
    """
    # Calculate the total state space size
    total_dim = np.prod(dimensions)

    # Print state vector information
    print(f"State vector size: {len(state_vector)}")
    print(f"Number of non-zero amplitudes: {np.count_nonzero(np.abs(state_vector) > 1e-10)}")

    # Print the state vector in a readable format
    print("\nState vector (non-zero amplitudes):")
    for i, amplitude in enumerate(state_vector):
        if abs(amplitude) > 1e-10:
            # Convert index to multi-qudit state representation
            qudit_states = []
            remaining_index = i
            for dim in reversed(dimensions):
                qudit_state = remaining_index % dim
                remaining_index //= dim
                qudit_states.insert(0, qudit_state)

            state_str = "".join(str(s) for s in qudit_states)
            print(f"|{state_str}⟩: {amplitude:.4f}")


def main():
    """Run the multi-qudit simulation example."""
    print("Running Multi-Qudit Simulation Example")

    # Create the multi-qudit circuit
    circuit = create_multi_qudit_circuit()

    # Print circuit information
    print(f"Created circuit with {circuit.num_qudits} qudits")
    print(f"Qudit dimensions: {circuit.dimensions}")
    print(f"Total state space size: {np.prod(circuit.dimensions)}")
    print(f"Number of gates: {len(circuit.gates)}")

    # Create a simulator
    simulator = HybridQuantumSimulator()

    # Simulate the circuit
    print("\nSimulating circuit...")
    final_state = simulator.simulate(circuit)

    # Get the state vector
    amplitudes = final_state.get_amplitudes()

    # Analyze the state
    analyze_multi_qudit_state(amplitudes, circuit.dimensions)

    # Simulate and measure
    print("\nPerforming measurements (1000 shots)...")
    results = simulator.simulate_and_measure(circuit, num_shots=1000)

    # Print measurement results
    print("\nMeasurement results:")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for outcome, count in sorted_results[:10]:  # Show top 10 results
        print(f"|{outcome}⟩: {count} shots ({count/10:.1f}%)")

    # Demonstrate how to interpret the measurement outcomes
    print("\nInterpreting measurement outcomes:")
    for outcome, count in list(sorted_results)[:3]:
        print(f"Outcome: |{outcome}⟩")
        print(f"  Qubit (d=2) state: {outcome[0]}")
        print(f"  Qutrit (d=3) state: {outcome[1]}")
        print(f"  Ququart (d=4) state: {outcome[2]}")
        print(f"  Frequency: {count/1000:.1%}")


if __name__ == "__main__":
    main()
