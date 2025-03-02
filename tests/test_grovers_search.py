"""
Test for Grover's search algorithm implementation.

This test suite verifies that the Grover's search algorithm implementation
produces results that match theoretical expectations.
"""

import unittest
import numpy as np
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


def calculate_theoretical_success_probability(num_qubits: int, num_iterations: int) -> float:
    """
    Calculate the theoretical success probability of Grover's algorithm.

    Args:
        num_qubits: Number of qubits in the circuit
        num_iterations: Number of Grover iterations performed

    Returns:
        The theoretical success probability
    """
    N = 2**num_qubits
    theta = np.arcsin(1 / np.sqrt(N))
    return np.sin((2 * num_iterations + 1) * theta) ** 2


class TestGroversSearch(unittest.TestCase):
    """Test case for Grover's search algorithm."""

    def setUp(self):
        """Set up the test case."""
        self.simulator = HybridQuantumSimulator()

    def test_initialization_state(self):
        """Test that the initialization creates an equal superposition."""
        num_qubits = 3
        N = 2**num_qubits

        # Create a circuit with just initialization (Hadamard on all qubits)
        circuit = QuantumCircuit(num_qubits)
        for qubit in circuit.qudits:
            h_gate = HadamardGate(qubit)
            circuit.add_gate(h_gate)

        # Simulate the circuit
        state = self.simulator.simulate(circuit)
        amplitudes = state.get_amplitudes()

        # Check that we have the correct number of amplitudes
        self.assertEqual(len(amplitudes), N)

        # Check that all amplitudes have equal magnitude 1/sqrt(N)
        expected_amplitude = 1.0 / np.sqrt(N)
        for amp in amplitudes:
            self.assertAlmostEqual(abs(amp), expected_amplitude, places=6)

    def test_oracle_phase_flip(self):
        """Test that the oracle correctly flips the phase of the marked state."""
        num_qubits = 2
        marked_state = "10"  # |10⟩

        # Create a circuit with initialization
        circuit = QuantumCircuit(num_qubits)
        for qubit in circuit.qudits:
            h_gate = HadamardGate(qubit)
            circuit.add_gate(h_gate)

        # Apply the oracle
        create_oracle(circuit, marked_state)

        # Simulate the circuit
        state = self.simulator.simulate(circuit)
        amplitudes = state.get_amplitudes()

        # The marked state should have its phase flipped (negative amplitude)
        marked_index = int(marked_state, 2)

        # Check that the amplitude for the marked state has a negative sign
        # (allowing for global phase differences)
        non_marked_indices = [i for i in range(2**num_qubits) if i != marked_index]

        # All non-marked states should have the same phase
        reference_phase = np.angle(amplitudes[non_marked_indices[0]])
        marked_phase = np.angle(amplitudes[marked_index])

        # The phase difference should be approximately π (allowing for numerical precision)
        phase_diff = abs((marked_phase - reference_phase + np.pi) % (2 * np.pi) - np.pi)
        self.assertAlmostEqual(phase_diff, np.pi, places=6)

    def test_theoretical_probability_one_iteration(self):
        """Test that one iteration of Grover's algorithm matches theoretical probability."""
        # Use 3 qubits for a manageable test
        num_qubits = 3
        marked_state = "101"  # |101⟩

        # Create a circuit with one Grover iteration
        circuit = create_grovers_circuit(num_qubits, marked_state, 1)

        # Simulate the circuit
        state = self.simulator.simulate(circuit)

        # Calculate the probability of measuring the marked state
        marked_index = int(marked_state, 2)
        amplitudes = state.get_amplitudes()
        measured_prob = abs(amplitudes[marked_index]) ** 2

        # Calculate the theoretical probability
        theoretical_prob = calculate_theoretical_success_probability(num_qubits, 1)

        # Check that the measured probability matches the theoretical probability
        self.assertAlmostEqual(measured_prob, theoretical_prob, places=6)

    def test_optimal_iterations(self):
        """Test that the optimal number of iterations gives the highest success probability."""
        # Use 3 qubits for a manageable test
        num_qubits = 3
        marked_state = "110"  # |110⟩

        # Calculate the optimal number of iterations
        optimal_iterations = calculate_optimal_iterations(num_qubits)

        # Create circuits with different numbers of iterations
        circuits = {
            i: create_grovers_circuit(num_qubits, marked_state, i)
            for i in range(optimal_iterations + 2)  # Test up to optimal+1
        }

        # Simulate each circuit and calculate the success probability
        probs = {}
        for i, circuit in circuits.items():
            state = self.simulator.simulate(circuit)
            amplitudes = state.get_amplitudes()
            marked_index = int(marked_state, 2)
            probs[i] = abs(amplitudes[marked_index]) ** 2

        # The optimal number of iterations should give the highest probability
        # (or at least very close to it)
        optimal_prob = probs[optimal_iterations]

        # Check that the optimal probability is close to 1
        self.assertGreater(optimal_prob, 0.9)

        # Check that the optimal probability is higher than or equal to
        # the probability with fewer iterations
        for i in range(optimal_iterations):
            self.assertGreaterEqual(optimal_prob, probs[i])

    def test_probability_scaling_with_iterations(self):
        """Test that the success probability follows the expected pattern as iterations increase."""
        # Use 2 qubits for a simple test
        num_qubits = 2
        marked_state = "01"  # |01⟩
        N = 2**num_qubits

        # Test for several iterations
        max_iterations = 3

        for iterations in range(max_iterations + 1):
            # Create and simulate the circuit
            circuit = create_grovers_circuit(num_qubits, marked_state, iterations)
            state = self.simulator.simulate(circuit)

            # Calculate the measured probability
            marked_index = int(marked_state, 2)
            amplitudes = state.get_amplitudes()
            measured_prob = abs(amplitudes[marked_index]) ** 2

            # Calculate the theoretical probability
            theta = np.arcsin(1 / np.sqrt(N))
            theoretical_prob = np.sin((2 * iterations + 1) * theta) ** 2

            # Check that the measured probability matches the theoretical probability
            self.assertAlmostEqual(measured_prob, theoretical_prob, places=6)

    def test_multiple_marked_states(self):
        """Test Grover's algorithm with multiple marked states (not implemented in this example)."""
        # This is a placeholder for a more advanced test
        # In a real implementation, you would need to modify the oracle to mark multiple states
        pass


if __name__ == "__main__":
    unittest.main()
