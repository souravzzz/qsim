"""
Tests for execution components (simulator and hybrid execution manager).
"""

import unittest
import numpy as np
from qsim.core.qudit import Qudit
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.phase import PhaseGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.controlled import ControlledGate
from qsim.states.state_vector import StateVector
from qsim.states.sparse_state_vector import SparseStateVector
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.execution.hybrid_execution_manager import HybridExecutionManager
from qsim.analysis.circuit_analyzer import CircuitAnalyzer


class TestHybridExecutionManager(unittest.TestCase):
    """Test case for HybridExecutionManager."""

    def setUp(self):
        """Set up the test case."""
        self.manager = HybridExecutionManager()
        self.analyzer = CircuitAnalyzer()

    def test_determine_simulation_method(self):
        """Test determining the optimal simulation method."""
        # Create a simple circuit with low entanglement
        simple_circuit = QuantumCircuit(2)
        simple_circuit.add_gate(HadamardGate(simple_circuit.qudits[0]))

        # Should use state vector for simple circuits
        method, params = self.manager.determine_simulation_method(simple_circuit)
        self.assertEqual(method, "state_vector")

        # Create a circuit with high entanglement
        entangled_circuit = QuantumCircuit(10)
        # Add Hadamard to all qubits
        for i in range(10):
            entangled_circuit.add_gate(HadamardGate(entangled_circuit.qudits[i]))
        # Add CNOT gates between adjacent qubits
        for i in range(9):
            x_gate = PermutationGate(entangled_circuit.qudits[i + 1], [1, 0])
            entangled_circuit.add_gate(ControlledGate(x_gate, entangled_circuit.qudits[i], 1))

        # For highly entangled circuits with many qubits, should use tensor network
        method, params = self.manager.determine_simulation_method(entangled_circuit)
        self.assertEqual(method, "tensor_network")

        # Create a circuit with sparse state
        sparse_circuit = QuantumCircuit(20)  # Large number of qubits
        # Only add a few gates to keep the state sparse
        sparse_circuit.add_gate(HadamardGate(sparse_circuit.qudits[0]))
        sparse_circuit.add_gate(PermutationGate(sparse_circuit.qudits[1], [1, 0]))

        # For sparse states, should use sparse state vector
        method, params = self.manager.determine_simulation_method(sparse_circuit)
        self.assertEqual(method, "sparse_state_vector")

    def test_create_initial_state(self):
        """Test creating the initial state based on the simulation method."""
        # Test state vector method
        circuit = QuantumCircuit(2)
        state = self.manager.create_initial_state(circuit, "state_vector")
        self.assertIsInstance(state, StateVector)
        self.assertEqual(state.num_qudits, 2)

        # Test sparse state vector method
        state = self.manager.create_initial_state(circuit, "sparse_state_vector")
        self.assertIsInstance(state, SparseStateVector)
        self.assertEqual(state.num_qudits, 2)

        # Test with mixed dimensions
        circuit = QuantumCircuit(2, [2, 3])
        state = self.manager.create_initial_state(circuit, "state_vector")
        self.assertEqual(state.dimensions, [2, 3])
        self.assertEqual(state.total_dimension, 6)


class TestHybridQuantumSimulator(unittest.TestCase):
    """Test case for HybridQuantumSimulator."""

    def setUp(self):
        """Set up the test case."""
        self.simulator = HybridQuantumSimulator()

    def test_simulate(self):
        """Test simulating a quantum circuit."""
        # Create a Bell state circuit
        circuit = QuantumCircuit(2)
        circuit.add_gate(HadamardGate(circuit.qudits[0]))
        x_gate = PermutationGate(circuit.qudits[1], [1, 0])
        circuit.add_gate(ControlledGate(x_gate, circuit.qudits[0], 1))

        # Simulate the circuit
        state = self.simulator.simulate(circuit)

        # Check that we get the Bell state (|00⟩ + |11⟩)/√2
        amplitudes = state.get_amplitudes()
        self.assertEqual(len(amplitudes), 4)
        self.assertAlmostEqual(abs(amplitudes[0]), 1 / np.sqrt(2))
        self.assertAlmostEqual(abs(amplitudes[3]), 1 / np.sqrt(2))
        self.assertAlmostEqual(abs(amplitudes[1]), 0)
        self.assertAlmostEqual(abs(amplitudes[2]), 0)

    def test_simulate_and_measure(self):
        """Test simulating and measuring a quantum circuit."""
        # Create a GHZ state circuit for 3 qubits
        circuit = QuantumCircuit(3)
        circuit.add_gate(HadamardGate(circuit.qudits[0]))

        # CNOT from qubit 0 to qubit 1
        x_gate1 = PermutationGate(circuit.qudits[1], [1, 0])
        circuit.add_gate(ControlledGate(x_gate1, circuit.qudits[0], 1))

        # CNOT from qubit 0 to qubit 2
        x_gate2 = PermutationGate(circuit.qudits[2], [1, 0])
        circuit.add_gate(ControlledGate(x_gate2, circuit.qudits[0], 1))

        # Simulate and measure
        results = self.simulator.simulate_and_measure(circuit, num_shots=1000)

        # Should only get 000 and 111 with roughly equal probability
        self.assertIn("000", results)
        self.assertIn("111", results)

        # Check that other outcomes are not present or very rare
        for outcome in ["001", "010", "011", "100", "101", "110"]:
            self.assertTrue(
                results.get(outcome, 0) < 50
            )  # Allow for small statistical fluctuations

        # Check that 000 and 111 have roughly equal counts
        count_000 = results.get("000", 0)
        count_111 = results.get("111", 0)
        self.assertTrue(abs(count_000 - count_111) < 0.1 * 1000)

    def test_multi_qudit_simulation(self):
        """Test simulating a circuit with qudits of different dimensions."""
        # Create a circuit with a qubit and a qutrit
        circuit = QuantumCircuit(2, [2, 3])

        # Apply Hadamard to qubit
        circuit.add_gate(HadamardGate(circuit.qudits[0]))

        # Apply a permutation to qutrit (cycle: 0->1->2->0)
        circuit.add_gate(PermutationGate(circuit.qudits[1], [1, 2, 0]))

        # Simulate the circuit
        state = self.simulator.simulate(circuit)

        # Check dimensions
        self.assertEqual(state.dimensions, [2, 3])
        self.assertEqual(state.total_dimension, 6)

        # Check amplitudes
        amplitudes = state.get_amplitudes()
        self.assertEqual(len(amplitudes), 6)

        # Initial state |00⟩ transformed to (|01⟩ + |11⟩)/√2
        self.assertAlmostEqual(abs(amplitudes[1]), 1 / np.sqrt(2))
        self.assertAlmostEqual(abs(amplitudes[4]), 1 / np.sqrt(2))

        # All other amplitudes should be zero
        for i in [0, 2, 3, 5]:
            self.assertAlmostEqual(abs(amplitudes[i]), 0)


if __name__ == "__main__":
    unittest.main()
