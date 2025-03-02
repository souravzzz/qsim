"""
Tests for circuit analysis components.
"""

import unittest
import numpy as np
from qsim.core.qudit import Qudit
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.phase import PhaseGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.controlled import ControlledGate
from qsim.analysis.circuit_analyzer import CircuitAnalyzer


class TestCircuitAnalyzer(unittest.TestCase):
    """Test case for CircuitAnalyzer."""

    def setUp(self):
        """Set up the test case."""
        self.analyzer = CircuitAnalyzer()

    def test_analyze_circuit_complexity(self):
        """Test analyzing circuit complexity."""
        # Create a simple circuit with low entanglement
        simple_circuit = QuantumCircuit(2)
        simple_circuit.add_gate(HadamardGate(simple_circuit.qudits[0]))
        simple_circuit.add_gate(PhaseGate(simple_circuit.qudits[1], np.pi / 4))

        # Analyze the circuit
        analysis = self.analyzer.analyze_circuit(simple_circuit)

        # Check analysis results
        self.assertEqual(analysis["num_qudits"], 2)
        self.assertEqual(analysis["num_gates"], 2)
        self.assertEqual(analysis["entangling_gates"], 0)
        self.assertFalse(analysis["is_highly_entangled"])
        self.assertTrue(analysis["is_sparse"])
        self.assertFalse(analysis["requires_tensor_network"])

    def test_analyze_entangled_circuit(self):
        """Test analyzing a highly entangled circuit."""
        # Create a circuit with high entanglement
        entangled_circuit = QuantumCircuit(4)

        # Add Hadamard to all qubits
        for i in range(4):
            entangled_circuit.add_gate(HadamardGate(entangled_circuit.qudits[i]))

        # Add CNOT gates between all pairs of qubits
        for i in range(3):
            for j in range(i + 1, 4):
                x_gate = PermutationGate(entangled_circuit.qudits[j], [1, 0])
                entangled_circuit.add_gate(ControlledGate(x_gate, entangled_circuit.qudits[i], 1))

        # Analyze the circuit
        analysis = self.analyzer.analyze_circuit(entangled_circuit)

        # Check analysis results
        self.assertEqual(analysis["num_qudits"], 4)
        self.assertEqual(analysis["num_gates"], 4 + 6)  # 4 Hadamards + 6 CNOTs
        self.assertEqual(analysis["entangling_gates"], 6)
        self.assertTrue(analysis["is_highly_entangled"])
        self.assertFalse(analysis["is_sparse"])

        # For 4 qubits, tensor network might not be required yet
        # but the analyzer should recognize the high entanglement
        self.assertEqual(analysis["entanglement_ratio"], 6 / 4)  # 6 entangling gates / 4 qubits

    def test_analyze_large_sparse_circuit(self):
        """Test analyzing a large but sparse circuit."""
        # Create a large circuit with few gates
        sparse_circuit = QuantumCircuit(20)

        # Add only a few gates
        sparse_circuit.add_gate(HadamardGate(sparse_circuit.qudits[0]))
        sparse_circuit.add_gate(PhaseGate(sparse_circuit.qudits[1], np.pi / 2))
        sparse_circuit.add_gate(PermutationGate(sparse_circuit.qudits[2], [1, 0]))

        # Add one entangling gate
        x_gate = PermutationGate(sparse_circuit.qudits[1], [1, 0])
        sparse_circuit.add_gate(ControlledGate(x_gate, sparse_circuit.qudits[0], 1))

        # Analyze the circuit
        analysis = self.analyzer.analyze_circuit(sparse_circuit)

        # Check analysis results
        self.assertEqual(analysis["num_qudits"], 20)
        self.assertEqual(analysis["num_gates"], 4)
        self.assertEqual(analysis["entangling_gates"], 1)
        self.assertFalse(analysis["is_highly_entangled"])
        self.assertTrue(analysis["is_sparse"])
        self.assertTrue(analysis["requires_sparse_simulation"])

    def test_analyze_block_structure(self):
        """Test analyzing block structure of a circuit."""
        # Create a circuit with block structure
        block_circuit = QuantumCircuit(6)

        # First block: qubits 0, 1, 2
        for i in range(3):
            block_circuit.add_gate(HadamardGate(block_circuit.qudits[i]))

        # Add entanglement within first block
        for i in range(2):
            x_gate = PermutationGate(block_circuit.qudits[i + 1], [1, 0])
            block_circuit.add_gate(ControlledGate(x_gate, block_circuit.qudits[i], 1))

        # Second block: qubits 3, 4, 5
        for i in range(3, 6):
            block_circuit.add_gate(HadamardGate(block_circuit.qudits[i]))

        # Add entanglement within second block
        for i in range(3, 5):
            x_gate = PermutationGate(block_circuit.qudits[i + 1], [1, 0])
            block_circuit.add_gate(ControlledGate(x_gate, block_circuit.qudits[i], 1))

        # Analyze the circuit
        analysis = self.analyzer.analyze_circuit(block_circuit)

        # Check analysis results
        self.assertEqual(analysis["num_qudits"], 6)
        self.assertEqual(analysis["num_gates"], 6 + 4)  # 6 Hadamards + 4 CNOTs
        self.assertEqual(analysis["entangling_gates"], 4)

        # The circuit has a block structure
        self.assertTrue(analysis["has_block_structure"])

        # Check that the analyzer identifies the two blocks
        blocks = analysis.get("blocks", [])
        self.assertEqual(len(blocks), 2)

        # Check that the blocks contain the correct qubits
        block_qubits = [[qudit.index for qudit in block] for block in blocks]
        self.assertIn(set([0, 1, 2]), [set(qubits) for qubits in block_qubits])
        self.assertIn(set([3, 4, 5]), [set(qubits) for qubits in block_qubits])

    def test_analyze_mixed_dimensions(self):
        """Test analyzing a circuit with mixed qudit dimensions."""
        # Create a circuit with mixed dimensions
        mixed_circuit = QuantumCircuit(3, [2, 3, 2])

        # Add some gates
        mixed_circuit.add_gate(HadamardGate(mixed_circuit.qudits[0]))
        mixed_circuit.add_gate(PermutationGate(mixed_circuit.qudits[1], [1, 2, 0]))
        mixed_circuit.add_gate(PhaseGate(mixed_circuit.qudits[2], np.pi / 2))

        # Add an entangling gate between qubit 0 and qutrit 1
        # (This is a simplified example, as controlled gates between different dimensions
        # would need more complex implementation)
        x_gate = PermutationGate(mixed_circuit.qudits[2], [1, 0])
        mixed_circuit.add_gate(ControlledGate(x_gate, mixed_circuit.qudits[0], 1))

        # Analyze the circuit
        analysis = self.analyzer.analyze_circuit(mixed_circuit)

        # Check analysis results
        self.assertEqual(analysis["num_qudits"], 3)
        self.assertEqual(analysis["dimensions"], [2, 3, 2])
        self.assertEqual(analysis["total_dimension"], 12)  # 2 × 3 × 2
        self.assertEqual(analysis["num_gates"], 4)
        self.assertEqual(analysis["entangling_gates"], 1)
        self.assertTrue(analysis["has_mixed_dimensions"])


if __name__ == "__main__":
    unittest.main()
