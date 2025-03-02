"""
Tests for circuit builder utility functions.
"""

import unittest
import numpy as np
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.phase import PhaseGate
from qsim.gates.controlled import ControlledGate
from qsim.utils.circuit_builders import (
    create_bell_state_circuit,
    create_ghz_state_circuit,
    create_quantum_fourier_transform_circuit,
)
from qsim.execution.simulator import HybridQuantumSimulator


class TestCircuitBuilders(unittest.TestCase):
    """Test case for circuit builder utility functions."""

    def setUp(self):
        """Set up the test case."""
        self.simulator = HybridQuantumSimulator()

    def test_bell_state_circuit_structure(self):
        """Test the structure of the Bell state circuit."""
        circuit = create_bell_state_circuit()

        # Check circuit properties
        self.assertEqual(circuit.num_qudits, 2)
        self.assertEqual(len(circuit.gates), 2)

        # Check first gate is Hadamard on qubit 0
        self.assertIsInstance(circuit.gates[0], HadamardGate)
        self.assertEqual(circuit.gates[0].qudit, circuit.qudits[0])

        # Check second gate is CNOT (Controlled-X)
        self.assertIsInstance(circuit.gates[1], ControlledGate)
        self.assertEqual(circuit.gates[1].control_qudit, circuit.qudits[0])
        self.assertEqual(circuit.gates[1].control_value, 1)

        # Check target gate of CNOT is X (permutation)
        target_gate = circuit.gates[1].target_gate
        self.assertIsInstance(target_gate, PermutationGate)
        self.assertEqual(target_gate.qudit, circuit.qudits[1])
        self.assertEqual(target_gate.permutation, [1, 0])

    def test_bell_state_circuit_simulation(self):
        """Test that the Bell state circuit produces the correct quantum state."""
        circuit = create_bell_state_circuit()
        state = self.simulator.simulate(circuit)

        # Bell state should be (|00⟩ + |11⟩)/√2
        amplitudes = state.get_amplitudes()
        self.assertEqual(len(amplitudes), 4)

        # Check amplitudes match Bell state
        self.assertAlmostEqual(abs(amplitudes[0]), 1 / np.sqrt(2))
        self.assertAlmostEqual(abs(amplitudes[3]), 1 / np.sqrt(2))
        self.assertAlmostEqual(abs(amplitudes[1]), 0)
        self.assertAlmostEqual(abs(amplitudes[2]), 0)

        # Check that the phase relationship is correct (both terms have same phase)
        # This assumes the convention that the first term is real and positive
        if abs(amplitudes[0]) > 0:
            phase0 = np.angle(amplitudes[0])
            phase3 = np.angle(amplitudes[3])
            self.assertAlmostEqual(phase0, phase3, places=5)

    def test_ghz_state_circuit_structure(self):
        """Test the structure of the GHZ state circuit with different qubit counts."""
        # Test with 3 qubits
        circuit3 = create_ghz_state_circuit(3)
        self.assertEqual(circuit3.num_qudits, 3)
        self.assertEqual(len(circuit3.gates), 3)  # 1 Hadamard + 2 CNOTs

        # Test with 5 qubits
        circuit5 = create_ghz_state_circuit(5)
        self.assertEqual(circuit5.num_qudits, 5)
        self.assertEqual(len(circuit5.gates), 5)  # 1 Hadamard + 4 CNOTs

        # Check first gate is always Hadamard on qubit 0
        self.assertIsInstance(circuit5.gates[0], HadamardGate)
        self.assertEqual(circuit5.gates[0].qudit, circuit5.qudits[0])

        # Check that all other gates are CNOTs in sequence
        for i in range(1, 5):
            self.assertIsInstance(circuit5.gates[i], ControlledGate)
            self.assertEqual(circuit5.gates[i].control_qudit, circuit5.qudits[i - 1])
            self.assertEqual(circuit5.gates[i].control_value, 1)

            target_gate = circuit5.gates[i].target_gate
            self.assertIsInstance(target_gate, PermutationGate)
            self.assertEqual(target_gate.qudit, circuit5.qudits[i])
            self.assertEqual(target_gate.permutation, [1, 0])

    def test_ghz_state_circuit_simulation(self):
        """Test that the GHZ state circuit produces the correct quantum state."""
        # Test with 3 qubits
        circuit = create_ghz_state_circuit(3)
        state = self.simulator.simulate(circuit)

        # GHZ state should be (|000⟩ + |111⟩)/√2
        amplitudes = state.get_amplitudes()
        self.assertEqual(len(amplitudes), 8)

        # Check amplitudes match GHZ state
        self.assertAlmostEqual(abs(amplitudes[0]), 1 / np.sqrt(2))
        self.assertAlmostEqual(abs(amplitudes[7]), 1 / np.sqrt(2))

        # All other amplitudes should be zero
        for i in range(1, 7):
            self.assertAlmostEqual(abs(amplitudes[i]), 0)

        # Test with 4 qubits
        circuit4 = create_ghz_state_circuit(4)
        state4 = self.simulator.simulate(circuit4)

        # 4-qubit GHZ state should be (|0000⟩ + |1111⟩)/√2
        amplitudes4 = state4.get_amplitudes()
        self.assertEqual(len(amplitudes4), 16)

        # Check amplitudes match 4-qubit GHZ state
        self.assertAlmostEqual(abs(amplitudes4[0]), 1 / np.sqrt(2))
        self.assertAlmostEqual(abs(amplitudes4[15]), 1 / np.sqrt(2))

        # All other amplitudes should be zero
        for i in range(1, 15):
            self.assertAlmostEqual(abs(amplitudes4[i]), 0)

    def test_qft_circuit_structure(self):
        """Test the structure of the Quantum Fourier Transform circuit."""
        # Test with 3 qubits
        circuit = create_quantum_fourier_transform_circuit(3)
        self.assertEqual(circuit.num_qudits, 3)

        # QFT on 3 qubits should have:
        # - 3 Hadamard gates
        # - 3 controlled phase gates (1 for qubit 0->1, 1 for 0->2, 1 for 1->2)
        # - 3 CNOT gates for the SWAP (to swap qubit 0 and 2)
        expected_gates = 3 + 3 + 3
        self.assertEqual(len(circuit.gates), expected_gates)

        # First gate should be Hadamard on qubit 0
        self.assertIsInstance(circuit.gates[0], HadamardGate)
        self.assertEqual(circuit.gates[0].qudit, circuit.qudits[0])

        # Check that we have the right number of each gate type
        hadamard_count = 0
        controlled_phase_count = 0
        swap_cnot_count = 0

        for gate in circuit.gates:
            if isinstance(gate, HadamardGate):
                hadamard_count += 1
            elif isinstance(gate, ControlledGate) and isinstance(gate.target_gate, PhaseGate):
                controlled_phase_count += 1
            elif isinstance(gate, ControlledGate) and isinstance(gate.target_gate, PermutationGate):
                swap_cnot_count += 1

        self.assertEqual(hadamard_count, 3)
        self.assertEqual(controlled_phase_count, 3)
        self.assertEqual(swap_cnot_count, 3)

    def test_qft_circuit_simulation_simple(self):
        """Test QFT circuit with a simple input state."""
        # Create a 2-qubit QFT circuit
        circuit = create_quantum_fourier_transform_circuit(2)

        # Prepare input state |00⟩ (already the default)
        # QFT on |00⟩ should give equal superposition of all states
        state = self.simulator.simulate(circuit)
        amplitudes = state.get_amplitudes()

        # All amplitudes should have equal magnitude 1/2
        for amp in amplitudes:
            self.assertAlmostEqual(abs(amp), 0.5)

        # Check that phases follow the QFT pattern
        # For |00⟩ input, all amplitudes should have the same phase
        phase0 = np.angle(amplitudes[0])
        for i in range(1, 4):
            self.assertAlmostEqual(np.angle(amplitudes[i]), phase0, places=5)

    def test_qft_circuit_edge_cases(self):
        """Test edge cases for the QFT circuit."""
        # Test with 1 qubit (should be equivalent to Hadamard)
        circuit1 = create_quantum_fourier_transform_circuit(1)
        self.assertEqual(circuit1.num_qudits, 1)
        self.assertEqual(len(circuit1.gates), 1)
        self.assertIsInstance(circuit1.gates[0], HadamardGate)

        # Test with 4 qubits (larger circuit)
        circuit4 = create_quantum_fourier_transform_circuit(4)
        self.assertEqual(circuit4.num_qudits, 4)

        # Count expected gates:
        # - 4 Hadamard gates
        # - 6 controlled phase gates (1+2+3)
        # - 6 CNOT gates for the SWAPs (2 qubits to swap = 3 CNOTs per swap * 2)
        expected_gates = 4 + 6 + 6
        self.assertEqual(len(circuit4.gates), expected_gates)


if __name__ == "__main__":
    unittest.main()
