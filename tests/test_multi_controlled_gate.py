"""
Tests for multi-controlled gate implementation.
"""

import unittest
import numpy as np
from qsim.core.qudit import Qudit
from qsim.gates.permutation import PermutationGate
from qsim.gates.hadamard import HadamardGate
from qsim.gates.multi_controlled import MultiControlledGate
from qsim.states.state_vector import StateVector


class TestMultiControlledGate(unittest.TestCase):
    """Test case for multi-controlled gate implementation."""

    def test_multi_controlled_gate_properties(self):
        """Test basic properties of multi-controlled gates."""
        # Create qubits
        qubit0 = Qudit(0)
        qubit1 = Qudit(1)
        qubit2 = Qudit(2)

        # Create a target gate (X gate on qubit2)
        x_gate = PermutationGate(qubit2, [1, 0])

        # Create a multi-controlled gate (controlled on qubit0=1 and qubit1=0)
        mcx_gate = MultiControlledGate(x_gate, [qubit0, qubit1], [1, 0])

        # Check properties
        self.assertEqual(mcx_gate.name, "MCPerm")
        self.assertEqual(len(mcx_gate.qudits), 3)
        self.assertEqual(mcx_gate.control_qudits, [qubit0, qubit1])
        self.assertEqual(mcx_gate.target_qudits, [qubit2])
        self.assertEqual(mcx_gate.control_values, [1, 0])
        self.assertFalse(mcx_gate.is_local)
        self.assertTrue(mcx_gate.is_entangling)

    def test_multi_controlled_gate_matrix(self):
        """Test matrix representation of multi-controlled gates."""
        # Create qubits
        qubit0 = Qudit(0)
        qubit1 = Qudit(1)
        qubit2 = Qudit(2)

        # Create a target gate (X gate on qubit2)
        x_gate = PermutationGate(qubit2, [1, 0])

        # Create a multi-controlled gate (controlled on qubit0=1 and qubit1=0)
        mcx_gate = MultiControlledGate(x_gate, [qubit0, qubit1], [1, 0])

        # The matrix should be an 8x8 matrix (2^3)
        self.assertEqual(mcx_gate.matrix.shape, (8, 8))

        # The matrix should be identity except for the block corresponding to qubit0=1, qubit1=0
        # For a 3-qubit system with qubits 0,1,2, the computational basis states are:
        # |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩
        # The block corresponding to qubit0=1, qubit1=0 is for states |100⟩ and |101⟩
        # So the X gate should be applied to the block at indices 4,5 (0-indexed)

        # Expected matrix: identity with X gate at the block for qubit0=1, qubit1=0
        expected_matrix = np.eye(8, dtype=complex)
        # Swap rows/columns 4 and 5 (corresponding to |100⟩ and |101⟩)
        expected_matrix[4, 4] = 0
        expected_matrix[4, 5] = 1
        expected_matrix[5, 4] = 1
        expected_matrix[5, 5] = 0

        np.testing.assert_array_almost_equal(mcx_gate.matrix, expected_matrix)

    def test_multi_controlled_gate_application(self):
        """Test applying multi-controlled gates to quantum states."""
        # Create a 3-qubit state vector
        state = StateVector(3, [2, 2, 2])

        # Apply Hadamard to the first qubit to get (|000⟩ + |100⟩)/√2
        h_gate = HadamardGate(Qudit(0))
        h_gate.apply(state)

        # Apply X to the second qubit to get (|000⟩ + |110⟩)/√2
        x_gate1 = PermutationGate(Qudit(1), [1, 0])
        cx_gate = MultiControlledGate(x_gate1, [Qudit(0)], [1])
        cx_gate.apply(state)

        # The state should now be (|000⟩ + |110⟩)/√2
        expected_amplitudes = np.zeros(8, dtype=complex)
        expected_amplitudes[0] = 1 / np.sqrt(2)  # |000⟩
        expected_amplitudes[6] = 1 / np.sqrt(2)  # |110⟩
        np.testing.assert_array_almost_equal(state.amplitudes, expected_amplitudes)

        # Now apply a multi-controlled X gate on the third qubit, controlled on qubits 0 and 1
        # being in state |11⟩
        x_gate2 = PermutationGate(Qudit(2), [1, 0])
        mcx_gate = MultiControlledGate(x_gate2, [Qudit(0), Qudit(1)], [1, 1])
        mcx_gate.apply(state)

        # The state should now be (|000⟩ + |111⟩)/√2
        expected_amplitudes = np.zeros(8, dtype=complex)
        expected_amplitudes[0] = 1 / np.sqrt(2)  # |000⟩
        expected_amplitudes[7] = 1 / np.sqrt(2)  # |111⟩
        np.testing.assert_array_almost_equal(state.amplitudes, expected_amplitudes)

    def test_invalid_control_values(self):
        """Test that an error is raised when control qudits and values don't match."""
        # Create qubits
        qubit0 = Qudit(0)
        qubit1 = Qudit(1)
        qubit2 = Qudit(2)

        # Create a target gate (X gate on qubit2)
        x_gate = PermutationGate(qubit2, [1, 0])

        # Try to create a multi-controlled gate with mismatched control qudits and values
        with self.assertRaises(ValueError):
            MultiControlledGate(x_gate, [qubit0, qubit1], [1])


if __name__ == "__main__":
    unittest.main()
