"""
Tests for quantum gate implementations.
"""

import unittest
import numpy as np
from qsim.core.qudit import Qudit
from qsim.gates.base import Gate
from qsim.gates.hadamard import HadamardGate
from qsim.gates.phase import PhaseGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.controlled import ControlledGate
from qsim.states.state_vector import StateVector


class TestGateProperties(unittest.TestCase):
    """Test case for gate properties."""

    def setUp(self):
        """Set up the test case."""
        self.qubit0 = Qudit(0)
        self.qubit1 = Qudit(1)
        self.qutrit0 = Qudit(0, 3)

    def test_gate_properties(self):
        """Test basic gate properties."""
        # Test single-qubit gate
        h_gate = HadamardGate(self.qubit0)
        self.assertTrue(h_gate.is_local)
        self.assertFalse(h_gate.is_entangling)
        self.assertEqual(repr(h_gate), "H(0)")

        # Test two-qubit gate
        cx_gate = ControlledGate(
            HadamardGate(self.qubit1), control_qudit=self.qubit0, control_value=1
        )
        self.assertFalse(cx_gate.is_local)
        self.assertTrue(cx_gate.is_entangling)
        self.assertEqual(repr(cx_gate), "CH(0, 1)")


class TestGateMatrices(unittest.TestCase):
    """Test case for gate matrix representations."""

    def setUp(self):
        """Set up the test case."""
        self.qubit0 = Qudit(0)
        self.qubit1 = Qudit(1)
        self.qutrit0 = Qudit(0, 3)

    def test_hadamard_matrix(self):
        """Test Hadamard gate matrix."""
        h_gate = HadamardGate(self.qubit0)
        expected_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(h_gate.matrix, expected_matrix)

    def test_phase_matrix(self):
        """Test Phase gate matrix."""
        # Test with pi/4
        p_gate = PhaseGate(self.qubit0, np.pi / 4)
        expected_matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        np.testing.assert_array_almost_equal(p_gate.matrix, expected_matrix)

        # Test with pi/2 (S gate)
        s_gate = PhaseGate(self.qubit0, np.pi / 2)
        expected_matrix = np.array([[1, 0], [0, 1j]])
        np.testing.assert_array_almost_equal(s_gate.matrix, expected_matrix)

        # Test with pi (Z gate)
        z_gate = PhaseGate(self.qubit0, np.pi)
        expected_matrix = np.array([[1, 0], [0, -1]])
        np.testing.assert_array_almost_equal(z_gate.matrix, expected_matrix)

    def test_permutation_matrix(self):
        """Test Permutation gate matrix."""
        # Test X gate (NOT gate)
        x_gate = PermutationGate(self.qubit0, [1, 0])
        expected_matrix = np.array([[0, 1], [1, 0]])
        np.testing.assert_array_almost_equal(x_gate.matrix, expected_matrix)

        # Test qutrit permutation
        perm_gate = PermutationGate(self.qutrit0, [1, 2, 0])
        expected_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        np.testing.assert_array_almost_equal(perm_gate.matrix, expected_matrix)

    def test_controlled_matrix(self):
        """Test Controlled gate matrix."""
        # Test controlled-X (CNOT)
        x_gate = PermutationGate(self.qubit1, [1, 0])
        cx_gate = ControlledGate(x_gate, self.qubit0, 1)

        # CNOT matrix should be:
        # [1 0 0 0]
        # [0 1 0 0]
        # [0 0 0 1]
        # [0 0 1 0]
        expected_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        np.testing.assert_array_almost_equal(cx_gate.matrix, expected_matrix)


class TestGateApplication(unittest.TestCase):
    """Test case for applying gates to states."""

    def test_hadamard_application(self):
        """Test applying Hadamard gate to a state."""
        # Create a single qubit in |0⟩ state
        state = StateVector(1, [2])

        # Apply Hadamard
        h_gate = HadamardGate(Qudit(0))
        h_gate.apply(state)

        # Should get |+⟩ = (|0⟩ + |1⟩)/√2
        expected_amplitudes = np.array([1, 1]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(state.amplitudes, expected_amplitudes)

        # Apply Hadamard again
        h_gate.apply(state)

        # Should get back to |0⟩
        expected_amplitudes = np.array([1, 0])
        np.testing.assert_array_almost_equal(state.amplitudes, expected_amplitudes)

    def test_phase_application(self):
        """Test applying Phase gate to a state."""
        # Create a single qubit in |+⟩ state
        state = StateVector(1, [2])
        h_gate = HadamardGate(Qudit(0))
        h_gate.apply(state)

        # Apply Phase gate (S gate)
        s_gate = PhaseGate(Qudit(0), np.pi / 2)
        s_gate.apply(state)

        # Should get (|0⟩ + i|1⟩)/√2
        expected_amplitudes = np.array([1, 1j]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(state.amplitudes, expected_amplitudes)

    def test_permutation_application(self):
        """Test applying Permutation gate to a state."""
        # Create a single qubit in |0⟩ state
        state = StateVector(1, [2])

        # Apply X gate
        x_gate = PermutationGate(Qudit(0), [1, 0])
        x_gate.apply(state)

        # Should get |1⟩
        expected_amplitudes = np.array([0, 1])
        np.testing.assert_array_almost_equal(state.amplitudes, expected_amplitudes)

        # Apply X gate again
        x_gate.apply(state)

        # Should get back to |0⟩
        expected_amplitudes = np.array([1, 0])
        np.testing.assert_array_almost_equal(state.amplitudes, expected_amplitudes)

    def test_controlled_application(self):
        """Test applying Controlled gate to a state."""
        # Create a two-qubit state |00⟩
        state = StateVector(2, [2, 2])

        # Apply Hadamard to first qubit to get (|00⟩ + |10⟩)/√2
        h_gate = HadamardGate(Qudit(0))
        h_gate.apply(state)

        # Apply CNOT (control=0, target=1)
        x_gate = PermutationGate(Qudit(1), [1, 0])
        cx_gate = ControlledGate(x_gate, Qudit(0), 1)
        cx_gate.apply(state)

        # Should get (|00⟩ + |11⟩)/√2 (Bell state)
        expected_amplitudes = np.zeros(4)
        expected_amplitudes[0] = 1 / np.sqrt(2)
        expected_amplitudes[3] = 1 / np.sqrt(2)
        np.testing.assert_array_almost_equal(state.amplitudes, expected_amplitudes)


if __name__ == "__main__":
    unittest.main()
