"""
Test for Generalized Bell state preparation and measurement with qudits.
"""

import unittest
import numpy as np
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.utils.circuit_builders import create_generalized_bell_state_circuit


class TestGeneralizedBellState(unittest.TestCase):
    """Test case for Generalized Bell state preparation and measurement with qudits."""

    def setUp(self):
        """Set up the test case."""
        self.simulator = HybridQuantumSimulator()

    def test_qubit_bell_state(self):
        """Test that the standard Bell state (d=2) simulation produces the correct state vector."""
        # Create a Bell state circuit for qubits (d=2)
        bell_circuit = create_generalized_bell_state_circuit(dimension=2)

        # Simulate the circuit
        state = self.simulator.simulate(bell_circuit)

        # The Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 should have equal amplitudes for |00⟩ and |11⟩
        amplitudes = state.get_amplitudes()

        # Check that we have the correct number of amplitudes
        self.assertEqual(len(amplitudes), 4)

        # Check that |00⟩ and |11⟩ have amplitude 1/√2 ≈ 0.7071
        self.assertAlmostEqual(abs(amplitudes[0]), 1 / np.sqrt(2), places=4)
        self.assertAlmostEqual(abs(amplitudes[3]), 1 / np.sqrt(2), places=4)

        # Check that |01⟩ and |10⟩ have amplitude 0
        self.assertAlmostEqual(abs(amplitudes[1]), 0, places=4)
        self.assertAlmostEqual(abs(amplitudes[2]), 0, places=4)

    def test_qutrit_bell_state(self):
        """Test that the qutrit Bell state (d=3) simulation produces the correct state vector."""
        # Create a Bell state circuit for qutrits (d=3)
        bell_circuit = create_generalized_bell_state_circuit(dimension=3)

        # Simulate the circuit
        state = self.simulator.simulate(bell_circuit)

        # The qutrit Bell state |Φ⟩ = (|00⟩ + |11⟩ + |22⟩)/√3 should have equal amplitudes
        amplitudes = state.get_amplitudes()

        # Check that we have the correct number of amplitudes
        self.assertEqual(len(amplitudes), 9)  # 3^2 = 9 possible states

        # Expected non-zero indices for |00⟩, |11⟩, |22⟩
        expected_indices = [0, 4, 8]  # 0*3+0, 1*3+1, 2*3+2
        expected_amplitude = 1 / np.sqrt(3)  # 1/√3 ≈ 0.5774

        # Check that |00⟩, |11⟩, and |22⟩ have amplitude 1/√3
        for idx in expected_indices:
            self.assertAlmostEqual(abs(amplitudes[idx]), expected_amplitude, places=4)

        # Check that all other states have amplitude 0
        for idx in range(9):
            if idx not in expected_indices:
                self.assertAlmostEqual(abs(amplitudes[idx]), 0, places=4)

    def test_ququart_bell_state(self):
        """Test that the ququart Bell state (d=4) simulation produces the correct state vector."""
        # Create a Bell state circuit for ququarts (d=4)
        bell_circuit = create_generalized_bell_state_circuit(dimension=4)

        # Simulate the circuit
        state = self.simulator.simulate(bell_circuit)

        # The ququart Bell state |Φ⟩ = (|00⟩ + |11⟩ + |22⟩ + |33⟩)/2 should have equal amplitudes
        amplitudes = state.get_amplitudes()

        # Check that we have the correct number of amplitudes
        self.assertEqual(len(amplitudes), 16)  # 4^2 = 16 possible states

        # Expected non-zero indices for |00⟩, |11⟩, |22⟩, |33⟩
        expected_indices = [0, 5, 10, 15]  # 0*4+0, 1*4+1, 2*4+2, 3*4+3
        expected_amplitude = 0.5  # 1/2

        # Check that |00⟩, |11⟩, |22⟩, and |33⟩ have amplitude 1/2
        for idx in expected_indices:
            self.assertAlmostEqual(abs(amplitudes[idx]), expected_amplitude, places=4)

        # Check that all other states have amplitude 0
        for idx in range(16):
            if idx not in expected_indices:
                self.assertAlmostEqual(abs(amplitudes[idx]), 0, places=4)

    def test_generalized_bell_state_measurement(self):
        """Test that measuring generalized Bell states gives the expected distribution."""
        # Test for different dimensions
        for dimension in [2, 3, 4, 5]:
            with self.subTest(dimension=dimension):
                # Create a Bell state circuit for the given dimension
                bell_circuit = create_generalized_bell_state_circuit(dimension=dimension)

                # Simulate and measure the circuit with a large number of shots
                results = self.simulator.simulate_and_measure(bell_circuit, num_shots=10000)

                # We expect to see only |00⟩, |11⟩, |22⟩, etc. outcomes with roughly equal probability
                expected_outcomes = [f"{i}{i}" for i in range(dimension)]

                # Check that all expected outcomes are present
                for outcome in expected_outcomes:
                    self.assertIn(outcome, results)

                # Check that unexpected outcomes are not present or very rare
                total_expected_count = 0
                for outcome in expected_outcomes:
                    total_expected_count += results.get(outcome, 0)

                # Expected count per outcome
                expected_count_per_outcome = total_expected_count / dimension

                # Check that each expected outcome has roughly the expected count (within 10%)
                for outcome in expected_outcomes:
                    count = results.get(outcome, 0)
                    self.assertTrue(
                        abs(count - expected_count_per_outcome) < 0.1 * expected_count_per_outcome,
                        f"Count for {outcome} is {count}, expected around {expected_count_per_outcome}",
                    )

                # Check that the total count of unexpected outcomes is very small
                unexpected_count = 0
                for outcome, count in results.items():
                    if outcome not in expected_outcomes:
                        unexpected_count += count

                self.assertTrue(
                    unexpected_count < 0.01 * total_expected_count,
                    f"Unexpected outcomes have count {unexpected_count}, should be very small",
                )


if __name__ == "__main__":
    unittest.main()
