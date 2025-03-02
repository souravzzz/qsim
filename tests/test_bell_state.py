"""
Test for Bell state preparation and measurement.
"""

import unittest
import numpy as np
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.utils.circuit_builders import create_bell_state_circuit


class TestBellState(unittest.TestCase):
    """Test case for Bell state preparation and measurement."""

    def setUp(self):
        """Set up the test case."""
        self.simulator = HybridQuantumSimulator()
        self.bell_circuit = create_bell_state_circuit()

    def test_bell_state_simulation(self):
        """Test that the Bell state simulation produces the correct state vector."""
        # Simulate the circuit
        state = self.simulator.simulate(self.bell_circuit)

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

    def test_bell_state_measurement(self):
        """Test that measuring the Bell state gives the expected distribution."""
        # Simulate and measure the circuit with a large number of shots
        results = self.simulator.simulate_and_measure(
            self.bell_circuit, num_shots=10000
        )

        # We expect to see only |00⟩ and |11⟩ outcomes with roughly equal probability
        self.assertIn("00", results)
        self.assertIn("11", results)

        # Check that we don't have |01⟩ or |10⟩ outcomes (or they're very rare)
        self.assertTrue(
            results.get("01", 0) < 100
        )  # Allow for small statistical fluctuations
        self.assertTrue(results.get("10", 0) < 100)

        # Check that |00⟩ and |11⟩ have roughly equal counts
        # (within 5% of each other for a large number of shots)
        count_00 = results.get("00", 0)
        count_11 = results.get("11", 0)

        self.assertTrue(abs(count_00 - count_11) < 0.05 * 10000)


if __name__ == "__main__":
    unittest.main()
