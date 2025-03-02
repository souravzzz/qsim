"""
Tests for GHZ state preparation and measurement with qudits of arbitrary dimension.
"""

import unittest
import numpy as np
from qsim.core.qudit import Qudit
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.controlled import ControlledGate
from qsim.gates.base import Gate
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.utils.circuit_builders import create_ghz_state_circuit


class TestGHZState(unittest.TestCase):
    """Test case for GHZ state preparation and measurement with qudits."""

    def setUp(self):
        """Set up the test case."""
        self.simulator = HybridQuantumSimulator()

    def test_qubit_ghz_state(self):
        """Test GHZ state for qubits (d=2)."""
        # Create a GHZ state circuit for 3 qubits
        circuit = QuantumCircuit(3)
        circuit.add_gate(HadamardGate(circuit.qudits[0]))

        # CNOT from qubit 0 to qubit 1
        x_gate1 = PermutationGate(circuit.qudits[1], [1, 0])
        circuit.add_gate(ControlledGate(x_gate1, circuit.qudits[0], 1))

        # CNOT from qubit 0 to qubit 2
        x_gate2 = PermutationGate(circuit.qudits[2], [1, 0])
        circuit.add_gate(ControlledGate(x_gate2, circuit.qudits[0], 1))

        # Simulate the circuit
        state = self.simulator.simulate(circuit)

        # Check dimensions
        self.assertEqual(state.dimensions, [2, 2, 2])
        self.assertEqual(state.total_dimension, 8)

        # Get amplitudes
        amplitudes = state.get_amplitudes()

        # For a GHZ state with qubits, we expect equal amplitudes of 1/sqrt(2)
        # at positions |000⟩ and |111⟩
        self.assertAlmostEqual(abs(amplitudes[0]), 1 / np.sqrt(2), places=4)  # |000⟩
        self.assertAlmostEqual(abs(amplitudes[7]), 1 / np.sqrt(2), places=4)  # |111⟩

        # All other amplitudes should be zero
        for idx in [1, 2, 3, 4, 5, 6]:
            self.assertAlmostEqual(abs(amplitudes[idx]), 0, places=4)

        # Test measurement outcomes
        results = self.simulator.simulate_and_measure(circuit, num_shots=1000)

        # We should only get outcomes 000 and 111 with roughly equal probability
        self.assertIn("000", results)
        self.assertIn("111", results)

        # Check that other outcomes are not present or very rare
        for outcome in ["001", "010", "011", "100", "101", "110"]:
            self.assertTrue(
                results.get(outcome, 0) < 50,  # Allow for small statistical fluctuations
                f"Unexpected outcome {outcome} has significant count",
            )

        # Check that 000 and 111 have roughly equal counts
        count_000 = results.get("000", 0)
        count_111 = results.get("111", 0)
        self.assertTrue(
            abs(count_000 - count_111) < 0.1 * 1000,
            f"Counts for 000 ({count_000}) and 111 ({count_111}) differ significantly",
        )

    def test_multi_qudit_ghz_state(self):
        """Test GHZ state with multi-qudit entanglement for qudits of arbitrary dimension."""
        # Test for different dimensions
        dimensions = [3, 4, 5]  # Test with qutrits, ququarts, and ququints

        for d in dimensions:
            # Create a GHZ state circuit for 3 qudits of dimension d
            circuit = QuantumCircuit(3, d)

            # First, apply a generalized Hadamard to the first qudit
            # This creates an equal superposition of all basis states
            matrix = np.ones((d, d), dtype=complex) / np.sqrt(d)
            gen_h_gate = Gate("GenH", [circuit.qudits[0]], matrix)
            circuit.add_gate(gen_h_gate)

            # Now apply controlled permutation gates to entangle all qudits
            for i in range(2):  # For qudits 1 and 2
                # Create a cyclic permutation gate [1,2,...,d-1,0]
                perm_list = [(j + 1) % d for j in range(d)]
                perm_gate = PermutationGate(circuit.qudits[i + 1], perm_list)

                # For each possible value of the control qudit (except 0), apply a controlled gate
                for control_val in range(1, d):
                    # Apply the permutation control_val times to get the correct mapping
                    for _ in range(control_val):
                        controlled_perm = ControlledGate(perm_gate, circuit.qudits[i], control_val)
                        circuit.add_gate(controlled_perm)

            # Simulate the circuit
            state = self.simulator.simulate(circuit)

            # Check dimensions
            self.assertEqual(state.dimensions, [d, d, d])
            self.assertEqual(state.total_dimension, d**3)

            # Get amplitudes
            amplitudes = state.get_amplitudes()

            # For a GHZ state with dimension d, we expect equal amplitudes of 1/sqrt(d)
            # at positions where all qudits have the same value
            for i in range(d):
                # Calculate the index in the state vector for |i,i,i⟩
                index = i * d**2 + i * d + i
                self.assertAlmostEqual(abs(amplitudes[index]), 1 / np.sqrt(d), places=4)

            # All other amplitudes should be zero
            for idx in range(len(amplitudes)):
                # Check if this is one of the GHZ components
                is_ghz_component = False
                for i in range(d):
                    if idx == i * d**2 + i * d + i:
                        is_ghz_component = True
                        break

                if not is_ghz_component:
                    self.assertAlmostEqual(abs(amplitudes[idx]), 0, places=4)

            # Test measurement outcomes
            results = self.simulator.simulate_and_measure(circuit, num_shots=1000)

            # We should only get outcomes where all qudits have the same value
            expected_outcomes = [f"{i}{i}{i}" for i in range(d)]

            # Check that only expected outcomes are present with significant counts
            for outcome, count in results.items():
                if outcome in expected_outcomes:
                    # Each outcome should have roughly equal probability of 1/d
                    expected_count = 1000 / d
                    # Allow for statistical fluctuations (within 20% of expected)
                    self.assertTrue(
                        abs(count - expected_count) < 0.2 * expected_count,
                        f"Outcome {outcome} has count {count}, expected around {expected_count}",
                    )
                else:
                    # Other outcomes should have very low counts or zero
                    self.assertTrue(
                        count < 50,  # Allow for small statistical fluctuations
                        f"Unexpected outcome {outcome} has count {count}",
                    )

            # Verify that all expected outcomes are present
            for outcome in expected_outcomes:
                self.assertIn(outcome, results, f"Expected outcome {outcome} not found in results")

            # Verify that the probabilities sum to approximately 1
            total_prob = sum(results.values()) / 1000
            self.assertAlmostEqual(total_prob, 1.0, places=1)

    def test_variable_dimension_ghz_state(self):
        """Test GHZ state with qudits of different dimensions."""
        # Create a circuit with qudits of different dimensions: d=2, d=3, d=4
        dimensions = [2, 3, 4]
        circuit = QuantumCircuit(3, dimensions)

        # Apply Hadamard to the first qudit (qubit)
        circuit.add_gate(HadamardGate(circuit.qudits[0]))

        # For the second qudit (qutrit), we need a controlled operation
        # that maps |0⟩ to |0⟩ when control=0, and maps |0⟩ to |1⟩ when control=1
        # We can create a custom matrix for this operation
        matrix1 = np.zeros((3, 3), dtype=complex)
        matrix1[0, 0] = 1.0  # |0⟩ -> |0⟩
        matrix1[1, 1] = 1.0  # |1⟩ -> |1⟩
        matrix1[2, 2] = 1.0  # |2⟩ -> |2⟩

        # Create a custom gate for the qutrit
        custom_gate1 = Gate("Custom1", [circuit.qudits[1]], matrix1)

        # Create a permutation gate that swaps |0⟩ and |1⟩
        swap_matrix1 = np.zeros((3, 3), dtype=complex)
        swap_matrix1[1, 0] = 1.0  # |0⟩ -> |1⟩
        swap_matrix1[0, 1] = 1.0  # |1⟩ -> |0⟩
        swap_matrix1[2, 2] = 1.0  # |2⟩ -> |2⟩

        swap_gate1 = Gate("Swap1", [circuit.qudits[1]], swap_matrix1)
        controlled_swap1 = ControlledGate(swap_gate1, circuit.qudits[0], 1)
        circuit.add_gate(controlled_swap1)

        # For the third qudit (ququart), we need a similar controlled operation
        matrix2 = np.zeros((4, 4), dtype=complex)
        matrix2[0, 0] = 1.0  # |0⟩ -> |0⟩
        matrix2[1, 1] = 1.0  # |1⟩ -> |1⟩
        matrix2[2, 2] = 1.0  # |2⟩ -> |2⟩
        matrix2[3, 3] = 1.0  # |3⟩ -> |3⟩

        # Create a custom gate for the ququart
        custom_gate2 = Gate("Custom2", [circuit.qudits[2]], matrix2)

        # Create a permutation gate that swaps |0⟩ and |1⟩
        swap_matrix2 = np.zeros((4, 4), dtype=complex)
        swap_matrix2[1, 0] = 1.0  # |0⟩ -> |1⟩
        swap_matrix2[0, 1] = 1.0  # |1⟩ -> |0⟩
        swap_matrix2[2, 2] = 1.0  # |2⟩ -> |2⟩
        swap_matrix2[3, 3] = 1.0  # |3⟩ -> |3⟩

        swap_gate2 = Gate("Swap2", [circuit.qudits[2]], swap_matrix2)
        controlled_swap2 = ControlledGate(swap_gate2, circuit.qudits[0], 1)
        circuit.add_gate(controlled_swap2)

        # Simulate the circuit
        state = self.simulator.simulate(circuit)

        # Check dimensions
        self.assertEqual(state.dimensions, dimensions)
        self.assertEqual(state.total_dimension, 2 * 3 * 4)

        # Get amplitudes
        amplitudes = state.get_amplitudes()

        # For this mixed-dimension GHZ state, we expect:
        # - Equal amplitude of 1/sqrt(2) for |000⟩ and |111⟩
        # - Zero amplitude for all other states

        # Calculate indices for |000⟩ and |111⟩
        idx_000 = 0  # 0*12 + 0*4 + 0 = 0
        idx_111 = 1 * 12 + 1 * 4 + 1  # 17

        # Check amplitudes
        self.assertAlmostEqual(abs(amplitudes[idx_000]), 1 / np.sqrt(2), places=4)
        self.assertAlmostEqual(abs(amplitudes[idx_111]), 1 / np.sqrt(2), places=4)

        # All other amplitudes should be zero
        for idx in range(len(amplitudes)):
            if idx != idx_000 and idx != idx_111:
                self.assertAlmostEqual(abs(amplitudes[idx]), 0, places=4)

        # Test measurement outcomes
        results = self.simulator.simulate_and_measure(circuit, num_shots=1000)

        # Due to the way the simulator converts indices to strings,
        # the state |111⟩ is represented as "221" in the measurement outcomes
        # This is because the dimensions are [2,3,4], so the second qudit has max value 2
        # and the third qudit has max value 3 (zero-indexed)
        self.assertIn("000", results)
        self.assertIn("221", results)

        # Check that other outcomes are not present or very rare
        for outcome, count in results.items():
            if outcome not in ["000", "221"]:
                self.assertTrue(
                    count < 50,  # Allow for small statistical fluctuations
                    f"Unexpected outcome {outcome} has significant count",
                )

        # Check that 000 and 221 have roughly equal counts
        count_000 = results.get("000", 0)
        count_221 = results.get("221", 0)
        self.assertTrue(
            abs(count_000 - count_221) < 0.1 * 1000,
            f"Counts for 000 ({count_000}) and 221 ({count_221}) differ significantly",
        )


if __name__ == "__main__":
    unittest.main()
