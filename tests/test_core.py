"""
Tests for core components (Qudit and QuantumCircuit).
"""

import unittest
import numpy as np
from qsim.core.qudit import Qudit
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.phase import PhaseGate
from qsim.gates.controlled import ControlledGate


class TestQudit(unittest.TestCase):
    """Test case for Qudit class."""

    def test_qudit_initialization(self):
        """Test that qudits are initialized correctly."""
        # Test default qubit
        qubit = Qudit(0)
        self.assertEqual(qubit.index, 0)
        self.assertEqual(qubit.dimension, 2)

        # Test qutrit
        qutrit = Qudit(1, 3)
        self.assertEqual(qutrit.index, 1)
        self.assertEqual(qutrit.dimension, 3)

        # Test representation
        self.assertEqual(repr(qubit), "Qudit(0, d=2)")
        self.assertEqual(repr(qutrit), "Qudit(1, d=3)")


class TestQuantumCircuit(unittest.TestCase):
    """Test case for QuantumCircuit class."""

    def test_circuit_initialization(self):
        """Test that circuits are initialized correctly."""
        # Test default circuit with qubits
        circuit = QuantumCircuit(3)
        self.assertEqual(circuit.num_qudits, 3)
        self.assertEqual(len(circuit.qudits), 3)
        self.assertEqual(len(circuit.gates), 0)
        self.assertEqual(circuit.dimensions, [2, 2, 2])

        # Test circuit with mixed dimensions
        circuit = QuantumCircuit(3, [2, 3, 4])
        self.assertEqual(circuit.num_qudits, 3)
        self.assertEqual(len(circuit.qudits), 3)
        self.assertEqual(circuit.dimensions, [2, 3, 4])
        self.assertEqual(circuit.qudits[0].dimension, 2)
        self.assertEqual(circuit.qudits[1].dimension, 3)
        self.assertEqual(circuit.qudits[2].dimension, 4)

    def test_add_gate(self):
        """Test adding gates to a circuit."""
        circuit = QuantumCircuit(2)

        # Add Hadamard gate
        h_gate = HadamardGate(circuit.qudits[0])
        circuit.add_gate(h_gate)
        self.assertEqual(len(circuit.gates), 1)
        self.assertEqual(circuit.gates[0], h_gate)

        # Add Phase gate
        p_gate = PhaseGate(circuit.qudits[1], np.pi / 4)
        circuit.add_gate(p_gate)
        self.assertEqual(len(circuit.gates), 2)
        self.assertEqual(circuit.gates[1], p_gate)

        # Add Controlled gate
        target_gate = HadamardGate(circuit.qudits[1])
        cx_gate = ControlledGate(
            target_gate=target_gate, control_qudit=circuit.qudits[0], control_value=1
        )
        circuit.add_gate(cx_gate)
        self.assertEqual(len(circuit.gates), 3)
        self.assertEqual(circuit.gates[2], cx_gate)

        # Test representation
        self.assertEqual(repr(circuit), "QuantumCircuit(2 qudits, 3 gates)")


if __name__ == "__main__":
    unittest.main()
