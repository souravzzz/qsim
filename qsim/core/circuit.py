"""
Quantum circuit representation.
"""

from typing import List, Union
from qsim.core.qudit import Qudit
from qsim.gates.base import Gate


class QuantumCircuit:
    """Representation of a quantum circuit."""

    def __init__(self, num_qudits: int, dimensions: Union[int, List[int]] = 2):
        """
        Initialize a quantum circuit.

        Args:
            num_qudits: Number of qudits in the circuit
            dimensions: Dimension of each qudit (either a single value for all qudits or a list)
        """
        if isinstance(dimensions, int):
            dimensions = [dimensions] * num_qudits

        self.qudits = [Qudit(i, d) for i, d in enumerate(dimensions)]
        self.gates = []
        self.num_qudits = num_qudits
        self.dimensions = dimensions

    def add_gate(self, gate: Gate):
        """Add a gate to the circuit."""
        self.gates.append(gate)
        return self

    def __repr__(self):
        return f"QuantumCircuit({self.num_qudits} qudits, {len(self.gates)} gates)"
