"""
Hadamard gate implementation.
"""

import numpy as np
from qsim.core.qudit import Qudit
from qsim.gates.base import Gate


class HadamardGate(Gate):
    """Hadamard gate that creates superposition."""

    def __init__(self, qudit: Qudit):
        """
        Initialize a Hadamard gate.

        Args:
            qudit: Qudit the gate acts on
        """
        d = qudit.dimension
        matrix = np.ones((d, d), dtype=complex) / np.sqrt(d)
        for i in range(d):
            for j in range(d):
                if i > 0 and j > 0:
                    omega = np.exp(2j * np.pi * i * j / d)
                    matrix[i, j] *= omega
        super().__init__("H", [qudit], matrix)

    @property
    def is_entangling(self) -> bool:
        return False  # Not entangling on its own, but can lead to entanglement

    def apply(self, state):
        """Apply the Hadamard gate to a quantum state."""
        # Use the matrix representation for application
        qudit_indices = [q.index for q in self.qudits]
        state._apply_gate_tensor(self.matrix, qudit_indices)
