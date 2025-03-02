"""
Phase gate implementation.
"""

import numpy as np
from qsim.core.qudit import Qudit
from qsim.gates.base import Gate


class PhaseGate(Gate):
    """Phase gate that adds a phase to specific computational basis states."""

    def __init__(self, qudit: Qudit, phase: float):
        """
        Initialize a phase gate.

        Args:
            qudit: Qudit the gate acts on
            phase: Phase to add (in radians)
        """
        matrix = np.diag([np.exp(1j * phase * i) for i in range(qudit.dimension)])
        super().__init__("Phase", [qudit], matrix)
        self.phase = phase

    @property
    def is_entangling(self) -> bool:
        return False

    def apply(self, state):
        """Apply the phase gate to a quantum state."""
        # Use the matrix representation for application
        qudit_indices = [q.index for q in self.qudits]
        state._apply_gate_tensor(self.matrix, qudit_indices)
