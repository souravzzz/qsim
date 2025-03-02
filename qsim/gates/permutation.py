"""
Permutation gate implementation.
"""

import numpy as np
from qsim.core.qudit import Qudit
from qsim.gates.base import Gate


class PermutationGate(Gate):
    """Permutation gate that performs a permutation of computational basis states."""

    def __init__(self, qudit: Qudit, permutation: list):
        """
        Initialize a permutation gate.

        Args:
            qudit: Qudit the gate acts on
            permutation: List specifying the permutation of states
        """
        if len(permutation) != qudit.dimension:
            raise ValueError(
                f"Permutation length {len(permutation)} must match qudit dimension {qudit.dimension}"
            )

        matrix = np.zeros((qudit.dimension, qudit.dimension), dtype=complex)
        for i, j in enumerate(permutation):
            matrix[j, i] = 1.0
        super().__init__("Perm", [qudit], matrix)
        self.permutation = permutation

    @property
    def is_entangling(self) -> bool:
        return False

    def apply(self, state):
        """Apply the permutation gate to a quantum state."""
        # Use the matrix representation for application
        qudit_indices = [q.index for q in self.qudits]
        state._apply_gate_tensor(self.matrix, qudit_indices)
