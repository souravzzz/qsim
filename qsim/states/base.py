"""
Base class for quantum state representations.
"""

import numpy as np
from typing import List


class QuantumState:
    """Base class for quantum state representations."""

    def __init__(self, num_qudits: int, dimensions: List[int]):
        """
        Initialize a quantum state.

        Args:
            num_qudits: Number of qudits
            dimensions: Dimension of each qudit
        """
        self.num_qudits = num_qudits
        self.dimensions = dimensions
        self.total_dimension = np.prod(dimensions)

    def get_probability(self, index: int) -> float:
        """Get the probability of measuring a specific computational basis state."""
        raise NotImplementedError

    def get_probabilities(self) -> np.ndarray:
        """Get the probabilities of all computational basis states."""
        raise NotImplementedError

    def apply_gate(self, gate):
        """Apply a gate to the state."""
        raise NotImplementedError
