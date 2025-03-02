"""
Qudit representation for quantum simulation.
"""


class Qudit:
    """Representation of a qudit with d levels."""

    def __init__(self, index: int, dimension: int = 2):
        """
        Initialize a qudit.

        Args:
            index: Index of the qudit
            dimension: Dimension of the qudit (default: 2 for qubits)
        """
        self.index = index
        self.dimension = dimension

    def __repr__(self):
        return f"Qudit({self.index}, d={self.dimension})"
