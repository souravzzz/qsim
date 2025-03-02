"""
Base class for quantum gates.
"""

from typing import List, Optional
import numpy as np
from qsim.core.qudit import Qudit


class Gate:
    """Base class for quantum gates."""

    def __init__(self, name: str, qudits: List[Qudit], matrix: Optional[np.ndarray] = None):
        """
        Initialize a quantum gate.

        Args:
            name: Name of the gate
            qudits: List of qudits the gate acts on
            matrix: Unitary matrix representation of the gate (optional)
        """
        self.name = name
        self.qudits = qudits
        self.matrix = matrix

    def __repr__(self):
        return f"{self.name}({', '.join(str(q.index) for q in self.qudits)})"

    @property
    def qudit(self) -> Qudit:
        """
        Get the qudit for single-qudit gates.

        Returns:
            The single qudit this gate acts on

        Raises:
            AttributeError: If the gate acts on multiple qudits
        """
        if len(self.qudits) != 1:
            raise AttributeError("Gate acts on multiple qudits, use qudits property instead")
        return self.qudits[0]

    @property
    def is_local(self) -> bool:
        """Check if the gate is local (acts on a single qudit)."""
        return len(self.qudits) == 1

    @property
    def is_entangling(self) -> bool:
        """Check if the gate is entangling (default implementation: multi-qudit gates are entangling)."""
        return len(self.qudits) > 1

    def apply(self, state):
        """Apply the gate to a quantum state."""
        raise NotImplementedError("Subclasses must implement apply method")
