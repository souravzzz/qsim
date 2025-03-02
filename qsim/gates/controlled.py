"""
Controlled gate implementation.
"""

import numpy as np
import itertools
from typing import List, Optional
from qsim.core.qudit import Qudit
from qsim.gates.base import Gate


class ControlledGate(Gate):
    """Gate that is controlled on the state of one or more qudits."""

    def __init__(
        self,
        target_gate: Gate,
        control_qudit: Qudit,
        control_value: int = 1,
    ):
        """
        Initialize a controlled gate.

        Args:
            target_gate: Gate to apply when control condition is met
            control_qudit: Control qudit
            control_value: Value the control qudit must have (default: 1)
        """
        # Extract target qudits from the target gate
        target_qudits = target_gate.qudits

        # Create a name for the controlled gate
        name = "C" + target_gate.name

        # Combine control and target qudits
        all_qudits = [control_qudit] + target_qudits

        super().__init__(name, all_qudits)

        self.control_qudits = [control_qudit]
        self.target_qudits = target_qudits
        self.control_values = [control_value]
        self.target_gate = target_gate

        # Compute the full matrix representation if possible
        if target_gate.matrix is not None:
            self._compute_matrix()

    @property
    def control_qudit(self) -> Qudit:
        """
        Get the control qudit for gates with a single control.

        Returns:
            The single control qudit

        Raises:
            AttributeError: If the gate has multiple control qudits
        """
        if len(self.control_qudits) != 1:
            raise AttributeError(
                "Gate has multiple control qudits, use control_qudits property instead"
            )
        return self.control_qudits[0]

    @property
    def control_value(self) -> int:
        """
        Get the control value for gates with a single control.

        Returns:
            The single control value

        Raises:
            AttributeError: If the gate has multiple control values
        """
        if len(self.control_values) != 1:
            raise AttributeError(
                "Gate has multiple control values, use control_values property instead"
            )
        return self.control_values[0]

    def _compute_matrix(self):
        """Compute the full matrix representation of the controlled gate."""
        # Get dimensions and sizes
        control_dims = [q.dimension for q in self.control_qudits]
        target_dims = [q.dimension for q in self.target_qudits]

        control_size = np.prod(control_dims)
        target_size = np.prod(target_dims)
        total_size = control_size * target_size

        # Create the matrix
        self.matrix = np.eye(total_size, dtype=complex)

        # Find the indices where the control condition is met
        control_indices = []
        for control_values in itertools.product(*[range(dim) for dim in control_dims]):
            if list(control_values) == self.control_values:
                # Convert control values to linear index
                idx = 0
                for i, val in enumerate(control_values):
                    idx = idx * control_dims[i] + val
                control_indices.append(idx)

        # Apply the target gate to the appropriate submatrix
        target_matrix = self.target_gate.matrix
        for idx in control_indices:
            start_row = idx * target_size
            start_col = idx * target_size

            # Replace the identity block with the target matrix
            for i in range(target_size):
                for j in range(target_size):
                    self.matrix[start_row + i, start_col + j] = target_matrix[i, j]

    def apply(self, state):
        """Apply the controlled gate to a quantum state."""
        if self.matrix is not None:
            # Use the matrix representation for application
            qudit_indices = [q.index for q in self.qudits]
            state._apply_gate_tensor(self.matrix, qudit_indices)
        else:
            # Implement a custom application method if matrix is not available
            raise NotImplementedError(
                "Custom application for controlled gates without matrix is not implemented"
            )
