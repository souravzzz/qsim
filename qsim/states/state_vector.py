"""
Dense state vector representation of a quantum state.
"""

import numpy as np
from typing import List
from qsim.states.base import QuantumState
from qsim.gates.base import Gate


class StateVector(QuantumState):
    """Dense state vector representation of a quantum state."""

    def __init__(self, num_qudits: int, dimensions: List[int]):
        """
        Initialize a state vector in the |0...0⟩ state.

        Args:
            num_qudits: Number of qudits
            dimensions: Dimension of each qudit
        """
        super().__init__(num_qudits, dimensions)
        self.vector = np.zeros(self.total_dimension, dtype=complex)
        self.vector[0] = 1.0  # Initialize to |0...0⟩

    @property
    def amplitudes(self) -> np.ndarray:
        """Get the complex amplitudes of the state vector."""
        return self.vector

    @amplitudes.setter
    def amplitudes(self, new_amplitudes: np.ndarray):
        """Set the complex amplitudes of the state vector."""
        if len(new_amplitudes) != self.total_dimension:
            raise ValueError(
                f"Amplitude array size {len(new_amplitudes)} does not match state dimension {self.total_dimension}"
            )
        self.vector = new_amplitudes

    def get_probability(self, index: int) -> float:
        """Get the probability of measuring a specific computational basis state."""
        if index < 0 or index >= self.total_dimension:
            raise ValueError(
                f"Index {index} out of range for state vector of dimension {self.total_dimension}"
            )
        return np.abs(self.vector[index]) ** 2

    def get_probabilities(self) -> np.ndarray:
        """Get the probabilities of all computational basis states."""
        return np.abs(self.vector) ** 2

    def get_amplitudes(self) -> np.ndarray:
        """Get the complex amplitudes of all computational basis states."""
        return self.vector

    def apply_gate(self, gate: Gate):
        """Apply a gate to the state vector."""
        if gate.matrix is not None:
            # Get indices of qudits the gate acts on
            gate_qudits = [q.index for q in gate.qudits]

            # Apply the gate using tensor operations
            self._apply_gate_tensor(gate.matrix, gate_qudits)
        else:
            # Use gate's custom apply method
            gate.apply(self)

    def _apply_gate_tensor(self, gate_matrix: np.ndarray, gate_qudits: List[int]):
        """Apply a gate to the state vector using tensor operations."""
        # Get the dimensions of the system
        dims = self.dimensions

        # Reshape the state vector to a multidimensional array
        state_tensor = self.vector.reshape(dims)

        # Transpose the state tensor to bring the gate qudits to the front
        other_qudits = [i for i in range(self.num_qudits) if i not in gate_qudits]
        transpose_indices = gate_qudits + other_qudits
        inverse_transpose = np.argsort(transpose_indices)

        state_tensor = np.transpose(state_tensor, transpose_indices)

        # Reshape for matrix multiplication
        gate_dims = [dims[i] for i in gate_qudits]
        gate_size = np.prod(gate_dims)
        other_size = self.total_dimension // gate_size

        state_tensor = state_tensor.reshape((gate_size, other_size))

        # Apply the gate
        state_tensor = np.dot(gate_matrix, state_tensor)

        # Reshape back
        state_tensor = state_tensor.reshape([dims[i] for i in transpose_indices])
        state_tensor = np.transpose(state_tensor, inverse_transpose)

        # Update the state vector
        self.vector = state_tensor.reshape(self.total_dimension)

    def measure(self) -> str:
        """
        Perform a measurement on the quantum state.

        Returns:
            A string representation of the measured state in the computational basis.
        """
        # Get probabilities
        probs = self.get_probabilities()

        # Sample from the probability distribution
        outcome = np.random.choice(self.total_dimension, p=probs)

        # Convert to binary representation
        result = []
        for dim in self.dimensions:
            result.insert(0, outcome % dim)
            outcome //= dim

        # Convert to string
        return "".join(str(d) for d in result)
