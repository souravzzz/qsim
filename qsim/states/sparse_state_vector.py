"""
Sparse representation of a quantum state using a hash map.
"""

import numpy as np
from typing import List, Dict
from qsim.states.base import QuantumState
from qsim.states.state_vector import StateVector
from qsim.gates.base import Gate
from qsim.gates.phase import PhaseGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.controlled import ControlledGate


class SparseStateVector(QuantumState):
    """Sparse representation of a quantum state using a hash map."""

    def __init__(self, num_qudits: int, dimensions: List[int]):
        """
        Initialize a sparse state vector in the |0...0⟩ state.

        Args:
            num_qudits: Number of qudits
            dimensions: Dimension of each qudit
        """
        super().__init__(num_qudits, dimensions)
        self.amplitudes = {0: 1.0}  # Dictionary mapping basis state indices to amplitudes

    def get_probability(self, index: int) -> float:
        """Get the probability of measuring a specific computational basis state."""
        if index < 0 or index >= self.total_dimension:
            raise ValueError(
                f"Index {index} out of range for state vector of dimension {self.total_dimension}"
            )
        return np.abs(self.amplitudes.get(index, 0.0)) ** 2

    def get_probabilities(self) -> np.ndarray:
        """Get the probabilities of all computational basis states."""
        probs = np.zeros(self.total_dimension, dtype=float)
        for idx, amp in self.amplitudes.items():
            probs[idx] = np.abs(amp) ** 2
        return probs

    def get_amplitudes(self) -> np.ndarray:
        """
        Get the amplitudes of all computational basis states.

        Returns:
            A dense array of complex amplitudes
        """
        amplitudes = np.zeros(self.total_dimension, dtype=complex)
        for idx, amp in self.amplitudes.items():
            amplitudes[idx] = amp
        return amplitudes

    def apply_gate(self, gate: Gate):
        """Apply a gate to the sparse state vector."""
        # Different implementation based on gate type
        if isinstance(gate, PhaseGate):
            self._apply_phase_gate(gate)
        elif isinstance(gate, PermutationGate):
            self._apply_permutation_gate(gate)
        elif isinstance(gate, ControlledGate):
            self._apply_controlled_gate(gate)
        elif gate.is_local:
            self._apply_local_gate(gate)
        else:
            self._apply_general_gate(gate)

    def _apply_phase_gate(self, gate: PhaseGate):
        """Apply a phase gate to the sparse state vector."""
        qudit_index = gate.qudits[0].index
        qudit_dim = self.dimensions[qudit_index]
        phase = gate.phase

        # Compute the divisor and modulus for extracting the qudit state
        divisor = 1
        for i in range(qudit_index):
            divisor *= self.dimensions[i]

        modulus = divisor * qudit_dim

        # Apply the phase to each basis state in the superposition
        new_amplitudes = {}
        for idx, amp in self.amplitudes.items():
            # Extract the qudit state
            qudit_state = (idx // divisor) % qudit_dim

            # Apply the phase
            new_amplitudes[idx] = amp * np.exp(1j * phase * qudit_state)

        self.amplitudes = new_amplitudes

    def _apply_permutation_gate(self, gate: PermutationGate):
        """Apply a permutation gate to the sparse state vector."""
        qudit_index = gate.qudits[0].index
        qudit_dim = self.dimensions[qudit_index]
        permutation = gate.permutation

        # Compute the divisor and modulus for extracting the qudit state
        divisor = 1
        for i in range(qudit_index):
            divisor *= self.dimensions[i]

        modulus = divisor * qudit_dim

        # Apply the permutation to each basis state in the superposition
        new_amplitudes = {}
        for idx, amp in self.amplitudes.items():
            # Extract the parts of the index
            prefix = idx // modulus
            qudit_state = (idx // divisor) % qudit_dim
            suffix = idx % divisor

            # Apply the permutation
            new_qudit_state = permutation[qudit_state]

            # Compute the new index
            new_idx = prefix * modulus + new_qudit_state * divisor + suffix

            new_amplitudes[new_idx] = amp

        self.amplitudes = new_amplitudes

    def _apply_controlled_gate(self, gate: ControlledGate):
        """Apply a controlled gate to the sparse state vector."""
        # Extract control and target information
        control_indices = [q.index for q in gate.control_qudits]
        control_dims = [self.dimensions[i] for i in control_indices]
        control_values = gate.control_values

        # Find basis states that satisfy the control condition
        matching_indices = []
        for idx in self.amplitudes:
            # Check if this basis state satisfies the control condition
            matches = True
            for control_idx, control_dim, control_val in zip(
                control_indices, control_dims, control_values
            ):
                # Extract the control qudit state
                divisor = 1
                for i in range(control_idx):
                    divisor *= self.dimensions[i]

                qudit_state = (idx // divisor) % control_dim

                if qudit_state != control_val:
                    matches = False
                    break

            if matches:
                matching_indices.append(idx)

        # Apply the target gate to the matching basis states
        target_gate = gate.target_gate
        target_indices = [q.index for q in gate.target_qudits]
        target_dims = [self.dimensions[i] for i in target_indices]

        # Create temporary dense representation for the target qudits
        target_dim = np.prod(target_dims)

        # For each matching basis state
        new_amplitudes = dict(self.amplitudes)  # Start with a copy

        for idx in matching_indices:
            # Remove this amplitude from the new amplitudes
            amplitude = new_amplitudes.pop(idx)

            # Extract the target qudits' state
            target_state_indices = []
            for target_idx in target_indices:
                divisor = 1
                for j in range(target_idx):
                    divisor *= self.dimensions[j]

                qudit_state = (idx // divisor) % self.dimensions[target_idx]
                target_state_indices.append(qudit_state)

            # Convert to linear index
            target_state = 0
            for i, state in enumerate(target_state_indices):
                multiplier = 1
                for j in range(i):
                    multiplier *= target_dims[j]
                target_state += state * multiplier

            # Apply the target gate's matrix to the target state
            if target_gate.matrix is not None:
                # For each possible output state of the target qudits
                for j in range(target_dim):
                    if abs(target_gate.matrix[j, target_state]) > 1e-10:
                        # Convert the output state to qudit states
                        output_states = []
                        temp = j
                        for dim in reversed(target_dims):
                            output_states.insert(0, temp % dim)
                            temp //= dim

                        # Compute the new state vector index
                        new_idx = idx
                        for target_idx, output_state in zip(target_indices, output_states):
                            # Clear the old value
                            divisor = 1
                            for k in range(target_idx):
                                divisor *= self.dimensions[k]

                            modulus = divisor * self.dimensions[target_idx]
                            prefix = new_idx // modulus
                            suffix = new_idx % divisor

                            # Set the new value
                            new_idx = prefix * modulus + output_state * divisor + suffix

                        # Add to the new amplitudes
                        new_amp = amplitude * target_gate.matrix[j, target_state]
                        if abs(new_amp) > 1e-10:
                            if new_idx in new_amplitudes:
                                new_amplitudes[new_idx] += new_amp
                            else:
                                new_amplitudes[new_idx] = new_amp

        self.amplitudes = new_amplitudes

    def _apply_local_gate(self, gate: Gate):
        """Apply a local gate to the sparse state vector."""
        qudit_index = gate.qudits[0].index
        qudit_dim = self.dimensions[qudit_index]

        # Compute the divisor and modulus for extracting the qudit state
        divisor = 1
        for i in range(qudit_index):
            divisor *= self.dimensions[i]

        modulus = divisor * qudit_dim

        # Apply the gate to each basis state in the superposition
        new_amplitudes = {}
        for idx, amp in self.amplitudes.items():
            # Extract the parts of the index
            prefix = idx // modulus
            qudit_state = (idx // divisor) % qudit_dim
            suffix = idx % divisor

            # Apply the gate to the qudit state
            for output_state in range(qudit_dim):
                matrix_element = gate.matrix[output_state, qudit_state]
                if abs(matrix_element) > 1e-10:
                    # Compute the new index
                    new_idx = prefix * modulus + output_state * divisor + suffix

                    # Compute the new amplitude
                    new_amp = amp * matrix_element

                    if new_idx in new_amplitudes:
                        new_amplitudes[new_idx] += new_amp
                    else:
                        new_amplitudes[new_idx] = new_amp

        self.amplitudes = new_amplitudes

    def _apply_general_gate(self, gate: Gate):
        """Apply a general gate to the sparse state vector."""
        # Convert to dense, apply the gate, and convert back to sparse
        dense_state = self.to_state_vector()
        dense_state.apply_gate(gate)

        # Update amplitudes from dense state
        self.amplitudes = {}
        for idx, amp in enumerate(dense_state.amplitudes):
            if abs(amp) > 1e-10:  # Only store non-zero amplitudes
                self.amplitudes[idx] = amp

    def _apply_gate_tensor(self, gate_matrix: np.ndarray, gate_qudits: List[int]):
        """
        Apply a gate to the sparse state vector using tensor operations.

        Args:
            gate_matrix: Matrix representation of the gate
            gate_qudits: Indices of qudits the gate acts on
        """
        # For sparse state vectors, we'll handle this by using the appropriate
        # specialized methods based on the gate structure

        # If it's a single-qudit gate
        if len(gate_qudits) == 1:
            self._apply_local_gate_matrix(gate_matrix, gate_qudits[0])
        else:
            # For multi-qudit gates, convert to dense, apply, and convert back
            dense_state = self.to_state_vector()
            dense_state._apply_gate_tensor(gate_matrix, gate_qudits)

            # Update amplitudes from dense state
            self.amplitudes = {}
            for idx, amp in enumerate(dense_state.amplitudes):
                if abs(amp) > 1e-10:  # Only store non-zero amplitudes
                    self.amplitudes[idx] = amp

    def _apply_local_gate_matrix(self, gate_matrix: np.ndarray, qudit_index: int):
        """
        Apply a single-qudit gate matrix to the sparse state vector.

        Args:
            gate_matrix: Matrix representation of the gate
            qudit_index: Index of the qudit the gate acts on
        """
        qudit_dim = self.dimensions[qudit_index]

        # Compute the divisor and modulus for extracting the qudit state
        divisor = 1
        for i in range(qudit_index):
            divisor *= self.dimensions[i]

        modulus = divisor * qudit_dim

        # Apply the gate to each basis state in the superposition
        new_amplitudes = {}
        for idx, amp in self.amplitudes.items():
            # Extract the parts of the index
            prefix = idx // modulus
            qudit_state = (idx // divisor) % qudit_dim
            suffix = idx % divisor

            # Apply the gate to the qudit state
            for output_state in range(qudit_dim):
                matrix_element = gate_matrix[output_state, qudit_state]
                if abs(matrix_element) > 1e-10:
                    # Compute the new index
                    new_idx = prefix * modulus + output_state * divisor + suffix

                    # Compute the new amplitude
                    new_amp = amp * matrix_element

                    if new_idx in new_amplitudes:
                        new_amplitudes[new_idx] += new_amp
                    else:
                        new_amplitudes[new_idx] = new_amp

        self.amplitudes = new_amplitudes

    def to_state_vector(self):
        """
        Convert the sparse state vector to a dense state vector.

        Returns:
            A StateVector object with the same state.
        """
        from qsim.states.state_vector import StateVector

        # Create a new dense state vector
        dense_state = StateVector(self.num_qudits, self.dimensions)

        # Initialize with zeros
        dense_state.amplitudes = np.zeros(self.total_dimension, dtype=complex)

        # Fill in the non-zero amplitudes
        for idx, amp in self.amplitudes.items():
            dense_state.amplitudes[idx] = amp

        return dense_state

    @classmethod
    def from_state_vector(cls, state_vector):
        """
        Create a sparse state vector from a dense state vector.

        Args:
            state_vector: A StateVector object

        Returns:
            A SparseStateVector object with the same state.
        """
        # Create a new sparse state vector
        sparse_state = cls(state_vector.num_qudits, state_vector.dimensions)

        # Fill in the non-zero amplitudes
        sparse_state.amplitudes = {}
        for idx, amp in enumerate(state_vector.amplitudes):
            if abs(amp) > 1e-10:  # Only store non-zero amplitudes
                sparse_state.amplitudes[idx] = amp

        return sparse_state

    def measure(self) -> str:
        """
        Perform a measurement on the quantum state.

        Returns:
            A string representation of the measured state in the computational basis.
        """
        # Get probabilities for non-zero amplitudes
        indices = list(self.amplitudes.keys())
        probs = [np.abs(self.amplitudes[idx]) ** 2 for idx in indices]

        # Normalize probabilities
        total_prob = sum(probs)
        if abs(total_prob - 1.0) > 1e-10:
            probs = [p / total_prob for p in probs]

        # Sample from the probability distribution
        outcome_idx = np.random.choice(len(indices), p=probs)
        outcome = indices[outcome_idx]

        # Convert to binary representation
        result = []
        for dim in self.dimensions:
            result.insert(0, outcome % dim)
            outcome //= dim

        # Convert to string
        return "".join(str(d) for d in result)
