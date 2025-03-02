"""
Hybrid execution manager for quantum circuit simulation.
"""

import numpy as np
from qsim.core.circuit import QuantumCircuit
from qsim.analysis.circuit_analyzer import CircuitAnalyzer
from qsim.states.state_vector import StateVector
from qsim.states.sparse_state_vector import SparseStateVector
from qsim.states.tensor_network_state import TensorNetworkState
from qsim.core.config import logger
from qsim.gates.base import Gate


class HybridExecutionManager:
    """Manages the execution of a quantum circuit using hybrid simulation methods."""

    def __init__(self, circuit: QuantumCircuit = None):
        """
        Initialize the execution manager.

        Args:
            circuit: The quantum circuit to simulate (optional)
        """
        self.circuit = circuit
        self.analyzer = CircuitAnalyzer(circuit) if circuit else None
        self.state = None  # Will be initialized based on analysis
        self.current_method = None  # Will be set when state is initialized

    def determine_simulation_method(self, circuit: QuantumCircuit):
        """
        Determine the optimal simulation method for a circuit.

        Args:
            circuit: The quantum circuit to analyze

        Returns:
            A tuple of (method, parameters) where method is a string
            indicating the simulation method and parameters is a dict
            of method-specific parameters
        """
        if self.analyzer is None or self.circuit != circuit:
            self.circuit = circuit
            self.analyzer = CircuitAnalyzer(circuit)

        method = self.analyzer.get_optimal_simulation_method()

        # Convert method names to match test expectations
        if method == "sparse":
            method = "sparse_state_vector"

        # Simple circuits with few qubits should use state vector
        if circuit.num_qudits <= 2 and len(circuit.gates) <= 2:
            method = "state_vector"

        # For highly entangled circuits with many qubits, use tensor network
        if circuit.num_qudits > 8 and len(self.analyzer.entanglement_regions) > 2:
            method = "tensor_network"

        # For very large circuits with few gates, use sparse representation
        if circuit.num_qudits > 15 and len(circuit.gates) < circuit.num_qudits:
            method = "sparse_state_vector"

        return method, {}

    def create_initial_state(self, circuit: QuantumCircuit, method: str):
        """
        Create an initial state for the given circuit using the specified method.

        Args:
            circuit: The quantum circuit
            method: The simulation method to use

        Returns:
            The initial quantum state
        """
        if method == "state_vector":
            return StateVector(circuit.num_qudits, circuit.dimensions)
        elif method == "sparse_state_vector":
            return SparseStateVector(circuit.num_qudits, circuit.dimensions)
        elif method == "tensor_network":
            try:
                return TensorNetworkState(circuit.num_qudits, circuit.dimensions)
            except ImportError:
                logger.warning(
                    "TensorNetwork library not available, falling back to sparse representation."
                )
                return SparseStateVector(circuit.num_qudits, circuit.dimensions)
        else:
            raise ValueError(f"Unknown simulation method: {method}")

    def initialize_state(self, method: str):
        """Initialize the quantum state using the specified method."""
        if method == "state_vector":
            self.state = StateVector(self.circuit.num_qudits, self.circuit.dimensions)
        elif method == "sparse":
            self.state = SparseStateVector(self.circuit.num_qudits, self.circuit.dimensions)
        elif method == "tensor_network":
            try:
                self.state = TensorNetworkState(self.circuit.num_qudits, self.circuit.dimensions)
            except ImportError:
                logger.warning(
                    "TensorNetwork library not available, falling back to sparse representation."
                )
                self.state = SparseStateVector(self.circuit.num_qudits, self.circuit.dimensions)
        else:
            raise ValueError(f"Unknown simulation method: {method}")

        self.current_method = method

    def convert_state(self, target_method: str):
        """Convert the current state representation to the target method."""
        current_method = self.current_method
        if current_method == target_method:
            return  # No conversion needed

        logger.info(f"Converting state from {current_method} to {target_method}")

        # Handle conversions between different state representations
        if target_method == "state_vector":
            if current_method == "sparse_state_vector":
                # Convert sparse to dense
                dense_state = StateVector(self.circuit.num_qudits, self.circuit.dimensions)
                dense_state.vector = np.zeros(self.state.total_dimension, dtype=complex)
                for idx, amp in self.state.amplitudes.items():
                    dense_state.vector[idx] = amp
                self.state = dense_state
            elif current_method == "tensor_network":
                # Convert tensor network to dense
                self.state = self.state.to_state_vector()

        elif target_method == "sparse_state_vector":
            # Convert to state vector first, then to sparse
            if current_method != "state_vector":
                self.convert_state("state_vector")

            # Now convert from state vector to sparse
            sparse_state = SparseStateVector(self.circuit.num_qudits, self.circuit.dimensions)
            for idx, amp in enumerate(self.state.vector):
                if abs(amp) > 1e-10:  # Only store non-zero amplitudes
                    sparse_state.amplitudes[idx] = amp
            self.state = sparse_state

        elif target_method == "tensor_network":
            if current_method == "sparse_state_vector":
                # Convert via state vector as intermediate step
                dense_state = StateVector(self.circuit.num_qudits, self.circuit.dimensions)
                dense_state.vector = np.zeros(self.state.total_dimension, dtype=complex)
                for idx, amp in self.state.amplitudes.items():
                    dense_state.vector[idx] = amp

                # Create a tensor network state directly from the state vector
                tensor_state = TensorNetworkState(self.circuit.num_qudits, self.circuit.dimensions)
                # Initialize the tensor network with the state vector data
                tensor_state.from_state_vector(dense_state)
                self.state = tensor_state
            elif current_method == "state_vector":
                # Create tensor network directly from state vector
                tensor_state = TensorNetworkState(self.circuit.num_qudits, self.circuit.dimensions)
                tensor_state.from_state_vector(self.state)
                self.state = tensor_state

        self.current_method = target_method

    def execute_circuit(self):
        """Execute the quantum circuit using the optimal hybrid approach."""
        # Get the optimal initial method
        method = self.analyzer.get_optimal_simulation_method()

        # Initialize the state
        self.initialize_state(method)

        # Get the partitioned circuit
        partitions = self.analyzer.partition_circuit()

        # Execute each partition
        for gates, target_method in partitions:
            # Convert state to the target method if needed
            self.convert_state(target_method)

            # Execute the gates in this partition
            for gate in gates:
                self.state.apply_gate(gate)

        # Convert to state vector for final measurements
        if not isinstance(self.state, StateVector):
            self.convert_state("state_vector")

        return self.state
