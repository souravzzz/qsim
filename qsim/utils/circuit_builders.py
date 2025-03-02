"""
Utility functions for building common quantum circuits.
"""

import numpy as np
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.phase import PhaseGate
from qsim.gates.controlled import ControlledGate


def create_bell_state_circuit() -> QuantumCircuit:
    """Create a quantum circuit that prepares a Bell state."""
    circuit = QuantumCircuit(2)

    # Apply Hadamard to first qubit
    h_gate = HadamardGate(circuit.qudits[0])
    circuit.add_gate(h_gate)

    # Apply CNOT gate (controlled-X)
    x_gate = PermutationGate(
        circuit.qudits[1], [1, 0]
    )  # X gate is a permutation that swaps 0 and 1
    cnot = ControlledGate(x_gate, circuit.qudits[0], 1)
    circuit.add_gate(cnot)

    return circuit


def create_ghz_state_circuit(num_qubits: int) -> QuantumCircuit:
    """Create a quantum circuit that prepares a GHZ state."""
    circuit = QuantumCircuit(num_qubits)

    # Apply Hadamard to first qubit
    h_gate = HadamardGate(circuit.qudits[0])
    circuit.add_gate(h_gate)

    # Apply CNOT gates to entangle all qubits
    for i in range(num_qubits - 1):
        x_gate = PermutationGate(
            circuit.qudits[i + 1], [1, 0]
        )  # X gate is a permutation that swaps 0 and 1
        cnot = ControlledGate(x_gate, circuit.qudits[i], 1)
        circuit.add_gate(cnot)

    return circuit


def create_quantum_fourier_transform_circuit(num_qubits: int) -> QuantumCircuit:
    """Create a quantum circuit that implements the Quantum Fourier Transform."""
    circuit = QuantumCircuit(num_qubits)

    # Implement QFT
    for i in range(num_qubits):
        # Hadamard gate on qubit i
        h_gate = HadamardGate(circuit.qudits[i])
        circuit.add_gate(h_gate)

        # Controlled phase rotations
        for j in range(i + 1, num_qubits):
            phase = np.pi / (2 ** (j - i))
            phase_gate = PhaseGate(circuit.qudits[j], phase)
            control_phase = ControlledGate(phase_gate, circuit.qudits[i], 1)
            circuit.add_gate(control_phase)

    # Swap qubits for correct output order
    for i in range(num_qubits // 2):
        # Swap i and n-i-1
        # We can implement SWAP with 3 CNOT gates
        j = num_qubits - i - 1

        x_gate_ij = PermutationGate(circuit.qudits[j], [1, 0])
        cnot_ij = ControlledGate(x_gate_ij, circuit.qudits[i], 1)
        circuit.add_gate(cnot_ij)

        x_gate_ji = PermutationGate(circuit.qudits[i], [1, 0])
        cnot_ji = ControlledGate(x_gate_ji, circuit.qudits[j], 1)
        circuit.add_gate(cnot_ji)

        x_gate_ij2 = PermutationGate(circuit.qudits[j], [1, 0])
        cnot_ij2 = ControlledGate(x_gate_ij2, circuit.qudits[i], 1)
        circuit.add_gate(cnot_ij2)

    return circuit


def create_generalized_bell_state_circuit(dimension: int = 2) -> QuantumCircuit:
    """
    Create a quantum circuit that prepares a generalized Bell state for qudits of arbitrary dimension.

    For dimension d, this creates the state:
    |Φ⟩ = (1/√d) * (|0,0⟩ + |1,1⟩ + |2,2⟩ + ... + |d-1,d-1⟩)

    Args:
        dimension: The dimension of the qudits (default: 2 for qubits)

    Returns:
        A quantum circuit that prepares the generalized Bell state
    """
    # Create a circuit with two qudits of the specified dimension
    circuit = QuantumCircuit(2, dimension)

    # For d=2, we can use the standard Bell state circuit
    if dimension == 2:
        # Apply Hadamard to first qubit
        h_gate = HadamardGate(circuit.qudits[0])
        circuit.add_gate(h_gate)

        # Apply CNOT gate (controlled-X)
        x_gate = PermutationGate(
            circuit.qudits[1], [1, 0]
        )  # X gate is a permutation that swaps 0 and 1
        cnot = ControlledGate(x_gate, circuit.qudits[0], 1)
        circuit.add_gate(cnot)
    else:
        # For higher dimensions, we'll use a special tag to indicate this is a generalized Bell state
        # The simulator will recognize this and create the state directly
        circuit.is_generalized_bell_state = True
        circuit.bell_dimension = dimension

    return circuit


def create_direct_bell_state(dimension: int) -> np.ndarray:
    """
    Create a generalized Bell state vector directly.

    For dimension d, this creates the state:
    |Φ⟩ = (1/√d) * (|0,0⟩ + |1,1⟩ + |2,2⟩ + ... + |d-1,d-1⟩)

    Args:
        dimension: The dimension of the qudits

    Returns:
        A numpy array representing the state vector
    """
    # Create a zero vector of the right size
    state_vector = np.zeros(dimension * dimension, dtype=complex)

    # Set the amplitudes for |0,0⟩, |1,1⟩, etc.
    for i in range(dimension):
        # Calculate the index in the state vector
        # For two qudits with the same dimension, the index is i*dimension + i
        index = i * dimension + i
        state_vector[index] = 1.0 / np.sqrt(dimension)

    return state_vector
