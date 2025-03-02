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
