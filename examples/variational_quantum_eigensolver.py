#!/usr/bin/env python3
"""
Variational Quantum Eigensolver (VQE) Example

This example demonstrates how to implement a simple Variational Quantum Eigensolver (VQE)
using the QSim quantum circuit simulator. VQE is a hybrid quantum-classical algorithm
used to find the ground state energy of a quantum system.

In this example, we:
1. Define a simple Hamiltonian (e.g., for a hydrogen molecule)
2. Create a parameterized quantum circuit (ansatz)
3. Use classical optimization to find the parameters that minimize the energy
4. Demonstrate the hybrid quantum-classical approach of VQE
"""

import numpy as np
from typing import List, Tuple, Callable
from scipy.optimize import minimize
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.phase import PhaseGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.controlled import ControlledGate


class VQEHamiltonian:
    """Class representing a Hamiltonian for VQE."""

    def __init__(self, pauli_terms: List[Tuple[str, float]]):
        """
        Initialize a Hamiltonian with Pauli terms.

        Args:
            pauli_terms: List of (Pauli string, coefficient) tuples
                         e.g., [("ZZ", 0.5), ("XX", 0.3), ("YY", 0.3), ("II", -1.5)]
        """
        self.pauli_terms = pauli_terms

    def get_expectation_value(self, state_vector: np.ndarray) -> float:
        """
        Calculate the expectation value of the Hamiltonian for a given state vector.

        Args:
            state_vector: The quantum state vector

        Returns:
            The expectation value <ψ|H|ψ>
        """
        expectation = 0.0

        for pauli_string, coefficient in self.pauli_terms:
            # Calculate the expectation value for each Pauli term
            term_expectation = self._get_pauli_expectation(pauli_string, state_vector)
            expectation += coefficient * term_expectation

        return expectation

    def _get_pauli_expectation(self, pauli_string: str, state_vector: np.ndarray) -> float:
        """
        Calculate the expectation value of a Pauli string for a given state vector.

        Args:
            pauli_string: String of Pauli operators (e.g., "ZZ", "XY")
            state_vector: The quantum state vector

        Returns:
            The expectation value <ψ|P|ψ> where P is the Pauli operator
        """
        num_qubits = int(np.log2(len(state_vector)))

        # Ensure the Pauli string has the right length
        if len(pauli_string) != num_qubits:
            raise ValueError(
                f"Pauli string length ({len(pauli_string)}) must match number of qubits ({num_qubits})"
            )

        # Identity matrix
        I = np.array([[1, 0], [0, 1]], dtype=complex)

        # Pauli matrices
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        # Map from Pauli character to matrix
        pauli_map = {"I": I, "X": X, "Y": Y, "Z": Z}

        # Construct the full operator using tensor products
        operator = np.array([[1]], dtype=complex)

        for pauli_char in pauli_string:
            pauli_matrix = pauli_map.get(pauli_char, I)
            operator = np.kron(operator, pauli_matrix)

        # Calculate the expectation value
        expectation = np.real(np.vdot(state_vector, operator @ state_vector))

        return expectation


def create_h2_hamiltonian() -> VQEHamiltonian:
    """
    Create a simplified Hamiltonian for the hydrogen molecule (H2).

    This is a simplified model with typical values for the H2 molecule
    at equilibrium bond length.

    Returns:
        A VQEHamiltonian object representing the H2 molecule
    """
    # Simplified H2 Hamiltonian terms
    # These coefficients are typical for H2 at equilibrium bond length
    pauli_terms = [
        ("II", -1.052373245772859),
        ("IZ", 0.39793742484318045),
        ("ZI", -0.39793742484318045),
        ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423156),
    ]

    return VQEHamiltonian(pauli_terms)


def create_vqe_ansatz(parameters: List[float]) -> QuantumCircuit:
    """
    Create a parameterized quantum circuit (ansatz) for VQE.

    This implements a simple hardware-efficient ansatz with rotations and entanglement.

    Args:
        parameters: List of rotation angles for the parameterized gates

    Returns:
        A parameterized quantum circuit
    """
    # For H2, we need 2 qubits
    circuit = QuantumCircuit(2)

    # Initial state preparation - Hadamard on both qubits
    for qubit in circuit.qudits:
        h_gate = HadamardGate(qubit)
        circuit.add_gate(h_gate)

    # First layer of parameterized rotations (Rz gates)
    for i, qubit in enumerate(circuit.qudits):
        phase_gate = PhaseGate(qubit, parameters[i])
        circuit.add_gate(phase_gate)

    # Entanglement layer - CNOT between qubits
    # Create a permutation list for X gate (bit flip) [1, 0]
    x_perm_list = [1, 0]  # For a qubit, this swaps states 0 and 1
    x_gate = PermutationGate(circuit.qudits[1], x_perm_list)
    cnot = ControlledGate(
        x_gate, circuit.qudits[0], 1  # Target gate  # Control qudit  # Control value
    )
    circuit.add_gate(cnot)

    # Second layer of parameterized rotations
    for i, qubit in enumerate(circuit.qudits):
        phase_gate = PhaseGate(qubit, parameters[i + len(circuit.qudits)])
        circuit.add_gate(phase_gate)

    return circuit


def vqe_objective_function(parameters: List[float], hamiltonian: VQEHamiltonian) -> float:
    """
    Objective function for VQE optimization.

    Args:
        parameters: List of circuit parameters to optimize
        hamiltonian: The Hamiltonian to calculate the energy for

    Returns:
        The energy expectation value for the given parameters
    """
    # Create the parameterized circuit
    circuit = create_vqe_ansatz(parameters)

    # Create a simulator
    simulator = HybridQuantumSimulator()

    # Simulate the circuit
    final_state = simulator.simulate(circuit)

    # Get the state vector
    amplitudes = final_state.get_amplitudes()

    # Calculate the energy expectation value
    energy = hamiltonian.get_expectation_value(amplitudes)

    return energy


def run_vqe(hamiltonian: VQEHamiltonian, initial_params: List[float]) -> Tuple[float, List[float]]:
    """
    Run the VQE algorithm to find the ground state energy.

    Args:
        hamiltonian: The Hamiltonian to minimize
        initial_params: Initial parameters for the ansatz

    Returns:
        Tuple of (minimum energy, optimal parameters)
    """
    # Define the objective function (capture the hamiltonian in the closure)
    objective = lambda params: vqe_objective_function(params, hamiltonian)

    # Run the classical optimization
    result = minimize(objective, initial_params, method="COBYLA", options={"maxiter": 100})

    # Return the minimum energy and optimal parameters
    return result.fun, result.x


def main():
    """Run the VQE example."""
    print("Running Variational Quantum Eigensolver (VQE) for H2 molecule")

    # Create the H2 Hamiltonian
    h2_hamiltonian = create_h2_hamiltonian()

    # Initial parameters (random starting point)
    np.random.seed(42)  # For reproducibility
    num_parameters = 4  # 2 qubits * 2 layers
    initial_params = np.random.uniform(0, 2 * np.pi, num_parameters)

    print(f"Initial parameters: {initial_params}")

    # Calculate initial energy
    initial_energy = vqe_objective_function(initial_params, h2_hamiltonian)
    print(f"Initial energy: {initial_energy:.6f} Hartree")

    # Run VQE
    print("\nOptimizing parameters...")
    min_energy, opt_params = run_vqe(h2_hamiltonian, initial_params)

    print("\nOptimization complete!")
    print(f"Optimal parameters: {opt_params}")
    print(f"Ground state energy: {min_energy:.6f} Hartree")

    # The exact ground state energy of H2 at equilibrium bond length is approximately -1.137 Hartree
    exact_energy = -1.137
    print(f"Exact ground state energy: {exact_energy:.6f} Hartree")
    print(f"Error: {abs(min_energy - exact_energy):.6f} Hartree")

    # Create the optimized circuit
    optimized_circuit = create_vqe_ansatz(opt_params)

    # Simulate and measure the optimized circuit
    simulator = HybridQuantumSimulator()
    results = simulator.simulate_and_measure(optimized_circuit, num_shots=1000)

    # Print measurement results
    print("\nGround state measurement results (1000 shots):")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for outcome, count in sorted_results:
        print(f"|{outcome}⟩: {count} shots ({count/10:.1f}%)")


if __name__ == "__main__":
    main()
