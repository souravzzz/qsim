#!/usr/bin/env python3
"""
Hybrid Simulation Benchmark Example

This example demonstrates the hybrid simulation capabilities of the QSim quantum circuit
simulator by creating different types of circuits that trigger different simulation methods.
It benchmarks the performance of the simulator on these different circuit types.

In this example, we:
1. Create circuits with different properties (sparse, dense, highly entangled)
2. Analyze the circuits to determine the optimal simulation method
3. Simulate the circuits and measure performance
4. Compare the results of different simulation methods
"""

import time
import numpy as np
from typing import Dict, List, Tuple
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.phase import PhaseGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.controlled import ControlledGate
from qsim.analysis.circuit_analyzer import CircuitAnalyzer
from qsim.utils.circuit_builders import (
    create_ghz_state_circuit,
    create_quantum_fourier_transform_circuit,
)


def create_sparse_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Create a circuit that produces a sparse state vector.

    This circuit applies Hadamard gates to only a few qubits,
    resulting in a state with many zero amplitudes.

    Args:
        num_qubits: Number of qubits in the circuit

    Returns:
        A quantum circuit that produces a sparse state
    """
    circuit = QuantumCircuit(num_qubits)

    # Apply Hadamard gates to only the first 3 qubits
    for i in range(min(3, num_qubits)):
        h_gate = HadamardGate(circuit.qudits[i])
        circuit.add_gate(h_gate)

    # Apply a few controlled operations to maintain sparsity
    if num_qubits >= 4:
        for i in range(2):
            control_qubit = circuit.qudits[i]
            target_qubit = circuit.qudits[i + 2]

            # Create a permutation list for X gate (bit flip) [1, 0]
            x_perm_list = [1, 0]  # For a qubit, this swaps states 0 and 1
            x_gate = PermutationGate(target_qubit, x_perm_list)
            cnot = ControlledGate(
                x_gate, control_qubit, 1  # Target gate  # Control qudit  # Control value
            )
            circuit.add_gate(cnot)

    return circuit


def create_dense_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Create a circuit that produces a dense state vector.

    This circuit applies Hadamard gates to all qubits,
    resulting in a state with all non-zero amplitudes.

    Args:
        num_qubits: Number of qubits in the circuit

    Returns:
        A quantum circuit that produces a dense state
    """
    circuit = QuantumCircuit(num_qubits)

    # Apply Hadamard gates to all qubits
    for qubit in circuit.qudits:
        h_gate = HadamardGate(qubit)
        circuit.add_gate(h_gate)

    # Apply some phase gates to make the state more interesting
    for i, qubit in enumerate(circuit.qudits):
        phase = np.pi / (i + 2)
        phase_gate = PhaseGate(qubit, phase)
        circuit.add_gate(phase_gate)

    return circuit


def create_highly_entangled_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Create a circuit with high entanglement between qubits.

    This circuit creates a highly entangled state by applying
    a series of controlled operations between many qubit pairs.

    Args:
        num_qubits: Number of qubits in the circuit

    Returns:
        A quantum circuit that produces a highly entangled state
    """
    # Start with a GHZ state circuit
    circuit = create_ghz_state_circuit(num_qubits)

    # Add additional entangling gates
    for i in range(num_qubits - 1):
        for j in range(i + 1, min(i + 3, num_qubits)):
            control_qubit = circuit.qudits[i]
            target_qubit = circuit.qudits[j]

            # Apply controlled phase gate
            phase_gate = PhaseGate(target_qubit, np.pi / 4)
            controlled_phase = ControlledGate(
                phase_gate, control_qubit, 1  # Target gate  # Control qudit  # Control value
            )
            circuit.add_gate(controlled_phase)

    return circuit


def create_block_structured_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Create a circuit with block structure (localized operations).

    This circuit applies operations only within local blocks of qubits,
    which should be efficiently simulable using block simulation methods.

    Args:
        num_qubits: Number of qubits in the circuit

    Returns:
        A quantum circuit with block structure
    """
    circuit = QuantumCircuit(num_qubits)

    # Apply Hadamard gates to all qubits
    for qubit in circuit.qudits:
        h_gate = HadamardGate(qubit)
        circuit.add_gate(h_gate)

    # Apply operations only within blocks of 4 qubits
    block_size = 4
    for block_start in range(0, num_qubits, block_size):
        block_end = min(block_start + block_size, num_qubits)

        # Apply operations within this block
        for i in range(block_start, block_end - 1):
            control_qubit = circuit.qudits[i]
            target_qubit = circuit.qudits[i + 1]

            # Apply CNOT gate
            # Create a permutation list for X gate (bit flip) [1, 0]
            x_perm_list = [1, 0]  # For a qubit, this swaps states 0 and 1
            x_gate = PermutationGate(target_qubit, x_perm_list)
            cnot = ControlledGate(
                x_gate, control_qubit, 1  # Target gate  # Control qudit  # Control value
            )
            circuit.add_gate(cnot)

    return circuit


def benchmark_circuit(circuit: QuantumCircuit, name: str) -> Dict:
    """
    Benchmark the simulation of a circuit.

    Args:
        circuit: The quantum circuit to benchmark
        name: A name for the circuit

    Returns:
        Dictionary with benchmark results
    """
    # Analyze the circuit
    analyzer = CircuitAnalyzer(circuit)
    optimal_method = analyzer.get_optimal_simulation_method()

    # Create a simulator
    simulator = HybridQuantumSimulator()

    # Measure simulation time
    start_time = time.time()
    final_state = simulator.simulate(circuit)
    end_time = time.time()

    simulation_time = end_time - start_time

    # Get some properties of the final state
    amplitudes = final_state.get_amplitudes()
    num_nonzero = np.count_nonzero(np.abs(amplitudes) > 1e-10)
    sparsity = 1.0 - (num_nonzero / len(amplitudes))

    # Return benchmark results
    return {
        "name": name,
        "num_qubits": circuit.num_qudits,
        "num_gates": len(circuit.gates),
        "optimal_method": optimal_method,
        "simulation_time": simulation_time,
        "state_size": len(amplitudes),
        "nonzero_amplitudes": num_nonzero,
        "sparsity": sparsity,
    }


def run_benchmarks() -> List[Dict]:
    """
    Run benchmarks on different types of circuits.

    Returns:
        List of benchmark results
    """
    benchmarks = []

    # Test different circuit sizes
    qubit_counts = [4, 8, 12, 16]

    for num_qubits in qubit_counts:
        print(f"\nBenchmarking circuits with {num_qubits} qubits...")

        # Create and benchmark sparse circuit
        sparse_circuit = create_sparse_circuit(num_qubits)
        sparse_results = benchmark_circuit(sparse_circuit, f"Sparse ({num_qubits} qubits)")
        benchmarks.append(sparse_results)
        print(
            f"  Sparse circuit: {sparse_results['simulation_time']:.4f} seconds, method: {sparse_results['optimal_method']}"
        )

        # Create and benchmark dense circuit
        dense_circuit = create_dense_circuit(num_qubits)
        dense_results = benchmark_circuit(dense_circuit, f"Dense ({num_qubits} qubits)")
        benchmarks.append(dense_results)
        print(
            f"  Dense circuit: {dense_results['simulation_time']:.4f} seconds, method: {dense_results['optimal_method']}"
        )

        # Create and benchmark highly entangled circuit
        entangled_circuit = create_highly_entangled_circuit(num_qubits)
        entangled_results = benchmark_circuit(entangled_circuit, f"Entangled ({num_qubits} qubits)")
        benchmarks.append(entangled_results)
        print(
            f"  Entangled circuit: {entangled_results['simulation_time']:.4f} seconds, method: {entangled_results['optimal_method']}"
        )

        # Create and benchmark block structured circuit
        block_circuit = create_block_structured_circuit(num_qubits)
        block_results = benchmark_circuit(block_circuit, f"Block ({num_qubits} qubits)")
        benchmarks.append(block_results)
        print(
            f"  Block circuit: {block_results['simulation_time']:.4f} seconds, method: {block_results['optimal_method']}"
        )

        # Create and benchmark QFT circuit
        qft_circuit = create_quantum_fourier_transform_circuit(num_qubits)
        qft_results = benchmark_circuit(qft_circuit, f"QFT ({num_qubits} qubits)")
        benchmarks.append(qft_results)
        print(
            f"  QFT circuit: {qft_results['simulation_time']:.4f} seconds, method: {qft_results['optimal_method']}"
        )

    return benchmarks


def print_benchmark_summary(benchmarks: List[Dict]) -> None:
    """
    Print a summary of benchmark results.

    Args:
        benchmarks: List of benchmark results
    """
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(
        f"{'Circuit Type':<25} {'Qubits':<8} {'Gates':<8} {'Method':<15} {'Time (s)':<10} {'Sparsity':<10}"
    )
    print("-" * 80)

    for result in benchmarks:
        print(
            f"{result['name']:<25} {result['num_qubits']:<8} {result['num_gates']:<8} "
            f"{result['optimal_method']:<15} {result['simulation_time']:<10.4f} "
            f"{result['sparsity']:<10.4f}"
        )

    print("=" * 80)

    # Group by number of qubits and find the fastest method for each size
    qubit_counts = sorted(set(result["num_qubits"] for result in benchmarks))

    print("\nFastest simulation method by circuit size:")
    for num_qubits in qubit_counts:
        size_results = [r for r in benchmarks if r["num_qubits"] == num_qubits]
        fastest = min(size_results, key=lambda r: r["simulation_time"])
        print(
            f"{num_qubits} qubits: {fastest['name']} ({fastest['optimal_method']}) - {fastest['simulation_time']:.4f} seconds"
        )


def main():
    """Run the hybrid simulation benchmark example."""
    print("Running Hybrid Simulation Benchmark")
    print("This example tests the simulator's ability to select the optimal simulation method")
    print("for different types of quantum circuits.")

    # Run benchmarks
    benchmarks = run_benchmarks()

    # Print summary
    print_benchmark_summary(benchmarks)


if __name__ == "__main__":
    main()
