#!/usr/bin/env python3
"""
GHZ State Scaling Benchmark

This benchmark tests how well the simulator scales with increasing system size using GHZ state circuits.
GHZ circuits generate highly entangled states but remain sparse, making them ideal for testing
block simulation and sparse matrix optimizations.

The benchmark creates GHZ states with different qudit dimensions:
- GHZ (qubits): d=2, up to 32 qudits
- GHZ (qutrits): d=3, up to 16 qudits
- GHZ (ququads): d=4, up to 8 qudits
- GHZ (ququints): d=5, up to 4 qudits

It measures execution time vs. number of qudits and compares sparse vs. dense simulation methods.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import traceback
import gc
import os
import argparse
from typing import Dict, List, Tuple, Optional, Any
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.controlled import ControlledGate
from qsim.analysis.circuit_analyzer import CircuitAnalyzer

# Number of iterations for each benchmark to compute average execution time
NUM_ITERATIONS = 3
# Memory threshold (in GB) to consider a test at risk of OOM
MEMORY_THRESHOLD_GB = 0.9 * psutil.virtual_memory().total / (1024**3)


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="GHZ State Scaling Benchmark")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save benchmark plots (default: current directory)",
    )
    return parser.parse_args()


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.

    Returns:
        Dictionary with memory usage in MB and percentage
    """
    process = psutil.Process()
    memory_info = process.memory_info()

    return {
        "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
        "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
        "percent": process.memory_percent(),
        "available_gb": psutil.virtual_memory().available / (1024**3),
    }


def create_ghz_state_circuit_d(num_qudits: int, dimension: int) -> QuantumCircuit:
    """
    Create a quantum circuit that prepares a GHZ state for qudits of arbitrary dimension.

    For dimension d, this creates the state: (|0...0⟩ + |1...1⟩ + ... + |(d-1)...(d-1)⟩)/sqrt(d)

    Args:
        num_qudits: Number of qudits in the circuit
        dimension: Dimension of each qudit

    Returns:
        A quantum circuit that prepares a GHZ state
    """
    circuit = QuantumCircuit(num_qudits, dimension)

    # First, apply a Hadamard-like gate to the first qudit to create superposition
    # For qubits (d=2), this is just the standard Hadamard
    if dimension == 2:
        h_gate = HadamardGate(circuit.qudits[0])
        circuit.add_gate(h_gate)
    else:
        # For higher dimensions, we need to create a generalized Hadamard
        # This is a permutation that creates equal superposition of all basis states
        # We'll use a permutation gate with a custom matrix

        # Create a d×d matrix with 1/sqrt(d) in all entries
        d = dimension
        matrix = np.ones((d, d), dtype=complex) / np.sqrt(d)

        # Apply this gate to the first qudit
        from qsim.gates.base import Gate

        gen_h_gate = Gate("GenH", [circuit.qudits[0]], matrix)
        circuit.add_gate(gen_h_gate)

    # Now apply controlled permutation gates to entangle all qudits
    for i in range(num_qudits - 1):
        # Create a cyclic permutation gate [1,2,...,d-1,0] for the target qudit
        # This maps |0⟩→|1⟩, |1⟩→|2⟩, ..., |d-1⟩→|0⟩
        perm_list = [(j + 1) % dimension for j in range(dimension)]
        perm_gate = PermutationGate(circuit.qudits[i + 1], perm_list)

        # For each possible value of the control qudit (except 0), apply a controlled gate
        for control_val in range(1, dimension):
            # Create a controlled gate that applies the permutation when control qudit is |control_val⟩
            controlled_perm = ControlledGate(perm_gate, circuit.qudits[i], control_val)
            circuit.add_gate(controlled_perm)

            # For higher dimensions, we need to apply the permutation multiple times
            # to get the correct mapping for each control value
            for _ in range(control_val - 1):
                controlled_perm = ControlledGate(perm_gate, circuit.qudits[i], control_val)
                circuit.add_gate(controlled_perm)

    return circuit


def benchmark_circuit(
    circuit: QuantumCircuit, name: str, force_method: Optional[str] = None, num_iterations: int = 1
) -> Dict[str, Any]:
    """
    Benchmark the simulation of a circuit.

    Args:
        circuit: The quantum circuit to benchmark
        name: A name for the circuit
        force_method: Optional method to force the simulator to use
        num_iterations: Number of iterations to run for averaging execution time

    Returns:
        Dictionary with benchmark results
    """
    # Analyze the circuit
    analyzer = CircuitAnalyzer(circuit)
    optimal_method = analyzer.get_optimal_simulation_method()

    # Initialize result dictionary
    result = {
        "name": name,
        "num_qudits": circuit.num_qudits,
        "dimension": circuit.dimensions[0] if len(set(circuit.dimensions)) == 1 else "mixed",
        "num_gates": len(circuit.gates),
        "optimal_method": optimal_method,
        "used_method": force_method if force_method else optimal_method,
        "simulation_time": 0.0,
        "simulation_times": [],
        "memory_usage_mb": 0.0,
        "peak_memory_mb": 0.0,
        "success": False,
        "error": None,
    }

    # Check if the simulation might exceed memory limits
    state_size = np.prod([circuit.dimensions[i] for i in range(circuit.num_qudits)])
    estimated_memory_gb = state_size * 16 / (1024**3)  # Rough estimate for dense state

    if force_method == "dense" and estimated_memory_gb > MEMORY_THRESHOLD_GB:
        result["error"] = (
            f"Skipped: Estimated memory usage ({estimated_memory_gb:.2f} GB) exceeds threshold"
        )
        return result

    # Create a simulator
    simulator = HybridQuantumSimulator()

    # Run multiple iterations to get average execution time
    successful_iterations = 0
    peak_memory = 0

    # Print iteration information
    print(f"  Running {num_iterations} iterations for {name} ({force_method})...")

    for iteration in range(num_iterations):
        try:
            # Force garbage collection before each run
            gc.collect()

            # Record initial memory usage
            initial_memory = get_memory_usage()

            # Measure simulation time
            start_time = time.time()

            # If force_method is specified, we'll manually apply the gates using the forced method
            if force_method:
                from qsim.execution.hybrid_execution_manager import HybridExecutionManager
                from qsim.states.state_vector import StateVector
                from qsim.states.sparse_state_vector import SparseStateVector
                from qsim.states.tensor_network_state import TensorNetworkState

                # Create the appropriate state based on the forced method
                if force_method == "dense":
                    state = StateVector(circuit.num_qudits, circuit.dimensions)
                elif force_method == "sparse":
                    state = SparseStateVector(circuit.num_qudits, circuit.dimensions)
                elif force_method == "tensor_network":
                    state = TensorNetworkState(circuit.num_qudits, circuit.dimensions)
                else:
                    raise ValueError(f"Unknown simulation method: {force_method}")

                # Apply all gates to the state
                for gate in circuit.gates:
                    state.apply_gate(gate)

                final_state = state
            else:
                final_state = simulator.simulate(circuit)

            end_time = time.time()
            simulation_time = end_time - start_time

            # Record final memory usage
            final_memory = get_memory_usage()
            memory_used = final_memory["rss_mb"] - initial_memory["rss_mb"]
            peak_memory = max(peak_memory, final_memory["rss_mb"])

            # Get some properties of the final state
            try:
                amplitudes = final_state.get_amplitudes()
                num_nonzero = np.count_nonzero(np.abs(amplitudes) > 1e-10)
                sparsity = 1.0 - (num_nonzero / len(amplitudes))

                # Only set these properties on the first successful iteration
                if successful_iterations == 0:
                    result["state_size"] = len(amplitudes)
                    result["nonzero_amplitudes"] = num_nonzero
                    result["sparsity"] = sparsity
            except Exception as e:
                # If we can't get amplitudes, just continue
                if successful_iterations == 0:
                    result["state_size"] = state_size
                    result["nonzero_amplitudes"] = "unknown"
                    result["sparsity"] = "unknown"

            # Record the time for this iteration
            result["simulation_times"].append(simulation_time)
            successful_iterations += 1

            # Print progress
            print(
                f"    Iteration {iteration+1}/{num_iterations}: time={simulation_time:.4f}s, mem={memory_used:.1f}MB"
            )

        except Exception as e:
            # If we encounter an error, record it and break the loop
            result["error"] = f"{type(e).__name__}: {str(e)}"
            print(f"    ERROR: {type(e).__name__}")
            break

    # Calculate average time if we had any successful iterations
    if successful_iterations > 0:
        result["simulation_time"] = sum(result["simulation_times"]) / successful_iterations
        result["memory_usage_mb"] = memory_used
        result["peak_memory_mb"] = peak_memory
        result["success"] = True

    # Force garbage collection after benchmark
    gc.collect()

    return result


def run_ghz_benchmarks(max_iterations: int = NUM_ITERATIONS) -> List[Dict]:
    """
    Run benchmarks on GHZ state circuits with different qudit dimensions.

    Args:
        max_iterations: Maximum number of iterations for each benchmark

    Returns:
        List of benchmark results
    """
    benchmarks = []

    # Define the dimensions and maximum number of qudits to test
    dimensions_to_test = {
        2: [2, 4, 8, 16, 32],  # qubits: up to 32 qudits
        3: [2, 4, 8, 16],  # qutrits: up to 16 qudits
        4: [2, 4, 8],  # ququads: up to 8 qudits
        5: [2, 4],  # ququints: up to 4 qudits
    }

    print(f"Running benchmarks for {len(dimensions_to_test)} dimensions...")

    for dimension, num_qudits_range in dimensions_to_test.items():
        dimension_name = {2: "qubit", 3: "qutrit", 4: "ququad", 5: "ququint"}[dimension]
        print(f"\nTesting dimension d={dimension} ({dimension_name}s)")

        for num_qudits in num_qudits_range:
            print(f"\nGHZ with {num_qudits} {dimension_name}s (d={dimension})")

            # Create the circuit once for both simulation methods
            circuit = create_ghz_state_circuit_d(num_qudits, dimension)

            # Check if we've had too many failures and should skip larger circuits
            if any(
                b["error"]
                and "memory" in b["error"].lower()
                and b["dimension"] == dimension
                and b["num_qudits"] < num_qudits
                for b in benchmarks
            ):
                print(f"  Skipping {num_qudits} {dimension_name}s due to previous memory errors")
                continue

            # Benchmark with sparse simulation
            print(f"  Benchmarking GHZ state with {num_qudits} {dimension_name}s (sparse)...")
            sparse_results = benchmark_circuit(
                circuit, f"GHZ-{dimension_name} (n={num_qudits}, sparse)", "sparse", max_iterations
            )
            benchmarks.append(sparse_results)

            if sparse_results["success"]:
                print(
                    f"    Sparse simulation: {sparse_results['simulation_time']:.4f} seconds, "
                    f"Memory: {sparse_results['memory_usage_mb']:.2f} MB"
                )
            else:
                print(f"    Sparse simulation failed: {sparse_results['error']}")
                # If sparse fails, dense will likely fail too, so skip it
                continue

            # Benchmark with dense simulation
            print(f"  Benchmarking GHZ state with {num_qudits} {dimension_name}s (dense)...")
            dense_results = benchmark_circuit(
                circuit, f"GHZ-{dimension_name} (n={num_qudits}, dense)", "dense", max_iterations
            )
            benchmarks.append(dense_results)

            if dense_results["success"]:
                print(
                    f"    Dense simulation: {dense_results['simulation_time']:.4f} seconds, "
                    f"Memory: {dense_results['memory_usage_mb']:.2f} MB"
                )
            else:
                print(f"    Dense simulation failed: {dense_results['error']}")

    return benchmarks


def plot_benchmark_results(benchmarks: List[Dict], output_dir: str) -> None:
    """
    Plot the benchmark results.

    Args:
        benchmarks: List of benchmark results
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Filter out failed benchmarks
    successful_benchmarks = [b for b in benchmarks if b["success"]]

    if not successful_benchmarks:
        print("No successful benchmarks to plot.")
        return

    # Create a figure with subplots - 2 rows, 2 columns for time and memory
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    time_axs = axs[0]
    mem_axs = axs[1]

    # Colors for different dimensions
    colors = {2: "blue", 3: "green", 4: "red", 5: "purple"}

    # Markers for different simulation methods
    markers = {"sparse": "o", "dense": "s"}

    # Group benchmarks by dimension
    for dim_idx, dimension in enumerate([2, 3, 4, 5]):
        dimension_name = {2: "qubit", 3: "qutrit", 4: "ququad", 5: "ququint"}[dimension]
        time_ax = time_axs[dim_idx % 2]
        mem_ax = mem_axs[dim_idx % 2]

        # Filter benchmarks for this dimension
        dim_benchmarks = [b for b in successful_benchmarks if b["dimension"] == dimension]

        if not dim_benchmarks:
            continue

        # Group by simulation method
        for method in ["sparse", "dense"]:
            method_benchmarks = [b for b in dim_benchmarks if b["used_method"] == method]

            if not method_benchmarks:
                continue

            # Sort by number of qudits
            method_benchmarks.sort(key=lambda b: b["num_qudits"])

            # Extract data for plotting
            num_qudits = [b["num_qudits"] for b in method_benchmarks]
            times = [b["simulation_time"] for b in method_benchmarks]
            memories = [b["peak_memory_mb"] for b in method_benchmarks]

            # Plot time
            time_ax.plot(
                num_qudits,
                times,
                marker=markers[method],
                color=colors[dimension],
                linestyle="-" if method == "sparse" else "--",
                label=f"d={dimension}, {method.capitalize()}",
            )

            # Plot memory
            mem_ax.plot(
                num_qudits,
                memories,
                marker=markers[method],
                color=colors[dimension],
                linestyle="-" if method == "sparse" else "--",
                label=f"d={dimension}, {method.capitalize()}",
            )

        # Set plot labels and title
        time_ax.set_xlabel("Number of Qudits")
        time_ax.set_ylabel("Simulation Time (s)")
        time_ax.set_title(f"Execution Time - GHZ State with d={dimension}")
        time_ax.grid(True)
        time_ax.legend()

        mem_ax.set_xlabel("Number of Qudits")
        mem_ax.set_ylabel("Memory Usage (MB)")
        mem_ax.set_title(f"Memory Usage - GHZ State with d={dimension}")
        mem_ax.grid(True)
        mem_ax.legend()

        # Use log scale for y-axis if values vary significantly
        if len(times) > 1 and max(times) > 10 * min(times):
            time_ax.set_yscale("log")

        if len(memories) > 1 and max(memories) > 10 * min(memories):
            mem_ax.set_yscale("log")

    # Add a title for the entire figure
    fig.suptitle("GHZ State Scaling Benchmark: Performance Metrics", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle

    # Save the figure
    output_path = os.path.join(output_dir, "ghz_benchmark_results.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nBenchmark plot saved as '{output_path}'")

    # Create a combined plot for all dimensions - execution time
    plt.figure(figsize=(10, 8))

    for dimension in [2, 3, 4, 5]:
        dimension_name = {2: "qubit", 3: "qutrit", 4: "ququad", 5: "ququint"}[dimension]

        # Filter benchmarks for this dimension and sparse method
        dim_benchmarks = [
            b
            for b in successful_benchmarks
            if b["dimension"] == dimension and b["used_method"] == "sparse"
        ]

        if not dim_benchmarks:
            continue

        # Sort by number of qudits
        dim_benchmarks.sort(key=lambda b: b["num_qudits"])

        # Extract data for plotting
        num_qudits = [b["num_qudits"] for b in dim_benchmarks]
        times = [b["simulation_time"] for b in dim_benchmarks]

        # Plot
        plt.plot(
            num_qudits,
            times,
            marker="o",
            color=colors[dimension],
            label=f"d={dimension} ({dimension_name}s)",
        )

    # Set plot labels and title
    plt.xlabel("Number of Qudits")
    plt.ylabel("Simulation Time (s)")
    plt.title("GHZ State Scaling Benchmark: Comparison Across Qudit Dimensions (Sparse Simulation)")
    plt.grid(True)
    plt.legend()
    plt.yscale("log")

    # Save the figure
    output_path = os.path.join(output_dir, "ghz_benchmark_comparison_time.png")
    plt.savefig(output_path, dpi=300)
    print(f"Time comparison plot saved as '{output_path}'")

    # Create a combined plot for all dimensions - memory usage
    plt.figure(figsize=(10, 8))

    for dimension in [2, 3, 4, 5]:
        dimension_name = {2: "qubit", 3: "qutrit", 4: "ququad", 5: "ququint"}[dimension]

        # Filter benchmarks for this dimension and sparse method
        dim_benchmarks = [
            b
            for b in successful_benchmarks
            if b["dimension"] == dimension and b["used_method"] == "sparse"
        ]

        if not dim_benchmarks:
            continue

        # Sort by number of qudits
        dim_benchmarks.sort(key=lambda b: b["num_qudits"])

        # Extract data for plotting
        num_qudits = [b["num_qudits"] for b in dim_benchmarks]
        memories = [b["peak_memory_mb"] for b in dim_benchmarks]

        # Plot
        plt.plot(
            num_qudits,
            memories,
            marker="o",
            color=colors[dimension],
            label=f"d={dimension} ({dimension_name}s)",
        )

    # Set plot labels and title
    plt.xlabel("Number of Qudits")
    plt.ylabel("Memory Usage (MB)")
    plt.title(
        "GHZ State Scaling Benchmark: Memory Usage Across Qudit Dimensions (Sparse Simulation)"
    )
    plt.grid(True)
    plt.legend()
    plt.yscale("log")

    # Save the figure
    output_path = os.path.join(output_dir, "ghz_benchmark_comparison_memory.png")
    plt.savefig(output_path, dpi=300)
    print(f"Memory comparison plot saved as '{output_path}'")


def print_benchmark_summary(benchmarks: List[Dict]) -> None:
    """
    Print a summary of benchmark results.

    Args:
        benchmarks: List of benchmark results
    """
    print("\n" + "=" * 120)
    print("GHZ STATE BENCHMARK SUMMARY")
    print("=" * 120)
    print(
        f"{'Circuit Type':<25} {'Dimension':<10} {'Qudits':<8} {'Gates':<8} "
        f"{'Method':<10} {'Time (s)':<10} {'Memory (MB)':<12} {'Sparsity':<10} {'Status':<10}"
    )
    print("-" * 120)

    # Group benchmarks by dimension
    for dimension in [2, 3, 4, 5]:
        dimension_name = {2: "qubit", 3: "qutrit", 4: "ququad", 5: "ququint"}[dimension]

        # Filter benchmarks for this dimension
        dim_benchmarks = [b for b in benchmarks if b["dimension"] == dimension]

        if not dim_benchmarks:
            continue

        # Sort by number of qudits and then by simulation method
        dim_benchmarks.sort(key=lambda b: (b["num_qudits"], b["used_method"]))

        # Print results
        for result in dim_benchmarks:
            status = "Success" if result["success"] else "Failed"
            sparsity = result.get("sparsity", "N/A")
            if isinstance(sparsity, float):
                sparsity_str = f"{sparsity:.4f}"
            else:
                sparsity_str = str(sparsity)

            print(
                f"{result['name']:<25} {result['dimension']:<10} {result['num_qudits']:<8} "
                f"{result['num_gates']:<8} {result['used_method']:<10} "
                f"{result['simulation_time']:<10.4f} {result['peak_memory_mb']:<12.2f} "
                f"{sparsity_str:<10} {status:<10}"
            )

            # If there was an error, print it indented
            if not result["success"] and result["error"]:
                print(f"    Error: {result['error']}")

        print("-" * 120)

    # Print speedup of sparse vs. dense simulation
    print("\nSpeedup of Sparse vs. Dense Simulation:")

    for dimension in [2, 3, 4, 5]:
        dimension_name = {2: "qubit", 3: "qutrit", 4: "ququad", 5: "ququint"}[dimension]

        # Filter successful benchmarks for this dimension
        dim_benchmarks = [b for b in benchmarks if b["dimension"] == dimension and b["success"]]

        if not dim_benchmarks:
            continue

        print(f"\nDimension {dimension} ({dimension_name}s):")

        # Group by number of qudits
        qudit_counts = sorted(set(b["num_qudits"] for b in dim_benchmarks))

        for num_qudits in qudit_counts:
            # Get sparse and dense results for this number of qudits
            sparse_result = next(
                (
                    b
                    for b in dim_benchmarks
                    if b["num_qudits"] == num_qudits and b["used_method"] == "sparse"
                ),
                None,
            )
            dense_result = next(
                (
                    b
                    for b in dim_benchmarks
                    if b["num_qudits"] == num_qudits and b["used_method"] == "dense"
                ),
                None,
            )

            if sparse_result and dense_result:
                time_speedup = dense_result["simulation_time"] / sparse_result["simulation_time"]
                memory_ratio = dense_result["peak_memory_mb"] / sparse_result["peak_memory_mb"]
                print(
                    f"  {num_qudits} qudits: "
                    f"Sparse=[{sparse_result['simulation_time']:.4f}s, {sparse_result['peak_memory_mb']:.2f}MB], "
                    f"Dense=[{dense_result['simulation_time']:.4f}s, {dense_result['peak_memory_mb']:.2f}MB], "
                    f"Speedup={time_speedup:.2f}x, Memory Ratio={memory_ratio:.2f}x"
                )

    print("=" * 120)

    # Print iteration details for successful benchmarks
    print("\nIteration Details (successful benchmarks only):")
    print("-" * 120)
    print(
        f"{'Circuit Type':<25} {'Dimension':<10} {'Qudits':<8} {'Method':<10} {'Iterations':<10} {'Times (s)'}"
    )
    print("-" * 120)

    for result in [b for b in benchmarks if b["success"]]:
        times_str = ", ".join([f"{t:.4f}" for t in result["simulation_times"]])
        print(
            f"{result['name']:<25} {result['dimension']:<10} {result['num_qudits']:<8} "
            f"{result['used_method']:<10} {len(result['simulation_times']):<10} [{times_str}]"
        )

    print("=" * 120)


def main():
    """Run the GHZ state scaling benchmark."""
    # Parse command line arguments
    args = parse_args()

    print("Running GHZ State Scaling Benchmark")
    print("This benchmark tests how well the simulator scales with increasing system size")
    print("using GHZ state circuits with different qudit dimensions.")
    print(f"Each test will be run up to {NUM_ITERATIONS} times to compute average execution time.")
    print(f"Memory threshold for skipping tests: {MEMORY_THRESHOLD_GB:.2f} GB")
    print(f"Output directory for plots: {args.output_dir}")

    try:
        # Run benchmarks
        benchmarks = run_ghz_benchmarks(NUM_ITERATIONS)

        # Print summary
        print_benchmark_summary(benchmarks)

        # Plot results
        plot_benchmark_results(benchmarks, args.output_dir)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nBenchmark failed with error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
