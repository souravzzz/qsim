# QSim Examples

This directory contains example scripts that demonstrate the capabilities of the QSim quantum circuit simulator. Each example focuses on a different aspect of quantum computing and showcases different features of the simulator.

## Running the Examples

All examples can be run directly from the command line. Make sure you have installed the QSim package first:

```bash
# From the root directory of the project
pip install -e .
```

Then run any example using Python:

```bash
python examples/basic_usage.py
```

You can also use the Makefile targets to run examples:

```bash
# Run basic examples
make examples

# Run advanced examples
make advanced-examples

# Run all examples
make all-examples
```

## Available Examples

### Basic Usage

**File:** `basic_usage.py`

A simple introduction to the QSim simulator, demonstrating how to create a quantum circuit, apply gates, and measure the results. This example creates a 3-qubit circuit with Hadamard gates and CNOT gates to create a simple entangled state.

### Quantum Phase Estimation

**File:** `quantum_phase_estimation.py`

Demonstrates the Quantum Phase Estimation (QPE) algorithm, which is a fundamental quantum algorithm used to estimate the eigenvalue of a unitary operator. This example shows how to:

- Create a circuit with a target qubit and several estimation qubits
- Apply controlled phase operations
- Implement the inverse Quantum Fourier Transform
- Measure and interpret the results

### Grover's Search Algorithm

**File:** `grovers_search.py`

Implements Grover's search algorithm, which provides a quadratic speedup for searching an unstructured database. This example demonstrates:

- Creating an oracle that marks a specific state
- Implementing the diffusion operator
- Iterating the algorithm for the optimal number of steps
- Measuring to find the marked state

### Quantum Error Correction

**File:** `quantum_error_correction.py`

Shows how to implement a simple quantum error correction code (the 3-qubit bit flip code). This example demonstrates:

- Encoding a logical qubit into three physical qubits
- Simulating a bit flip error
- Detecting and correcting the error using syndrome measurements
- Decoding the logical qubit and verifying the correction worked

### Variational Quantum Eigensolver (VQE)

**File:** `variational_quantum_eigensolver.py`

Implements a simple Variational Quantum Eigensolver (VQE) for finding the ground state energy of a hydrogen molecule. This example demonstrates:

- Defining a Hamiltonian with Pauli terms
- Creating a parameterized quantum circuit (ansatz)
- Using classical optimization to find the optimal parameters
- Calculating expectation values of quantum observables

### Hybrid Simulation Benchmark

**File:** `hybrid_simulation_benchmark.py`

Benchmarks the hybrid simulation capabilities of QSim by creating different types of circuits that trigger different simulation methods. This example demonstrates:

- Creating circuits with different properties (sparse, dense, highly entangled, block-structured)
- Analyzing circuits to determine the optimal simulation method
- Measuring simulation performance
- Comparing the results of different simulation methods

### Multi-Qudit Simulation

**File:** `multi_qudit_simulation.py`

Demonstrates QSim's ability to handle quantum systems with qudits of different dimensions (not just qubits). This example shows:

- Creating a circuit with qudits of dimensions 2 (qubit), 3 (qutrit), and 4 (ququart)
- Applying appropriate gates to these qudits
- Simulating the circuit and analyzing the results
- Interpreting measurement outcomes for multi-qudit systems

## Advanced Usage

These examples are designed to showcase the full capabilities of the QSim simulator. You can use them as a starting point for your own quantum algorithms and simulations. Feel free to modify them or combine different techniques to explore more complex quantum systems.

For more information about the QSim simulator and its features, refer to the main project documentation. 