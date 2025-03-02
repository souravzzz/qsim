# QSim: Hybrid Quantum Circuit Simulator

QSim is a high-performance quantum circuit simulator that efficiently handles various types of quantum circuits by dynamically selecting the most appropriate simulation method based on circuit properties.

## Features

- **Hybrid Simulation Architecture**: Dynamically selects optimal simulation methods (sparse, dense, tensor network)
- **Multi-qudit Support**: Simulates systems with arbitrary qudit dimensions (not just qubits)
- **Circuit Analysis**: Automatically analyzes circuit structure to determine the best simulation strategy
- **Performance Optimization**: Sparse representation for states with many zero amplitudes
- **Common Circuit Builders**: Utility functions for standard quantum circuits (Bell states, GHZ states, QFT)
- **Benchmarking Tools**: Comprehensive benchmarks for performance analysis
- **GPU Acceleration**: Optional GPU acceleration for large-scale simulations (Work in Progress)

## Architecture

The simulator is organized around these core components:

- **Core Components**: `QuantumCircuit` and `Qudit` implementations
- **Quantum Gates**: Hadamard, Phase, Permutation, and Controlled gate implementations
- **Quantum State Representations**: `StateVector`, `SparseStateVector`, `TensorNetworkState`
- **Circuit Analysis**: `CircuitAnalyzer` for determining optimal simulation strategies
- **Execution Management**: `HybridExecutionManager` and `HybridQuantumSimulator`
- **Utilities**: Helper functions and circuit builders

## Installation

```bash
# Clone the repository
git clone https://github.com/souravzzz/qsim.git
cd qsim

# Install with pip
pip install -e .

# Or with uv (recommended)
uv pip install -e .
```

## Basic Usage

```python
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.controlled import ControlledGate
from qsim.gates.permutation import PermutationGate

# Create a Bell state circuit
circuit = QuantumCircuit(2)
circuit.add_gate(HadamardGate(circuit.qudits[0]))
circuit.add_gate(ControlledGate(PermutationGate(circuit.qudits[1], [1, 0]), circuit.qudits[0], 1))

# Simulate and measure
simulator = HybridQuantumSimulator()
results = simulator.simulate_and_measure(circuit, num_shots=1000)

# Print results
for outcome, count in sorted(results.items()):
    print(f"|{outcome}⟩: {count} shots ({count/10:.1f}%)")
```

## Examples

Run the included examples:

```bash
# Basic examples (Bell state, GHZ state, QFT)
python -m qsim.main
python examples/basic_usage.py

# Advanced examples
python examples/quantum_phase_estimation.py
python examples/grovers_search.py
python examples/quantum_error_correction.py

# Run all examples
make examples
```

## Benchmarks

The package includes benchmarking tools to evaluate performance:

```bash
# Run hybrid simulation benchmarks
python examples/hybrid_simulation_benchmark.py

# Run scaling benchmarks for GHZ states
python benchmarks/scaling_ghz.py --output-dir=benchmarks/plots
```

## Development

### Setting Up a Development Environment

```bash
# Clone and set up development environment
git clone https://github.com/souravzzz/qsim.git
cd qsim

# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Unix/macOS
uv pip install -e ".[dev]"

# Or using make
make dev-setup
```

### Testing and Code Quality

```bash
# Run tests
make test
make coverage

# Format and lint code
make format
make lint

# Show all available commands
make help
```

### Project Structure

```
qsim/
├── core/           # Core components (Circuit, Qudit)
├── gates/          # Quantum gate implementations
├── states/         # Quantum state representations
├── analysis/       # Circuit analysis tools
├── execution/      # Execution management
└── utils/          # Utility functions and helpers
examples/           # Example scripts
benchmarks/         # Performance benchmarks
tests/              # Test suite
```

## Requirements

- Python 3.8+
- NumPy 1.20.0+
- SciPy 1.7.0+
- TensorNetwork 0.4.0+
- psutil 5.9.0+
- matplotlib 3.5.0+

## License

This project is licensed under the MIT License - see the LICENSE file for details.
