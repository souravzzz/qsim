# QSim: Hybrid Quantum Circuit Simulator

QSim is a high-performance quantum circuit simulator that efficiently handles various types of quantum circuits by dynamically selecting the most appropriate simulation method.

## Features

- **Hybrid Simulation Architecture**: Dynamically selects the optimal simulation method based on circuit properties
- **Multiple State Representations**: Supports dense state vectors, sparse state vectors, and tensor network states
- **Circuit Analysis**: Automatically analyzes circuit structure to determine the best simulation strategy
- **Common Circuit Builders**: Utility functions for creating standard quantum circuits (Bell states, GHZ states, QFT)
- **GPU Acceleration**: Optional GPU acceleration for large-scale simulations (Work in Progress)

## Architecture

The simulator is organized around these core components:

- **Quantum State Representations**: StateVector, SparseStateVector, TensorNetworkState
- **Circuit Analysis and Partitioning**: CircuitAnalyzer
- **Hybrid Execution Management**: HybridExecutionManager
- **Gate Implementations**: Local, permutation, phase, controlled, and entangling gates

## Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/souravzzz/qsim.git
cd qsim

# Install the package
pip install -e .
```

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/souravzzz/qsim.git
cd qsim

# Install the package
uv pip install -e .
```

## Usage

```python
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.controlled import ControlledGate
from qsim.gates.permutation import PermutationGate

# Create a quantum circuit
circuit = QuantumCircuit(2)
circuit.add_gate(HadamardGate(circuit.qudits[0]))
circuit.add_gate(ControlledGate(PermutationGate(circuit.qudits[1], [1, 0]), circuit.qudits[0], 1))

# Create a simulator and run the circuit
simulator = HybridQuantumSimulator()
results = simulator.simulate_and_measure(circuit, num_shots=1000)

# Print the results
for outcome, count in sorted(results.items()):
    print(f"|{outcome}⟩: {count} shots ({count/10:.1f}%)")
```

## Examples

The package includes several example circuits:

- Bell state preparation
- GHZ state preparation
- Quantum Fourier Transform

Run the main script to see these examples in action:

```bash
python -m qsim.main
```

Or try the basic usage example:

```bash
python examples/basic_usage.py
```

## Development

### Setting Up a Development Environment

We recommend using the `uv` package manager for Python development. Follow these steps to set up your development environment:

```bash
# Clone the repository
git clone https://github.com/souravzzz/qsim.git
cd qsim

# Create and activate a virtual environment with uv
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Install the package in development mode with development dependencies
uv pip install -e ".[dev]"
```

Alternatively, you can use the provided Makefile:

```bash
# Create and activate a virtual environment first, then:
make dev-setup
```

### Running Tests

There are several ways to run the tests:

```bash
# Using the provided script
python run_tests.py

# Using unittest directly
python -m unittest discover tests

# Using pytest
pytest tests/

# Run a specific test file
pytest tests/test_bell_state.py

# Run tests with coverage report
pytest --cov=qsim tests/
```

With the Makefile:

```bash
# Run tests
make test

# Run tests with coverage
make coverage
```

### Continuous Testing During Development

For continuous testing during development, you can use pytest-watch:

```bash
# Watch for changes and run tests automatically
ptw tests/
```

### Code Formatting and Linting

We use several tools to maintain code quality:

```bash
# Format code with Black
black qsim tests examples

# Sort imports with isort
isort qsim tests examples

# Lint code with flake8
flake8 qsim tests examples

# Type checking with mypy
mypy qsim
```

With the Makefile:

```bash
# Format code
make format

# Run linting
make lint

# Run both linting and tests
make all
```

### Using the Makefile

The project includes a Makefile with common development tasks:

```bash
# Show available commands
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
```

## Requirements

- Python 3.8+
- NumPy 1.20.0+
- SciPy 1.7.0+
- NetworkX 2.6.0+
- TensorNetwork 0.4.0+

## License

This project is licensed under the MIT License - see the LICENSE file for details.
