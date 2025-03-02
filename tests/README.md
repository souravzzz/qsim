# QSim Tests

This directory contains test cases for the QSim quantum circuit simulator. The tests are organized by component and are designed to verify the correctness of the simulator's functionality.

## Running the Tests

You can run all tests using the Makefile from the project root:

```bash
# Run all tests
make run-tests

# Run tests with coverage report
make coverage
```

Or you can run individual test files directly:

```bash
# Run a specific test file
pytest tests/test_core.py

# Run a specific test class
pytest tests/test_gates.py::TestGateMatrices

# Run a specific test method
pytest tests/test_states.py::TestStateVector::test_measure
```

## Test Organization

The tests are organized by component:

- `test_core.py`: Tests for core components (Qudit and QuantumCircuit)
- `test_gates.py`: Tests for quantum gate implementations
- `test_states.py`: Tests for quantum state representations
- `test_execution.py`: Tests for execution components (simulator and hybrid execution manager)
- `test_analysis.py`: Tests for circuit analysis components
- `test_bell_state.py`: Integration test for Bell state preparation and measurement

## Test Coverage

The tests aim to cover all major functionality of the QSim simulator:

1. **Core Components**
   - Qudit initialization and properties
   - QuantumCircuit creation and gate addition

2. **Quantum Gates**
   - Gate properties (local vs. entangling)
   - Matrix representations
   - Gate application to states

3. **Quantum States**
   - State initialization
   - Probability calculations
   - Measurement
   - State conversions between representations

4. **Execution**
   - Simulation method determination
   - Initial state creation
   - Circuit simulation
   - Measurement statistics

5. **Analysis**
   - Circuit complexity analysis
   - Entanglement detection
   - Block structure identification
   - Mixed dimension analysis

## Adding New Tests

When adding new functionality to the simulator, please add corresponding tests. Follow these guidelines:

1. Place tests in the appropriate file based on the component being tested
2. Use descriptive test method names that explain what is being tested
3. Include docstrings for test classes and methods
4. Use assertions to verify expected behavior
5. For probabilistic tests, allow for statistical variation

## Test Dependencies

The tests use the following dependencies:

- `unittest`: Python's built-in testing framework
- `pytest`: For test discovery and running
- `numpy`: For numerical assertions and array operations

These dependencies are included in the development requirements and will be installed when you run `make dev-setup`. 