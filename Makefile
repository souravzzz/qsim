.PHONY: test lint format clean dev-setup basic-examples advanced-examples examples run-tests benchmark

# Default target
all: lint test

# Run tests
test:
	python run_tests.py

# Run all tests with pytest
run-tests:
	pytest tests/

# Run pytest with coverage
coverage:
	pytest --cov=qsim tests/

# Run linting
lint:
	flake8 qsim tests examples
	mypy qsim

# Format code
format:
	black qsim tests examples
	isort qsim tests examples

# Clean up build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

# Install development dependencies
dev-setup:
	uv pip install -e ".[dev]"

# Run the basic examples
basic-examples:
	python -m qsim.main
	python examples/basic_usage.py

# Run the advanced examples
advanced-examples:
	@echo "Running Quantum Phase Estimation example..."
	python examples/quantum_phase_estimation.py
	@echo "\nRunning Grover's Search Algorithm example..."
	python examples/grovers_search.py
	@echo "\nRunning Quantum Error Correction example..."
	python examples/quantum_error_correction.py
	@echo "\nRunning Variational Quantum Eigensolver example..."
	python examples/variational_quantum_eigensolver.py
	@echo "\nRunning Multi-Qudit Simulation example..."
	python examples/multi_qudit_simulation.py
	@echo "\nRunning Hybrid Simulation Benchmark example..."
	python examples/hybrid_simulation_benchmark.py

# Run all examples (basic and advanced)
examples: basic-examples advanced-examples

# Run the scaling benchmarks
benchmark:
	mkdir -p benchmarks/plots
	python benchmarks/scaling_ghz.py --output-dir=benchmarks/plots

# Help
help:
	@echo "Available targets:"
	@echo "  all            - Run linting and tests"
	@echo "  test           - Run tests using run_tests.py"
	@echo "  run-tests      - Run all tests using pytest"
	@echo "  coverage       - Run tests with coverage report"
	@echo "  lint           - Run linting tools"
	@echo "  format         - Format code with Black and isort"
	@echo "  clean          - Clean up build artifacts"
	@echo "  dev-setup      - Install development dependencies"
	@echo "  basic-examples    - Run basic example scripts"
	@echo "  advanced-examples - Run advanced example scripts"
	@echo "  all-examples   - Run all example scripts"
	@echo "  benchmark 		- Run scaling benchmarks"
