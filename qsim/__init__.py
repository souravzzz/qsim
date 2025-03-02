"""
QSim: Hybrid Quantum Circuit Simulator

A high-performance quantum circuit simulator that efficiently handles various types
of quantum circuits by dynamically selecting the most appropriate simulation method.
"""

__version__ = "0.1.0"
__author__ = "QSim Team"

# Import commonly used components for easier access
from qsim.core.circuit import QuantumCircuit
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.utils.circuit_builders import (
    create_bell_state_circuit,
    create_ghz_state_circuit,
    create_quantum_fourier_transform_circuit,
)

# Define what's available when using "from qsim import *"
__all__ = [
    "QuantumCircuit",
    "HybridQuantumSimulator",
    "create_bell_state_circuit",
    "create_ghz_state_circuit",
    "create_quantum_fourier_transform_circuit",
]
