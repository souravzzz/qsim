"""
Tests for Quantum Fourier Transform (QFT) implementation.

This test suite verifies the correctness of the QFT implementation by:
1. Testing the transformation of basis states
2. Verifying the Fourier basis transformation properties
3. Testing QFT on superposition states
4. Verifying QFT on multi-qudit systems with arbitrary dimensions
"""

import unittest
import numpy as np
from qsim.core.qudit import Qudit
from qsim.core.circuit import QuantumCircuit
from qsim.gates.hadamard import HadamardGate
from qsim.gates.phase import PhaseGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.controlled import ControlledGate
from qsim.gates.base import Gate
from qsim.execution.simulator import HybridQuantumSimulator
from qsim.utils.circuit_builders import create_quantum_fourier_transform_circuit
from qsim.states.state_vector import StateVector


class TestQuantumFourierTransform(unittest.TestCase):
    """Test case for Quantum Fourier Transform implementation."""

    def setUp(self):
        """Set up the test case."""
        self.simulator = HybridQuantumSimulator()

    def test_qft_basis_states_2qubit(self):
        """Test QFT on 2-qubit basis states and verify against theoretical results."""
        # Create a 2-qubit QFT circuit
        circuit = create_quantum_fourier_transform_circuit(2)

        # Theoretical QFT matrix for 2 qubits
        # QFT_2 = 1/2 * [1  1  1  1]
        #               [1  i -1 -i]
        #               [1 -1  1 -1]
        #               [1 -i -1  i]

        # Test each basis state
        basis_states = [
            [1, 0, 0, 0],  # |00⟩
            [0, 1, 0, 0],  # |01⟩
            [0, 0, 1, 0],  # |10⟩
            [0, 0, 0, 1],  # |11⟩
        ]

        # Expected results after QFT
        expected_results = [
            # QFT|00⟩ = 1/2(|00⟩ + |01⟩ + |10⟩ + |11⟩)
            [0.5, 0.5, 0.5, 0.5],
            # QFT|01⟩ = 1/2(|00⟩ + i|01⟩ - |10⟩ - i|11⟩)
            [0.5, 0.5j, -0.5, -0.5j],
            # QFT|10⟩ = 1/2(|00⟩ - |01⟩ + |10⟩ - |11⟩)
            [0.5, -0.5, 0.5, -0.5],
            # QFT|11⟩ = 1/2(|00⟩ - i|01⟩ - |10⟩ + i|11⟩)
            [0.5, -0.5j, -0.5, 0.5j],
        ]

        for i, basis_state in enumerate(basis_states):
            # Create a custom circuit with the basis state
            custom_circuit = QuantumCircuit(2)

            # Prepare the basis state
            if i in [1, 3]:  # States |01⟩ and |11⟩ need X on qubit 1
                custom_circuit.add_gate(PermutationGate(custom_circuit.qudits[1], [1, 0]))
            if i in [2, 3]:  # States |10⟩ and |11⟩ need X on qubit 0
                custom_circuit.add_gate(PermutationGate(custom_circuit.qudits[0], [1, 0]))

            # Add QFT gates
            for gate in circuit.gates:
                if isinstance(gate, HadamardGate):
                    custom_circuit.add_gate(
                        HadamardGate(custom_circuit.qudits[gate.qudits[0].index])
                    )
                elif isinstance(gate, ControlledGate):
                    if isinstance(gate.target_gate, PhaseGate):
                        phase_gate = PhaseGate(
                            custom_circuit.qudits[gate.target_gate.qudits[0].index],
                            gate.target_gate.phase,
                        )
                        control_gate = ControlledGate(
                            phase_gate,
                            custom_circuit.qudits[gate.control_qudit.index],
                            gate.control_value,
                        )
                        custom_circuit.add_gate(control_gate)
                    elif isinstance(gate.target_gate, PermutationGate):
                        perm_gate = PermutationGate(
                            custom_circuit.qudits[gate.target_gate.qudits[0].index],
                            gate.target_gate.permutation,
                        )
                        control_gate = ControlledGate(
                            perm_gate,
                            custom_circuit.qudits[gate.control_qudit.index],
                            gate.control_value,
                        )
                        custom_circuit.add_gate(control_gate)

            # Simulate
            state = self.simulator.simulate(custom_circuit)
            amplitudes = state.get_amplitudes()

            # Verify against expected results
            for j, expected_amp in enumerate(expected_results[i]):
                # Check both real and imaginary parts
                self.assertAlmostEqual(
                    amplitudes[j].real,
                    expected_amp.real,
                    places=5,
                    msg=f"Real part mismatch for state |{i:02b}⟩ at position {j}",
                )
                self.assertAlmostEqual(
                    amplitudes[j].imag,
                    expected_amp.imag,
                    places=5,
                    msg=f"Imaginary part mismatch for state |{i:02b}⟩ at position {j}",
                )

    def test_qft_basis_states_3qubit(self):
        """Test QFT on 3-qubit basis states and verify against theoretical results."""
        # Create a 3-qubit QFT circuit
        circuit = create_quantum_fourier_transform_circuit(3)

        # Test |000⟩ state (already the default)
        state = self.simulator.simulate(circuit)
        amplitudes = state.get_amplitudes()

        # QFT|000⟩ should give equal superposition with same phase
        expected_magnitude = 1.0 / np.sqrt(8)
        for i in range(8):
            self.assertAlmostEqual(
                abs(amplitudes[i]),
                expected_magnitude,
                places=5,
                msg=f"Magnitude mismatch at position {i}",
            )

            # All amplitudes should have the same phase for |000⟩ input
            if i > 0:
                self.assertAlmostEqual(
                    np.angle(amplitudes[i]),
                    np.angle(amplitudes[0]),
                    places=5,
                    msg=f"Phase mismatch at position {i}",
                )

        # Test |001⟩ state
        custom_circuit = QuantumCircuit(3)
        custom_circuit.add_gate(PermutationGate(custom_circuit.qudits[2], [1, 0]))  # X on qubit 2

        # Add QFT gates
        for gate in circuit.gates:
            if isinstance(gate, HadamardGate):
                custom_circuit.add_gate(HadamardGate(custom_circuit.qudits[gate.qudits[0].index]))
            elif isinstance(gate, ControlledGate):
                if isinstance(gate.target_gate, PhaseGate):
                    phase_gate = PhaseGate(
                        custom_circuit.qudits[gate.target_gate.qudits[0].index],
                        gate.target_gate.phase,
                    )
                    control_gate = ControlledGate(
                        phase_gate,
                        custom_circuit.qudits[gate.control_qudit.index],
                        gate.control_value,
                    )
                    custom_circuit.add_gate(control_gate)
                elif isinstance(gate.target_gate, PermutationGate):
                    perm_gate = PermutationGate(
                        custom_circuit.qudits[gate.target_gate.qudits[0].index],
                        gate.target_gate.permutation,
                    )
                    control_gate = ControlledGate(
                        perm_gate,
                        custom_circuit.qudits[gate.control_qudit.index],
                        gate.control_value,
                    )
                    custom_circuit.add_gate(control_gate)

        # Simulate
        state = self.simulator.simulate(custom_circuit)
        amplitudes = state.get_amplitudes()

        # For |001⟩, the QFT should produce amplitudes with specific phase relationships
        # The magnitude should be 1/sqrt(8) for all
        for i in range(8):
            self.assertAlmostEqual(abs(amplitudes[i]), 1 / np.sqrt(8), places=5)

            # The phase should follow the pattern e^(2πi*j*1/8) for j=0..7
            expected_phase = 2 * np.pi * i * 1 / 8
            measured_phase = np.angle(amplitudes[i])

            # Adjust for phase wrapping
            phase_diff = (measured_phase - expected_phase) % (2 * np.pi)
            if phase_diff > np.pi:
                phase_diff = phase_diff - 2 * np.pi

            self.assertAlmostEqual(phase_diff, 0, places=5, msg=f"Phase mismatch at position {i}")

    def test_qft_superposition_states(self):
        """Test QFT on superposition states."""
        # Create a 2-qubit circuit with a superposition input
        circuit = QuantumCircuit(2)

        # Create |+⟩ state on first qubit (|0⟩ + |1⟩)/sqrt(2)
        circuit.add_gate(HadamardGate(circuit.qudits[0]))

        # Apply QFT
        qft_circuit = create_quantum_fourier_transform_circuit(2)
        for gate in qft_circuit.gates:
            if isinstance(gate, HadamardGate):
                circuit.add_gate(HadamardGate(circuit.qudits[gate.qudits[0].index]))
            elif isinstance(gate, ControlledGate):
                if isinstance(gate.target_gate, PhaseGate):
                    phase_gate = PhaseGate(
                        circuit.qudits[gate.target_gate.qudits[0].index], gate.target_gate.phase
                    )
                    control_gate = ControlledGate(
                        phase_gate, circuit.qudits[gate.control_qudit.index], gate.control_value
                    )
                    circuit.add_gate(control_gate)
                elif isinstance(gate.target_gate, PermutationGate):
                    perm_gate = PermutationGate(
                        circuit.qudits[gate.target_gate.qudits[0].index],
                        gate.target_gate.permutation,
                    )
                    control_gate = ControlledGate(
                        perm_gate, circuit.qudits[gate.control_qudit.index], gate.control_value
                    )
                    circuit.add_gate(control_gate)

        # Simulate
        state = self.simulator.simulate(circuit)
        amplitudes = state.get_amplitudes()

        # For input (|0⟩ + |1⟩)/sqrt(2) ⊗ |0⟩, the QFT should produce:
        # QFT[(|00⟩ + |10⟩)/sqrt(2)] = (QFT|00⟩ + QFT|10⟩)/sqrt(2)
        # = 1/sqrt(2) * [1/2(|00⟩ + |01⟩ + |10⟩ + |11⟩) + 1/2(|00⟩ - |01⟩ + |10⟩ - |11⟩)]
        # = 1/sqrt(2) * [1/2(2|00⟩ + 0|01⟩ + 2|10⟩ + 0|11⟩)]
        # = 1/sqrt(2) * [|00⟩ + |10⟩]

        # Expected non-zero amplitudes at positions 0 and 2
        expected_non_zero = [0, 2]
        for i in range(4):
            if i in expected_non_zero:
                self.assertAlmostEqual(
                    abs(amplitudes[i]),
                    1 / np.sqrt(2),
                    places=5,
                    msg=f"Amplitude at position {i} should be 1/sqrt(2)",
                )
            else:
                self.assertAlmostEqual(
                    abs(amplitudes[i]), 0, places=5, msg=f"Amplitude at position {i} should be 0"
                )

    def test_qft_inverse(self):
        """Test that QFT followed by inverse QFT returns the original state."""
        # Test with different numbers of qubits
        for num_qubits in range(1, 5):
            # Create a random state vector
            state_vector = StateVector(num_qubits, [2] * num_qubits)

            # Initialize with random amplitudes
            random_amplitudes = np.random.random(2**num_qubits) + 1j * np.random.random(
                2**num_qubits
            )
            # Normalize
            random_amplitudes = random_amplitudes / np.linalg.norm(random_amplitudes)
            state_vector.vector = random_amplitudes

            # Create QFT circuit
            qft_circuit = create_quantum_fourier_transform_circuit(num_qubits)

            # Apply QFT
            for gate in qft_circuit.gates:
                state_vector.apply_gate(gate)

            # Create inverse QFT by reversing gates and negating phases
            inverse_qft_gates = []
            for gate in reversed(qft_circuit.gates):
                if isinstance(gate, PhaseGate):
                    inverse_qft_gates.append(PhaseGate(gate.qudits[0], -gate.phase))
                elif isinstance(gate, ControlledGate) and isinstance(gate.target_gate, PhaseGate):
                    inverse_target = PhaseGate(gate.target_gate.qudits[0], -gate.target_gate.phase)
                    inverse_qft_gates.append(
                        ControlledGate(inverse_target, gate.control_qudit, gate.control_value)
                    )
                else:
                    inverse_qft_gates.append(gate)

            # Apply inverse QFT
            for gate in inverse_qft_gates:
                state_vector.apply_gate(gate)

            # Check that we get back the original state
            for i in range(len(random_amplitudes)):
                self.assertAlmostEqual(
                    state_vector.vector[i].real, random_amplitudes[i].real, places=5
                )
                self.assertAlmostEqual(
                    state_vector.vector[i].imag, random_amplitudes[i].imag, places=5
                )

    def test_qft_higher_dimensions(self):
        """Test QFT on qudits with dimension > 2."""
        # Create a circuit with a qutrit (d=3)
        circuit = QuantumCircuit(1, 3)

        # Apply generalized QFT (Hadamard for d=3 is already the QFT)
        circuit.add_gate(HadamardGate(circuit.qudits[0]))

        # Simulate
        state = self.simulator.simulate(circuit)
        amplitudes = state.get_amplitudes()

        # For a qutrit in state |0⟩, the QFT should produce:
        # QFT|0⟩ = 1/sqrt(3) * (|0⟩ + |1⟩ + |2⟩)
        expected_magnitude = 1.0 / np.sqrt(3)

        for i in range(3):
            self.assertAlmostEqual(abs(amplitudes[i]), expected_magnitude, places=5)

        # Test with a qutrit in state |1⟩
        circuit2 = QuantumCircuit(1, 3)

        # Prepare |1⟩ state
        perm = [1, 2, 0]  # Cyclic permutation |0⟩ -> |1⟩, |1⟩ -> |2⟩, |2⟩ -> |0⟩
        circuit2.add_gate(PermutationGate(circuit2.qudits[0], perm))

        # Apply QFT
        circuit2.add_gate(HadamardGate(circuit2.qudits[0]))

        # Simulate
        state2 = self.simulator.simulate(circuit2)
        amplitudes2 = state2.get_amplitudes()

        # For a qutrit in state |1⟩, the QFT should produce:
        # QFT|1⟩ = 1/sqrt(3) * (|0⟩ + ω|1⟩ + ω²|2⟩) where ω = e^(2πi/3)
        omega = np.exp(2j * np.pi / 3)

        expected_amplitudes = [1 / np.sqrt(3), omega / np.sqrt(3), omega**2 / np.sqrt(3)]

        for i in range(3):
            self.assertAlmostEqual(abs(amplitudes2[i]), 1 / np.sqrt(3), places=5)

            # Check phase relationships
            if i > 0:
                expected_phase = np.angle(expected_amplitudes[i])
                measured_phase = np.angle(amplitudes2[i])

                # Adjust for phase wrapping
                phase_diff = (measured_phase - expected_phase) % (2 * np.pi)
                if phase_diff > np.pi:
                    phase_diff = phase_diff - 2 * np.pi

                self.assertAlmostEqual(phase_diff, 0, places=5)

    def test_qft_linearity(self):
        """Test the linearity property of QFT."""
        # Create two different basis states
        basis_states = [
            np.array([1, 0, 0, 0], dtype=complex),  # |00⟩
            np.array([0, 1, 0, 0], dtype=complex),  # |01⟩
        ]

        # Create a superposition state (|00⟩ + |01⟩)/sqrt(2)
        superposition = (basis_states[0] + basis_states[1]) / np.sqrt(2)

        # Create QFT circuit
        qft_circuit = create_quantum_fourier_transform_circuit(2)

        # Apply QFT to each basis state
        qft_basis_states = []
        for basis_state in basis_states:
            # Create state vector
            state_vector = StateVector(2, [2, 2])
            state_vector.vector = basis_state  # Now using NumPy array instead of list

            # Apply QFT
            for gate in qft_circuit.gates:
                state_vector.apply_gate(gate)

            qft_basis_states.append(state_vector.vector)

        # Apply QFT to superposition
        superposition_state = StateVector(2, [2, 2])
        superposition_state.vector = superposition  # Already a NumPy array

        for gate in qft_circuit.gates:
            superposition_state.apply_gate(gate)

        # Check linearity: QFT(|00⟩ + |01⟩)/sqrt(2) = (QFT|00⟩ + QFT|01⟩)/sqrt(2)
        expected_result = (qft_basis_states[0] + qft_basis_states[1]) / np.sqrt(2)

        for i in range(4):
            self.assertAlmostEqual(
                superposition_state.vector[i].real, expected_result[i].real, places=5
            )
            self.assertAlmostEqual(
                superposition_state.vector[i].imag, expected_result[i].imag, places=5
            )

    def test_qft_periodicity(self):
        """Test the periodicity properties of QFT."""
        # Create a periodic state: |0⟩ + |2⟩ (period 2 in a 4-qubit system)
        num_qubits = 4
        periodic_state = np.zeros(2**num_qubits, dtype=complex)

        # Set amplitudes for |0000⟩, |0010⟩, |0100⟩, |0110⟩, etc.
        for i in range(0, 2**num_qubits, 4):
            periodic_state[i] = 1 / np.sqrt(4)
            periodic_state[i + 2] = 1 / np.sqrt(4)

        # Create state vector
        state_vector = StateVector(num_qubits, [2] * num_qubits)
        state_vector.vector = periodic_state

        # Create QFT circuit
        qft_circuit = create_quantum_fourier_transform_circuit(num_qubits)

        # Apply QFT
        for gate in qft_circuit.gates:
            state_vector.apply_gate(gate)

        # For a state with period 2, the QFT should have peaks at positions
        # that are multiples of 2^(n-1) = 2^3 = 8
        amplitudes = state_vector.vector

        # Expected peaks at positions 0 and 8
        peak_positions = [0, 8]

        for i in range(2**num_qubits):
            if i in peak_positions:
                self.assertGreater(
                    abs(amplitudes[i]), 0.7 / np.sqrt(2), msg=f"Expected peak at position {i}"
                )
            else:
                self.assertLess(abs(amplitudes[i]), 0.3, msg=f"Unexpected peak at position {i}")


if __name__ == "__main__":
    unittest.main()
