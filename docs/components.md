# QSIM: Hybrid Quantum Circuit Simulator

This document outlines the key components and interfaces of the qsim codebase, providing a comprehensive reference for developers and AI assistants working with the library.

## Core Components

### Qudit (`qsim/core/qudit.py`)

Represents a quantum system with d levels (d=2 for qubits, d>2 for qudits).

```python
class Qudit:
    def __init__(self, index: int, dimension: int = 2):
        """
        Initialize a qudit with specified index and dimension.
        
        Args:
            index: Unique identifier for the qudit within a circuit
            dimension: Number of levels in the qudit system (2 for qubits)
        """
        
    @property
    def index(self) -> int:
        """Index of the qudit in the circuit."""
        
    @property
    def dimension(self) -> int:
        """Dimension of the qudit (2 for qubits, >2 for qudits)."""
```

**Usage Example:**
```python
# Create a qubit (2-level system)
qubit = Qudit(index=0, dimension=2)

# Create a qutrit (3-level system)
qutrit = Qudit(index=1, dimension=3)
```

### QuantumCircuit (`qsim/core/circuit.py`)

Represents a quantum circuit containing qudits and quantum gates.

```python
class QuantumCircuit:
    def __init__(self, num_qudits: int, dimensions: Union[int, List[int]] = 2):
        """
        Initialize a quantum circuit with specified number of qudits.
        
        Args:
            num_qudits: Number of qudits in the circuit
            dimensions: Either a single dimension for all qudits or a list of dimensions
                        for each qudit individually
        """
        
    @property
    def qudits(self) -> List[Qudit]:
        """List of qudits in the circuit."""
        
    @property
    def gates(self) -> List[Gate]:
        """List of gates in the circuit, in order of application."""
        
    @property
    def num_qudits(self) -> int:
        """Number of qudits in the circuit."""
        
    @property
    def dimensions(self) -> List[int]:
        """Dimensions of each qudit in the circuit."""
        
    def add_gate(self, gate: Gate) -> 'QuantumCircuit':
        """
        Add a gate to the circuit.
        
        Args:
            gate: The quantum gate to add
            
        Returns:
            Self, allowing for method chaining
            
        Raises:
            ValueError: If the gate references qudits not in the circuit
        """
```

**Usage Example:**
```python
# Create a circuit with 3 qubits
circuit = QuantumCircuit(num_qudits=3)

# Create a circuit with mixed dimensions (qubit, qutrit, qubit)
circuit = QuantumCircuit(num_qudits=3, dimensions=[2, 3, 2])

# Add gates to the circuit
circuit.add_gate(HadamardGate(circuit.qudits[0]))
```

## Gate Implementations

### Gate (`qsim/gates/base.py`)

Base class for all quantum gates.

```python
class Gate:
    def __init__(self, name: str, qudits: List[Qudit], matrix: Optional[np.ndarray] = None):
        """
        Initialize a quantum gate.
        
        Args:
            name: Name of the gate
            qudits: Qudits the gate acts on
            matrix: Unitary matrix representation (optional)
        """
        
    @property
    def name(self) -> str:
        """Name of the gate."""
        
    @property
    def qudits(self) -> List[Qudit]:
        """Qudits the gate acts on."""
        
    @property
    def matrix(self) -> np.ndarray:
        """Unitary matrix representation of the gate."""
        
    @property
    def qudit(self) -> Qudit:
        """For single-qudit gates, returns the qudit it acts on."""
        
    @property
    def is_local(self) -> bool:
        """Whether gate acts on a single qudit."""
        
    @property
    def is_entangling(self) -> bool:
        """Whether gate can create entanglement between qudits."""
        
    def apply(self, state) -> None:
        """
        Apply gate to a quantum state.
        
        Args:
            state: Quantum state to apply the gate to
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
```

### HadamardGate (`qsim/gates/hadamard.py`)

Implements the Hadamard gate, which creates superposition.

```python
class HadamardGate(Gate):
    def __init__(self, qudit: Qudit):
        """
        Initialize a Hadamard gate.
        
        Args:
            qudit: Qudit to apply the Hadamard gate to
            
        Note:
            For qudits with dimension > 2, this implements the generalized
            quantum Fourier transform.
        """
        
    def apply(self, state) -> None:
        """
        Apply Hadamard to a quantum state.
        
        Args:
            state: Quantum state to apply the gate to
        """
```

**Usage Example:**
```python
# Apply Hadamard gate to the first qudit
h_gate = HadamardGate(circuit.qudits[0])
circuit.add_gate(h_gate)
```

### PhaseGate (`qsim/gates/phase.py`)

Implements the phase gate, which adds a phase to the |1⟩ state.

```python
class PhaseGate(Gate):
    def __init__(self, qudit: Qudit, phase: float):
        """
        Initialize a phase gate.
        
        Args:
            qudit: Qudit to apply the phase gate to
            phase: Phase angle in radians
        """
        
    @property
    def phase(self) -> float:
        """Phase angle in radians."""
        
    def apply(self, state) -> None:
        """
        Apply phase gate to a quantum state.
        
        Args:
            state: Quantum state to apply the gate to
        """
```

**Usage Example:**
```python
# Apply π/4 phase gate (T gate) to the first qudit
t_gate = PhaseGate(circuit.qudits[0], phase=np.pi/4)
circuit.add_gate(t_gate)
```

### PermutationGate (`qsim/gates/permutation.py`)

Implements permutation gates, including the X/NOT gate.

```python
class PermutationGate(Gate):
    def __init__(self, qudit: Qudit, permutation: List[int]):
        """
        Initialize a permutation gate.
        
        Args:
            qudit: Qudit to apply the permutation to
            permutation: List specifying the permutation mapping
            
        Note:
            For a qubit, the permutation [1, 0] corresponds to the X/NOT gate.
            For higher dimensions, arbitrary permutations are supported.
        """
        
    @property
    def permutation(self) -> List[int]:
        """Permutation mapping."""
        
    def apply(self, state) -> None:
        """
        Apply permutation to a quantum state.
        
        Args:
            state: Quantum state to apply the gate to
        """
```

**Usage Example:**
```python
# Apply X/NOT gate to a qubit
x_gate = PermutationGate(circuit.qudits[0], permutation=[1, 0])

# Apply a cyclic permutation to a qutrit
cycle_gate = PermutationGate(circuit.qudits[1], permutation=[1, 2, 0])
```

### ControlledGate (`qsim/gates/controlled.py`)

Implements controlled versions of other gates.

```python
class ControlledGate(Gate):
    def __init__(self, target_gate: Gate, control_qudit: Qudit, control_value: int = 1):
        """
        Initialize a controlled gate.
        
        Args:
            target_gate: Gate to apply conditionally
            control_qudit: Qudit that controls the gate
            control_value: Value of control qudit that activates the gate (default: 1)
        """
        
    @property
    def target_gate(self) -> Gate:
        """Gate to apply conditionally."""
        
    @property
    def control_qudit(self) -> Qudit:
        """Qudit that controls the gate."""
        
    @property
    def control_value(self) -> int:
        """Value of control qudit that activates the gate."""
        
    def apply(self, state) -> None:
        """
        Apply controlled gate to a quantum state.
        
        Args:
            state: Quantum state to apply the gate to
        """
```

**Usage Example:**
```python
# Create a CNOT gate (controlled-X)
x_gate = PermutationGate(circuit.qudits[1], permutation=[1, 0])
cnot = ControlledGate(target_gate=x_gate, control_qudit=circuit.qudits[0])

# Create a controlled-H gate with control value 2 (for qutrits)
h_gate = HadamardGate(circuit.qudits[1])
controlled_h = ControlledGate(
    target_gate=h_gate, 
    control_qudit=circuit.qudits[0], 
    control_value=2
)
```

### MultiControlledGate (`qsim/gates/multi_controlled.py`)

Implements gates with multiple control qudits.

```python
class MultiControlledGate(Gate):
    def __init__(self, target_gate: Gate, control_qudits: List[Qudit], control_values: List[int] = None):
        """
        Initialize a multi-controlled gate.
        
        Args:
            target_gate: Gate to apply conditionally
            control_qudits: List of qudits that control the gate
            control_values: List of values for each control qudit that activates the gate
                           (defaults to all 1s if not specified)
        """
        
    @property
    def target_gate(self) -> Gate:
        """Gate to apply conditionally."""
        
    @property
    def control_qudits(self) -> List[Qudit]:
        """Qudits that control the gate."""
        
    @property
    def control_values(self) -> List[int]:
        """Values of control qudits that activate the gate."""
        
    def apply(self, state) -> None:
        """
        Apply multi-controlled gate to a quantum state.
        
        Args:
            state: Quantum state to apply the gate to
        """
```

**Usage Example:**
```python
# Create a Toffoli gate (controlled-controlled-X)
x_gate = PermutationGate(circuit.qudits[2], permutation=[1, 0])
toffoli = MultiControlledGate(
    target_gate=x_gate, 
    control_qudits=[circuit.qudits[0], circuit.qudits[1]]
)

# Create a gate with custom control values
custom_gate = MultiControlledGate(
    target_gate=HadamardGate(circuit.qudits[2]),
    control_qudits=[circuit.qudits[0], circuit.qudits[1]],
    control_values=[0, 2]  # Control on |0⟩ for first qudit and |2⟩ for second qudit
)
```

## Quantum State Representations

### QuantumState (`qsim/states/base.py`)

Base class for quantum state representations.

```python
class QuantumState:
    def __init__(self, num_qudits: int, dimensions: List[int]):
        """
        Initialize a quantum state.
        
        Args:
            num_qudits: Number of qudits in the system
            dimensions: Dimensions of each qudit
        """
        
    @property
    def num_qudits(self) -> int:
        """Number of qudits in the system."""
        
    @property
    def dimensions(self) -> List[int]:
        """Dimension of each qudit."""
        
    @property
    def total_dimension(self) -> int:
        """Total Hilbert space dimension (product of all qudit dimensions)."""
        
    def get_probability(self, index: int) -> float:
        """
        Get probability of a specific computational basis state.
        
        Args:
            index: Index of the computational basis state
            
        Returns:
            Probability of measuring the specified state
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        
    def get_probabilities(self) -> np.ndarray:
        """
        Get probabilities for all computational basis states.
        
        Returns:
            Array of probabilities
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        
    def apply_gate(self, gate: Gate) -> None:
        """
        Apply a gate to the state.
        
        Args:
            gate: Gate to apply
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
```

### StateVector (`qsim/states/state_vector.py`)

Dense state vector representation of a quantum state.

```python
class StateVector(QuantumState):
    def __init__(self, num_qudits: int, dimensions: List[int]):
        """
        Initialize a state vector.
        
        Args:
            num_qudits: Number of qudits in the system
            dimensions: Dimensions of each qudit
            
        Note:
            Initializes to |0...0⟩ state by default.
        """
        
    @property
    def vector(self) -> np.ndarray:
        """Dense complex vector of amplitudes."""
        
    def get_amplitudes(self) -> np.ndarray:
        """
        Get all amplitudes.
        
        Returns:
            Complex numpy array of amplitudes
        """
        
    def apply_gate(self, gate: Gate) -> None:
        """
        Apply a gate to the state.
        
        Args:
            gate: Gate to apply
        """
        
    def measure(self) -> str:
        """
        Perform a measurement of the quantum state.
        
        Returns:
            String representation of the measured state
        """
```

**Usage Example:**
```python
# Create a state vector for 3 qubits
state = StateVector(num_qudits=3, dimensions=[2, 2, 2])

# Apply a gate
h_gate = HadamardGate(circuit.qudits[0])
state.apply_gate(h_gate)

# Get measurement probabilities
probabilities = state.get_probabilities()

# Perform a measurement
result = state.measure()
```

### SparseStateVector (`qsim/states/sparse_state_vector.py`)

Sparse representation of a quantum state using hash maps.

```python
class SparseStateVector(QuantumState):
    def __init__(self, num_qudits: int, dimensions: List[int]):
        """
        Initialize a sparse state vector.
        
        Args:
            num_qudits: Number of qudits in the system
            dimensions: Dimensions of each qudit
            
        Note:
            Initializes to |0...0⟩ state by default.
            Efficient for states with many zero amplitudes.
        """
        
    @property
    def amplitudes(self) -> Dict[int, complex]:
        """Map from basis state indices to amplitudes."""
        
    def get_amplitudes(self) -> np.ndarray:
        """
        Get all amplitudes as dense array.
        
        Returns:
            Complex numpy array of amplitudes
        """
        
    def apply_gate(self, gate: Gate) -> None:
        """
        Apply a gate to the state.
        
        Args:
            gate: Gate to apply
        """
        
    def to_state_vector(self) -> StateVector:
        """
        Convert to dense representation.
        
        Returns:
            Equivalent StateVector object
        """
        
    @classmethod
    def from_state_vector(cls, state_vector: StateVector) -> 'SparseStateVector':
        """
        Create from dense representation.
        
        Args:
            state_vector: Dense state vector to convert
            
        Returns:
            New SparseStateVector object
        """
        
    def measure(self) -> str:
        """
        Perform a measurement of the quantum state.
        
        Returns:
            String representation of the measured state
        """
```

**Usage Example:**
```python
# Create a sparse state vector for a large system
state = SparseStateVector(num_qudits=20, dimensions=[2]*20)

# Apply gates that maintain sparsity
x_gate = PermutationGate(circuit.qudits[0], permutation=[1, 0])
state.apply_gate(x_gate)

# Convert to dense representation if needed
dense_state = state.to_state_vector()
```

### TensorNetworkState (`qsim/states/tensor_network_state.py`)

Tensor network representation of a quantum state.

```python
class TensorNetworkState(QuantumState):
    def __init__(self, num_qudits: int, dimensions: List[int]):
        """
        Initialize a tensor network state.
        
        Args:
            num_qudits: Number of qudits in the system
            dimensions: Dimensions of each qudit
            
        Note:
            Efficient for states with specific entanglement structures.
        """
        
    @property
    def nodes(self) -> List[tn.Node]:
        """Tensor network nodes representing the state."""
        
    def to_state_vector(self) -> StateVector:
        """
        Convert to dense representation.
        
        Returns:
            Equivalent StateVector object
            
        Note:
            This operation can be expensive for large systems.
        """
        
    @classmethod
    def from_state_vector(cls, state_vector: StateVector) -> 'TensorNetworkState':
        """
        Create from dense representation.
        
        Args:
            state_vector: Dense state vector to convert
            
        Returns:
            New TensorNetworkState object
        """
        
    def apply_gate(self, gate: Gate) -> None:
        """
        Apply a gate to the state.
        
        Args:
            gate: Gate to apply
        """
```

**Usage Example:**
```python
# Create a tensor network state for a system with specific entanglement
state = TensorNetworkState(num_qudits=10, dimensions=[2]*10)

# Apply gates efficiently using tensor network methods
cnot = ControlledGate(
    target_gate=PermutationGate(circuit.qudits[1], [1, 0]),
    control_qudit=circuit.qudits[0]
)
state.apply_gate(cnot)
```

## Circuit Analysis

### CircuitAnalyzer (`qsim/analysis/circuit_analyzer.py`)

Analyzes circuits to determine optimal simulation strategy.

```python
class CircuitAnalyzer:
    def __init__(self, circuit: Optional[QuantumCircuit] = None):
        """
        Initialize a circuit analyzer.
        
        Args:
            circuit: Optional circuit to analyze
        """
        
    @property
    def circuit(self) -> QuantumCircuit:
        """Circuit being analyzed."""
        
    @property
    def connectivity_graph(self) -> nx.Graph:
        """Graph of qudit connectivity based on two-qudit gates."""
        
    @property
    def gate_classifications(self) -> Dict[str, List]:
        """Gates classified by type (local, entangling, etc.)."""
        
    @property
    def entanglement_regions(self) -> List[Set[int]]:
        """Regions of entangled qudits identified in the circuit."""
        
    @property
    def sparsity_estimate(self) -> float:
        """Estimated state sparsity (0.0-1.0) after circuit execution."""
        
    def analyze_circuit(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Analyze a circuit to determine its properties.
        
        Args:
            circuit: Circuit to analyze
            
        Returns:
            Dictionary of analysis results including:
            - entanglement_structure
            - estimated_sparsity
            - circuit_depth
            - gate_counts
            - recommended_simulation_method
        """
        
    def get_optimal_simulation_method(self) -> str:
        """
        Get best simulation method based on circuit analysis.
        
        Returns:
            String indicating recommended method:
            - 'state_vector': For small circuits
            - 'sparse': For circuits with high sparsity
            - 'tensor_network': For circuits with specific entanglement patterns
            - 'hybrid': For circuits that benefit from mixed approaches
        """
        
    def partition_circuit(self) -> List[Tuple[List[Gate], str]]:
        """
        Partition circuit for hybrid execution.
        
        Returns:
            List of (gates, method) tuples indicating which simulation
            method to use for each partition of gates
        """
```

**Usage Example:**
```python
# Analyze a circuit to determine optimal simulation strategy
analyzer = CircuitAnalyzer()
results = analyzer.analyze_circuit(circuit)

# Get recommended simulation method
method = analyzer.get_optimal_simulation_method()

# Partition circuit for hybrid execution
partitions = analyzer.partition_circuit()
```

## Execution Management

### HybridExecutionManager (`qsim/execution/hybrid_execution_manager.py`)

Manages hybrid simulation execution.

```python
class HybridExecutionManager:
    def __init__(self, circuit: Optional[QuantumCircuit] = None):
        """
        Initialize a hybrid execution manager.
        
        Args:
            circuit: Optional circuit to simulate
        """
        
    @property
    def circuit(self) -> QuantumCircuit:
        """Circuit to simulate."""
        
    @property
    def analyzer(self) -> CircuitAnalyzer:
        """Circuit analyzer used for optimization."""
        
    @property
    def state(self) -> QuantumState:
        """Current quantum state during simulation."""
        
    @property
    def current_method(self) -> str:
        """Current simulation method being used."""
        
    def determine_simulation_method(self, circuit: QuantumCircuit) -> Tuple[str, Dict]:
        """
        Determine best simulation method for a circuit.
        
        Args:
            circuit: Circuit to analyze
            
        Returns:
            Tuple of (method_name, configuration_parameters)
        """
        
    def create_initial_state(self, circuit: QuantumCircuit, method: str) -> QuantumState:
        """
        Create initial state for simulation.
        
        Args:
            circuit: Circuit to simulate
            method: Simulation method to use
            
        Returns:
            Initialized quantum state
        """
        
    def initialize_state(self, method: str) -> None:
        """
        Initialize state for simulation.
        
        Args:
            method: Simulation method to use
        """
        
    def convert_state(self, target_method: str) -> None:
        """
        Convert between state representations.
        
        Args:
            target_method: Target simulation method
            
        Note:
            Converts the current state to the representation
            required by the target method.
        """
        
    def execute_circuit(self) -> QuantumState:
        """
        Execute the circuit using hybrid simulation.
        
        Returns:
            Final quantum state after execution
            
        Note:
            Automatically switches between simulation methods
            based on circuit analysis.
        """
```

**Usage Example:**
```python
# Create execution manager with a circuit
manager = HybridExecutionManager(circuit)

# Execute the circuit with automatic method selection
final_state = manager.execute_circuit()

# Get measurement probabilities
probabilities = final_state.get_probabilities()
```

### HybridQuantumSimulator (`qsim/execution/simulator.py`)

Main simulator interface.

```python
class HybridQuantumSimulator:
    def __init__(self, use_gpu: Optional[bool] = None):
        """
        Initialize a hybrid quantum simulator.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
                     (None = auto-detect)
        """
        
    @property
    def use_gpu(self) -> bool:
        """Whether GPU acceleration is being used."""
        
    def simulate(self, circuit: QuantumCircuit) -> QuantumState:
        """
        Simulate a quantum circuit.
        
        Args:
            circuit: Circuit to simulate
            
        Returns:
            Final quantum state after simulation
        """
        
    def simulate_and_measure(self, circuit: QuantumCircuit, num_shots: int = 1024) -> Dict[str, int]:
        """
        Simulate a circuit and perform multiple measurements.
        
        Args:
            circuit: Circuit to simulate
            num_shots: Number of measurements to perform
            
        Returns:
            Dictionary mapping measurement outcomes to counts
        """
```

**Usage Example:**
```python
# Create a simulator with GPU acceleration if available
simulator = HybridQuantumSimulator()

# Simulate a circuit
final_state = simulator.simulate(circuit)

# Simulate and get measurement statistics
results = simulator.simulate_and_measure(circuit, num_shots=10000)
```

## Constants and Configuration

### Constants (`qsim/core/constants.py`)

```python
# Threshold for using tensor networks (entanglement measure)
TENSOR_NETWORK_THRESHOLD: float = 0.7  

# Threshold for using sparse representation (estimated sparsity)
SPARSITY_THRESHOLD: float = 0.01  
```

### Configuration (`qsim/core/config.py`)

```python
# Whether GPU acceleration is available
HAS_GPU: bool = True  

# Whether tensor network library is available
HAS_TENSOR_NETWORK: bool = True  

# Logger for the qsim package
logger: Logger = logging.getLogger("qsim")
```

**Usage Example:**
```python
from qsim.core.constants import SPARSITY_THRESHOLD
from qsim.core.config import HAS_GPU, logger

# Adjust sparsity threshold for specific use case
custom_threshold = SPARSITY_THRESHOLD * 0.5

# Check if GPU is available
if HAS_GPU:
    logger.info("Using GPU acceleration")
else:
    logger.info("GPU acceleration not available")
```

