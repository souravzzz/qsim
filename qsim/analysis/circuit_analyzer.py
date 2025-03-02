"""
Circuit analyzer for determining optimal simulation strategies.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Set, Any
from qsim.core.circuit import QuantumCircuit
from qsim.gates.base import Gate
from qsim.gates.phase import PhaseGate
from qsim.gates.permutation import PermutationGate
from qsim.gates.hadamard import HadamardGate
from qsim.gates.controlled import ControlledGate
from qsim.core.constants import TENSOR_NETWORK_THRESHOLD, SPARSITY_THRESHOLD


class CircuitAnalyzer:
    """Analyzes quantum circuits to determine the optimal simulation strategy."""

    def __init__(self, circuit: QuantumCircuit = None):
        """
        Initialize a circuit analyzer.

        Args:
            circuit: Optional quantum circuit to analyze
        """
        self.circuit = circuit
        if circuit:
            self.connectivity_graph = self._build_connectivity_graph()
            self.gate_classifications = self._classify_gates()
            self.entanglement_regions = self._identify_entanglement_regions()
            self.sparsity_estimate = self._estimate_sparsity()

    def analyze_circuit(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Analyze a quantum circuit and return a dictionary of analysis results.

        Args:
            circuit: The quantum circuit to analyze

        Returns:
            A dictionary containing analysis results
        """
        self.circuit = circuit
        self.connectivity_graph = self._build_connectivity_graph()
        self.gate_classifications = self._classify_gates()
        self.entanglement_regions = self._identify_entanglement_regions()
        self.sparsity_estimate = self._estimate_sparsity()

        # Create analysis dictionary
        analysis = {
            "num_qudits": self.circuit.num_qudits,
            "dimensions": self.circuit.dimensions,
            "total_dimension": np.prod(self.circuit.dimensions),
            "num_gates": len(self.circuit.gates),
            "entangling_gates": len(self.gate_classifications["entangling"]),
            "is_highly_entangled": any(len(region) > 2 for region in self.entanglement_regions),
            "is_sparse": self.sparsity_estimate > SPARSITY_THRESHOLD,
            "requires_tensor_network": any(len(region) > 2 for region in self.entanglement_regions),
            "requires_sparse_simulation": self.sparsity_estimate > SPARSITY_THRESHOLD,
            "entanglement_ratio": (
                len(self.gate_classifications["entangling"]) / self.circuit.num_qudits
                if self.circuit.num_qudits > 0
                else 0
            ),
            "has_mixed_dimensions": len(set(self.circuit.dimensions)) > 1,
            "optimal_method": self.get_optimal_simulation_method(),
        }

        # Add block structure analysis
        blocks = self._identify_blocks()
        analysis["has_block_structure"] = len(blocks) > 1
        analysis["blocks"] = blocks

        return analysis

    def _build_connectivity_graph(self) -> nx.Graph:
        """Build a connectivity graph of the circuit, where nodes are qudits and edges represent interactions."""
        graph = nx.Graph()

        # Add nodes for each qudit
        for i in range(self.circuit.num_qudits):
            graph.add_node(i, dimension=self.circuit.dimensions[i])

        # Add edges for each multi-qudit gate
        for gate in self.circuit.gates:
            if len(gate.qudits) > 1:
                # Add edges between all pairs of qudits the gate acts on
                qudit_indices = [q.index for q in gate.qudits]
                for i in range(len(qudit_indices)):
                    for j in range(i + 1, len(qudit_indices)):
                        if not graph.has_edge(qudit_indices[i], qudit_indices[j]):
                            graph.add_edge(qudit_indices[i], qudit_indices[j], gates=[])
                        graph[qudit_indices[i]][qudit_indices[j]]["gates"].append(gate)

        return graph

    def _classify_gates(self) -> Dict[str, List[Gate]]:
        """Classify gates into different categories."""
        classifications = {
            "local": [],
            "permutation": [],
            "phase": [],
            "controlled": [],
            "entangling": [],
            "other": [],
        }

        for gate in self.circuit.gates:
            if isinstance(gate, PhaseGate):
                classifications["phase"].append(gate)
            elif isinstance(gate, PermutationGate):
                classifications["permutation"].append(gate)
            elif isinstance(gate, ControlledGate):
                classifications["controlled"].append(gate)
                if gate.is_entangling:
                    classifications["entangling"].append(gate)
            elif gate.is_local:
                classifications["local"].append(gate)
            elif gate.is_entangling:
                classifications["entangling"].append(gate)
            else:
                classifications["other"].append(gate)

        return classifications

    def _identify_entanglement_regions(self) -> List[Set[int]]:
        """Identify regions of high entanglement in the circuit."""
        # Start with connected components of the connectivity graph
        components = list(nx.connected_components(self.connectivity_graph))

        # Further analyze each component
        regions = []
        for component in components:
            # If the component is a single qudit, it's not entangled
            if len(component) == 1:
                continue

            # Check for entangling gates between qudits in this component
            subgraph = self.connectivity_graph.subgraph(component)
            entangling_gates = 0
            total_gates = 0

            for u, v, data in subgraph.edges(data=True):
                for gate in data["gates"]:
                    total_gates += 1
                    if gate.is_entangling:
                        entangling_gates += 1

            # If a significant fraction of gates are entangling, consider it an entanglement region
            if total_gates > 0 and entangling_gates / total_gates > TENSOR_NETWORK_THRESHOLD:
                regions.append(set(component))
            else:
                # Check individual qudit pairs
                for u, v, data in subgraph.edges(data=True):
                    entangling_gates = sum(1 for gate in data["gates"] if gate.is_entangling)
                    if entangling_gates > 0:
                        regions.append({u, v})

        return regions

    def _identify_blocks(self) -> List[List]:
        """
        Identify independent blocks of qudits in the circuit.

        Returns:
            A list of lists, where each inner list contains the qudits in a block
        """
        # Use connected components of the connectivity graph to identify blocks
        components = list(nx.connected_components(self.connectivity_graph))

        # Convert to lists of qudit objects
        blocks = []
        for component in components:
            block = [self.circuit.qudits[i] for i in component]
            if block:  # Only add non-empty blocks
                blocks.append(block)

        # Add isolated qudits as their own blocks
        isolated_qudits = (
            set(range(self.circuit.num_qudits)) - set().union(*components)
            if components
            else set(range(self.circuit.num_qudits))
        )
        for i in isolated_qudits:
            blocks.append([self.circuit.qudits[i]])

        return blocks

    def _estimate_sparsity(self) -> float:
        """Estimate the sparsity of the quantum state throughout the circuit execution."""
        # Start with a perfectly sparse state (only one nonzero amplitude)
        sparsity = 1.0

        # For circuits with very few gates relative to the number of qudits,
        # the state is likely to remain sparse
        if len(self.circuit.gates) < self.circuit.num_qudits / 2:
            # Adjust sparsity based on the ratio of gates to qudits
            gate_to_qudit_ratio = len(self.circuit.gates) / self.circuit.num_qudits
            # Higher ratio means less sparse
            sparsity = max(0.95, 1.0 - gate_to_qudit_ratio)
            return sparsity

        # Count Hadamard and other superposition-creating gates
        hadamard_count = 0
        entangling_count = 0

        # Analyze how each gate affects sparsity
        for gate in self.circuit.gates:
            if isinstance(gate, HadamardGate):
                # Hadamard creates a full superposition, reducing sparsity
                hadamard_count += 1
                qudit_dim = gate.qudits[0].dimension
                sparsity *= 1.0 / qudit_dim
            elif gate.is_entangling:
                # Entangling gates can reduce sparsity
                entangling_count += 1
                sparsity *= 0.9  # Approximate factor
            elif isinstance(gate, PermutationGate) or isinstance(gate, PhaseGate):
                # These gates maintain sparsity
                pass
            else:
                # Other gates can reduce sparsity slightly
                sparsity *= 0.95

        # If there are very few Hadamard gates, the state is likely to remain sparse
        if hadamard_count <= 1 and self.circuit.num_qudits > 10:
            sparsity = max(sparsity, 0.95)

        # If there are no entangling gates, the state is likely to be more sparse
        if entangling_count == 0:
            sparsity = max(sparsity, 0.92)

        return max(sparsity, 0.0)

    def get_optimal_simulation_method(self) -> str:
        """Determine the optimal simulation method for the circuit."""
        # Decision based on circuit properties

        # If the state is expected to be sparse, use sparse simulation
        if self.sparsity_estimate > SPARSITY_THRESHOLD:
            return "sparse"

        # If there are significant entanglement regions, use tensor networks
        if any(len(region) > 2 for region in self.entanglement_regions):
            return "tensor_network"

        # For small circuits, dense state vector is fine
        if self.circuit.num_qudits <= 20:
            return "state_vector"

        # Default to sparse representation
        return "sparse"

    def partition_circuit(self) -> List[Tuple[List[Gate], str]]:
        """
        Partition the circuit into segments that can be simulated efficiently with different methods.

        Returns:
            List of (gates, method) tuples, where method is the recommended simulation method
        """
        partitions = []
        current_partition = []
        current_method = "state_vector"  # Start with state vector as default

        # Keep track of entanglement buildup
        entanglement_level = 0.0

        for gate in self.circuit.gates:
            # Check if this gate significantly changes the simulation method
            if gate.is_entangling:
                entanglement_level += 0.1

            if isinstance(gate, HadamardGate) and entanglement_level > TENSOR_NETWORK_THRESHOLD:
                # High entanglement with Hadamard suggests using tensor networks
                if current_method != "tensor_network" and current_partition:
                    partitions.append((current_partition, current_method))
                    current_partition = []
                    current_method = "tensor_network"

            current_partition.append(gate)

            # If we've accumulated enough gates, create a partition
            if len(current_partition) >= 10:
                partitions.append((current_partition, current_method))
                current_partition = []

                # Re-evaluate method based on current state
                if entanglement_level > TENSOR_NETWORK_THRESHOLD:
                    current_method = "tensor_network"
                elif self.sparsity_estimate > SPARSITY_THRESHOLD:
                    current_method = "sparse"
                else:
                    current_method = "state_vector"

        # Add the last partition if not empty
        if current_partition:
            partitions.append((current_partition, current_method))

        return partitions
