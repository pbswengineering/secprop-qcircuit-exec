"""
A multi-device scheduler for quantum circuit execution based on circuit cutting.
The scheduler also provides security features such as:
   - confidentiality (devices can be rated 1-10 with respect to confidentiality 
     and each one will receive a number of circuit fragments based on this metric);
   - integrity: each QPU will be tested with benchmark circuits with known
     expected values prior to using them and an integrity trust index will be
     computed, so that dependable QPUs will influence the result more than
     unreliable ones.
The following class is exported:
   - QScheduler
"""

from collections import defaultdict
import logging
import random
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import networkx.classes.multidigraph
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.tape.qscript import QuantumScript


ResultType = Union[int, Tuple[int, int]]
SabotageFuncType = Optional[Callable[[float], float]]


def weighted_avg(a, weight_a: float, b, weight_b: float):
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            raise ValueError("Tuples a and b must be of the same length.")
        return tuple(
            ((a[i] * weight_a) + (b[i] * weight_b)) / (weight_a + weight_b)
            for i in range(len(a))
        )
    else:
        return ((a * weight_a) + (b * weight_b)) / (weight_a + weight_b)


def random_circuit(
    num_qubits: int, depth: int, rng: np.random.Generator
) -> QuantumTape:
    assert num_qubits > 1, "Qubits should be strictly positive"
    assert depth > 1, "Circuit depth should be strictly positive"
    single_qubit_gates = [qml.RX, qml.RY, qml.RZ]
    two_qubit_gates = [qml.CNOT, qml.CZ, qml.SWAP]
    with QuantumTape() as tape:
        for _ in range(depth):
            for qubit in range(num_qubits):
                gate = rng.choice(single_qubit_gates)
                param = rng.uniform(0, 2 * np.pi)
                gate(param, wires=qubit)
            if num_qubits > 1:
                qubit1, qubit2 = rng.choice(range(num_qubits), size=2, replace=False)
                gate = rng.choice(two_qubit_gates)
                gate(wires=[qubit1, qubit2])
        qml.expval(qml.pauli.string_to_pauli_word("Z" * num_qubits))
    return tape


class QPU:
    """A Quantum Processing Unit."""

    label: str
    device: qml.Device
    confidentiality_score: int
    sabotage_func: SabotageFuncType

    def __init__(
        self,
        label: str,
        device: qml.Device,
        confidentiality_score: int,
        sabotage_func: SabotageFuncType = None,
    ):
        assert (
            1 <= confidentiality_score <= 10
        ), "Confidentiality trust should be 1..10, inclusive"
        self.label = label
        self.device = device
        self.confidentiality_score = confidentiality_score
        self.sabotage_func = sabotage_func

    def __hash__(self):
        """Use the label as the hash value."""
        return hash(self.label)

    def __eq__(self, other):
        """Check equality based on the label."""
        if isinstance(other, QPU):
            return self.label == other.label
        return False


last_real_result = None


class QProcess:
    """A single quantum computation executed on a single QPU."""

    script: QuantumScript
    qpu: QPU
    label: str
    result: Optional[float]

    def __init__(
        self,
        script: QuantumScript,
        qpu: QPU,
        label: str,
        result: Optional[float] = None,
    ):
        self.script = script
        self.qpu = qpu
        self.label = label
        self.result = result

    def execute(self) -> ResultType:
        global last_real_result
        if self.result is not None:
            if last_real_result is not None:
                self.result += random.random()
            return self.result
        res = qml.execute([self.script], self.qpu.device, gradient_fn=None)[0]
        if self.qpu.sabotage_func:
            res = self.qpu.sabotage_func(res)
        last_real_result = res
        return res


class QPlan:
    """A schedule of multiple quantum computation on multiple QPUs."""

    process_labels: Set[str]
    qpu_schedules: Dict[QPU, List[QProcess]]
    process_qpus: Dict[str, QPU]
    results: Dict[str, ResultType]

    def __init__(self):
        self.clear()

    def add_process(self, process: QProcess):
        dev = process.qpu
        if not dev.label in self.qpu_schedules:
            self.qpu_schedules[dev.label] = []
        assert (
            process.label not in self.process_labels
        ), f"There's already a process labeled '{process.label}'"
        self.qpu_schedules[dev.label].append(process)
        self.process_labels.add(process.label)
        self.process_qpus[process.label] = dev

    def has_process(self, label: str):
        return label in self.process_labels

    def clear(self):
        self.process_labels = set()
        self.qpu_schedules = {}
        self.process_qpus = {}
        self.results = {}

    def execute(self):
        for process_list in self.qpu_schedules.values():
            for process in process_list:
                res = process.execute()
                self.results[process.label] = res

    def dump(self, logger: logging.Logger):
        logger.debug("PROCESS DISTRIBUTION:")
        tot = len(self.process_qpus)
        for qpu, process_list in self.qpu_schedules.items():
            processes = len(process_list)
            perc = processes * 100 / tot
            logger.debug(f"   {qpu}: {processes} {perc}%")


class QScheduler:
    """Execute quantum circuits by applying circuit cutting (both normal and secure cutting)."""

    devices: List[QPU]
    redundancy_factor: int
    integrity_scores: Dict[QPU, float]
    real_circuit: QuantumTape
    rng: Optional[np.random.Generator]
    obfuscation_factor: float
    ignore_fake_circuits: bool
    fake_circuits_output_range: Optional[Tuple[float]]

    def __init__(self):
        self.devices = []
        self.redundancy_factor = 2
        self.integrity_scores = {}
        self.real_circuit = None
        self.rng = None
        self.obfuscation_factor = 0
        self.ignore_fake_circuits = True
        self.fake_circuits_output_range = None

    def add_device(self, device: QPU):
        """Add a quantum processor (it may be simulated or physical) with its trust levels (1..10, inclusive)."""
        assert device is not None, "Please specify a device"
        self.devices.append(device)

    def set_real_circuit(self, circuit: QuantumTape):
        """Set the real quantum circuit that must be executed."""
        assert circuit is not None, "Please specify a circuit"
        self.real_circuit = circuit

    def __cut_tape(
        self, original_tape: QuantumTape, wires: qml.wires.Wires
    ) -> Tuple[List[QuantumScript], networkx.classes.multidigraph.MultiDiGraph]:
        graph = qml.qcut.tape_to_graph(original_tape)
        qml.qcut.replace_wire_cut_nodes(graph)
        fragments, communication_graph = qml.qcut.fragment_graph(graph)
        fragment_tapes = [qml.qcut.graph_to_tape(f) for f in fragments]
        fragment_tapes = [
            qml.map_wires(t, dict(zip(t.wires, wires)))[0][0] for t in fragment_tapes
        ]
        return fragment_tapes, communication_graph

    def __expand_sub_circuit(self, fragment_tapes: List[QuantumScript]) -> Tuple[
        List[QuantumScript],
        List[List[qml.qcut.utils.PrepareNode]],
        List[List[qml.qcut.utils.MeasureNode]],
    ]:
        expanded = [qml.qcut.expand_fragment_tape(t) for t in fragment_tapes]
        configurations = []
        prepare_nodes = []
        measure_nodes = []
        for t, p, m in expanded:
            configurations.append(t)
            prepare_nodes.append(p)
            measure_nodes.append(m)
        sub_tapes = tuple(tape for c in configurations for tape in c)
        return sub_tapes, prepare_nodes, measure_nodes

    def exec_with_singledev_cut(self, logger: logging.Logger):
        """Execute a quantum circuit with a standard cutting technique using the first device."""
        wires = self.devices[0].device.wires
        fragment_tapes, communication_graph = self.__cut_tape(self.real_circuit, wires)
        sub_tapes, prepare_nodes, measure_nodes = self.__expand_sub_circuit(
            fragment_tapes
        )
        results = []
        for t in sub_tapes:
            dev = self.devices[0]
            res = qml.execute([t], dev.device, gradient_fn=None)[0]
            if dev.sabotage_func:
                res = dev.sabotage_func(res)
            results.append(res)
        # results = qml.execute(sub_tapes, self.devices[0].device, gradient_fn=None)
        result_with_cut = qml.qcut.qcut_processing_fn(
            results,
            communication_graph,
            prepare_nodes,
            measure_nodes,
        )
        return result_with_cut

    def exec_with_normal_multidev_cut(self, logger: logging.Logger):
        """Execute a quantum circuit with a standard multiple-device cutting technique."""
        wires = self.devices[0].device.wires
        fragment_tapes, communication_graph = self.__cut_tape(self.real_circuit, wires)
        sub_tapes, prepare_nodes, measure_nodes = self.__expand_sub_circuit(
            fragment_tapes
        )
        plan = QPlan()
        for i, script in enumerate(sub_tapes):
            qpu = self.devices[i % len(self.devices)]
            process = QProcess(script, qpu, f"sub-tape-{i}")
            plan.add_process(process)
        plan.execute()
        results = [plan.results[f"sub-tape-{i}"] for i in range(len(sub_tapes))]
        result_with_cut = qml.qcut.qcut_processing_fn(
            results,
            communication_graph,
            prepare_nodes,
            measure_nodes,
        )
        return result_with_cut

    def get_test_tape(self):
        with QuantumTape() as test_tape:
            qml.RX(np.pi / 4, wires=0)
            qml.RY(np.pi / 6, wires=0)
            qml.expval(qml.PauliZ(0))
        expected_result = 0.61237
        return test_tape, expected_result

    def configure_redundancy(self, redundancy_factor: int):
        assert (
            1 <= redundancy_factor <= len(self.devices)
        ), f"Redundancy factor must be between 2 and {len(self.devices)}"
        self.redundancy_factor = redundancy_factor

    def configure_obfuscation(
        self,
        rng: np.random.Generator,
        obfuscation_factor: float,
        ignore_fake_circuits: bool,
        fake_circuits_output_range: Optional[Tuple[float]] = None,
    ):
        assert rng is not None, "Please specify a random number generator"
        assert obfuscation_factor >= 0, "Obfuscation factor must be >= 0"
        self.rng = rng
        self.obfuscation_factor = obfuscation_factor
        self.ignore_fake_circuits = ignore_fake_circuits
        self.fake_circuits_output_range = fake_circuits_output_range

    def exec_with_secure_multidev_cut(self, logger: logging.Logger):
        """
        Execute a quantum circuit with a secure multiple-device cutting technique:
           - Integrity protection via pre-computed circuit evaluation.
           - Integrity protection via redundant subcircuit evaluation.
           - Confidentiality protection via dummy circuit evaluation.
        """
        wires = self.devices[0].device.wires
        fragment_tapes, communication_graph = self.__cut_tape(self.real_circuit, wires)
        sub_tapes, prepare_nodes, measure_nodes = self.__expand_sub_circuit(
            fragment_tapes
        )
        logger.debug(f"{len(sub_tapes)} sub-tapes")

        plan = QPlan()

        # Configure integrity evaluation processes
        expected_test_results = {}
        for qpu_index, qpu in enumerate(self.devices):
            test_process_label = f"test-tape-{qpu_index}"
            test_tape, expected_result = self.get_test_tape()
            test_process = QProcess(test_tape, qpu, test_process_label)
            plan.add_process(test_process)
            expected_test_results[qpu_index] = expected_result

        # Execute integrity test processes
        plan.execute()

        # Compute integrity scores
        for qpu_index, qpu in enumerate(self.devices):
            test_process_label = f"test-tape-{qpu_index}"
            result = plan.results[test_process_label]
            expected = expected_test_results[qpu_index]
            error = abs(expected - result)
            score = max(1, 10 - int(error / expected * 10))  # 1-10 scaling
            self.integrity_scores[qpu] = score
            logger.debug(f"Device {qpu.label} integrity score = {score}")

        # Assign sub-circuits to QPUs based on integrity scores AND confidence factors
        # tot_confidentiality_scores = sum(
        #     qpu.confidentiality_score for qpu in self.devices
        # )
        # tot_integrity_scores = sum(self.integrity_scores.values())
        # qpu_probabilities = [
        #     (self.integrity_scores[qpu] + qpu.confidentiality_score)
        #     / (tot_integrity_scores + tot_confidentiality_scores)
        #     for qpu in self.devices
        # ]
        # Assign sub-circuits to QPUs based on exponential weighting of integrity and confidentiality scores
        qpu_weights = [
            np.exp(self.integrity_scores[qpu] + qpu.confidentiality_score)
            for qpu in self.devices
        ]
        total_weight = sum(qpu_weights)
        qpu_probabilities = [weight / total_weight for weight in qpu_weights]
        qpu_assignments = defaultdict(list)
        for i, script in enumerate(sub_tapes):
            assigned_qpus = set()
            while len(assigned_qpus) < self.redundancy_factor:
                # At least redundancy_factor QPUs must receive the sub-circuit
                selected_qpu = random.choices(
                    self.devices, weights=qpu_probabilities, k=1
                )[0]
                if selected_qpu not in assigned_qpus:
                    assigned_qpus.add(selected_qpu)
                    qpu_assignments[selected_qpu].append((i, script))

        plan.clear()

        # Add processes for each assigned QPU
        for qpu, assignments in qpu_assignments.items():
            for i, script in assignments:
                plan.add_process(QProcess(script, qpu, f"sub-tape-{i}-{qpu.label}"))

        # If there are fake processes, add them to the plan
        fake_circuits_len = round(len(sub_tapes) * self.obfuscation_factor)
        logger.debug(f"Adding {fake_circuits_len} fake circuits")
        fake_circuits = [
            random_circuit(len(wires), self.rng.integers(2, 7), self.rng)
            for i in range(fake_circuits_len)
        ]
        logger.debug(f"{len(fake_circuits)} fake circuits created")
        devices_sorted_by_conf = sorted(
            self.devices, key=lambda qpu: qpu.confidentiality_score
        )
        devices_len = len(self.devices)
        for i, fake_circuit in enumerate(fake_circuits):
            qpu = devices_sorted_by_conf[i % devices_len]
            if self.fake_circuits_output_range:
                plan.add_process(
                    QProcess(
                        fake_circuit,
                        qpu,
                        f"fake-tape-{i}-{qpu.label}",
                        self.rng.uniform(
                            self.fake_circuits_output_range[0],
                            self.fake_circuits_output_range[1],
                        ),
                    )
                )
            else:
                plan.add_process(
                    QProcess(fake_circuit, qpu, f"fake-tape-{i}-{qpu.label}")
                )

        # Execute the processes
        plan.dump(logger)  # Debugging and descriptive statistics
        plan.execute()

        # Combine results from sub-tapes
        results = []
        for i in range(len(sub_tapes)):
            qpu_results = []
            for qpu in self.devices:
                process_label = f"sub-tape-{i}-{qpu.label}"
                if process_label in plan.results:
                    raw_result = plan.results[process_label]
                    qpu_results.append((raw_result, self.integrity_scores[qpu]))
                    if fake_circuits and not self.ignore_fake_circuits:
                        for k in range(len(fake_circuits)):
                            fake_process_label = f"fake-tape-{k}-{qpu.label}"
                            if fake_process_label in plan.results:
                                fake_result = plan.results[fake_process_label]
                                qpu_results.append(
                                    (fake_result, self.integrity_scores[qpu])
                                )

            if len(qpu_results) < 1:
                logger.error(
                    f"Insufficient results for sub-tape-{i} (we have {len(sub_tapes)} sub-tapes)"
                )
                continue

            if self.ignore_fake_circuits:
                # Compute weighted average
                weighted_result = None
                for raw_result, score in qpu_results:
                    if weighted_result is None:
                        weighted_result = (raw_result, score)
                    else:
                        weighted_result = (
                            weighted_avg(
                                weighted_result[0],
                                weighted_result[1],
                                raw_result,
                                score,
                            ),
                            weighted_result[1] + score,
                        )
                results.append(weighted_result[0])
            else:
                for raw_result, score in qpu_results:
                    results.append(raw_result)

        if self.ignore_fake_circuits:
            # Classical post-processing
            result_with_cut = qml.qcut.qcut_processing_fn(
                results,
                communication_graph,
                prepare_nodes,
                measure_nodes,
            )
            return result_with_cut
        else:
            # Raw low-level output
            return results
