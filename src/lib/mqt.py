from datetime import datetime
import logging
from pathlib import Path

import pennylane as qml


def load_mqt_bench_dataset(
    logger: logging.Logger = None,
) -> qml.data.base.dataset.Dataset:
    log = print if logger is None else logger.info
    log("Loading MQT-BENCH dataset...")
    path = Path.home() / "datasets"
    start = datetime.now()
    dataset = qml.data.load("other", name="mqt-bench", folder_path=path)[0]
    elapsed = datetime.now() - start
    log(f"Loaded in {elapsed}.")
    return dataset


def get_mqt_tape(
    dataset: qml.data.base.dataset.Dataset, circuit_name: str, n_qubits: int
) -> qml.tape.QuantumTape:
    operations = getattr(dataset, circuit_name)[str(n_qubits)]
    cuts = {1, 3}
    with qml.tape.QuantumTape() as tape:
        for i, op in enumerate(operations):
            qml.apply(op)
            if len(op.wires) == 1 and op.wires[0] in cuts:
                qml.WireCut(wires=op.wires[0])
                cuts.remove(op.wires[0])
        qml.expval(qml.pauli.string_to_pauli_word("Z" * n_qubits))
    return tape


if __name__ == "__main__":
    dataset = load_mqt_bench_dataset()
    tape = get_mqt_tape(dataset, "vqe", 15)
    print(tape.draw())
