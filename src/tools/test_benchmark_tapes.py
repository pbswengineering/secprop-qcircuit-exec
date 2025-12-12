import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import pennylane as qml
from qiskit_aer import AerSimulator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.qscheduler import QScheduler, QPU
from lib.benchmark_15qubit import (
    n_qubits,
    benchmark_tape_cut,
    benchmark_tape_cut1,
    benchmark_tape_cut2,
    benchmark_tape_cut3,
    benchmark_tape_cut4,
    benchmark_tape_cut5,
)

if __name__ == "__main__":
    devices = 6
    shots = 1000
    sched = QScheduler()
    for d in range(devices):
        devname = f"DEV{d}"
        sched.add_device(
            QPU(
                devname,
                qml.device(
                    "qiskit.aer",
                    wires=n_qubits,
                    backend=AerSimulator(),
                    shots=shots,
                ),
                10,
            )
        )
    tapes = [
        benchmark_tape_cut,
        benchmark_tape_cut1,
        benchmark_tape_cut2,
        benchmark_tape_cut3,
        benchmark_tape_cut4,
        benchmark_tape_cut5,
    ]
    for i, tape in enumerate(tapes):
        sched.set_real_circuit(tape)
        results = [
            sched.exec_with_normal_multidev_cut(logging.getLogger()) for i in range(10)
        ]
        avg = sum(results) / len(results)
        print(f"RESULT[{i}] = {avg}")
        qml.drawer.tape_mpl(tape)
        circuit_fig_file = os.path.join(
            os.path.dirname(__file__), "..", "tmp", f"benchmark_tape_cut{i}.png"
        )
        plt.savefig(circuit_fig_file)
        plt.close()
    print("Circuit drawings were saved in the tmp directory.")
