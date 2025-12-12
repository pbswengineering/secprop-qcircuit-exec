from datetime import datetime
import logging
import os
import sys
import warnings

# C:\Users\rnd\Envs\thesis\Lib\site-packages\pennylane\ops\op_math\composite.py:185: FutureWarning: functools.partial will be a method descriptor in future Python versions; wrap it in staticmethod() if you want to preserve the old behavior
# return self._math_op(math.vstack(eigvals), axis=0)
warnings.simplefilter(action="ignore", category=FutureWarning)

import pennylane as qml
from pennylane import numpy as np
from qiskit_aer import AerSimulator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.qscheduler import QPU, QScheduler
from lib.benchmark_15qubit import benchmark_tape, benchmark_tape_cut, n_qubits


def check_error(reference_res, other_res, label):
    error = reference_res - other_res
    error_perc = abs(error * 100 / reference_res)
    print(f"ERROR ({label}) = {error:0.5f} ({error_perc:0.2f}%)")
    return error_perc


if __name__ == "__main__":
    shots = 1000

    start_time = datetime.now()

    sched = QScheduler()

    sched.add_device(
        QPU(
            "DEV0",
            qml.device(
                "qiskit.aer", wires=n_qubits, backend=AerSimulator(), shots=shots
            ),
            10,
        )
    )
    sched.add_device(
        QPU(
            "DEV1",
            qml.device(
                "qiskit.aer", wires=n_qubits, backend=AerSimulator(), shots=shots
            ),
            5,
        )
    )
    sched.add_device(
        QPU(
            "DEV2",
            qml.device(
                "qiskit.aer", wires=n_qubits, backend=AerSimulator(), shots=shots
            ),
            7,
        )
    )

    baseline_tape = benchmark_tape
    original_tape = benchmark_tape_cut

    sched.set_real_circuit(original_tape)
    qml.drawer.tape_mpl(original_tape)

    dev = qml.device("default.qubit", wires=n_qubits)
    # baseline = qml.execute([baseline_tape], dev)[0]
    baseline = sched.exec_with_singledev_cut(logging.getLogger())
    print("RESULT (baseline) =", baseline)

    singledev_errors = []
    multidev_normal_errors = []
    multidev_secure_errors = []
    for n in range(10):
        singledev = sched.exec_with_singledev_cut(logging.getLogger())
        print("RESULT (with single-device cutting) =", singledev)
        multidev_normal = sched.exec_with_normal_multidev_cut(logging.getLogger())
        print("RESULT (with normal multiple-device cutting) =", multidev_normal)
        multidev_secure = sched.exec_with_secure_multidev_cut(logging.getLogger())
        if not multidev_secure:
            print("Secure execution skipped")
            continue
        print("RESULT (with secure multiple-device cutting) =", multidev_secure)
        singledev_errors.append(
            check_error(baseline, singledev, "baseline VS singledev")
        )
        multidev_normal_errors.append(
            check_error(baseline, multidev_normal, "baseline VS multidev normal")
        )
        multidev_secure_errors.append(
            check_error(baseline, multidev_secure, "baseline VS multidev secure")
        )

    print("-" * 30)
    print(
        f"Single-device: MIN {min(singledev_errors):0.2f}%, MAX {max(singledev_errors):0.2f}%, AVG {(sum(singledev_errors) / len(singledev_errors)):0.2f}%"
    )
    print(
        f"Multiple-device, normal: MIN {min(multidev_normal_errors):0.2f}%, MAX {max(multidev_normal_errors):0.2f}%, AVG {(sum(multidev_normal_errors) / len(multidev_normal_errors)):0.2f}%"
    )
    print(
        f"Multiple-device, secure: MIN {min(multidev_secure_errors):0.2f}%, MAX {max(multidev_secure_errors):0.2f}%, AVG {(sum(multidev_secure_errors) / len(multidev_secure_errors)):0.2f}%"
    )
    print("-" * 30)

    end_time = datetime.now()
    print(f"Duration: {end_time - start_time}")
