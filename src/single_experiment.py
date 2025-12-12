from datetime import datetime, timedelta
import importlib
import json
import logging
import math
import os
from random import random
import warnings
from typing import Callable, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt

import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

from lib.qscheduler import QPU, QScheduler
from lib.utils import create_logger


def attack(val: Union[float, Tuple[float]]) -> Union[float, Tuple[float]]:
    def perturbation(x):
        return x * (1.5 + random())

    if type(val) is tuple:
        return tuple(perturbation(x) for x in val)
    else:
        return perturbation(val)


def flatten_mixed(input_list):
    flat_list = []

    for item in input_list:
        if isinstance(item, list) or isinstance(
            item, tuple
        ):  # If the item is a list, recurse
            flat_list.extend(flatten_mixed(item))
        elif isinstance(item, np.ndarray):  # If the item is a NumPy array, flatten it
            flat_list.extend(item.flatten())
        else:  # Otherwise, add the item directly (it's a classical value)
            flat_list.append(item)

    return flat_list


def perform_experiment(
    test_name: str,
    n_qubits: int,
    benchmark_tape: QuantumTape,
    test_count: int,
    bins: int,
    bin_range: Tuple[int],
    shots: int,
    devices: int,
    confidence_trust_factors: List[int],
    redundancy_factor: int,
    saboteurs: Set[int],
    execution_function: Callable[[logging.Logger], Union[float, Tuple[float]]],
    obfuscation_factor: int,
    ignore_fake_circuits: bool,
    rng: Optional[np.random.Generator],
    fake_circuits_output_range: Optional[Tuple[float]],
    log_level: int,
):
    start_time = datetime.now()

    # C:\Users\rnd\Envs\thesis\Lib\site-packages\pennylane\ops\op_math\composite.py:185: FutureWarning: functools.partial will be a method descriptor in future Python versions; wrap it in staticmethod() if you want to preserve the old behavior
    #  return self._math_op(math.vstack(eigvals), axis=0)
    warnings.simplefilter(action="ignore", category=FutureWarning)

    test_dir = os.path.join(
        "tmp", f"{test_name}_{start_time.strftime("%Y-%m-%d_%H-%M-%S")}"
    )
    if not os.path.exists(test_name):
        os.makedirs(test_dir)

    test_log = os.path.join(test_dir, "test.log")
    logger = create_logger("perform_test", log_level, test_log)
    logger.info(f"Experiment {test_name.upper()} started at {start_time}")

    sched = QScheduler()

    logger.info("PARAMETERS:")
    logger.info(f"   test_name = {test_name}")
    logger.info(f"   n_qubits = {n_qubits}")
    logger.info(f"   benchmark_tape = {benchmark_tape}")
    logger.info(f"   test_count = {test_count}")
    logger.info(f"   bins = {bins}")
    logger.info(f"   bin_range = {bin_range}")
    logger.info(f"   shots = {shots}")
    logger.info(f"   devices = {devices}")
    logger.info(f"   confidence_trust_factors = {confidence_trust_factors}")
    logger.info(f"   saboteurs = {saboteurs}")
    logger.info(f"   execution_function = {execution_function}")
    logger.info(f"   redundancy_factor = {redundancy_factor}")
    logger.info(f"   obfuscation_factor = {obfuscation_factor}")
    logger.info(f"   rng = {rng}")
    logger.info(f"   ignore_fake_circuits = {ignore_fake_circuits}")
    logger.info(f"   fake_circuits_output_range = {fake_circuits_output_range}")

    logger.info("DEVICES:")
    # Check whether GPU is available
    available_devices = AerSimulator().available_devices()
    device = "GPU" if "GPU" in available_devices else "CPU"

    # Noise model configuration
    noise_model = NoiseModel()
    cnot_error = depolarizing_error(0.01, 2)
    rx_error = thermal_relaxation_error(50e-6, 30e-6, 0.02)
    ry_error = thermal_relaxation_error(50e-6, 30e-6, 0.015)
    noise_model.add_all_qubit_quantum_error(cnot_error, ["cx"])
    noise_model.add_all_qubit_quantum_error(rx_error, ["rx"])
    noise_model.add_all_qubit_quantum_error(ry_error, ["ry"])

    for d in range(devices):
        devname = f"DEV{d}"
        sabotage_func = attack if d in saboteurs else None
        sched.add_device(
            QPU(
                devname,
                qml.device(
                    "qiskit.aer",
                    wires=n_qubits,
                    backend=AerSimulator(device=device, noise_model=noise_model),
                    shots=shots,
                    seed=42,
                ),
                confidence_trust_factors[d],
                sabotage_func=sabotage_func,
            )
        )
        logger.info(
            f"   Device {devname}, confidence {confidence_trust_factors[d]}, sabotage {sabotage_func}, device {device}"
        )

    original_tape = benchmark_tape
    if obfuscation_factor > 0:
        assert benchmark_tape is not None, "Please specify the benchmark tape"
        sched.configure_obfuscation(
            rng, obfuscation_factor, ignore_fake_circuits, fake_circuits_output_range
        )
    sched.configure_redundancy(redundancy_factor)

    sched.set_real_circuit(original_tape)
    qml.drawer.tape_mpl(original_tape)
    circuit_fig_file = os.path.join(test_dir, f"{test_name}_circuit.png")
    logger.info(f"Saving circuit drawing to {circuit_fig_file}...")
    plt.savefig(circuit_fig_file)
    plt.close()

    results = []
    execution_func = getattr(sched, execution_function)
    start_execs = datetime.now()
    padding = math.ceil(math.log10(test_count)) + 1
    for i in range(test_count):
        res = execution_func(logger)
        results.append(res)
        if test_count > 20 and (i + 1) % (test_count // 20) == 0:
            elapsed = int((datetime.now() - start_execs).total_seconds())
            total = elapsed * test_count // i
            remaining = timedelta(seconds=total - elapsed)
            # 10 characters-long progress bar
            bar_remaining = round(remaining.total_seconds() * 10 / total)
            bar_done = 10 - bar_remaining
            logger.info(
                f"{i+1: >{padding}} / {test_count} [{'#' * bar_done}{'_' * bar_remaining}] {remaining} left... "
            )
    if not ignore_fake_circuits:
        results = flatten_mixed(results)

    # Save expected values to disk
    test_results_file = os.path.join(test_dir, f"{test_name}_results.json")
    logger.info(f"Saving test results to {test_results_file}...")
    with open(test_results_file, "w") as f:
        json.dump(results, f)

    # Create a histogram and numerical distribution
    counts, bin_edges = np.histogram(
        results, bins=bins, range=bin_range, density=True
    )  # Normalized histogram
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Compute bin centers
    # Numerical distribution (bin_centers, normalized probabilities)
    numerical_distribution = list(zip(bin_centers, counts))
    # Save histogram to disk
    test_hist_file = os.path.join(test_dir, f"{test_name}_histogram.json")
    logger.info(f"Saving test histogram data to {test_hist_file}...")
    with open(test_hist_file, "w") as f:
        json.dump(numerical_distribution, f)

    plt.bar(
        bin_centers,
        counts,
        width=(bin_edges[1] - bin_edges[0]),
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
    )
    plt.title(f"Histogram of Expectation Values (shots={shots})")
    if bin_range:
        plt.xlim(*bin_range)
    plt.xlabel("Expectation Value")
    plt.ylabel("Probability Density")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    hist_fig_file = os.path.join(test_dir, f"{test_name}_hist.png")
    logger.info(f"Saving histogram drawing to {hist_fig_file}...")
    plt.savefig(hist_fig_file)
    plt.close()

    end_time = datetime.now()
    logger.info(f"Experiment {test_name.upper()} ended at {start_time}")
    logger.info(f"Duration: {end_time - start_time}")


if __name__ == "__main__":
    from lib.benchmark_15qubit import benchmark_tape_cut, n_qubits
    from lib.mqt import load_mqt_bench_dataset, get_mqt_tape

    # datasets = load_mqt_bench_dataset()
    # tape = get_mqt_tape(datasets, "dj", 15)
    tape = benchmark_tape_cut

    test_name = "test_random"
    benchmark_tape = tape
    test_count = 10
    bins = 120
    bin_range = (-0.02, 0.02)
    shots = 1000  # 2000
    devices = 6
    confidence_trust_factors = [1, 1, 1, 1, 1, 1]
    redundancy_factor = 1
    saboteurs = set()
    # execution_function = "exec_with_singledev_cut"
    execution_function = "exec_with_normal_multidev_cut"
    # execution_function = "exec_with_secure_multidev_cut"
    obfuscation_factor = 2
    ignore_fake_circuits = False
    rng = np.random.default_rng(12345)
    fake_circuits_output_range = None  # (-0.02, 0.02)

    perform_experiment(
        test_name,
        n_qubits,
        benchmark_tape,
        test_count,
        bins,
        bin_range,
        shots,
        devices,
        confidence_trust_factors,
        redundancy_factor,
        saboteurs,
        execution_function,
        obfuscation_factor,
        ignore_fake_circuits,
        rng,
        fake_circuits_output_range,
        logging.INFO,
    )
