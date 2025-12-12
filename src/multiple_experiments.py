from datetime import datetime
import logging
from single_experiment import perform_experiment

import numpy as np

import lib.benchmark_15qubit
from lib.mqt import load_mqt_bench_dataset, get_mqt_tape

datasets = load_mqt_bench_dataset()
tape = get_mqt_tape(datasets, "vqe", 15)

prefix = "vqe1000_saboteurs"
n_qubits = lib.benchmark_15qubit.n_qubits
benchmark_tape = tape
test_count = 1000
devices = 6
bins = 120
bin_range = (-0.1, 0.1)
shots = 2000
confidence_trust_factors = [10, 5, 7, 10, 5, 7]
redundancy_factor = 3
saboteurs = set(range(devices))
rng = np.random.default_rng(12345)
# execution_function = "exec_with_singledev_cut"
# execution_function = "exec_with_normal_multidev_cut"
execution_function = "exec_with_secure_multidev_cut"


def experiment_integrity():
    for sab_to_remove in range(0, devices + 1):
        test_name = f"{prefix}_{len(saboteurs)}"
        perform_experiment(
            test_name,
            n_qubits,
            benchmark_tape,
            test_count,
            bins,
            bin_range,
            shots,
            devices,
            [1, 1, 1, 1, 1, 1],
            redundancy_factor,
            saboteurs,
            execution_function,
            0,
            True,
            rng,
            None,
            logging.INFO,
        )
        if sab_to_remove in saboteurs:
            saboteurs.remove(sab_to_remove)


start = datetime.now()
experiment_integrity()
print("Time elapsed:", datetime.now() - start)
print("THAT'S ALL, FOLKS!")
