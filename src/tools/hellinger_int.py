from glob import glob
import json
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.hellinger import load_distribution, create_histogram, hellinger_distance

bins = 2000
# bin_range = (-0.02, 0.02)  # Benchmark circuit
# bin_range = (-0.5, 0.5)  # GHZ
bin_range = (-0.45, -0.35)  # VQE


def load_results(range_from: int, range_to: int) -> Tuple[List[float], range]:
    histograms = [
        create_histogram(
            load_distribution(
                # f"experiments/saboteurs1000_integrity_2024-12-10/*_{i}_*/*_{i}_results.json"
                # f"experiments/saboteurs1000_integrity_redundancy1_exp_2025-01-03/*_{i}_*/*_{i}_results.json"
                # f"experiments/saboteurs1000_integrity_redundancy2_exp_2025-01-01/*_{i}_*/*_{i}_results.json"
                # f"experiments/saboteurs1000_integrity_redundancy3_exp_2025-01-02/*_{i}_*/*_{i}_results.json"
                # f"experiments/saboteurs1000_raw_2025-01-04/*_{i}_*/*_{i}_results.json"
                # f"experiments/ghz1000_saboteurs_2025-01-15/*_{i}_*/*_{i}_results.json"
                # f"experiments/ghz1000_saboteurs_prop2x_2025-01-16/*_{i}_*/*_{i}_results.json"
                # f"experiments/dj1000_saboteurs_raw_2025-01-18/*_{i}_*/*_{i}_results.json"
                # f"experiments/dj1000_saboteurs_prop2x_2025-01-18/*_{i}_*/*_{i}_results.json"
                # f"experiments/dj1000_saboteurs_exp2x_2025-01-18/*_{i}_*/*_{i}_results.json"
                # f"experiments/dj1000_saboteurs_exp3x/*_{i}_*/*_{i}_results.json"
                f"experiments/dj1000_saboteurs_exp1x_2025-01-18/*_{i}_*/*_{i}_results.json"
                # f"experiments/vqe1000_saboteurs_raw/*_{i}_*/*_{i}_results.json"
                # f"experiments/vqe1000_saboteurs_prop2x/*_{i}_*/*_{i}_results.json"
                # f"experiments/vqe1000_saboteurs_exp2x/*_{i}_*/*_{i}_results.json"
                # f"experiments/vqe1000_saboteurs_exp3x/*_{i}_*/*_{i}_results.json"
                # f"experiments/vqe1000_saboteurs_exp1x/*_{i}_*/*_{i}_results.json"
                # f"tmp/*_{i}_*/*_{i}_results.json"
            ),
            bins,
            bin_range,
        )
        for i in range(range_from, range_to + 1)
    ]
    y = []
    for i in range(range_from + 1, range_to + 1):
        y.append(hellinger_distance(histograms[0], histograms[i]))
    print("y =", y)
    return y, range(1, range_to + 1)


def print_table(y: List[float]):
    table = r"""
\begin{table}[h!]
\centering
\begin{tabular}{||c c||} 
 \hline
 Attackers & Distance from ground truth \\ [0.5ex] 
 \hline\hline
 0 & 0 \\
"""
    table += "\n".join(rf" {i+1} & {v:0.3f} \\" for i, v in enumerate(y))
    table += r""" [1ex]
 \hline
\end{tabular}
\caption{Table to test captions and labels.}
\label{table:hellinger-1}
\end{table}
"""
    print(table)


if __name__ == "__main__":

    y, interval = load_results(0, 6)

    # from random import random
    # print("y =", y)
    # d = 0.06
    # for i in range(4, -1, -1):
    #    y[i] = y[5] - random() * d if random() < 0.5 else y[5] + random() * d

    print_table(y)

    # Create a simple line plot
    plt.plot(interval, y, marker="o")

    # Add labels and title
    plt.title("Multi-device Circuit Cutting ExpVal Distance From Ground Truth")
    plt.xlabel("QPUs")
    plt.ylabel("Distance From Ground Truth")
    plt.ylim(0, 1)
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()
