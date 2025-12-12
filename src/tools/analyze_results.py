from glob import glob
import json
import os
from statistics import mean, stdev, variance

import matplotlib.pyplot as plt
import numpy as np


def load_distribution(dist_file_pattern):
    dist_files = glob(dist_file_pattern)
    if len(dist_files) > 1:
        raise ValueError(f"{dist_file_pattern} corresponds to more than 1 file")
    elif len(dist_files) == 0:
        raise ValueError(f"{dist_file_pattern} doesn't correspond to any file")
    with open(dist_files[0]) as f:
        return json.load(f)


bins = 120
bin_range = (-0.02, 0.02)  # Benchmark circuit
results_file = os.path.join(
    os.path.dirname(__file__),
    "..",
    "tmp",
    "test_random_2025-01-30_12-05-48",
    "test_random_results.json",
)
print(f"Loading {results_file}...")
results = load_distribution(results_file)

# Basic stats
meanres = mean(results)
var = variance(results)
stddev = stdev(results)
minres = min(results)
maxres = max(results)
max_distance = max(abs(meanres - minres), abs(meanres - maxres))
max_distance_perc = max_distance / abs(minres - maxres)
print(f"Mean             : {meanres}")
print(f"Variance         : {var}")
print(f"Standard Dev.    : {stddev}")
print(f"Min              : {minres}")
print(f"Max              : {maxres}")
print(f"Max distance     : {max_distance}")
print(f"Max distance perc: {max_distance_perc * 100:.2f}%")
print(f"4*stdev          : {abs(4 * stddev)}")
print(f"5*stdev          : {abs(5 * stddev)}")
print(f"Mean + 4*stdev   : {meanres + 4 * stddev}")
print(f"Mean + 5*stdev   : {meanres + 5 * stddev}")
print(f"Mean - 4*stdev   : {meanres - 4 * stddev}")
print(f"Mean - 5*stdev   : {meanres - 5 * stddev}")

# Frequency histogram
counts, bin_edges = np.histogram(
    results, bins=bins, density=True, range=bin_range
)  # Normalized histogram
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Compute bin centers
plt.figure(1)
plt.bar(
    bin_centers,
    counts,
    width=(bin_edges[1] - bin_edges[0]),
    color="skyblue",
    edgecolor="black",
    alpha=0.7,
)
plt.title(f"Histogram of Expectation Values")
plt.xlabel("Expectation Value")
plt.ylabel("Probability Density")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Results boxplot
plt.figure(2)
plt.title("Expectation Values")
plt.ylabel("Expectation Value")
plt.boxplot(results)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show plots
plt.show()
plt.close()
