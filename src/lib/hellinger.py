from glob import glob
import json
from typing import List, Tuple

import numpy as np


def load_distribution(dist_file_pattern: str) -> np.ndarray:
    dist_files = glob(dist_file_pattern)
    if len(dist_files) > 1:
        raise ValueError(f"{dist_file_pattern} corresponds to more than 1 file")
    elif len(dist_files) == 0:
        raise ValueError(f"{dist_file_pattern} doesn't correspond to any file")
    with open(dist_files[0]) as f:
        dist = json.load(f)
    return np.array(dist)


def create_histogram(
    results: List[int],
    bins: int,
    bin_range: Tuple[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, bin_edges = np.histogram(
        results, bins=bins, range=bin_range, density=True
    )  # Normalized histogram
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    numerical_distribution = list(zip(bin_centers, counts))
    return numerical_distribution


def hellinger_distance(p_hist: np.ndarray, q_hist: np.ndarray) -> float:
    p_counts = []
    q_counts = []
    for p_item, q_item in zip(p_hist, q_hist):
        if p_item[0] != q_item[0]:
            raise ValueError(
                "Hellinger Distance: the p and q histogram centers are different"
            )
        p_counts.append(p_item[1])
        q_counts.append(q_item[1])
    p = np.array(p_counts, dtype=float)
    q = np.array(q_counts, dtype=float)
    p_prob = p / np.sum(p)
    q_prob = q / np.sum(q)
    return np.sqrt(0.5 * np.sum((np.sqrt(p_prob) - np.sqrt(q_prob)) ** 2))
