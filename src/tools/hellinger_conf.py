from glob import glob
import os
import sys
from typing import List

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.hellinger import load_distribution, create_histogram, hellinger_distance

prefix = os.path.join(os.path.dirname(__file__), "..")
bins = 120
bin_range = (-0.02, 0.02)


def compare(d1: str, d2: str, label: str):
    f1 = glob(os.path.join(prefix, d1, "*_results.json"))[0]
    f2 = glob(os.path.join(prefix, d2, "*_results.json"))[0]
    r1 = load_distribution(f1)
    r2 = load_distribution(f2)
    h1 = create_histogram(r1, bins, bin_range)
    h2 = create_histogram(r2, bins, bin_range)
    hellinger = hellinger_distance(h1, h2)
    print(f"{label} = {hellinger:.3f}")
    return hellinger


def plot(values: List[int], labels: List[str], title: str, filename: str):
    min_value = min(values)
    max_value = max(values)
    plt.bar(labels, values)
    plt.axhline(
        y=min_value, color="red", linestyle="--", label=f"Min Value ({min_value})"
    )
    plt.axhline(
        y=max_value, color="red", linestyle="--", label=f"Max Value ({max_value})"
    )
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.title(title)
    plt.xlabel("Circuit")
    plt.ylabel("Hellinger Distance")
    plt.savefig(os.path.join(prefix, "tmp", filename))
    plt.close()


def plot_recap(
    alt_values: List[float],
    ghz_values: List[float],
    dj_values: List[float],
    title: str,
    filename: str,
):
    x_points = [2, 5, 10]
    plt.plot(x_points, alt_values, marker="o", label="Alternative Benchmark")
    plt.plot(x_points, ghz_values, marker="s", label="GHZ")
    plt.plot(x_points, dj_values, marker="^", label="Deutsch-Jozsa")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.title(title)
    plt.xlabel("Fake Circuits Multiplier")
    plt.ylabel("Hellinger Distance From Benchmark")
    plt.legend()
    plt.savefig(os.path.join(prefix, "tmp", filename))
    plt.close()


if __name__ == "__main__":
    alt_raw = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_raw",
        "experiments/test_fake_circuits_2025-01-25/test_alt_raw",
        "[BENCHMARK VS ALT] RAW",
    )
    alt_rnd_2x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_random_2x",
        "experiments/test_fake_circuits_2025-01-25/test_alt_random_2x",
        "[BENCHMARK VS ALT] 2X RANDOM ",
    )
    alt_rnd_5x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_random_5x",
        "experiments/test_fake_circuits_2025-01-25/test_alt_random_5x",
        "[BENCHMARK VS ALT] 5X RANDOM ",
    )
    alt_rnd_10x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_random_10x",
        "experiments/test_fake_circuits_2025-01-25/test_alt_random_10x",
        "[BENCHMARK VS ALT] 10X RANDOM",
    )
    alt_cal_2x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_calibrated_2x",
        "experiments/test_fake_circuits_2025-01-25/test_alt_calibrated_2x",
        "[BENCHMARK VS ALT] 2X CALIBRATED ",
    )
    alt_cal_5x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_calibrated_5x",
        "experiments/test_fake_circuits_2025-01-25/test_alt_calibrated_5x",
        "[BENCHMARK VS ALT] 5X CALIBRATED ",
    )
    alt_cal_10x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_calibrated_10x",
        "experiments/test_fake_circuits_2025-01-25/test_alt_calibrated_10x",
        "[BENCHMARK VS ALT] 10X CALIBRATED",
    )
    print(30 * "-")
    ghz_raw = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_raw",
        "experiments/test_fake_circuits_2025-01-25/test_ghz_raw",
        "[BENCHMARK VS GHZ] RAW",
    )
    ghz_rnd_2x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_random_2x",
        "experiments/test_fake_circuits_2025-01-25/test_ghz_random_2x",
        "[BENCHMARK VS GHZ] 2X RANDOM ",
    )
    ghz_rnd_5x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_random_5x",
        "experiments/test_fake_circuits_2025-01-25/test_ghz_random_5x",
        "[BENCHMARK VS GHZ] 5X RANDOM ",
    )
    ghz_rnd_10x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_random_10x",
        "experiments/test_fake_circuits_2025-01-25/test_ghz_random_10x",
        "[BENCHMARK VS GHZ] 10X RANDOM",
    )
    ghz_cal_2x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_calibrated_2x",
        "experiments/test_fake_circuits_2025-01-25/test_ghz_calibrated_2x",
        "[BENCHMARK VS GHZ] 2X CALIBRATED ",
    )
    ghz_cal_5x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_calibrated_5x",
        "experiments/test_fake_circuits_2025-01-25/test_ghz_calibrated_5x",
        "[BENCHMARK VS GHZ] 5X CALIBRATED ",
    )
    ghz_cal_10x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_calibrated_10x",
        "experiments/test_fake_circuits_2025-01-25/test_ghz_calibrated_10x",
        "[BENCHMARK VS GHZ] 10X CALIBRATED",
    )
    print(30 * "-")
    dj_raw = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_raw",
        "experiments/test_fake_circuits_2025-01-25/test_dj_raw",
        "[BENCHMARK VS DJ] RAW",
    )
    dj_rnd_2x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_random_2x",
        "experiments/test_fake_circuits_2025-01-25/test_dj_random_2x",
        "[BENCHMARK VS DJ] 2X RANDOM ",
    )
    dj_rnd_5x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_random_5x",
        "experiments/test_fake_circuits_2025-01-25/test_dj_random_5x",
        "[BENCHMARK VS DJ] 5X RANDOM ",
    )
    dj_rnd_10x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_random_10x",
        "experiments/test_fake_circuits_2025-01-25/test_dj_random_10x",
        "[BENCHMARK VS DJ] 10X RANDOM",
    )
    dj_cal_2x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_calibrated_2x",
        "experiments/test_fake_circuits_2025-01-25/test_dj_calibrated_2x",
        "[BENCHMARK VS DJ] 2X CALIBRATED ",
    )
    dj_cal_5x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_calibrated_5x",
        "experiments/test_fake_circuits_2025-01-25/test_dj_calibrated_5x",
        "[BENCHMARK VS DJ] 5X CALIBRATED ",
    )
    dj_cal_10x = compare(
        "experiments/test_fake_circuits_2025-01-25/test_benchmark_calibrated_10x",
        "experiments/test_fake_circuits_2025-01-25/test_dj_calibrated_10x",
        "[BENCHMARK VS DJ] 10X CALIBRATED",
    )
    plot(
        [alt_raw, ghz_raw, dj_raw],
        ["Alternative Benchmark", "GHZ", "Deutsch-Jozsa"],
        "Hellinger Distance from Benchmark (Raw Circuit Cutting)",
        "conf_raw_chart.png",
    )
    plot(
        [alt_rnd_2x, ghz_rnd_2x, dj_rnd_2x],
        ["Alternative Benchmark", "GHZ", "Deutsch-Jozsa"],
        "Hellinger Distance from Benchmark (2X random fake circuits)",
        "conf_rnd_2x_chart.png",
    )
    plot(
        [alt_cal_2x, ghz_cal_2x, dj_cal_2x],
        ["Alternative Benchmark", "GHZ", "Deutsch-Jozsa"],
        "Hellinger Distance from Benchmark (2X calibrated fake circuits)",
        "conf_cal_2x_chart.png",
    )
    plot(
        [alt_rnd_5x, ghz_rnd_5x, dj_rnd_5x],
        ["Alternative Benchmark", "GHZ", "Deutsch-Jozsa"],
        "Hellinger Distance from Benchmark (5X random fake circuits)",
        "conf_rnd_5x_chart.png",
    )
    plot(
        [alt_cal_5x, ghz_cal_5x, dj_cal_5x],
        ["Alternative Benchmark", "GHZ", "Deutsch-Jozsa"],
        "Hellinger Distance from Benchmark (5X calibrated fake circuits)",
        "conf_cal_5x_chart.png",
    )
    plot(
        [alt_rnd_10x, ghz_rnd_10x, dj_rnd_10x],
        ["Alternative Benchmark", "GHZ", "Deutsch-Jozsa"],
        "Hellinger Distance from Benchmark (10X random fake circuits)",
        "conf_rnd_10x_chart.png",
    )
    plot(
        [alt_cal_10x, ghz_cal_10x, dj_cal_10x],
        ["Alternative Benchmark", "GHZ", "Deutsch-Jozsa"],
        "Hellinger Distance from Benchmark (10X calibrated fake circuits)",
        "conf_cal_10x_chart.png",
    )
    plot_recap(
        [alt_rnd_2x, alt_rnd_5x, alt_rnd_10x],
        [ghz_rnd_2x, ghz_rnd_5x, ghz_rnd_10x],
        [dj_rnd_2x, dj_rnd_5x, dj_rnd_10x],
        "Random Fake Circuits",
        "conf_rnd_recap.png",
    )
    plot_recap(
        [alt_cal_2x, alt_cal_5x, alt_cal_10x],
        [ghz_cal_2x, ghz_cal_5x, ghz_cal_10x],
        [dj_cal_2x, dj_cal_5x, dj_cal_10x],
        "Calibrated Fake Circuits",
        "conf_cal_recap.png",
    )
