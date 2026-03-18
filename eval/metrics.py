"""Metrics computation and persistence for the evaluation suite."""

import json
import os

import numpy as np

from benchmark.fitting import compute_metrics


def compute_all_metrics(
    acc_matrix: list,
    r_t_history: list,
    total_replay_samples: int,
    wall_clock: float,
    n_classes_per_task: int,
    pre_task_acc: list,
) -> dict:
    """Extend compute_metrics() with FWT and replay budget.

    Args:
        acc_matrix: lower-triangular accuracy matrix
        r_t_history: replay fraction per task
        total_replay_samples: cumulative replay samples used
        wall_clock: wall clock seconds for the full run
        n_classes_per_task: used to compute random-chance baseline for FWT
        pre_task_acc: accuracy on task t before training on it (len = T-1)

    Returns:
        dict with AA, AF, BWT, FWT, replay_budget, wall_clock_seconds
    """
    base = compute_metrics(acc_matrix)
    T = len(acc_matrix)
    a_random = 1.0 / n_classes_per_task

    # FWT: mean(pre_task_acc[j] - a_random) for j in 1..T-1
    if pre_task_acc:
        fwt = float(np.mean([p - a_random for p in pre_task_acc]))
    else:
        fwt = 0.0

    return {
        **base,
        "fwt": fwt,
        "total_replay_samples": total_replay_samples,
        "wall_clock_seconds": wall_clock,
    }


def save_metrics(metrics: dict, path: str) -> None:
    """Persist metrics dict as JSON, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def write_summary_csv(bench_name: str, all_results: dict, output_dir: str) -> str:
    """Write per-benchmark summary CSV aggregated across seeds.

    Args:
        bench_name: benchmark identifier
        all_results: {method: {seed: metrics_dict}}
        output_dir: root output directory

    Returns:
        Path to written CSV file.
    """
    import csv

    metric_keys = [
        ("avg_final_acc", "AA"),
        ("avg_forgetting", "AF"),
        ("bwt", "BWT"),
        ("fwt", "FWT"),
        ("total_replay_samples", "replay_budget"),
    ]
    fieldnames = ["method"]
    for _, col in metric_keys:
        fieldnames += [f"{col}_mean", f"{col}_std"]

    rows = []
    for method, seed_map in all_results.items():
        seed_metrics = list(seed_map.values())
        row = {"method": method}
        for key, col in metric_keys:
            vals = [float(m.get(key, 0)) for m in seed_metrics]
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals))
        rows.append(row)

    path = os.path.join(output_dir, bench_name, "summary.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path
