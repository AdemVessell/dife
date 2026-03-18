#!/usr/bin/env python3
"""Enumerate missing (bench, seed, method) jobs and run each as a fresh subprocess.

Usage:
    python run_missing.py [--bench perm_mnist split_cifar] [--device cpu]

Each job runs as an isolated `python run_one_job.py` invocation so a crash or
OOM cannot lose already-completed results.  Grid-search params are cached to
disk after the first run and reused by every subsequent subprocess.
"""

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

ALL_METHODS = [
    "FT", "EWC", "SI",
    "ConstReplay_0.1", "ConstReplay_0.3", "RandReplay",
    "DIFE_only", "MV_only", "DIFE_MV",
]

BENCH_SEEDS = {
    "perm_mnist": 5,
    "split_cifar": 3,
}


def missing_jobs(bench: str, n_seeds: int, results_root: str = "results") -> list:
    jobs = []
    for seed in range(n_seeds):
        for method in ALL_METHODS:
            path = os.path.join(results_root, bench, method, f"seed_{seed}", "metrics.json")
            if not os.path.exists(path):
                jobs.append((bench, seed, method))
    return jobs


def run_job(bench: str, seed: int, method: str, device: str, log_dir: str) -> bool:
    """Run one job as a subprocess. Returns True on success."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{bench}_seed{seed}_{method}.log")
    cmd = [
        sys.executable, os.path.join(_HERE, "run_one_job.py"),
        "--bench", bench,
        "--seed", str(seed),
        "--method", method,
        "--device", device,
    ]
    print(f"\n>>> {bench} seed={seed} method={method}  [log: {log_path}]")
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.decode(errors="replace")
        log_f.write(output)

    # Print last few lines to console for progress visibility
    lines = output.strip().splitlines()
    for line in lines[-6:]:
        print("  " + line)

    if result.returncode != 0:
        print(f"  [FAILED] exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench", nargs="+",
        default=["perm_mnist", "split_cifar"],
        choices=list(BENCH_SEEDS.keys()),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-dir", default="logs/jobs")
    args = parser.parse_args()

    all_missing = []
    for bench in args.bench:
        n_seeds = BENCH_SEEDS[bench]
        missing = missing_jobs(bench, n_seeds)
        all_missing.extend(missing)
        print(f"{bench}: {len(missing)} jobs missing (of {n_seeds * len(ALL_METHODS)})")

    if not all_missing:
        print("\nAll jobs already complete.")
        return

    print(f"\nTotal missing: {len(all_missing)} jobs — running sequentially as isolated processes.\n")

    failed = []
    for i, (bench, seed, method) in enumerate(all_missing, 1):
        print(f"[{i}/{len(all_missing)}]", end="")
        ok = run_job(bench, seed, method, args.device, args.log_dir)
        if not ok:
            failed.append((bench, seed, method))

    print(f"\n{'='*60}")
    print(f"Done. {len(all_missing) - len(failed)}/{len(all_missing)} jobs succeeded.")
    if failed:
        print("FAILED jobs:")
        for bench, seed, method in failed:
            print(f"  {bench} seed={seed} method={method}")
        sys.exit(1)


if __name__ == "__main__":
    main()
