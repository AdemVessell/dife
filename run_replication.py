#!/usr/bin/env python3
"""Task 1 — Replication Study.

Runs 5 seeds × 5 methods on split-CIFAR-10 at r_max=0.30, then generates
RESULTS_REPLICATION.md with aggregate mean±std tables and key Q&A.
"""
import subprocess
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

BENCH = "split_cifar"
SEEDS = ["0", "1", "2", "3", "4"]
EPOCHS_PER_TASK = "3"
METHODS = ["FT", "ConstReplay_0.1", "ConstReplay_0.3", "DIFE_only", "DIFE_MV"]
R_MAX = "0.30"
OUTPUT_ROOT = "results/replication_study"


def run_experiments():
    cmd = [
        sys.executable, "run_fast_track.py",
        "--bench", BENCH,
        "--seeds", *SEEDS,
        "--epochs-per-task", EPOCHS_PER_TASK,
        "--methods", *METHODS,
        "--r-max", R_MAX,
        "--output-root", OUTPUT_ROOT,
    ]
    print("Running replication study...")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=_HERE)
    if result.returncode != 0:
        print(f"ERROR: run_fast_track.py exited with code {result.returncode}")
        sys.exit(result.returncode)


def generate_report():
    cmd = [sys.executable, "generate_replication_report.py"]
    print("\nGenerating replication report...")
    result = subprocess.run(cmd, cwd=_HERE)
    if result.returncode != 0:
        print(f"ERROR: generate_replication_report.py exited with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    run_experiments()
    generate_report()
    print("\nTask 1 complete. See RESULTS_REPLICATION.md")
