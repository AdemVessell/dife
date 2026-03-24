#!/usr/bin/env python3
"""Parallel dispatcher for beta-bound reruns.

Runs all remaining (beta_min, method, seed) combos in parallel using N workers.
Each worker calls run_beta_bound_rerun.py --beta-min X --method Y --seed Z.
The underlying script's skip logic handles already-completed jobs safely.

Usage:
    python run_parallel_beta.py --workers 4
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.join(_HERE, "run_beta_bound_rerun.py")

BETA_MINS = [0.05, 0.10]
METHODS   = ["FT", "ConstReplay_0.1", "ConstReplay_0.3",
             "DIFE_only", "MV_only", "DIFE_MV"]
SEEDS     = [0, 1, 2, 3, 4]


def _metrics_path(beta_min: float, method: str, seed: int) -> str:
    label = f"{beta_min:.2f}".replace(".", "")
    return os.path.join(
        _HERE, "results", f"canonical_beta{label}",
        "split_cifar_rmax_0.30", method, f"seed_{seed}", "metrics.json"
    )


def _pending_jobs():
    jobs = []
    for beta_min in BETA_MINS:
        for method in METHODS:
            for seed in SEEDS:
                if not os.path.exists(_metrics_path(beta_min, method, seed)):
                    jobs.append((beta_min, method, seed))
    return jobs


def _run_one(args):
    beta_min, method, seed = args
    label = f"beta{beta_min:.2f}".replace(".", "")
    log_path = os.path.join(_HERE, "logs", f"{label}_{method}_seed{seed}.log")
    cmd = [
        sys.executable, _RUNNER,
        "--beta-min", str(beta_min),
        "--method",   method,
        "--seed",     str(seed),
        "--device",   "cpu",
    ]
    t0 = time.time()
    with open(log_path, "w") as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"ERR({result.returncode})"
    return (beta_min, method, seed, elapsed, status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs = _pending_jobs()
    total = len(jobs)
    print(f"Pending jobs: {total}  Workers: {args.workers}")
    for j in jobs:
        print(f"  beta={j[0]}  method={j[1]}  seed={j[2]}")
    print()

    if total == 0:
        print("Nothing to do — all jobs complete.")
        return

    done = 0
    errors = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, j): j for j in jobs}
        for fut in as_completed(futures):
            beta_min, method, seed, elapsed, status = fut.result()
            done += 1
            elapsed_total = time.time() - t_start
            print(
                f"[{done}/{total}] beta={beta_min} method={method} seed={seed} "
                f"→ {status}  ({elapsed:.0f}s)  total={elapsed_total:.0f}s"
            )
            if status != "OK":
                errors.append((beta_min, method, seed, status))

    print(f"\nFinished {total} jobs in {time.time() - t_start:.1f}s")
    if errors:
        print(f"{len(errors)} errors:")
        for e in errors:
            print(f"  beta={e[0]} method={e[1]} seed={e[2]} → {e[3]}")
    else:
        print("All jobs completed successfully.")


if __name__ == "__main__":
    main()
