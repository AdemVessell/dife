#!/usr/bin/env python3
"""Bulletproof beta-bound rerun: each job is an isolated subprocess with timeout.

One job crashes? Next one runs. No shared state, no fragile loop.
The underlying run_beta_bound_rerun.py already skips completed jobs.

Usage:
    python run_bulletproof.py --beta-min 0.05
    python run_bulletproof.py --beta-min 0.10
    python run_bulletproof.py --beta-min 0.05 --beta-min-2 0.10   # both
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ["FT", "ConstReplay_0.1", "ConstReplay_0.3",
           "DIFE_only", "MV_only", "DIFE_MV"]
SEEDS = [0, 1, 2, 3, 4]
TIMEOUT = 300  # 5 minutes per job — they take ~90s normally


def output_root(beta_min):
    label = f"{beta_min:.2f}".replace(".", "")
    return os.path.join(HERE, "results", f"canonical_beta{label}",
                        "split_cifar_rmax_0.30")


def is_done(beta_min, method, seed):
    path = os.path.join(output_root(beta_min), method, f"seed_{seed}", "metrics.json")
    return os.path.exists(path)


def run_one(beta_min, method, seed):
    """Run a single job as a subprocess. Returns (success, elapsed, msg)."""
    cmd = [
        sys.executable, os.path.join(HERE, "run_beta_bound_rerun.py"),
        "--beta-min", str(beta_min),
        "--method", method,
        "--seed", str(seed),
    ]
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT, cwd=HERE
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            return True, elapsed, "OK"
        else:
            # Get last 3 lines of stderr for diagnosis
            err_lines = result.stderr.strip().split("\n")[-3:]
            return False, elapsed, f"exit={result.returncode} | {' | '.join(err_lines)}"
    except subprocess.TimeoutExpired:
        return False, TIMEOUT, "TIMEOUT"
    except Exception as e:
        return False, time.time() - t0, str(e)


def run_batch(beta_min):
    """Run all missing jobs for a given beta_min."""
    missing = []
    for m in METHODS:
        for s in SEEDS:
            if not is_done(beta_min, m, s):
                missing.append((m, s))

    total = len(missing)
    if total == 0:
        print(f"\n  beta_min={beta_min}: ALL 30 JOBS ALREADY COMPLETE")
        return 0

    print(f"\n{'='*60}")
    print(f"  BETA_MIN={beta_min}: {total} missing jobs to run")
    print(f"{'='*60}\n")

    succeeded = 0
    failed = 0
    t_batch = time.time()

    for i, (method, seed) in enumerate(missing):
        tag = f"[{i+1}/{total}] beta={beta_min} {method}/seed_{seed}"
        print(f"{tag} ... ", end="", flush=True)
        ok, elapsed, msg = run_one(beta_min, method, seed)
        if ok:
            succeeded += 1
            print(f"DONE in {elapsed:.0f}s")
        else:
            failed += 1
            print(f"FAIL ({msg}) in {elapsed:.0f}s")

    batch_time = time.time() - t_batch
    print(f"\n  Batch complete: {succeeded}/{total} succeeded, "
          f"{failed} failed, {batch_time:.0f}s total")
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta-min", type=float, default=0.05)
    parser.add_argument("--beta-min-2", type=float, default=None,
                        help="Optional second beta_min to run after first")
    args = parser.parse_args()

    total_failures = 0
    total_failures += run_batch(args.beta_min)

    if args.beta_min_2 is not None:
        total_failures += run_batch(args.beta_min_2)

    if total_failures == 0:
        print("\nALL JOBS COMPLETED SUCCESSFULLY")
    else:
        print(f"\n{total_failures} TOTAL FAILURES — check output above")

    sys.exit(1 if total_failures > 0 else 0)


if __name__ == "__main__":
    main()
