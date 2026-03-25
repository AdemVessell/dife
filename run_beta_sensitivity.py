#!/usr/bin/env python3
"""Beta-min sensitivity sweep.

Tests DIFE_only, MV_only, and DIFE_MV at additional beta_min values
(0.02 and 0.20) to characterize the transition from cap-saturated
to adaptive behaviour. Uses the same setup as canonical beta-bound reruns
(r_max=0.30, 5 seeds, 3 epochs/task, split_cifar).

Results go to results/canonical_beta{label}/ (same convention as
run_beta_bound_rerun.py). FT and ConstReplay are also run so each
directory is self-contained.
"""
import subprocess
import sys
import os
import time

BETA_MIN_VALUES = [0.02, 0.20]
_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    for beta_min in BETA_MIN_VALUES:
        label = f"{beta_min:.2f}".replace(".", "")  # e.g. "002", "020"
        out_dir = os.path.join(_HERE, "results", f"canonical_beta{label}",
                               "split_cifar_rmax_0.30")
        # Check if already complete
        done = sum(
            1 for m in ["FT", "ConstReplay_0.1", "ConstReplay_0.3",
                        "DIFE_only", "MV_only", "DIFE_MV"]
            for s in range(5)
            if os.path.exists(os.path.join(out_dir, m, f"seed_{s}", "metrics.json"))
        )
        if done >= 30:
            print(f"[skip] beta_min={beta_min} — all 30 jobs already done")
            continue

        print(f"\n{'='*60}")
        print(f"Running beta_min={beta_min} ({done}/30 jobs already done)")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "run_beta_bound_rerun.py",
            "--beta-min", str(beta_min),
        ]
        t0 = time.time()
        result = subprocess.run(cmd, cwd=_HERE)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"ERROR: beta_min={beta_min} exited with code {result.returncode}")
        else:
            print(f"Done beta_min={beta_min} in {elapsed:.0f}s")

    print("\nBeta sensitivity sweep complete.")
    print("Results in results/canonical_beta002/ and results/canonical_beta020/")


if __name__ == "__main__":
    main()
