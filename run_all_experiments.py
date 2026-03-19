#!/usr/bin/env python3
"""Master orchestrator: runs all four experiment phases in sequence.

Phases:
  1. Replication study (split-CIFAR, 5 seeds, r_max=0.30, 5 methods)
  2. Budget sweep (DIFE_only + DIFE_MV, r_max ∈ {0.05,0.10,0.20,0.30}, 5 seeds)
  3. MV shape ablation (DIFE_only, DIFE_flatMatched, DIFE_MV, 5 seeds, r_max=0.30)
  4. Post-fix audit (static + runtime checks, generates AUDIT_POST_FIX.md)

Usage:
    python run_all_experiments.py              # run all phases
    python run_all_experiments.py --phase 1   # run only phase 1
    python run_all_experiments.py --phase 2   # run only phase 2
    python run_all_experiments.py --phase 3   # run only phase 3
    python run_all_experiments.py --phase 4   # run only phase 4
"""

import argparse
import subprocess
import sys
import time


def run_phase(name, script, phase_num):
    print(f"\n{'#'*70}")
    print(f"# PHASE {phase_num}: {name}")
    print(f"{'#'*70}\n")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], check=False)
    elapsed = time.time() - t0
    status = "DONE" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n[Phase {phase_num}] {name}: {status}  ({elapsed:.1f}s)")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Run only a specific phase (1-4). Default: run all.")
    args = parser.parse_args()

    phases = [
        (1, "Replication Study",      "run_replication.py"),
        (2, "Budget Sweep (Repaired)", "run_sweep_repaired.py"),
        (3, "MV Shape Ablation",       "run_ablation_mv_shape.py"),
        (4, "Post-Fix Audit",          "run_audit_post_fix.py"),
    ]

    if args.phase is not None:
        phases = [(n, name, script) for n, name, script in phases if n == args.phase]

    results = {}
    t_total = time.time()
    for phase_num, name, script in phases:
        ok = run_phase(name, script, phase_num)
        results[phase_num] = (name, ok)

    total_elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"ALL PHASES COMPLETE  ({total_elapsed:.1f}s total)")
    print(f"{'='*70}")
    for phase_num, (name, ok) in sorted(results.items()):
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  Phase {phase_num}: {name:<30}  {status}")

    print("\nOutputs:")
    print("  RESULTS_REPLICATION.md           (replication study)")
    print("  SUMMARY_SWEEP_REPAIRED.md        (budget sweep)")
    print("  results/sweep_repaired/af_vs_replay.png")
    print("  ABLATION_MV_SHAPE.md             (MV shape ablation)")
    print("  AUDIT_POST_FIX.md                (post-fix audit)")

    all_ok = all(ok for _, ok in results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
