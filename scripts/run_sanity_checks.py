#!/usr/bin/env python3
"""Automated sanity checks against saved results (no re-training needed).

Covers checks 1, 2, 7, 8 from SANITY_CHECKS.md.
Checks 3–6 require brief re-runs and are documented in SANITY_CHECKS.md.

Exit code: 0 if all pass, 1 if any fail.

Usage:
    python scripts/run_sanity_checks.py
"""

import glob
import json
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f": {detail}" if detail else ""))
        failures.append(name)


# ---------------------------------------------------------------------------
# Load all metrics.json files
# ---------------------------------------------------------------------------

all_files = sorted(glob.glob(os.path.join(_ROOT, "results", "perm_mnist", "*", "seed_*", "metrics.json")))
all_files += sorted(glob.glob(os.path.join(_ROOT, "results", "fast_track", "perm_mnist", "*", "seed_*", "metrics.json")))
all_files += sorted(glob.glob(os.path.join(_ROOT, "results", "split_cifar", "*", "seed_*", "metrics.json")))

all_data = []
for f in all_files:
    parts = f.split(os.sep)
    bench = "fast_track" if "fast_track" in f else parts[-4]
    method = parts[-3]
    seed = parts[-2]
    data = json.load(open(f))
    all_data.append({"bench": bench, "method": method, "seed": seed, "file": f, **data})

print(f"\nLoaded {len(all_data)} metrics.json files\n")

# ---------------------------------------------------------------------------
# Check 1: AF = -BWT for all files
# ---------------------------------------------------------------------------
print("Check 1 — AF = −BWT (by construction in codebase)")
violations = []
for d in all_data:
    af = d.get("avg_forgetting", None)
    bwt = d.get("bwt", None)
    if af is None or bwt is None:
        continue
    if abs(af + bwt) > 1e-9:
        violations.append(f"{d['bench']}/{d['method']}/{d['seed']}: AF={af}, BWT={bwt}")
check("AF = -BWT for all results", len(violations) == 0,
      f"{len(violations)} violations: {violations[:3]}")

# ---------------------------------------------------------------------------
# Check 2: Replay budget accounting for ConstReplay_0.1
# ---------------------------------------------------------------------------
print("\nCheck 2 — Replay budget accounting (ConstReplay_0.1, perm_mnist)")
# Formula: no replay on task 1 (buffer empty), then for tasks 2..5:
#   samples_per_task = n_batches * epochs * n_replay_per_batch
#   n_replay_per_batch = floor(0.1 * batch_size) = floor(0.1 * 256) = 25
#   n_batches per task: MNIST train = 60000 samples / 256 = ~234 batches
#   epochs_per_task = 5 (full run)
# Expected total = 4 tasks * 234 batches * 5 epochs * 25 = 117,000 (approx)
cr_files = [d for d in all_data if d["method"] == "ConstReplay_0.1" and d["bench"] == "perm_mnist"]
if cr_files:
    replay_vals = [d["total_replay_samples"] for d in cr_files]
    mean_replay = sum(replay_vals) / len(replay_vals)
    # Expected: 4 * 234 * 5 * 25 = 117,000 (rough; actual batches vary by dataset size)
    # Allow ±20% tolerance since batch count depends on dataset size
    expected_approx = 117_000
    tolerance = 0.25
    within_range = abs(mean_replay - expected_approx) / expected_approx < tolerance
    check(
        f"ConstReplay_0.1 replay budget near expected ({expected_approx:,} ± 25%)",
        within_range,
        f"mean={mean_replay:,.0f}, expected≈{expected_approx:,}"
    )
    print(f"    (actual mean={mean_replay:,.0f} across {len(cr_files)} seeds)")
else:
    print(f"  SKIP  No ConstReplay_0.1/perm_mnist results found")

# ---------------------------------------------------------------------------
# Check 7: r_t_history values in [0, 1]
# ---------------------------------------------------------------------------
print("\nCheck 7 — r_t_history values in [0.0, 1.0]")
out_of_range = []
for d in all_data:
    for i, r in enumerate(d.get("r_t_history", [])):
        if not (0.0 <= r <= 1.0):
            out_of_range.append(f"{d['bench']}/{d['method']}/{d['seed']} task {i}: r={r}")
check("All r_t values in [0, 1]", len(out_of_range) == 0,
      f"{len(out_of_range)} violations: {out_of_range[:3]}")

# ---------------------------------------------------------------------------
# Check 8: Efficiency sign consistency
# ---------------------------------------------------------------------------
print("\nCheck 8 — Efficiency sign (efficiency > 0 iff AF < FT_AF)")
# Load fast-track summary.csv for efficiency
import csv
csv_path = os.path.join(_ROOT, "results", "fast_track", "summary.csv")
eff_violations = []
if os.path.exists(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    ft_row = next((r for r in rows if r["method"] == "FT"), None)
    if ft_row:
        ft_af = float(ft_row["AF_mean"])
        for row in rows:
            method_af = float(row["AF_mean"])
            eff = float(row["efficiency"])
            mean_replay = float(row["replay_budget_mean"])
            expected_positive = method_af < ft_af and mean_replay > 0
            expected_zero = mean_replay == 0 or math.isclose(method_af, ft_af, rel_tol=0.01)
            if expected_positive and eff <= 0:
                eff_violations.append(f"{row['method']}: AF={method_af:.4f}<FT_AF={ft_af:.4f} but eff={eff}")
            if row["method"] == "FT" and eff != 0.0:
                eff_violations.append(f"FT efficiency should be 0 but is {eff}")
    check("Efficiency sign consistent with AF vs FT", len(eff_violations) == 0,
          "; ".join(eff_violations))
else:
    print("  SKIP  results/fast_track/summary.csv not found")

# ---------------------------------------------------------------------------
# Additional: AA in [0, 1] for all results
# ---------------------------------------------------------------------------
print("\nBonus — AA in [0.0, 1.0] (sanity on accuracy values)")
aa_violations = []
for d in all_data:
    aa = d.get("avg_final_acc")
    if aa is not None and not (0.0 <= aa <= 1.0):
        aa_violations.append(f"{d['bench']}/{d['method']}/{d['seed']}: AA={aa}")
check("All AA values in [0, 1]", len(aa_violations) == 0,
      f"{len(aa_violations)} violations")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
total = 4 + (1 if os.path.exists(csv_path) else 0)
passed = total - len(failures)
print(f"Result: {passed}/{total} checks passed")
if failures:
    print(f"Failed: {failures}")
    sys.exit(1)
else:
    print("All automated checks passed.")
    print("\nManual checks (see SANITY_CHECKS.md for instructions):")
    print("  Check 3 — Seed determinism")
    print("  Check 4 — Proxy non-degeneracy (needs 5-epoch run)")
    print("  Check 5 — DIFE causal ordering")
    print("  Check 6 — Buffer capacity enforcement")
