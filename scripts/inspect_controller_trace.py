#!/usr/bin/env python3
"""Inspect controller_trace.csv files from canonical runs.

Answers the core scientific question:
  Does DIFE materially influence replay online in the capped controller,
  or are the observed gains entirely from MV?

Usage:
    python scripts/inspect_controller_trace.py
    python scripts/inspect_controller_trace.py --trace-root results/canonical/split_cifar_rmax_0.30
"""

import argparse
import csv
import glob
import os
import sys
import math


def load_trace(path: str) -> list:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v in ("nan", ""):
                    parsed[k] = float("nan")
                elif k in ("task_id", "epoch_in_task", "global_epoch",
                           "replay_samples_this_epoch", "cumulative_replay_samples", "has_mv_fit"):
                    parsed[k] = int(float(v))
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def stats(vals):
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"n": 0, "min": float("nan"), "max": float("nan"),
                "mean": float("nan"), "std": float("nan")}
    n = len(vals)
    mn = min(vals)
    mx = max(vals)
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
    return {"n": n, "min": mn, "max": mx, "mean": mean, "std": std}


def fmt(s):
    if s["n"] == 0:
        return "no data"
    return f"min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} std={s['std']:.4f} (n={s['n']})"


def analyze_method(trace_root: str, method: str) -> dict:
    pattern = os.path.join(trace_root, method, "seed_*", "controller_trace.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return {"found": False}

    all_dife_envelopes = []
    all_before_cap = []
    all_after_cap = []
    all_replay_fracs = []
    tasks_where_dife_drops_below_rmax = 0
    epochs_dife_drops_below_rmax = 0
    total_epochs = 0
    total_tasks = 0
    r_max_seen = None

    for f in files:
        rows = load_trace(f)
        if not rows:
            continue

        r_max = rows[0].get("r_max", float("nan"))
        if not math.isnan(r_max):
            r_max_seen = r_max

        # Group by task
        tasks = {}
        for row in rows:
            tid = row["task_id"]
            tasks.setdefault(tid, []).append(row)

        for tid, task_rows in tasks.items():
            total_tasks += 1
            envelopes = [r["dife_envelope_value"] for r in task_rows
                         if not math.isnan(r.get("dife_envelope_value", float("nan")))]
            if envelopes and r_max_seen is not None:
                if min(envelopes) < r_max_seen:
                    tasks_where_dife_drops_below_rmax += 1

            for row in task_rows:
                total_epochs += 1
                env = row.get("dife_envelope_value", float("nan"))
                before = row.get("final_replay_fraction_before_cap", float("nan"))
                after = row.get("final_replay_fraction_after_cap", float("nan"))

                if not math.isnan(env):
                    all_dife_envelopes.append(env)
                if not math.isnan(before):
                    all_before_cap.append(before)
                if not math.isnan(after):
                    all_after_cap.append(after)
                all_replay_fracs.append(after if not math.isnan(after) else 0.0)

                if not math.isnan(env) and r_max_seen is not None and env < r_max_seen:
                    epochs_dife_drops_below_rmax += 1

    return {
        "found": True,
        "files": files,
        "r_max": r_max_seen,
        "total_tasks": total_tasks,
        "total_epochs": total_epochs,
        "tasks_where_dife_drops_below_rmax": tasks_where_dife_drops_below_rmax,
        "epochs_dife_drops_below_rmax": epochs_dife_drops_below_rmax,
        "dife_envelope_stats": stats(all_dife_envelopes),
        "before_cap_stats": stats(all_before_cap),
        "after_cap_stats": stats(all_after_cap),
    }


def compare_mv_vs_dife_mv(trace_root: str) -> dict:
    """Compare DIFE_MV and MV_only per-epoch replay fractions."""
    mv_fracs = {}    # (seed, task_id, epoch_in_task) -> after_cap
    dife_fracs = {}

    for method, store in [("MV_only", mv_fracs), ("DIFE_MV", dife_fracs)]:
        pattern = os.path.join(trace_root, method, "seed_*", "controller_trace.csv")
        for f in sorted(glob.glob(pattern)):
            rows = load_trace(f)
            seed = rows[0]["seed"] if rows else None
            for row in rows:
                key = (seed, row["task_id"], row["epoch_in_task"])
                store[key] = row.get("final_replay_fraction_after_cap", float("nan"))

    common_keys = set(mv_fracs.keys()) & set(dife_fracs.keys())
    diffs = []
    for k in sorted(common_keys):
        mv_v = mv_fracs[k]
        d_v  = dife_fracs[k]
        if not math.isnan(mv_v) and not math.isnan(d_v):
            diffs.append(abs(d_v - mv_v))

    if not diffs:
        return {"compared": False}

    n_materially_different = sum(1 for d in diffs if d > 1e-3)
    return {
        "compared": True,
        "n_comparable_epochs": len(diffs),
        "diff_stats": stats(diffs),
        "n_epochs_materially_different": n_materially_different,
        "pct_epochs_materially_different": 100.0 * n_materially_different / len(diffs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root",
                        default="results/canonical/split_cifar_rmax_0.30")
    args = parser.parse_args()

    trace_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.trace_root
    ) if not os.path.isabs(args.trace_root) else args.trace_root

    print("=" * 70)
    print(f"CONTROLLER TRACE INSPECTOR")
    print(f"Root: {trace_root}")
    print("=" * 70)

    methods_of_interest = ["DIFE_only", "MV_only", "DIFE_MV",
                           "ConstReplay_0.3", "ConstReplay_0.1", "FT"]

    analyses = {}
    for method in methods_of_interest:
        a = analyze_method(trace_root, method)
        analyses[method] = a
        if not a["found"]:
            continue

        print(f"\n--- {method} ({len(a['files'])} seed files) ---")
        print(f"  r_max: {a['r_max']}")
        print(f"  Total (task,epoch) pairs: {a['total_epochs']} across {a['total_tasks']} tasks")

        if method in ("DIFE_only", "DIFE_MV"):
            print(f"\n  DIFE ENVELOPE:")
            print(f"    {fmt(a['dife_envelope_stats'])}")
            print(f"  Tasks where envelope < r_max: {a['tasks_where_dife_drops_below_rmax']}"
                  f" / {a['total_tasks']}")
            print(f"  Epochs where envelope < r_max: {a['epochs_dife_drops_below_rmax']}"
                  f" / {a['total_epochs']}"
                  f" ({100.0*a['epochs_dife_drops_below_rmax']/max(a['total_epochs'],1):.1f}%)")

        print(f"\n  REPLAY FRACTION (before cap): {fmt(a['before_cap_stats'])}")
        print(f"  REPLAY FRACTION (after cap):  {fmt(a['after_cap_stats'])}")

    # Cross-method comparison
    print("\n" + "=" * 70)
    print("DIFE_MV vs MV_only — ARE THEY FUNCTIONALLY DIFFERENT?")
    print("=" * 70)
    cmp = compare_mv_vs_dife_mv(trace_root)
    if not cmp["compared"]:
        print("  Cannot compare — one or both methods have no trace data.")
    else:
        print(f"  Comparable epoch pairs: {cmp['n_comparable_epochs']}")
        print(f"  Absolute difference in after-cap replay fraction:")
        print(f"    {fmt(cmp['diff_stats'])}")
        print(f"  Epochs where |DIFE_MV - MV_only| > 0.001: "
              f"{cmp['n_epochs_materially_different']} / {cmp['n_comparable_epochs']} "
              f"({cmp['pct_epochs_materially_different']:.1f}%)")

    # Summary verdict
    print("\n" + "=" * 70)
    print("SUMMARY VERDICT")
    print("=" * 70)
    dife_a  = analyses.get("DIFE_only", {})
    difemv_a = analyses.get("DIFE_MV", {})
    r_max = dife_a.get("r_max")

    if dife_a.get("found") and r_max is not None:
        pct = 100.0 * dife_a["epochs_dife_drops_below_rmax"] / max(dife_a["total_epochs"], 1)
        if pct < 1.0:
            print(f"\n  [DIFE INOPERATIVE] DIFE envelope drops below r_max in only"
                  f" {pct:.1f}% of epochs.")
            print(f"  The DIFE component is NOT materially influencing replay.")
            print(f"  DIFE_only is functionally equivalent to ConstReplay_{r_max}.")
        elif pct < 20.0:
            print(f"\n  [DIFE MARGINAL] DIFE envelope drops below r_max in {pct:.1f}% of epochs.")
            print(f"  Small task-level differentiation exists but is not dominant.")
        else:
            print(f"\n  [DIFE ACTIVE] DIFE envelope drops below r_max in {pct:.1f}% of epochs.")
            print(f"  DIFE is materially differentiating replay across tasks.")

    if cmp.get("compared"):
        if cmp["pct_epochs_materially_different"] < 5.0:
            print(f"\n  [DIFE_MV ≈ MV_only] In {cmp['pct_epochs_materially_different']:.1f}%"
                  f" of epochs, DIFE_MV differs from MV_only by >0.001.")
            print(f"  DIFE_MV and MV_only are FUNCTIONALLY IDENTICAL in this run.")
            print(f"  Any performance difference is attributable to MV alone, not DIFE.")
        else:
            print(f"\n  [DIFE_MV ≠ MV_only] In {cmp['pct_epochs_materially_different']:.1f}%"
                  f" of epochs, DIFE_MV differs from MV_only by >0.001.")
            print(f"  DIFE is making a material contribution to DIFE_MV's replay schedule.")


if __name__ == "__main__":
    main()
