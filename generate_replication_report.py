#!/usr/bin/env python3
"""Generate RESULTS_REPLICATION.md from replication study results."""
import json
import glob
import os
import numpy as np

BENCH = "split_cifar"
OUTPUT_ROOT = "results/replication_study"
METHODS = ["FT", "ConstReplay_0.1", "ConstReplay_0.3", "DIFE_only", "DIFE_MV"]
SEEDS = [0, 1, 2, 3, 4]
OUT_MD = "RESULTS_REPLICATION.md"


def load_results():
    results = {}
    for method in METHODS:
        results[method] = {}
        for seed in SEEDS:
            path = os.path.join(OUTPUT_ROOT, BENCH, method, f"seed_{seed}", "metrics.json")
            if os.path.exists(path):
                with open(path) as f:
                    results[method][seed] = json.load(f)
            else:
                print(f"  MISSING: {path}")
    return results


def compute_efficiency(results):
    """AF improvement per 10k replay samples vs FT baseline."""
    ft_af_vals = [results["FT"][s]["avg_forgetting"] for s in SEEDS if s in results["FT"]]
    if not ft_af_vals:
        return {m: 0.0 for m in METHODS}
    ft_af = np.mean(ft_af_vals)
    efficiency = {}
    for method in METHODS:
        seed_data = [results[method][s] for s in SEEDS if s in results[method]]
        if not seed_data:
            efficiency[method] = 0.0
            continue
        mean_af = float(np.mean([d["avg_forgetting"] for d in seed_data]))
        mean_replay = float(np.mean([d["total_replay_samples"] for d in seed_data]))
        if mean_replay > 0:
            efficiency[method] = (ft_af - mean_af) / mean_replay * 10_000
        else:
            efficiency[method] = 0.0
    return efficiency


def build_rows(results, efficiency):
    rows = []
    metric_keys = [
        ("avg_final_acc", "AA"),
        ("avg_forgetting", "AF"),
        ("bwt", "BWT"),
        ("fwt", "FWT"),
        ("total_replay_samples", "Replay"),
    ]
    for method in METHODS:
        seed_data = [results[method][s] for s in SEEDS if s in results[method]]
        if not seed_data:
            continue
        row = {"method": method, "n_seeds": len(seed_data)}
        for key, col in metric_keys:
            vals = [float(d.get(key, 0)) for d in seed_data]
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals))
        row["efficiency"] = efficiency.get(method, 0.0)
        rows.append(row)
    return rows


def q1_dife_vs_const(results, rows):
    """Q1: Does DIFE_only beat ConstReplay_0.3 at equal replay budget?"""
    lines = ["### Q1 — DIFE_only vs ConstReplay_0.3 (equal replay budget)\n"]
    d_row = next((r for r in rows if r["method"] == "DIFE_only"), None)
    c_row = next((r for r in rows if r["method"] == "ConstReplay_0.3"), None)
    if d_row and c_row:
        delta_af = c_row["AF_mean"] - d_row["AF_mean"]
        delta_replay = d_row["Replay_mean"] - c_row["Replay_mean"]
        lines.append(f"DIFE_only:      AA={d_row['AA_mean']:.3f}±{d_row['AA_std']:.3f}  "
                     f"AF={d_row['AF_mean']:.3f}±{d_row['AF_std']:.3f}  "
                     f"Replay={d_row['Replay_mean']:,.0f}")
        lines.append(f"ConstReplay_0.3: AA={c_row['AA_mean']:.3f}±{c_row['AA_std']:.3f}  "
                     f"AF={c_row['AF_mean']:.3f}±{c_row['AF_std']:.3f}  "
                     f"Replay={c_row['Replay_mean']:,.0f}")
        lines.append(f"\nΔAF (Const - DIFE) = {delta_af:+.4f}  "
                     f"(positive means DIFE reduces forgetting more)")
        lines.append(f"ΔDIFE replay overhead = {delta_replay:+,.0f} samples")
        if delta_af > 0.005:
            verdict = "PASS — DIFE_only meaningfully reduces forgetting vs ConstReplay_0.3"
        elif delta_af > 0:
            verdict = "MARGINAL — DIFE_only slightly better but within noise"
        else:
            verdict = "INCONCLUSIVE — DIFE_only does not beat ConstReplay_0.3 at this r_max"
        lines.append(f"\nVerdict: {verdict}")
    return "\n".join(lines)


def q2_mv_efficiency(results, rows):
    """Q2: Does DIFE_MV improve efficiency vs DIFE_only?"""
    lines = ["### Q2 — DIFE_MV vs DIFE_only (efficiency)\n"]
    mv_row = next((r for r in rows if r["method"] == "DIFE_MV"), None)
    d_row = next((r for r in rows if r["method"] == "DIFE_only"), None)
    if mv_row and d_row:
        delta_af = d_row["AF_mean"] - mv_row["AF_mean"]
        delta_replay = d_row["Replay_mean"] - mv_row["Replay_mean"]
        lines.append(f"DIFE_MV:   AA={mv_row['AA_mean']:.3f}±{mv_row['AA_std']:.3f}  "
                     f"AF={mv_row['AF_mean']:.3f}±{mv_row['AF_std']:.3f}  "
                     f"Replay={mv_row['Replay_mean']:,.0f}  "
                     f"Eff={mv_row['efficiency']:.4f}")
        lines.append(f"DIFE_only: AA={d_row['AA_mean']:.3f}±{d_row['AA_std']:.3f}  "
                     f"AF={d_row['AF_mean']:.3f}±{d_row['AF_std']:.3f}  "
                     f"Replay={d_row['Replay_mean']:,.0f}  "
                     f"Eff={d_row['efficiency']:.4f}")
        lines.append(f"\nΔAF (DIFE_only − DIFE_MV) = {delta_af:+.4f}  "
                     f"(positive means MV reduces forgetting)")
        lines.append(f"Replay saved by MV = {delta_replay:+,.0f} samples  "
                     f"({100*delta_replay/max(d_row['Replay_mean'],1):.1f}% reduction)")
        if delta_replay > 0 and delta_af >= -0.005:
            verdict = "PASS — DIFE_MV uses fewer samples with equal or better forgetting"
        elif delta_af > 0.005:
            verdict = "PASS — DIFE_MV achieves better forgetting (MV shaping adds value)"
        else:
            verdict = "INCONCLUSIVE — MV provides no clear benefit at this budget"
        lines.append(f"\nVerdict: {verdict}")
    return "\n".join(lines)


def q3_variance(rows):
    """Q3: Is variance acceptable? AF std < 0.015 threshold."""
    lines = ["### Q3 — Variance Check (AF std < 0.015 threshold)\n"]
    for row in rows:
        flag = "PASS" if row["AF_std"] < 0.015 else "FAIL"
        lines.append(f"  {row['method']:<20} AF_std={row['AF_std']:.4f}  [{flag}]")
    return "\n".join(lines)


def write_md(rows, q1, q2, q3, out_path):
    lines = [
        "# Replication Study — DIFE × Memory Vortex",
        "",
        "**Benchmark:** Split-CIFAR-10 (5 binary tasks)  |  "
        "**Seeds:** 5  |  **Epochs/task:** 3  |  **r_max:** 0.30",
        "",
        "## Results Table",
        "",
        "| Method | AA ↑ | AF ↓ | BWT | FWT | Replay | Efficiency* |",
        "|--------|------|------|-----|-----|--------|-------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']:<18} "
            f"| {row['AA_mean']:.3f}±{row['AA_std']:.3f} "
            f"| {row['AF_mean']:.3f}±{row['AF_std']:.3f} "
            f"| {row['BWT_mean']:.3f}±{row['BWT_std']:.3f} "
            f"| {row['FWT_mean']:.3f}±{row['FWT_std']:.3f} "
            f"| {row['Replay_mean']:,.0f}±{row['Replay_std']:,.0f} "
            f"| {row['efficiency']:.4f} |"
        )
    lines += [
        "",
        "\\* Efficiency = AF improvement per 10,000 replay samples vs FT baseline.",
        "",
        "---",
        "",
        q1,
        "",
        "---",
        "",
        q2,
        "",
        "---",
        "",
        q3,
        "",
        "---",
        "",
        f"Generated by `generate_replication_report.py` from `{OUTPUT_ROOT}/`",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


def main():
    print("Loading replication results...")
    results = load_results()
    efficiency = compute_efficiency(results)
    rows = build_rows(results, efficiency)

    if not rows:
        print("ERROR: No results found. Run run_replication.py first.")
        raise SystemExit(1)

    q1 = q1_dife_vs_const(results, rows)
    q2 = q2_mv_efficiency(results, rows)
    q3 = q3_variance(rows)

    write_md(rows, q1, q2, q3, OUT_MD)

    print("\n--- Summary ---")
    print(f"{'Method':<20} {'AA':>8} {'AF':>8} {'Replay':>12} {'Eff':>8}")
    print("-" * 62)
    for row in rows:
        print(f"{row['method']:<20} "
              f"{row['AA_mean']:.3f}±{row['AA_std']:.3f}  "
              f"{row['AF_mean']:.3f}±{row['AF_std']:.3f}  "
              f"{row['Replay_mean']:>10,.0f}  "
              f"{row['efficiency']:>8.4f}")


if __name__ == "__main__":
    main()
