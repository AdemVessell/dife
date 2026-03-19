#!/usr/bin/env python3
"""Replication study: split-CIFAR fast-track, 5 seeds, r_max=0.30.

Methods: FT, ConstReplay_0.1, ConstReplay_0.3, DIFE_only, DIFE_MV
Seeds:   0, 1, 2, 3, 4
Settings: epochs_per_task=3, r_max=0.30, dataset=split_cifar
Output:  results/replication_study/
         RESULTS_REPLICATION.md
"""

import json
import os
import subprocess
import sys

import numpy as np

OUTPUT_ROOT = "results/replication_study"
BENCH = "split_cifar"
SEEDS = [0, 1, 2, 3, 4]
METHODS = ["FT", "ConstReplay_0.1", "ConstReplay_0.3", "DIFE_only", "DIFE_MV"]
R_MAX = "0.30"
EPOCHS = "3"


def run_training():
    """Run all training jobs via run_fast_track.py."""
    cmd = [
        sys.executable, "run_fast_track.py",
        "--bench", BENCH,
        "--seeds", *[str(s) for s in SEEDS],
        "--epochs-per-task", EPOCHS,
        "--methods", *METHODS,
        "--r-max", R_MAX,
        "--output-root", OUTPUT_ROOT,
    ]
    print("Running replication training...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def load_results():
    """Load all per-seed metrics from disk."""
    results = {m: {} for m in METHODS}
    for method in METHODS:
        for seed in SEEDS:
            path = os.path.join(OUTPUT_ROOT, BENCH, method, f"seed_{seed}", "metrics.json")
            if os.path.exists(path):
                with open(path) as f:
                    results[method][seed] = json.load(f)
            else:
                print(f"WARNING: missing {path}")
    return results


def compute_efficiency(results):
    ft_af = np.mean([results["FT"][s]["avg_forgetting"] for s in SEEDS if s in results["FT"]])
    eff = {}
    for m in METHODS:
        if m not in results:
            eff[m] = 0.0
            continue
        af_vals = [results[m][s]["avg_forgetting"] for s in SEEDS if s in results[m]]
        replay_vals = [results[m][s]["total_replay_samples"] for s in SEEDS if s in results[m]]
        mean_replay = float(np.mean(replay_vals)) if replay_vals else 0.0
        if mean_replay > 0 and af_vals:
            eff[m] = (ft_af - float(np.mean(af_vals))) / mean_replay * 10_000
        else:
            eff[m] = 0.0
    return eff


def build_summary_rows(results, eff):
    rows = []
    for m in METHODS:
        if m not in results or not results[m]:
            continue
        seeds_done = sorted(results[m].keys())
        aa = [results[m][s]["avg_final_acc"] for s in seeds_done]
        af = [results[m][s]["avg_forgetting"] for s in seeds_done]
        replay = [results[m][s]["total_replay_samples"] for s in seeds_done]
        rows.append({
            "method": m,
            "seeds_done": seeds_done,
            "AA_mean": float(np.mean(aa)),
            "AA_std": float(np.std(aa)),
            "AF_mean": float(np.mean(af)),
            "AF_std": float(np.std(af)),
            "replay_mean": float(np.mean(replay)),
            "replay_std": float(np.std(replay)),
            "efficiency": eff.get(m, 0.0),
        })
    return rows


def generate_report(results, rows, eff):
    lines = [
        "# RESULTS_REPLICATION — Split-CIFAR Replication Study",
        "",
        f"Benchmark: split_cifar | Seeds: {SEEDS} | Epochs/task: {EPOCHS} | r_max: {R_MAX}",
        "Repaired code: bounded L-BFGS-B for DIFE, MIN_OBS=6 for MV, per-epoch modulation active.",
        "",
        "## Summary Table (mean ± std over 5 seeds)",
        "",
        "| Method | AA ↑ | AF ↓ | Replay Used | Efficiency* |",
        "|--------|------|------|-------------|-------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']:<18} "
            f"| {r['AA_mean']:.3f}±{r['AA_std']:.3f} "
            f"| {r['AF_mean']:.3f}±{r['AF_std']:.3f} "
            f"| {r['replay_mean']:>10,.0f}±{r['replay_std']:>8,.0f} "
            f"| {r['efficiency']:.4f} |"
        )
    lines += [
        "",
        "\\* Efficiency = (FT_AF − Method_AF) / mean_replay × 10,000  (AF improvement per 10k samples vs FT)",
        "",
        "## Per-Seed Results",
        "",
    ]
    for m in METHODS:
        if m not in results or not results[m]:
            continue
        lines.append(f"### {m}")
        lines.append("| Seed | AA | AF | Replay |")
        lines.append("|------|----|----|--------|")
        for s in sorted(results[m]):
            d = results[m][s]
            lines.append(
                f"| {s} "
                f"| {d['avg_final_acc']:.4f} "
                f"| {d['avg_forgetting']:.4f} "
                f"| {d['total_replay_samples']:,} |"
            )
        lines.append("")

    # Answer the three questions
    lines += [
        "## Answers to Key Questions",
        "",
    ]

    # Q1: Does DIFE_only beat ConstReplay_0.3 at equal replay budget?
    dife_row = next((r for r in rows if r["method"] == "DIFE_only"), None)
    cr03_row = next((r for r in rows if r["method"] == "ConstReplay_0.3"), None)
    if dife_row and cr03_row:
        dife_af = dife_row["AF_mean"]
        cr03_af = cr03_row["AF_mean"]
        dife_replay = dife_row["replay_mean"]
        cr03_replay = cr03_row["replay_mean"]
        beat = dife_af < cr03_af
        lines += [
            "### Q1: Does DIFE_only beat ConstReplay_0.3 at equal replay budget?",
            "",
            f"- DIFE_only  AF = {dife_af:.4f} ± {dife_row['AF_std']:.4f}  |  Replay = {dife_replay:,.0f}",
            f"- ConstReplay_0.3  AF = {cr03_af:.4f} ± {cr03_row['AF_std']:.4f}  |  Replay = {cr03_replay:,.0f}",
            "",
        ]
        if dife_replay <= cr03_replay * 1.05:  # within 5% of same budget
            verdict = "YES" if beat else "NO"
            lines.append(
                f"**Verdict: {verdict}** — budgets are comparable ({dife_replay:,.0f} vs {cr03_replay:,.0f}); "
                f"DIFE_only AF {('<' if beat else '>=')} ConstReplay_0.3 AF."
            )
        else:
            verdict = "YES (lower AF)" if beat else "NO (higher AF)"
            diff_pct = (dife_replay - cr03_replay) / cr03_replay * 100
            lines.append(
                f"**Verdict: {verdict}** — DIFE_only uses {diff_pct:+.1f}% {'more' if diff_pct > 0 else 'less'} replay. "
                f"AF delta = {dife_af - cr03_af:+.4f} (negative = DIFE_only is better)."
            )
        lines.append("")

    # Q2: Does DIFE_MV improve efficiency vs DIFE_only?
    mv_row = next((r for r in rows if r["method"] == "DIFE_MV"), None)
    if dife_row and mv_row:
        eff_delta = mv_row["efficiency"] - dife_row["efficiency"]
        replay_delta = dife_row["replay_mean"] - mv_row["replay_mean"]
        af_delta = dife_row["AF_mean"] - mv_row["AF_mean"]
        lines += [
            "### Q2: Does DIFE_MV improve efficiency vs DIFE_only?",
            "",
            f"- DIFE_only  Efficiency = {dife_row['efficiency']:.4f}  |  AF = {dife_row['AF_mean']:.4f}",
            f"- DIFE_MV    Efficiency = {mv_row['efficiency']:.4f}  |  AF = {mv_row['AF_mean']:.4f}",
            f"- Delta efficiency = {eff_delta:+.4f}  |  Replay saved = {replay_delta:,.0f}  |  AF delta = {af_delta:+.4f}",
            "",
        ]
        if eff_delta > 0:
            lines.append(
                f"**Verdict: YES** — DIFE_MV has higher efficiency ({mv_row['efficiency']:.4f} vs "
                f"{dife_row['efficiency']:.4f}), using {replay_delta:,.0f} fewer replay samples."
            )
        else:
            lines.append(
                f"**Verdict: NO** — DIFE_MV efficiency ({mv_row['efficiency']:.4f}) does not exceed "
                f"DIFE_only ({dife_row['efficiency']:.4f}) at this budget."
            )
        lines.append("")

    # Q3: Is variance acceptable across seeds?
    lines += ["### Q3: Is variance acceptable across seeds?", ""]
    threshold = 0.015
    all_low = True
    for r in rows:
        if r["method"] == "FT":
            continue
        low = r["AF_std"] < threshold
        if not low:
            all_low = False
        lines.append(
            f"- {r['method']:<18}  AF std = {r['AF_std']:.4f}  "
            f"({'✓ acceptable' if low else '✗ high — investigate'})"
        )
    lines.append("")
    verdict_q3 = "YES" if all_low else "PARTIAL"
    lines.append(
        f"**Verdict: {verdict_q3}** — "
        f"{'All replay methods have AF std < ' + str(threshold) + '.' if all_low else 'Some methods exceed AF std threshold of ' + str(threshold) + '.'}"
    )
    lines.append("")

    return "\n".join(lines)


def main():
    # Step 1: train
    run_training()

    # Step 2: load results
    results = load_results()

    # Step 3: compute summary
    eff = compute_efficiency(results)
    rows = build_summary_rows(results, eff)

    # Step 4: print summary table
    print("\n" + "=" * 64)
    print("REPLICATION SUMMARY — split_cifar, 5 seeds, r_max=0.30")
    print("=" * 64)
    print(f"{'Method':<20} {'AA':>10} {'AF':>10} {'Replay':>14} {'Efficiency':>12}")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['method']:<20} "
            f"{r['AA_mean']:.3f}±{r['AA_std']:.3f}  "
            f"{r['AF_mean']:.3f}±{r['AF_std']:.3f}  "
            f"{r['replay_mean']:>10,.0f}  "
            f"{r['efficiency']:>12.4f}"
        )

    # Step 5: write RESULTS_REPLICATION.md
    report = generate_report(results, rows, eff)
    out_dir = os.path.join(OUTPUT_ROOT, BENCH)
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "RESULTS_REPLICATION.md")
    with open(md_path, "w") as f:
        f.write(report)
    print(f"\nReport written: {md_path}")

    # Also write to repo root for easy access
    root_path = "RESULTS_REPLICATION.md"
    with open(root_path, "w") as f:
        f.write(report)
    print(f"Report also written: {root_path}")


if __name__ == "__main__":
    main()
