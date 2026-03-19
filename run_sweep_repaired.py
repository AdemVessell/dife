#!/usr/bin/env python3
"""Repaired budget sweep for split-CIFAR.

Methods: DIFE_only, DIFE_MV
r_max:   0.05, 0.10, 0.20, 0.30
Seeds:   0, 1, 2, 3, 4
Output:  results/sweep_repaired/r_max_{val}/
         SUMMARY_SWEEP_REPAIRED.md
         results/sweep_repaired/af_vs_replay.png
"""

import json
import os
import subprocess
import sys

import numpy as np

OUTPUT_BASE = "results/sweep_repaired"
BENCH = "split_cifar"
SEEDS = [0, 1, 2, 3, 4]
METHODS = ["DIFE_only", "DIFE_MV"]
R_MAX_VALUES = [0.05, 0.10, 0.20, 0.30]
EPOCHS = "3"


def run_sweep():
    """Run all sweep jobs via run_fast_track.py."""
    for r_max in R_MAX_VALUES:
        out_root = os.path.join(OUTPUT_BASE, f"r_max_{r_max:.2f}")
        cmd = [
            sys.executable, "run_fast_track.py",
            "--bench", BENCH,
            "--seeds", *[str(s) for s in SEEDS],
            "--epochs-per-task", EPOCHS,
            "--methods", *METHODS,
            "--r-max", str(r_max),
            "--output-root", out_root,
        ]
        print(f"\n{'='*60}")
        print(f"Running sweep r_max={r_max:.2f} ...")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


def load_sweep_results():
    """Load all results, indexed by (r_max, method, seed)."""
    data = {}
    for r_max in R_MAX_VALUES:
        data[r_max] = {}
        out_root = os.path.join(OUTPUT_BASE, f"r_max_{r_max:.2f}")
        for method in METHODS:
            data[r_max][method] = {}
            for seed in SEEDS:
                path = os.path.join(out_root, BENCH, method, f"seed_{seed}", "metrics.json")
                if os.path.exists(path):
                    with open(path) as f:
                        data[r_max][method][seed] = json.load(f)
                else:
                    print(f"WARNING: missing {path}")
    return data


def aggregate(data):
    """Compute mean/std across seeds for each (r_max, method)."""
    agg = {}
    for r_max in R_MAX_VALUES:
        agg[r_max] = {}
        for method in METHODS:
            seed_data = [data[r_max][method][s] for s in SEEDS if s in data[r_max][method]]
            if not seed_data:
                continue
            aa = [d["avg_final_acc"] for d in seed_data]
            af = [d["avg_forgetting"] for d in seed_data]
            replay = [d["total_replay_samples"] for d in seed_data]
            # efficiency: need FT baseline — approximation from replication study
            agg[r_max][method] = {
                "AA_mean": float(np.mean(aa)),
                "AA_std": float(np.std(aa)),
                "AF_mean": float(np.mean(af)),
                "AF_std": float(np.std(af)),
                "replay_mean": float(np.mean(replay)),
                "replay_std": float(np.std(replay)),
                "n_seeds": len(seed_data),
            }
    return agg


def generate_summary_md(agg):
    lines = [
        "# SUMMARY_SWEEP_REPAIRED — Budget Sweep (Repaired DIFE/MV)",
        "",
        f"Benchmark: split_cifar | Seeds: {SEEDS} | Epochs/task: {EPOCHS}",
        "Repaired code: bounded L-BFGS-B, MIN_OBS=6, per-epoch modulation.",
        "",
        "## Main Table: mean ± std over 5 seeds",
        "",
        "| r_max | Method | AA ↑ | AF ↓ | Replay Used | ΔAF | ΔAA |",
        "|-------|--------|------|------|-------------|-----|-----|",
    ]

    crossover_found = False
    crossover_rmax = None

    for r_max in R_MAX_VALUES:
        if "DIFE_only" not in agg[r_max] or "DIFE_MV" not in agg[r_max]:
            continue
        d_only = agg[r_max]["DIFE_only"]
        d_mv = agg[r_max]["DIFE_MV"]
        delta_af = d_only["AF_mean"] - d_mv["AF_mean"]   # positive = MV reduces forgetting
        delta_aa = d_mv["AA_mean"] - d_only["AA_mean"]   # positive = MV improves accuracy

        for method, d in [("DIFE_only", d_only), ("DIFE_MV", d_mv)]:
            lines.append(
                f"| {r_max:.2f} "
                f"| {method:<10} "
                f"| {d['AA_mean']:.3f}±{d['AA_std']:.3f} "
                f"| {d['AF_mean']:.3f}±{d['AF_std']:.3f} "
                f"| {d['replay_mean']:>10,.0f}±{d['replay_std']:>7,.0f} "
                f"| {delta_af:+.4f} "
                f"| {delta_aa:+.4f} |"
            )

        if delta_af > 0 and not crossover_found:
            crossover_found = True
            crossover_rmax = r_max

    lines += ["", "ΔAF = AF(DIFE_only) - AF(DIFE_MV)  (positive = MV reduces forgetting)",
              "ΔAA = AA(DIFE_MV) - AA(DIFE_only)  (positive = MV improves accuracy)", ""]

    # Crossover statement
    lines += ["## Crossover Analysis", ""]
    if crossover_found:
        lines.append(
            f"**DIFE_MV begins to reliably beat DIFE_only at r_max = {crossover_rmax:.2f}** "
            f"(first r_max where ΔAF > 0)."
        )
    else:
        lines.append(
            "**No crossover observed in this sweep range.** "
            "DIFE_MV does not consistently outperform DIFE_only on AF across r_max ∈ {0.05, 0.10, 0.20, 0.30}."
        )

    lines += [
        "",
        "## Detailed Results by r_max",
        "",
    ]
    for r_max in R_MAX_VALUES:
        lines.append(f"### r_max = {r_max:.2f}")
        lines.append("")
        if "DIFE_only" not in agg[r_max] or "DIFE_MV" not in agg[r_max]:
            lines.append("(results unavailable)")
            lines.append("")
            continue
        d_only = agg[r_max]["DIFE_only"]
        d_mv = agg[r_max]["DIFE_MV"]
        lines += [
            f"- DIFE_only:  AA={d_only['AA_mean']:.4f}±{d_only['AA_std']:.4f}  "
            f"AF={d_only['AF_mean']:.4f}±{d_only['AF_std']:.4f}  "
            f"Replay={d_only['replay_mean']:,.0f}±{d_only['replay_std']:,.0f}  "
            f"(n={d_only['n_seeds']} seeds)",
            f"- DIFE_MV:    AA={d_mv['AA_mean']:.4f}±{d_mv['AA_std']:.4f}  "
            f"AF={d_mv['AF_mean']:.4f}±{d_mv['AF_std']:.4f}  "
            f"Replay={d_mv['replay_mean']:,.0f}±{d_mv['replay_std']:,.0f}  "
            f"(n={d_mv['n_seeds']} seeds)",
            f"- ΔAF = {d_only['AF_mean'] - d_mv['AF_mean']:+.4f}  "
            f"ΔAA = {d_mv['AA_mean'] - d_only['AA_mean']:+.4f}  "
            f"Replay saved = {d_only['replay_mean'] - d_mv['replay_mean']:+,.0f}",
            "",
        ]

    return "\n".join(lines)


def make_plot(agg):
    """AF vs Replay Used plot with two lines (DIFE_only and DIFE_MV)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot generation")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"DIFE_only": "#2196F3", "DIFE_MV": "#FF9800"}
    markers = {"DIFE_only": "o", "DIFE_MV": "s"}

    for method in METHODS:
        replay_vals = []
        af_vals = []
        af_std_vals = []
        replay_std_vals = []
        rmax_labels = []
        for r_max in R_MAX_VALUES:
            if method in agg[r_max]:
                d = agg[r_max][method]
                replay_vals.append(d["replay_mean"])
                af_vals.append(d["AF_mean"])
                af_std_vals.append(d["AF_std"])
                replay_std_vals.append(d["replay_std"])
                rmax_labels.append(r_max)

        if not replay_vals:
            continue

        ax.errorbar(
            replay_vals, af_vals,
            yerr=af_std_vals, xerr=replay_std_vals,
            label=method, color=colors[method], marker=markers[method],
            linewidth=2, markersize=8, capsize=4,
        )
        for x, y, r in zip(replay_vals, af_vals, rmax_labels):
            ax.annotate(f"r={r:.2f}", (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color=colors[method])

    ax.set_xlabel("Replay Used (mean over 5 seeds)", fontsize=12)
    ax.set_ylabel("Average Forgetting AF ↓", fontsize=12)
    ax.set_title("AF vs Replay Budget — Repaired DIFE/MV (split-CIFAR)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # lower AF is better, show lower values higher

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    plot_path = os.path.join(OUTPUT_BASE, "af_vs_replay.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {plot_path}")
    return plot_path


def main():
    # Step 1: run training
    run_sweep()

    # Step 2: load and aggregate
    data = load_sweep_results()
    agg = aggregate(data)

    # Step 3: print console summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY — split_cifar, r_max ∈ {0.05, 0.10, 0.20, 0.30}, 5 seeds")
    print("=" * 70)
    print(f"{'r_max':>6}  {'Method':<12}  {'AA':>10}  {'AF':>10}  {'Replay':>12}  {'ΔAF':>8}")
    print("-" * 66)
    for r_max in R_MAX_VALUES:
        if "DIFE_only" not in agg[r_max] or "DIFE_MV" not in agg[r_max]:
            continue
        d_only = agg[r_max]["DIFE_only"]
        d_mv = agg[r_max]["DIFE_MV"]
        delta_af = d_only["AF_mean"] - d_mv["AF_mean"]
        for method, d in [("DIFE_only", d_only), ("DIFE_MV", d_mv)]:
            print(
                f"{r_max:>6.2f}  {method:<12}  "
                f"{d['AA_mean']:.3f}±{d['AA_std']:.3f}  "
                f"{d['AF_mean']:.3f}±{d['AF_std']:.3f}  "
                f"{d['replay_mean']:>12,.0f}  "
                f"{delta_af:>+8.4f}"
            )

    # Step 4: generate markdown
    md = generate_summary_md(agg)
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    md_path = os.path.join(OUTPUT_BASE, "SUMMARY_SWEEP_REPAIRED.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\nMarkdown written: {md_path}")

    # Also to repo root
    with open("SUMMARY_SWEEP_REPAIRED.md", "w") as f:
        f.write(md)
    print("Markdown also written: SUMMARY_SWEEP_REPAIRED.md")

    # Step 5: plot
    make_plot(agg)


if __name__ == "__main__":
    main()
