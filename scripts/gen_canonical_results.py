#!/usr/bin/env python3
"""Generate canonical results summary, tables, and plots.

Reads from results/canonical/split_cifar_rmax_0.30/
Writes:
  results/canonical/split_cifar_rmax_0.30/summary.csv
  docs/CANONICAL_RESULTS.md
  results/canonical/split_cifar_rmax_0.30/plots/af_vs_replay.png
  results/canonical/split_cifar_rmax_0.30/plots/trace_seed0_DIFE_MV.png

Usage: python scripts/gen_canonical_results.py
"""

import csv
import glob
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_ROOT = os.path.join(_HERE, "results", "canonical", "split_cifar_rmax_0.30")
DOCS_DIR       = os.path.join(_HERE, "docs")
PLOTS_DIR      = os.path.join(CANONICAL_ROOT, "plots")


def load_all_metrics():
    results = {}
    for f in sorted(glob.glob(os.path.join(CANONICAL_ROOT, "*/seed_*/metrics.json"))):
        parts = f.replace(CANONICAL_ROOT + "/", "").split("/")
        method = parts[0]
        data   = json.load(open(f))
        results.setdefault(method, []).append(data)
    return results


def mean_std(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    s = (sum((v - m)**2 for v in vals) / n) ** 0.5
    return m, s


def fmt_ms(m, s):
    if math.isnan(m):
        return "—"
    return f"{m:.3f} ± {s:.3f}"


def generate_summary_csv(results: dict) -> str:
    path = os.path.join(CANONICAL_ROOT, "summary.csv")
    fields = ["method", "n_seeds",
              "AA_mean", "AA_std", "AF_mean", "AF_std",
              "BWT_mean", "BWT_std", "FWT_mean", "FWT_std",
              "replay_mean", "replay_std",
              "wall_clock_mean"]
    rows = []
    method_order = ["FT", "ConstReplay_0.1", "ConstReplay_0.3",
                    "DIFE_only", "MV_only", "DIFE_MV"]
    for method in method_order:
        runs = results.get(method, [])
        if not runs:
            continue
        aa_m, aa_s  = mean_std([r["avg_final_acc"]        for r in runs])
        af_m, af_s  = mean_std([r["avg_forgetting"]       for r in runs])
        bwt_m, bwt_s = mean_std([r["bwt"]                 for r in runs])
        fwt_m, fwt_s = mean_std([r.get("fwt", 0.0)        for r in runs])
        rp_m, rp_s  = mean_std([r["total_replay_samples"] for r in runs])
        wc_m, _     = mean_std([r["wall_clock_seconds"]   for r in runs])
        rows.append({
            "method": method, "n_seeds": len(runs),
            "AA_mean": aa_m, "AA_std": aa_s,
            "AF_mean": af_m, "AF_std": af_s,
            "BWT_mean": bwt_m, "BWT_std": bwt_s,
            "FWT_mean": fwt_m, "FWT_std": fwt_s,
            "replay_mean": rp_m, "replay_std": rp_s,
            "wall_clock_mean": wc_m,
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def generate_canonical_results_md(results: dict) -> str:
    path = os.path.join(DOCS_DIR, "CANONICAL_RESULTS.md")
    os.makedirs(DOCS_DIR, exist_ok=True)

    method_order = ["FT", "ConstReplay_0.1", "ConstReplay_0.3",
                    "DIFE_only", "MV_only", "DIFE_MV"]
    lines = []
    lines.append("# Canonical Results — split-CIFAR-10, r_max=0.30")
    lines.append("")
    lines.append("**Config:** 5 tasks, 3 epochs/task, seeds 0–4, r_max=0.30")
    lines.append("**Branch:** canonical/audit-rebuild")
    lines.append("**All results generated from the same code state.**")
    lines.append("")
    lines.append("## Performance Table")
    lines.append("")
    lines.append("| Method | Seeds | AA ↑ | AF ↓ | BWT | Replay Used |")
    lines.append("|--------|-------|------|------|-----|-------------|")

    for method in method_order:
        runs = results.get(method, [])
        if not runs:
            lines.append(f"| {method} | 0 | — | — | — | — |")
            continue
        aa_m, aa_s  = mean_std([r["avg_final_acc"]        for r in runs])
        af_m, af_s  = mean_std([r["avg_forgetting"]       for r in runs])
        bwt_m, bwt_s = mean_std([r["bwt"]                 for r in runs])
        rp_m, rp_s  = mean_std([r["total_replay_samples"] for r in runs])
        lines.append(
            f"| {method} | {len(runs)} | {fmt_ms(aa_m,aa_s)} | {fmt_ms(af_m,af_s)} "
            f"| {fmt_ms(bwt_m,bwt_s)} | {rp_m:,.0f} ± {rp_s:,.0f} |"
        )

    lines.append("")
    lines.append("## Per-Seed Detail")
    lines.append("")
    lines.append("| Method | Seed | AA | AF | BWT | Replay |")
    lines.append("|--------|------|----|----|-----|--------|")
    for method in method_order:
        for i, r in enumerate(results.get(method, [])):
            lines.append(
                f"| {method} | {i} | {r['avg_final_acc']:.3f} | {r['avg_forgetting']:.3f} "
                f"| {r['bwt']:.3f} | {r['total_replay_samples']:,} |"
            )

    lines.append("")
    lines.append("## DIFE Parameter History (post-task fit, mean across seeds)")
    lines.append("")
    for method in ("DIFE_only", "DIFE_MV"):
        runs = results.get(method, [])
        if not runs:
            continue
        lines.append(f"**{method}:**")
        lines.append("")
        lines.append("| Task | alpha (mean) | beta (mean) | r_t (mean) |")
        lines.append("|------|-------------|------------|------------|")
        n_tasks = max(len(r.get("dife_params_history", [])) for r in runs)
        for t in range(n_tasks):
            alphas = [r["dife_params_history"][t]["alpha"]
                      for r in runs if t < len(r.get("dife_params_history", []))]
            betas  = [r["dife_params_history"][t]["beta"]
                      for r in runs if t < len(r.get("dife_params_history", []))]
            r_ts   = [r["r_t_history"][t]
                      for r in runs if t < len(r.get("r_t_history", []))]
            alpha_m, _ = mean_std(alphas)
            beta_m, _  = mean_std(betas)
            rt_m, _    = mean_std(r_ts)
            lines.append(f"| {t} | {alpha_m:.4f} | {beta_m:.2e} | {rt_m:.4f} |")
        lines.append("")

    lines.append("## r_t History (task-level replay fraction)")
    lines.append("")
    lines.append("For capped methods, r_t should always equal r_max if DIFE envelope ≥ r_max.")
    lines.append("")
    for method in ("DIFE_only", "MV_only", "DIFE_MV", "ConstReplay_0.3"):
        runs = results.get(method, [])
        if not runs:
            continue
        all_rt = [r.get("r_t_history", []) for r in runs]
        lines.append(f"**{method}** r_t per seed:")
        for i, rt in enumerate(all_rt):
            lines.append(f"  seed_{i}: {[f'{x:.3f}' for x in rt]}")
        lines.append("")

    lines.append(f"\n*Generated: 2026-03-19 | Source: {CANONICAL_ROOT}*")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def generate_plots(results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [warn] matplotlib not available — skipping plots")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    method_order = ["FT", "ConstReplay_0.1", "ConstReplay_0.3",
                    "DIFE_only", "MV_only", "DIFE_MV"]
    colors = {"FT": "gray", "ConstReplay_0.1": "blue", "ConstReplay_0.3": "navy",
              "DIFE_only": "green", "MV_only": "orange", "DIFE_MV": "red"}
    markers = {"FT": "x", "ConstReplay_0.1": "s", "ConstReplay_0.3": "D",
               "DIFE_only": "^", "MV_only": "o", "DIFE_MV": "*"}

    # Plot 1: AF vs Replay Used
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in method_order:
        runs = results.get(method, [])
        if not runs:
            continue
        afs = [r["avg_forgetting"] for r in runs]
        rps = [r["total_replay_samples"] for r in runs]
        af_m, af_s = mean_std(afs)
        rp_m, _ = mean_std(rps)
        ax.errorbar(rp_m, af_m, yerr=af_s,
                    color=colors[method], marker=markers[method],
                    markersize=10, capsize=4, label=method, linewidth=1.5)
    ax.set_xlabel("Total Replay Samples (mean across seeds)")
    ax.set_ylabel("Average Forgetting (AF) ↓")
    ax.set_title("AF vs Replay Used — Canonical Run (split-CIFAR, r_max=0.30)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    p1 = os.path.join(PLOTS_DIR, "af_vs_replay.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {p1}")

    # Plot 2: Controller trace for seed 0, DIFE_MV
    import csv as csvmod
    trace_path = os.path.join(CANONICAL_ROOT, "DIFE_MV", "seed_0", "controller_trace.csv")
    if os.path.exists(trace_path):
        rows = []
        with open(trace_path) as f:
            for row in csvmod.DictReader(f):
                rows.append(row)

        ge    = [int(r["global_epoch"]) for r in rows]
        env   = [float(r["dife_envelope_value"]) if r["dife_envelope_value"] != "nan" else float("nan") for r in rows]
        mv_op = [float(r["mv_operator_value"])   if r["mv_operator_value"] != "nan"  else float("nan") for r in rows]
        after = [float(r["final_replay_fraction_after_cap"]) if r["final_replay_fraction_after_cap"] != "nan" else float("nan") for r in rows]
        r_max_val = float(rows[0]["r_max"]) if rows[0]["r_max"] != "nan" else 0.3

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ge, env,   "b-o", markersize=4, label="DIFE envelope",  linewidth=1.5)
        ax.plot(ge, mv_op, "g-s", markersize=4, label="MV operator",    linewidth=1.5)
        ax.plot(ge, after, "r-^", markersize=5, label="Final r (after cap)", linewidth=2.0)
        ax.axhline(r_max_val, color="black", linestyle="--", linewidth=1, label=f"r_max={r_max_val}")
        # Task boundaries
        n_epochs = max(ge) + 1
        n_tasks = 5
        ep_per_task = n_epochs // n_tasks
        for tk in range(1, n_tasks):
            ax.axvline(tk * ep_per_task - 0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Global Epoch")
        ax.set_ylabel("Replay Fraction")
        ax.set_title("Controller Trace — DIFE_MV seed_0 (canonical run)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
        p2 = os.path.join(PLOTS_DIR, "trace_seed0_DIFE_MV.png")
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {p2}")

        # Also plot MV_only trace for comparison
        mv_trace_path = os.path.join(CANONICAL_ROOT, "MV_only", "seed_0", "controller_trace.csv")
        if os.path.exists(mv_trace_path):
            mv_rows = []
            with open(mv_trace_path) as f:
                for row in csvmod.DictReader(f):
                    mv_rows.append(row)
            mv_after = [float(r["final_replay_fraction_after_cap"]) if r["final_replay_fraction_after_cap"] != "nan" else float("nan") for r in mv_rows]
            mv_ge = [int(r["global_epoch"]) for r in mv_rows]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(ge,    after,    "r-^", markersize=5, label="DIFE_MV final r", linewidth=2.0)
            ax.plot(mv_ge, mv_after, "g-o", markersize=5, label="MV_only final r", linewidth=2.0)
            ax.axhline(r_max_val, color="black", linestyle="--", linewidth=1, label=f"r_max={r_max_val}")
            for tk in range(1, n_tasks):
                ax.axvline(tk * ep_per_task - 0.5, color="gray", linestyle=":", linewidth=0.8)
            ax.set_xlabel("Global Epoch")
            ax.set_ylabel("Final Replay Fraction (after cap)")
            ax.set_title("DIFE_MV vs MV_only final replay fraction — seed_0 (canonical)")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_ylim(-0.05, 0.5)
            ax.grid(True, alpha=0.3)
            p3 = os.path.join(PLOTS_DIR, "trace_seed0_DIFE_MV_vs_MV_only.png")
            fig.savefig(p3, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Plot saved: {p3}")


def main():
    print("Loading canonical results...")
    results = load_all_metrics()
    if not results:
        print(f"ERROR: No metrics.json files found in {CANONICAL_ROOT}")
        sys.exit(1)

    for method, runs in results.items():
        print(f"  {method}: {len(runs)} seeds")

    print("\nGenerating summary.csv...")
    csv_path = generate_summary_csv(results)
    print(f"  Written: {csv_path}")

    print("\nGenerating CANONICAL_RESULTS.md...")
    md_path = generate_canonical_results_md(results)
    print(f"  Written: {md_path}")

    print("\nGenerating plots...")
    generate_plots(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
