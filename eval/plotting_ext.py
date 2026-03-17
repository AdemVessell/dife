"""Extended plotting for the 9-method evaluation suite."""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.plotting import (
    plot_accuracy_heatmap,
    plot_forgetting_curves,
    plot_method_comparison,
)

ALL_METHODS = [
    "FT", "EWC", "SI",
    "ConstReplay_0.1", "ConstReplay_0.3", "RandReplay",
    "DIFE_only", "MV_only", "DIFE_MV",
]

# Extended color palette for 9 methods
METHOD_COLORS = {
    "FT":              "#e05c5c",
    "EWC":             "#5c9ee0",
    "SI":              "#5cc47a",
    "ConstReplay_0.1": "#b07acc",
    "ConstReplay_0.3": "#7a4da0",
    "RandReplay":      "#e0a85c",
    "DIFE_only":       "#c8b820",
    "MV_only":         "#20b8c8",
    "DIFE_MV":         "#e0185c",
}


def plot_aa_af_bars(summary_rows: list, bench_name: str, out_dir: str) -> str:
    """Bar charts of AA and AF with error bars (mean ± std) for all 9 methods.

    Args:
        summary_rows: list of dicts with keys method, AA_mean, AA_std, AF_mean, AF_std
        bench_name: used for title and filename
        out_dir: output directory
    """
    os.makedirs(out_dir, exist_ok=True)
    methods = [r["method"] for r in summary_rows]
    colors = [METHOD_COLORS.get(m, "#999999") for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("AA_mean", "AA_std", "Avg Final Accuracy (AA) ↑"),
        ("AF_mean", "AF_std", "Avg Forgetting (AF) ↓"),
        ("BWT_mean", "BWT_std", "Backward Transfer (BWT) ↑"),
    ]

    for ax, (mean_key, std_key, title) in zip(axes, metrics):
        means = [r[mean_key] for r in summary_rows]
        stds = [r[std_key] for r in summary_rows]
        bars = ax.bar(range(len(methods)), means, color=colors,
                      edgecolor="black", linewidth=0.7)
        ax.errorbar(range(len(methods)), means, yerr=stds,
                    fmt="none", color="black", capsize=3, linewidth=1.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5)

    fig.suptitle(f"{bench_name} – Method Comparison (mean ± std)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{bench_name}_aa_af_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_replay_fractions(
    r_t_histories: dict,
    bench_name: str,
    out_dir: str,
) -> str:
    """Line plot of r_t over tasks for replay-scheduler methods.

    Args:
        r_t_histories: {method: list_of_r_t} for seed 42 (or seed 0)
        bench_name: benchmark identifier
        out_dir: output directory
    """
    os.makedirs(out_dir, exist_ok=True)
    replay_methods = [
        m for m in ALL_METHODS
        if m not in ("FT", "EWC", "SI")
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    for method in replay_methods:
        if method not in r_t_histories:
            continue
        r_ts = r_t_histories[method]
        ax.plot(
            range(1, len(r_ts) + 1), r_ts,
            marker="o", label=method,
            color=METHOD_COLORS.get(method, "#999999"), linewidth=2,
        )

    ax.set_xlabel("Task index")
    ax.set_ylabel("Replay fraction r_t")
    ax.set_title(f"{bench_name} – Replay fractions over tasks")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    path = os.path.join(out_dir, f"{bench_name}_replay_fractions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_accuracy_heatmaps_all(
    acc_matrices_by_method: dict,
    bench_name: str,
    out_dir: str,
) -> str:
    """One heatmap per method arranged in a grid."""
    return plot_accuracy_heatmap(acc_matrices_by_method, bench_name, out_dir)


def plot_forgetting_curves_all(
    acc_matrices_by_method: dict,
    fit_result: dict,
    bench_name: str,
    out_dir: str,
) -> str:
    """Forgetting curves per method (reuses benchmark/plotting.py)."""
    return plot_forgetting_curves(acc_matrices_by_method, fit_result, bench_name, out_dir)


def plot_ablation(summary_rows: list, bench_name: str, out_dir: str) -> str:
    """Side-by-side AA/AF/BWT for DIFE_only, MV_only, DIFE_MV only."""
    os.makedirs(out_dir, exist_ok=True)
    ablation_methods = ["DIFE_only", "MV_only", "DIFE_MV"]
    rows = [r for r in summary_rows if r["method"] in ablation_methods]
    if not rows:
        return ""

    methods = [r["method"] for r in rows]
    colors = [METHOD_COLORS.get(m, "#999999") for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    metrics = [
        ("AA_mean", "AA_std", "AA ↑"),
        ("AF_mean", "AF_std", "AF ↓"),
        ("BWT_mean", "BWT_std", "BWT ↑"),
    ]

    for ax, (mean_key, std_key, title) in zip(axes, metrics):
        means = [r[mean_key] for r in rows]
        stds = [r[std_key] for r in rows]
        ax.bar(range(len(methods)), means, color=colors,
               edgecolor="black", linewidth=0.7)
        ax.errorbar(range(len(methods)), means, yerr=stds,
                    fmt="none", color="black", capsize=4, linewidth=1.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5)

    fig.suptitle(f"{bench_name} – DIFE ∘ MV Ablation", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{bench_name}_ablation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def generate_all_plots(bench_name: str, all_results: dict, cfg, summary_rows: list) -> None:
    """Generate all plots for a benchmark.

    Args:
        bench_name: benchmark identifier
        all_results: {method: {seed: metrics_dict}}
        cfg: BenchConfig
        summary_rows: list of summary dicts (from write_summary_csv logic)
    """
    out_dir = os.path.join(cfg.output_dir, bench_name, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Use seed 0 results for per-method visualisations
    seed0 = 0
    acc_matrices_seed0 = {
        method: all_results[method][seed0]["acc_matrix"]
        for method in all_results
        if seed0 in all_results[method]
    }
    r_t_histories_seed0 = {
        method: all_results[method][seed0]["r_t_history"]
        for method in all_results
        if seed0 in all_results[method]
    }

    # AA/AF bar charts
    plot_aa_af_bars(summary_rows, bench_name, out_dir)

    # Replay fractions
    plot_replay_fractions(r_t_histories_seed0, bench_name, out_dir)

    # Accuracy heatmaps (all methods)
    plot_accuracy_heatmaps_all(acc_matrices_seed0, bench_name, out_dir)

    # Ablation
    plot_ablation(summary_rows, bench_name, out_dir)

    # Method comparison bar chart (reusing existing function)
    plot_method_comparison(acc_matrices_seed0, bench_name, out_dir)
