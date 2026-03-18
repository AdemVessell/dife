#!/usr/bin/env python3
"""Generate RESULTS_SNAPSHOT.md from saved metrics.json files and summary.csv.

Usage:
    python scripts/gen_results_snapshot.py          # prints to stdout
    python scripts/gen_results_snapshot.py --write  # writes RESULTS_SNAPSHOT.md
"""

import argparse
import csv
import glob
import json
import os
import sys
from datetime import date

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_perm_mnist_full():
    """Load all perm_mnist results from results/perm_mnist/."""
    order = [
        "FT", "EWC", "SI",
        "ConstReplay_0.1", "ConstReplay_0.3", "RandReplay",
        "DIFE_only", "MV_only", "DIFE_MV",
    ]
    rows = []
    for method in order:
        files = sorted(glob.glob(
            os.path.join(_ROOT, "results", "perm_mnist", method, "seed_*", "metrics.json")
        ))
        if not files:
            continue
        data = [json.load(open(f)) for f in files]
        keys = [
            ("avg_final_acc", "AA"),
            ("avg_forgetting", "AF"),
            ("bwt", "BWT"),
            ("fwt", "FWT"),
            ("total_replay_samples", "Replay"),
        ]
        row = {"method": method, "n": len(data)}
        for key, col in keys:
            vals = [float(d.get(key, 0)) for d in data]
            row[f"{col}_mean"] = np.mean(vals)
            row[f"{col}_std"] = np.std(vals)
        ft_af = None  # filled later
        rows.append(row)
    return rows


def _load_fast_track():
    """Load fast-track summary.csv."""
    path = os.path.join(_ROOT, "results", "fast_track", "summary.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_split_cifar_partial():
    """Load any split_cifar results found (fast-track path first, then full-run)."""
    files = sorted(glob.glob(
        os.path.join(_ROOT, "results", "fast_track", "split_cifar", "*", "seed_*", "metrics.json")
    ))
    if not files:
        files = sorted(glob.glob(
            os.path.join(_ROOT, "results", "split_cifar", "*", "seed_*", "metrics.json")
        ))
    rows = []
    for f in files:
        parts = f.split(os.sep)
        method = parts[-3]
        seed = parts[-2]
        data = json.load(open(f))
        rows.append({
            "method": method, "seed": seed,
            "AA": data["avg_final_acc"], "AF": data["avg_forgetting"],
            "replay": data["total_replay_samples"],
        })
    return rows


def _fmt(mean, std, decimals=3):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def generate(write_file=False):
    full = _load_perm_mnist_full()
    fast = _load_fast_track()
    partial = _load_split_cifar_partial()

    # Compute efficiency for fast-track rows
    ft_fast = next((r for r in fast if r["method"] == "FT"), None)
    ft_af = float(ft_fast["AF_mean"]) if ft_fast else None

    lines = [
        "# Results Snapshot — DIFE ∘ Memory Vortex",
        "",
        f"_Generated: {date.today()}  |  Source: results/ directory (do not hand-edit)_",
        "",
        "---",
        "",
        "## Metric Definitions",
        "",
        "| Metric | Full name | Definition |",
        "|--------|-----------|------------|",
        "| **AA** | Avg Final Accuracy | Mean accuracy on all tasks measured after the final task is trained |",
        "| **AF** | Avg Forgetting | Mean over tasks j of: max_acc(j) − final_acc(j); always ≥ 0 |",
        "| **BWT** | Backward Transfer | = −AF by construction; negative = forgetting, 0 = no forgetting |",
        "| **FWT** | Forward Transfer | Mean zero-shot acc on task j before training it, minus random baseline (0.1 for 10-class) |",
        "| **Replay** | Replay Budget | Total replay samples mixed into training batches across all tasks |",
        "| **Efficiency** | Sample Efficiency | (FT_AF − method_AF) / Replay × 10,000 — forgetting reduction per 10k samples; 0 for no-replay methods |",
        "",
        "---",
        "",
        "## Table 1 — Full perm_mnist Benchmark",
        "",
        f"5 tasks · 5 epochs/task · **{full[0]['n'] if full else '?'} seeds** · buffer_capacity=2000",
        "",
        "| Method | AA ↑ | AF ↓ | BWT | FWT | Replay Budget |",
        "|--------|------|------|-----|-----|---------------|",
    ]

    ft_full_af = None
    for row in full:
        if row["method"] == "FT":
            ft_full_af = row["AF_mean"]
        lines.append(
            f"| {row['method']:<18} "
            f"| {_fmt(row['AA_mean'], row['AA_std'])} "
            f"| {_fmt(row['AF_mean'], row['AF_std'])} "
            f"| {_fmt(row['BWT_mean'], row['BWT_std'])} "
            f"| {_fmt(row['FWT_mean'], row['FWT_std'])} "
            f"| {row['Replay_mean']:>12,.0f} ± {row['Replay_std']:>8,.0f} |"
        )

    lines += [
        "",
        "_std is across seeds. All methods use the same fixed seeds and data splits._",
        "",
        "---",
        "",
        "## Table 2 — Fast-Track perm_mnist",
        "",
        "5 tasks · **3 epochs/task** · 3 seeds · buffer_capacity=2000",
        "Methods: FT, ConstReplay_0.1/0.3, DIFE_only, MV_only, DIFE_MV (no EWC/SI/RandReplay)",
        "",
        "| Method | AA ↑ | AF ↓ | BWT | FWT | Replay Budget | Efficiency |",
        "|--------|------|------|-----|-----|---------------|------------|",
    ]

    for row in fast:
        eff = float(row.get("efficiency", 0))
        lines.append(
            f"| {row['method']:<18} "
            f"| {_fmt(float(row['AA_mean']), float(row['AA_std']))} "
            f"| {_fmt(float(row['AF_mean']), float(row['AF_std']))} "
            f"| {_fmt(float(row['BWT_mean']), float(row['BWT_std']))} "
            f"| {_fmt(float(row['FWT_mean']), float(row['FWT_std']))} "
            f"| {float(row['replay_budget_mean']):>12,.0f} ± {float(row['replay_budget_std']):>6,.0f} "
            f"| {eff:.4f} |"
        )

    lines += [
        "",
        "_Efficiency = AF improvement per 10k replay samples vs FT baseline._",
        "_Higher is better; 0 = no replay._",
        "",
        "---",
        "",
        "## Table 3 — split_cifar (Partial)",
        "",
    ]

    if partial:
        lines += [
            f"**{len(partial)} jobs complete** as of snapshot date.",
            "",
            "| Method | Seed | AA | AF | Replay |",
            "|--------|------|----|----|--------|",
        ]
        for row in partial:
            lines.append(
                f"| {row['method']:<18} | {row['seed']} "
                f"| {row['AA']:.3f} | {row['AF']:.3f} | {row['replay']:>10,} |"
            )
        lines += [
            "",
            "_Do not draw conclusions from partial data. See RESUME.md to continue this run._",
        ]
    else:
        lines.append("_No split_cifar results available yet. See RESUME.md._")

    lines += [
        "",
        "---",
        "",
        "## Reproducibility",
        "",
        "All runs use:",
        "- `torch.manual_seed(seed)` + `np.random.seed(seed)` before each job",
        "- Isolated subprocess per job (`run_one_job.py`) — no shared state",
        "- Skip logic: if `metrics.json` exists, job is skipped silently",
        "- Grid-search params cached to `results/{bench}/grid_search_params.json`",
        "",
        "To regenerate this file: `python scripts/gen_results_snapshot.py --write`",
    ]

    output = "\n".join(lines) + "\n"

    if write_file:
        out_path = os.path.join(_ROOT, "RESULTS_SNAPSHOT.md")
        with open(out_path, "w") as f:
            f.write(output)
        print(f"Written: {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write RESULTS_SNAPSHOT.md")
    args = parser.parse_args()
    generate(write_file=args.write)
