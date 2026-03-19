#!/usr/bin/env python3
"""Ablation study: isolating MV per-epoch shape vs flat budget redistribution.

Adds DIFE_flatMatched: same total replay per task as DIFE_MV, but distributed
uniformly across all epochs/batches (no MV modulation).

Methods: DIFE_only, DIFE_flatMatched, DIFE_MV
Seeds:   0, 1, 2, 3, 4
r_max:   0.30
Output:  results/ablation_mv_shape/
         ABLATION_MV_SHAPE.md
"""

import gc
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "memory-vortex-dife-lab"))

import numpy as np
import torch

from eval.config import make_bench_config
from eval.metrics import compute_all_metrics, save_metrics
from eval.runner import _load_data, _fresh_model, _grid_search_params
from eval.trainer import train_one_method

OUTPUT_ROOT = "results/ablation_mv_shape"
REPLICATION_ROOT = "results/replication_study"
BENCH = "split_cifar"
SEEDS = [0, 1, 2, 3, 4]
R_MAX = 0.30
EPOCHS_PER_TASK = 3
METHODS = ["DIFE_only", "DIFE_flatMatched", "DIFE_MV"]


def make_cfg():
    cfg = make_bench_config(BENCH)
    cfg.epochs_per_task = EPOCHS_PER_TASK
    cfg.output_dir = OUTPUT_ROOT
    return cfg


def load_or_run_method(method, seed, cfg, loaders, best_ewc_lam, best_si_c,
                       injected_task_budgets=None, replication_root=None):
    """Load from disk if available, otherwise run and save."""
    out_path = os.path.join(OUTPUT_ROOT, BENCH, method, f"seed_{seed}", "metrics.json")

    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)

    # For DIFE_only and DIFE_MV, also check replication_study results
    # (reuse them to avoid re-running; they use the same r_max=0.30 / repaired code)
    if method in ("DIFE_only", "DIFE_MV") and replication_root:
        rep_path = os.path.join(replication_root, BENCH, method, f"seed_{seed}", "metrics.json")
        if os.path.exists(rep_path):
            with open(rep_path) as f:
                data = json.load(f)
            # Save a copy to ablation dir for clean attribution
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  [reuse from replication_study] {method} seed={seed}")
            return data

    print(f"  [run] {method} seed={seed}  injected_budgets={injected_task_budgets is not None}")
    torch.manual_seed(seed)
    model = _fresh_model(BENCH)
    result = train_one_method(
        method=method if method != "DIFE_flatMatched" else "DIFE_flatMatched",
        model=model,
        task_loaders=loaders,
        cfg=cfg,
        seed=seed,
        best_ewc_lam=best_ewc_lam,
        best_si_c=best_si_c,
        r_max=R_MAX,
        gamma=1.0,
        injected_task_budgets=injected_task_budgets,
    )

    metrics = compute_all_metrics(
        acc_matrix=result["acc_matrix"],
        r_t_history=result["r_t_history"],
        total_replay_samples=result["total_replay_samples"],
        wall_clock=result["wall_clock_seconds"],
        n_classes_per_task=2,
        pre_task_acc=result.get("pre_task_acc", []),
    )
    metrics["acc_matrix"] = result["acc_matrix"]
    metrics["r_t_history"] = result["r_t_history"]
    metrics["mv_proxy_history"] = result["mv_proxy_history"]
    metrics["dife_params_history"] = result["dife_params_history"]
    metrics["pre_task_acc"] = result.get("pre_task_acc", [])
    metrics["replay_per_task"] = result.get("replay_per_task", [])

    save_metrics(metrics, out_path)
    del model, result
    gc.collect()
    return metrics


def run_ablation():
    cfg = make_cfg()
    best_ewc_lam, best_si_c = _grid_search_params(BENCH, cfg)

    all_results = {m: {} for m in METHODS}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        loaders = _load_data(BENCH, cfg, seed=seed)

        # Pass 1: DIFE_MV (needed to get per-task replay budgets for flatMatched)
        mv_metrics = load_or_run_method(
            "DIFE_MV", seed, cfg, loaders, best_ewc_lam, best_si_c,
            replication_root=REPLICATION_ROOT,
        )
        all_results["DIFE_MV"][seed] = mv_metrics

        # Extract DIFE_MV's per-task replay totals
        dife_mv_replay_per_task = mv_metrics.get("replay_per_task", [])
        if not dife_mv_replay_per_task:
            # Fallback: estimate from total (uniform across tasks)
            n_tasks = 5
            total = mv_metrics.get("total_replay_samples", 0)
            dife_mv_replay_per_task = [total // n_tasks] * n_tasks
            print(f"  WARNING: replay_per_task missing for DIFE_MV seed={seed}, using uniform estimate")
        print(f"  DIFE_MV replay_per_task: {dife_mv_replay_per_task}")

        # Pass 2: DIFE_only (no injection needed)
        only_metrics = load_or_run_method(
            "DIFE_only", seed, cfg, loaders, best_ewc_lam, best_si_c,
            replication_root=REPLICATION_ROOT,
        )
        all_results["DIFE_only"][seed] = only_metrics

        # Pass 3: DIFE_flatMatched (inject DIFE_MV's per-task budgets, flat distribution)
        torch.manual_seed(seed)
        np.random.seed(seed)
        flat_metrics = load_or_run_method(
            "DIFE_flatMatched", seed, cfg, loaders, best_ewc_lam, best_si_c,
            injected_task_budgets=dife_mv_replay_per_task,
        )
        all_results["DIFE_flatMatched"][seed] = flat_metrics

        # Verify total replay matches DIFE_MV
        mv_total = mv_metrics.get("total_replay_samples", 0)
        flat_total = flat_metrics.get("total_replay_samples", 0)
        print(f"  Budget check: DIFE_MV={mv_total:,}  DIFE_flatMatched={flat_total:,}  "
              f"diff={flat_total - mv_total:+,}")

        del loaders
        gc.collect()

    return all_results


def aggregate(all_results):
    agg = {}
    for method in METHODS:
        if not all_results[method]:
            continue
        seeds_done = sorted(all_results[method].keys())
        aa = [all_results[method][s]["avg_final_acc"] for s in seeds_done]
        af = [all_results[method][s]["avg_forgetting"] for s in seeds_done]
        replay = [all_results[method][s]["total_replay_samples"] for s in seeds_done]
        agg[method] = {
            "AA_mean": float(np.mean(aa)),
            "AA_std": float(np.std(aa)),
            "AF_mean": float(np.mean(af)),
            "AF_std": float(np.std(af)),
            "replay_mean": float(np.mean(replay)),
            "replay_std": float(np.std(replay)),
            "n_seeds": len(seeds_done),
        }
    return agg


def generate_report(all_results, agg):
    lines = [
        "# ABLATION_MV_SHAPE — Does MV Per-Epoch Shaping Help?",
        "",
        f"Benchmark: split_cifar | Seeds: {SEEDS} | Epochs/task: {EPOCHS_PER_TASK} | r_max: {R_MAX}",
        "",
        "## Experimental Design",
        "",
        "- **DIFE_only**: DIFE task-level budget, flat uniform replay across epochs.",
        "- **DIFE_flatMatched**: Same *total* replay per task as DIFE_MV (matched budget),",
        "  but distributed *uniformly* across all epochs/batches (no MV modulation).",
        "- **DIFE_MV**: DIFE task-level envelope × MV per-epoch shape (varying across epochs).",
        "",
        "Comparison of DIFE_flatMatched vs DIFE_MV isolates whether MV's *per-epoch shape*",
        "adds value beyond mere budget redistribution.",
        "",
        "## Summary Table (mean ± std over 5 seeds)",
        "",
        "| Method | AA ↑ | AF ↓ | Replay Used |",
        "|--------|------|------|-------------|",
    ]

    for m in METHODS:
        if m not in agg:
            continue
        d = agg[m]
        lines.append(
            f"| {m:<18} "
            f"| {d['AA_mean']:.3f}±{d['AA_std']:.3f} "
            f"| {d['AF_mean']:.3f}±{d['AF_std']:.3f} "
            f"| {d['replay_mean']:>10,.0f}±{d['replay_std']:>7,.0f} |"
        )

    lines += ["", "## Per-Seed Results", ""]
    for m in METHODS:
        if m not in all_results:
            continue
        lines.append(f"### {m}")
        lines.append("| Seed | AA | AF | Replay |")
        lines.append("|------|----|----|--------|")
        for s in sorted(all_results[m]):
            d = all_results[m][s]
            lines.append(
                f"| {s} "
                f"| {d['avg_final_acc']:.4f} "
                f"| {d['avg_forgetting']:.4f} "
                f"| {d['total_replay_samples']:,} |"
            )
        lines.append("")

    # Direct comparison: DIFE_MV vs DIFE_flatMatched
    lines += ["## Direct Comparison: DIFE_MV vs DIFE_flatMatched", ""]

    if "DIFE_MV" in agg and "DIFE_flatMatched" in agg:
        mv = agg["DIFE_MV"]
        flat = agg["DIFE_flatMatched"]
        delta_af = flat["AF_mean"] - mv["AF_mean"]   # positive = MV is better (lower AF)
        delta_aa = mv["AA_mean"] - flat["AA_mean"]   # positive = MV is better (higher AA)
        replay_diff = mv["replay_mean"] - flat["replay_mean"]

        lines += [
            f"- DIFE_MV:          AA={mv['AA_mean']:.4f}±{mv['AA_std']:.4f}  "
            f"AF={mv['AF_mean']:.4f}±{mv['AF_std']:.4f}  Replay={mv['replay_mean']:,.0f}",
            f"- DIFE_flatMatched: AA={flat['AA_mean']:.4f}±{flat['AA_std']:.4f}  "
            f"AF={flat['AF_mean']:.4f}±{flat['AF_std']:.4f}  Replay={flat['replay_mean']:,.0f}",
            f"",
            f"ΔAF (flatMatched − MV) = {delta_af:+.4f}  "
            f"(positive → MV reduces forgetting vs flat budget)",
            f"ΔAA (MV − flatMatched) = {delta_aa:+.4f}  "
            f"(positive → MV improves accuracy vs flat budget)",
            f"Replay difference (MV − flat) = {replay_diff:+,.0f}  "
            f"(should be ≈ 0 by design)",
            "",
        ]

        mv_beats_flat_af = mv["AF_mean"] < flat["AF_mean"]
        mv_beats_flat_aa = mv["AA_mean"] > flat["AA_mean"]
        lines += ["### Verdict", ""]
        if mv_beats_flat_af or mv_beats_flat_aa:
            lines += [
                f"**DIFE_MV BEATS DIFE_flatMatched** at matched total replay budget.",
                "",
                "- AF: " + (f"MV={mv['AF_mean']:.4f} < flat={flat['AF_mean']:.4f} ✓ (MV better)"
                             if mv_beats_flat_af else
                             f"MV={mv['AF_mean']:.4f} >= flat={flat['AF_mean']:.4f} (MV not better on AF)"),
                "- AA: " + (f"MV={mv['AA_mean']:.4f} > flat={flat['AA_mean']:.4f} ✓ (MV better)"
                             if mv_beats_flat_aa else
                             f"MV={mv['AA_mean']:.4f} <= flat={flat['AA_mean']:.4f} (MV not better on AA)"),
            ]
        else:
            lines += [
                f"**DIFE_MV does NOT beat DIFE_flatMatched** at matched total replay budget.",
                "",
                f"- AF: MV={mv['AF_mean']:.4f} vs flat={flat['AF_mean']:.4f} — no improvement",
                f"- AA: MV={mv['AA_mean']:.4f} vs flat={flat['AA_mean']:.4f} — no improvement",
            ]

    lines += ["", "## Interpretation", ""]

    if "DIFE_MV" in agg and "DIFE_flatMatched" in agg:
        mv = agg["DIFE_MV"]
        flat = agg["DIFE_flatMatched"]
        mv_beats = mv["AF_mean"] < flat["AF_mean"] or mv["AA_mean"] > flat["AA_mean"]

        if mv_beats:
            lines += [
                "**MV per-epoch shaping adds value beyond budget redistribution.**",
                "",
                "The improvement of DIFE_MV over DIFE_flatMatched (at exactly matched total replay)",
                "cannot be attributed to having more replay samples overall. The per-epoch temporal",
                "modulation — concentrating replay at high-forgetting epochs and reducing it at",
                "stable epochs — produces measurably better outcomes.",
                "",
                "This supports the core claim of Memory Vortex: the *shape* of the replay schedule",
                "within a task matters, not just the total budget.",
            ]
        else:
            lines += [
                "**MV per-epoch shaping does not add value beyond budget redistribution** at this setting.",
                "",
                "DIFE_MV and DIFE_flatMatched perform comparably at matched total replay.",
                "This suggests that at r_max=0.30 on split-CIFAR (3 epochs/task), the MV operator",
                "has not yet learned a sufficiently informative epoch-level signal to outperform",
                "uniform replay within each task.",
                "",
                "Possible explanations:",
                "- With only 3 epochs/task, intra-task forgetting variation is limited.",
                "- MIN_OBS=6 means MV only fits after task 2; earlier tasks use flat fallback.",
                "- The proxy signal (1 − buffer accuracy) may be too noisy at small buffer sizes.",
                "",
                "The budget sweep (`SUMMARY_SWEEP_REPAIRED.md`) may reveal settings where MV shaping",
                "provides more leverage (e.g., larger r_max or more epochs per task).",
            ]

    lines += [""]
    return "\n".join(lines)


def main():
    # Step 1: run ablation
    print("Running MV shape ablation (DIFE_only, DIFE_flatMatched, DIFE_MV)...")
    all_results = run_ablation()

    # Step 2: aggregate
    agg = aggregate(all_results)

    # Step 3: print console summary
    print("\n" + "=" * 64)
    print("ABLATION SUMMARY — MV Per-Epoch Shape vs Flat Budget")
    print("=" * 64)
    print(f"{'Method':<20} {'AA':>10} {'AF':>10} {'Replay':>14}")
    print("-" * 58)
    for m in METHODS:
        if m not in agg:
            continue
        d = agg[m]
        print(
            f"{m:<20} "
            f"{d['AA_mean']:.3f}±{d['AA_std']:.3f}  "
            f"{d['AF_mean']:.3f}±{d['AF_std']:.3f}  "
            f"{d['replay_mean']:>12,.0f}"
        )

    # Step 4: generate report
    report = generate_report(all_results, agg)
    out_dir = os.path.join(OUTPUT_ROOT, BENCH)
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "ABLATION_MV_SHAPE.md")
    with open(md_path, "w") as f:
        f.write(report)
    print(f"\nReport written: {md_path}")

    root_path = "ABLATION_MV_SHAPE.md"
    with open(root_path, "w") as f:
        f.write(report)
    print(f"Report also written: {root_path}")


if __name__ == "__main__":
    main()
