#!/usr/bin/env python3
"""Money comparison: fixed-rate vs adaptive-selection vs adaptive-scheduling vs combined.

Methods : ConstReplay_0.3 | MIR | DIFE_only | DIFE_MV
Bench   : split_cifar, r_max=0.30, beta_min=0.10, 3 epochs/task
Seeds   : 0-4  (run one at a time — safe against env resets)
Output  : results/money_comparison/split_cifar_rmax_0.30/

Usage:
    python run_money_comparison.py                      # all methods × all seeds
    python run_money_comparison.py --seed 2             # one seed, all methods
    python run_money_comparison.py --method MIR --seed 0
"""

import gc
import json
import os
import sys
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "memory-vortex-dife-lab"))

import torch
import numpy as np

# ── Patch BETA_BOUNDS before OnlineDIFEFitter is imported anywhere ─────────────
from eval.online_fitters import OnlineDIFEFitter
OnlineDIFEFitter.BETA_BOUNDS = (0.10, 1.0)
print(f"[patch] OnlineDIFEFitter.BETA_BOUNDS = {OnlineDIFEFitter.BETA_BOUNDS}")

from eval.config import make_bench_config
from eval.runner import _load_data, _fresh_model, _grid_search_params
from eval.trainer import train_one_method
from eval.metrics import compute_all_metrics, save_metrics

# ── Experiment config ───────────────────────────────────────────────────────────
BENCH   = "split_cifar"
RMAX    = 0.30
BETA_MIN = 0.10
EPOCHS  = 3
SEEDS   = [0, 1, 2, 3, 4]
METHODS = ["ConstReplay_0.3", "MIR", "DIFE_only", "DIFE_MV"]
OUTPUT_ROOT = os.path.join(_HERE, "results", "money_comparison",
                           "split_cifar_rmax_0.30")


def run_one(seed: int, method: str, cfg, best_ewc_lam: float, best_si_c: float) -> None:
    out_dir   = os.path.join(OUTPUT_ROOT, method, f"seed_{seed}")
    out_path  = os.path.join(out_dir, "metrics.json")
    trace_path = os.path.join(out_dir, "controller_trace.csv")

    if os.path.exists(out_path):
        print(f"[skip] {method} seed={seed}")
        return

    print(f"\n{'='*60}")
    print(f"  {method}  seed={seed}  r_max={RMAX}  beta_min={BETA_MIN}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    loaders = _load_data(BENCH, cfg, seed=seed)
    model   = _fresh_model(BENCH, cfg)

    result = train_one_method(
        method=method,
        model=model,
        task_loaders=loaders,
        cfg=cfg,
        seed=seed,
        best_ewc_lam=best_ewc_lam,
        best_si_c=best_si_c,
        r_max=RMAX,
        gamma=1.0,
        trace_path=trace_path,
    )

    metrics = compute_all_metrics(
        acc_matrix=result["acc_matrix"],
        r_t_history=result["r_t_history"],
        total_replay_samples=result["total_replay_samples"],
        wall_clock=result["wall_clock_seconds"],
        n_classes_per_task=cfg.n_classes_per_task,
        pre_task_acc=result.get("pre_task_acc", []),
    )
    metrics["acc_matrix"]          = result["acc_matrix"]
    metrics["r_t_history"]         = result["r_t_history"]
    metrics["mv_proxy_history"]    = result["mv_proxy_history"]
    metrics["dife_params_history"] = result["dife_params_history"]
    metrics["pre_task_acc"]        = result.get("pre_task_acc", [])
    metrics["run_config"] = {
        "bench": BENCH, "r_max": RMAX, "beta_min": BETA_MIN,
        "epochs_per_task": EPOCHS, "seed": seed, "method": method,
    }

    save_metrics(metrics, out_path)
    print(
        f"[done] {method} seed={seed}  "
        f"AA={metrics['avg_final_acc']:.3f}  AF={metrics['avg_forgetting']:.3f}  "
        f"replay={metrics['total_replay_samples']:,}"
    )

    del model, result, loaders
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    cfg = make_bench_config(BENCH, device=args.device)
    cfg.epochs_per_task = EPOCHS
    cfg.output_dir = OUTPUT_ROOT

    best_ewc_lam, best_si_c = _grid_search_params(BENCH, cfg)

    seeds   = [args.seed]   if args.seed   is not None else SEEDS
    methods = [args.method] if args.method is not None else METHODS

    total = len(methods) * len(seeds)
    done  = 0
    t0    = time.time()

    for method in methods:
        for seed in seeds:
            try:
                run_one(seed, method, cfg, best_ewc_lam, best_si_c)
            except Exception as exc:
                print(f"ERROR {method} seed={seed}: {exc}")
            done += 1
            print(f"  Progress: {done}/{total}  elapsed={time.time()-t0:.0f}s")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<20}  {'AA':>7}  {'AF':>7}  {'Replay':>10}")
    print("-" * 50)
    for method in METHODS:
        aas, afs, reps = [], [], []
        for seed in SEEDS:
            p = os.path.join(OUTPUT_ROOT, method, f"seed_{seed}", "metrics.json")
            if os.path.exists(p):
                d = json.load(open(p))
                aas.append(d["avg_final_acc"])
                afs.append(d["avg_forgetting"])
                reps.append(d["total_replay_samples"])
        if aas:
            print(f"{method:<20}  {np.mean(aas):.4f}  {np.mean(afs):.4f}  "
                  f"{int(np.mean(reps)):>10,}  (n={len(aas)})")
        else:
            print(f"{method:<20}  (no results yet)")


if __name__ == "__main__":
    main()
