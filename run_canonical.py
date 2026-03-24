#!/usr/bin/env python3
"""Canonical experiment runner for DIFE / MV audit.

Runs exactly one clean experiment family:
  Benchmark:  split_cifar
  r_max:      0.30
  Seeds:      0, 1, 2, 3, 4
  Epochs/task: 3
  Methods:    FT, ConstReplay_0.1, ConstReplay_0.3, DIFE_only, MV_only, DIFE_MV
  Output:     results/canonical/split_cifar_rmax_0.30/

All results are written fresh.  No skip logic based on old result folders.
Skip logic applies only within this canonical output path so interrupted runs
can be resumed cleanly.

Usage:
    python run_canonical.py [--device cpu] [--seed SEED] [--method METHOD]
    python run_canonical.py               # runs all seeds × methods sequentially
"""

import argparse
import gc
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "memory-vortex-dife-lab"))

import torch
import numpy as np

from eval.config import make_bench_config, BenchConfig
from eval.runner import _load_data, _fresh_model, _grid_search_params
from eval.trainer import train_one_method
from eval.metrics import compute_all_metrics, save_metrics

# ─── Canonical configuration ──────────────────────────────────────────────────
CANONICAL_OUTPUT_ROOT = os.path.join(_HERE, "results", "canonical", "split_cifar_rmax_0.30")
CANONICAL_BENCH       = "split_cifar"
CANONICAL_RMAX        = 0.30
CANONICAL_SEEDS       = [0, 1, 2, 3, 4]
CANONICAL_METHODS     = ["FT", "ConstReplay_0.1", "ConstReplay_0.3",
                          "DIFE_only", "MV_only", "DIFE_MV"]
CANONICAL_EPOCHS      = 3   # override config default of 5


def make_canonical_config(device: str = "cpu") -> BenchConfig:
    cfg = make_bench_config(CANONICAL_BENCH, device=device)
    # Override epochs to canonical value
    cfg.epochs_per_task = CANONICAL_EPOCHS
    cfg.output_dir = CANONICAL_OUTPUT_ROOT
    return cfg


def run_one(bench: str, seed: int, method: str, device: str = "cpu") -> None:
    out_dir  = os.path.join(CANONICAL_OUTPUT_ROOT, method, f"seed_{seed}")
    out_path = os.path.join(out_dir, "metrics.json")
    trace_path = os.path.join(out_dir, "controller_trace.csv")

    if os.path.exists(out_path):
        print(f"[skip] seed={seed} method={method} — already done in canonical output")
        return

    print(f"\n{'='*60}")
    print(f"  CANONICAL: bench={bench} seed={seed} method={method}")
    print(f"  epochs/task={CANONICAL_EPOCHS}  r_max={CANONICAL_RMAX}")
    print(f"{'='*60}")

    cfg = make_canonical_config(device=device)

    best_ewc_lam, best_si_c = _grid_search_params(bench, cfg)

    torch.manual_seed(seed)
    np.random.seed(seed)
    loaders = _load_data(bench, cfg, seed=seed)
    model = _fresh_model(bench)

    result = train_one_method(
        method=method,
        model=model,
        task_loaders=loaders,
        cfg=cfg,
        seed=seed,
        best_ewc_lam=best_ewc_lam,
        best_si_c=best_si_c,
        r_max=CANONICAL_RMAX,
        gamma=1.0,
        trace_path=trace_path,
    )

    n_classes = 2  # split_cifar
    metrics = compute_all_metrics(
        acc_matrix=result["acc_matrix"],
        r_t_history=result["r_t_history"],
        total_replay_samples=result["total_replay_samples"],
        wall_clock=result["wall_clock_seconds"],
        n_classes_per_task=n_classes,
        pre_task_acc=result.get("pre_task_acc", []),
    )
    metrics["acc_matrix"]          = result["acc_matrix"]
    metrics["r_t_history"]         = result["r_t_history"]
    metrics["mv_proxy_history"]    = result["mv_proxy_history"]
    metrics["dife_params_history"] = result["dife_params_history"]
    metrics["pre_task_acc"]        = result.get("pre_task_acc", [])
    metrics["canonical_config"] = {
        "bench": bench,
        "r_max": CANONICAL_RMAX,
        "epochs_per_task": CANONICAL_EPOCHS,
        "seed": seed,
        "method": method,
        "code_branch": "canonical/audit-rebuild",
    }

    save_metrics(metrics, out_path)
    print(
        f"[done] seed={seed} method={method} "
        f"AA={metrics['avg_final_acc']:.3f} AF={metrics['avg_forgetting']:.3f} "
        f"replay={metrics['total_replay_samples']:,}"
    )

    del model, result, loaders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Canonical DIFE/MV audit experiment")
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--seed",    type=int, default=None,
                        help="Run a single seed (default: all seeds)")
    parser.add_argument("--method",  default=None,
                        help="Run a single method (default: all methods)")
    args = parser.parse_args()

    seeds   = [args.seed]   if args.seed   is not None else CANONICAL_SEEDS
    methods = [args.method] if args.method is not None else CANONICAL_METHODS

    os.makedirs(CANONICAL_OUTPUT_ROOT, exist_ok=True)

    t_start = time.time()
    total = len(seeds) * len(methods)
    done  = 0

    for method in methods:
        for seed in seeds:
            run_one(CANONICAL_BENCH, seed, method, device=args.device)
            done += 1
            elapsed = time.time() - t_start
            print(f"  Progress: {done}/{total}  elapsed={elapsed:.0f}s")

    print(f"\nAll {total} jobs done in {time.time() - t_start:.1f}s")
    print(f"Results: {CANONICAL_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
