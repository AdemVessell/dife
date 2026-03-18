#!/usr/bin/env python3
"""Run exactly one (bench, seed, method) job and save metrics.json.

Usage:
    python run_one_job.py --bench perm_mnist --seed 3 --method EWC

Exits with code 0 on success, non-zero on failure.
If metrics.json already exists the job is skipped silently (exit 0).
"""

import argparse
import gc
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "memory-vortex-dife-lab"))

import torch
import numpy as np

from eval.config import make_bench_config
from eval.runner import _load_data, _fresh_model, _grid_search_params
from eval.trainer import train_one_method
from eval.metrics import compute_all_metrics, save_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True, choices=["perm_mnist", "split_cifar"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--method", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = make_bench_config(args.bench, device=args.device)

    out_path = os.path.join(
        cfg.output_dir, args.bench, args.method, f"seed_{args.seed}", "metrics.json"
    )
    if os.path.exists(out_path):
        print(f"[skip] {args.bench} seed={args.seed} method={args.method} — already done")
        return

    best_ewc_lam, best_si_c = _grid_search_params(args.bench, cfg)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    loaders = _load_data(args.bench, cfg, seed=args.seed)
    model = _fresh_model(args.bench)

    result = train_one_method(
        method=args.method,
        model=model,
        task_loaders=loaders,
        cfg=cfg,
        seed=args.seed,
        best_ewc_lam=best_ewc_lam,
        best_si_c=best_si_c,
    )

    n_classes = 10 if args.bench == "perm_mnist" else 2
    metrics = compute_all_metrics(
        acc_matrix=result["acc_matrix"],
        r_t_history=result["r_t_history"],
        total_replay_samples=result["total_replay_samples"],
        wall_clock=result["wall_clock_seconds"],
        n_classes_per_task=n_classes,
        pre_task_acc=result.get("pre_task_acc", []),
    )
    metrics["acc_matrix"] = result["acc_matrix"]
    metrics["r_t_history"] = result["r_t_history"]
    metrics["mv_proxy_history"] = result["mv_proxy_history"]
    metrics["dife_params_history"] = result["dife_params_history"]
    metrics["pre_task_acc"] = result.get("pre_task_acc", [])

    save_metrics(metrics, out_path)
    print(
        f"[done] {args.bench} seed={args.seed} method={args.method} "
        f"AA={metrics['avg_final_acc']:.3f} AF={metrics['avg_forgetting']:.3f}"
    )

    del model, result, loaders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
