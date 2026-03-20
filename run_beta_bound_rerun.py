#!/usr/bin/env python3
"""Beta-bound rerun: test whether raising DIFE beta lower bound enables online control.

Runs canonical split-CIFAR setup (r_max=0.30, 5 seeds, 3 epochs/task, 6 methods)
with a modified BETA_BOUNDS lower bound.

Usage:
    python run_beta_bound_rerun.py --beta-min 0.05
    python run_beta_bound_rerun.py --beta-min 0.10
    python run_beta_bound_rerun.py --beta-min 0.05 --seed 0 --method DIFE_only
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

# ─── Patch BETA_BOUNDS BEFORE anything else imports OnlineDIFEFitter ──────────
# We parse --beta-min from sys.argv early so the class attribute is set before
# train_one_method instantiates any OnlineDIFEFitter.
def _parse_beta_min():
    for i, arg in enumerate(sys.argv):
        if arg == "--beta-min" and i + 1 < len(sys.argv):
            return float(sys.argv[i + 1])
    return 0.05  # default

_BETA_MIN = _parse_beta_min()

from eval.online_fitters import OnlineDIFEFitter
OnlineDIFEFitter.BETA_BOUNDS = (_BETA_MIN, 1.0)
print(f"[patch] OnlineDIFEFitter.BETA_BOUNDS = {OnlineDIFEFitter.BETA_BOUNDS}")

from eval.config import make_bench_config, BenchConfig
from eval.runner import _load_data, _fresh_model, _grid_search_params
from eval.trainer import train_one_method
from eval.metrics import compute_all_metrics, save_metrics

# ─── Configuration ────────────────────────────────────────────────────────────
BENCH       = "split_cifar"
RMAX        = 0.30
SEEDS       = [0, 1, 2, 3, 4]
METHODS     = ["FT", "ConstReplay_0.1", "ConstReplay_0.3",
               "DIFE_only", "MV_only", "DIFE_MV"]
EPOCHS      = 3

def _output_root(beta_min: float) -> str:
    label = f"{beta_min:.2f}".replace(".", "")   # 0.05 -> "005", 0.10 -> "010"
    return os.path.join(_HERE, "results", f"canonical_beta{label}",
                        "split_cifar_rmax_0.30")


def run_one(seed: int, method: str, output_root: str, device: str = "cpu") -> None:
    out_dir   = os.path.join(output_root, method, f"seed_{seed}")
    out_path  = os.path.join(out_dir, "metrics.json")
    trace_path = os.path.join(out_dir, "controller_trace.csv")

    if os.path.exists(out_path):
        print(f"[skip] seed={seed} method={method} — already done")
        return

    print(f"\n{'='*60}")
    print(f"  BETA-RERUN: seed={seed} method={method} beta_min={_BETA_MIN}")
    print(f"  bench={BENCH}  epochs/task={EPOCHS}  r_max={RMAX}")
    print(f"{'='*60}")

    cfg = make_bench_config(BENCH, device=device)
    cfg.epochs_per_task = EPOCHS
    cfg.output_dir = output_root

    best_ewc_lam, best_si_c = _grid_search_params(BENCH, cfg)

    torch.manual_seed(seed)
    np.random.seed(seed)
    loaders = _load_data(BENCH, cfg, seed=seed)
    model   = _fresh_model(BENCH)

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
        n_classes_per_task=2,
        pre_task_acc=result.get("pre_task_acc", []),
    )
    metrics["acc_matrix"]          = result["acc_matrix"]
    metrics["r_t_history"]         = result["r_t_history"]
    metrics["mv_proxy_history"]    = result["mv_proxy_history"]
    metrics["dife_params_history"] = result["dife_params_history"]
    metrics["pre_task_acc"]        = result.get("pre_task_acc", [])
    metrics["canonical_config"] = {
        "bench": BENCH,
        "r_max": RMAX,
        "epochs_per_task": EPOCHS,
        "seed": seed,
        "method": method,
        "beta_min": _BETA_MIN,
        "code_branch": "canonical/beta-bound-rerun",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta-min", type=float, default=0.05, dest="beta_min")
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--seed",     type=int, default=None)
    parser.add_argument("--method",   default=None)
    args = parser.parse_args()

    # beta_min already patched at module load; confirm consistency
    assert abs(args.beta_min - _BETA_MIN) < 1e-9, \
        f"Mismatch: parsed {_BETA_MIN} at module load but args.beta_min={args.beta_min}"

    output_root = _output_root(args.beta_min)
    os.makedirs(output_root, exist_ok=True)

    seeds   = [args.seed]   if args.seed   is not None else SEEDS
    methods = [args.method] if args.method is not None else METHODS

    t_start = time.time()
    total   = len(seeds) * len(methods)
    done    = 0
    errors  = []

    for method in methods:
        for seed in seeds:
            try:
                run_one(seed, method, output_root, device=args.device)
            except Exception as e:
                msg = f"ERROR seed={seed} method={method}: {e}"
                print(msg)
                errors.append(msg)
            done += 1
            elapsed = time.time() - t_start
            print(f"  Progress: {done}/{total}  elapsed={elapsed:.0f}s")

    print(f"\nAll {total} jobs attempted in {time.time() - t_start:.1f}s")
    print(f"Results: {output_root}")
    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
