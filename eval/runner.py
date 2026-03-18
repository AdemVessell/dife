"""Orchestrates seeds, methods, and benchmarks for the full evaluation suite."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory-vortex-dife-lab"))

import torch
import numpy as np

from benchmark.data import permuted_mnist, split_cifar10
from benchmark.models import fresh_mlp, fresh_cnn
from eval.config import make_bench_config
from eval.grid_search import find_best_ewc_lambda, find_best_si_c
from eval.trainer import train_one_method
from eval.metrics import compute_all_metrics, save_metrics, write_summary_csv
from eval.plotting_ext import generate_all_plots

ALL_METHODS = [
    "FT", "EWC", "SI",
    "ConstReplay_0.1", "ConstReplay_0.3", "RandReplay",
    "DIFE_only", "MV_only", "DIFE_MV",
]

CANONICAL_SEED = 42


def _load_data(bench_name: str, cfg, seed: int) -> list:
    """Load task data loaders for the given benchmark and seed."""
    if bench_name == "perm_mnist":
        return permuted_mnist(
            n_tasks=cfg.n_tasks,
            batch_size=cfg.batch_size,
            data_root=cfg.data_root,
            seed=seed,
        )
    else:
        return split_cifar10(
            n_tasks=5,
            batch_size=cfg.batch_size,
            data_root=cfg.data_root,
        )[: cfg.n_tasks]


def _fresh_model(bench_name: str):
    """Return a freshly initialised model for the given benchmark."""
    if bench_name == "perm_mnist":
        return fresh_mlp()
    else:
        return fresh_cnn(output_dim=2)


def run_benchmark(bench_name: str, cfg, n_seeds: int) -> dict:
    """Run all methods × all seeds for one benchmark.

    Returns:
        Nested dict: {method: {seed: metrics_dict}}
    """
    seeds = list(range(n_seeds))

    # Grid search using seed 42 (run once, shared across all seeds)
    print(f"\n[{bench_name}] Running hyperparameter grid search (seed={CANONICAL_SEED})...")
    torch.manual_seed(CANONICAL_SEED)
    np.random.seed(CANONICAL_SEED)
    loaders_gs = _load_data(bench_name, cfg, seed=CANONICAL_SEED)

    model_factory = lambda: _fresh_model(bench_name)
    best_ewc_lam = find_best_ewc_lambda(
        loaders_gs, cfg.ewc_lambdas, cfg.epochs_per_task, cfg.lr, model_factory
    )
    best_si_c = find_best_si_c(
        loaders_gs, cfg.si_cs, cfg.epochs_per_task, cfg.lr, model_factory
    )
    print(f"[{bench_name}] Best EWC lambda={best_ewc_lam}, SI c={best_si_c}")

    all_results = {m: {} for m in ALL_METHODS}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        loaders = _load_data(bench_name, cfg, seed=seed)

        for method in ALL_METHODS:
            path = os.path.join(
                cfg.output_dir, bench_name, method, f"seed_{seed}", "metrics.json"
            )
            if os.path.exists(path):
                print(f"\n[{bench_name}] seed={seed}  method={method} — skipping (exists)")
                with open(path) as f:
                    all_results[method][seed] = json.load(f)
                continue

            print(f"\n[{bench_name}] seed={seed}  method={method}")
            model = _fresh_model(bench_name)
            result = train_one_method(
                method=method,
                model=model,
                task_loaders=loaders,
                cfg=cfg,
                seed=seed,
                best_ewc_lam=best_ewc_lam,
                best_si_c=best_si_c,
            )

            n_classes = 10 if bench_name == "perm_mnist" else 2
            metrics = compute_all_metrics(
                acc_matrix=result["acc_matrix"],
                r_t_history=result["r_t_history"],
                total_replay_samples=result["total_replay_samples"],
                wall_clock=result["wall_clock_seconds"],
                n_classes_per_task=n_classes,
                pre_task_acc=result.get("pre_task_acc", []),
            )
            # Attach raw histories for plotting
            metrics["acc_matrix"] = result["acc_matrix"]
            metrics["r_t_history"] = result["r_t_history"]
            metrics["mv_proxy_history"] = result["mv_proxy_history"]
            metrics["dife_params_history"] = result["dife_params_history"]
            metrics["pre_task_acc"] = result.get("pre_task_acc", [])

            path = os.path.join(
                cfg.output_dir, bench_name, method, f"seed_{seed}", "metrics.json"
            )
            save_metrics(metrics, path)
            all_results[method][seed] = metrics
            print(
                f"  [{bench_name}] {method} seed={seed} "
                f"AA={metrics['avg_final_acc']:.3f} "
                f"AF={metrics['avg_forgetting']:.3f} "
                f"BWT={metrics['bwt']:.3f} "
                f"FWT={metrics['fwt']:.3f}"
            )

    return all_results


def build_summary_rows(all_results: dict) -> list:
    """Build summary row dicts for all methods from all_results.

    Returns:
        list of dicts matching the CSV fieldname schema.
    """
    metric_keys = [
        ("avg_final_acc", "AA"),
        ("avg_forgetting", "AF"),
        ("bwt", "BWT"),
        ("fwt", "FWT"),
        ("total_replay_samples", "replay_budget"),
    ]
    rows = []
    for method, seed_map in all_results.items():
        seed_metrics = list(seed_map.values())
        row = {"method": method}
        for key, col in metric_keys:
            vals = [float(m.get(key, 0)) for m in seed_metrics]
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals))
        rows.append(row)
    return rows
