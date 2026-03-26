"""Orchestrates seeds, methods, and benchmarks for the full evaluation suite."""

import sys
import os
import json
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory-vortex-dife-lab"))

import torch
import numpy as np

from benchmark.data import permuted_mnist, split_cifar10, split_cifar100
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
    "MIR",
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
    elif bench_name == "split_cifar100":
        return split_cifar100(
            n_tasks=cfg.n_tasks,
            batch_size=cfg.batch_size,
            data_root=cfg.data_root,
        )
    else:
        return split_cifar10(
            n_tasks=5,
            batch_size=cfg.batch_size,
            data_root=cfg.data_root,
        )[: cfg.n_tasks]


def _fresh_model(bench_name: str, cfg=None):
    """Return a freshly initialised model for the given benchmark."""
    if bench_name == "perm_mnist":
        return fresh_mlp()
    n_out = cfg.n_classes_per_task if cfg is not None else 2
    return fresh_cnn(output_dim=n_out)


def _grid_search_params(bench_name: str, cfg) -> tuple:
    """Return (best_ewc_lam, best_si_c), loading from cache if available."""
    cache_path = os.path.join(cfg.output_dir, bench_name, "grid_search_params.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            p = json.load(f)
        print(f"[{bench_name}] Loaded cached grid search params: EWC lambda={p['ewc_lam']}, SI c={p['si_c']}")
        return p["ewc_lam"], p["si_c"]

    print(f"\n[{bench_name}] Running hyperparameter grid search (seed={CANONICAL_SEED})...")
    torch.manual_seed(CANONICAL_SEED)
    np.random.seed(CANONICAL_SEED)
    loaders_gs = _load_data(bench_name, cfg, seed=CANONICAL_SEED)
    model_factory = lambda: _fresh_model(bench_name, cfg)
    best_ewc_lam = find_best_ewc_lambda(
        loaders_gs, cfg.ewc_lambdas, cfg.epochs_per_task, cfg.lr, model_factory
    )
    best_si_c = find_best_si_c(
        loaders_gs, cfg.si_cs, cfg.epochs_per_task, cfg.lr, model_factory
    )
    del loaders_gs
    gc.collect()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"ewc_lam": best_ewc_lam, "si_c": best_si_c}, f)
    print(f"[{bench_name}] Best EWC lambda={best_ewc_lam}, SI c={best_si_c}")
    return best_ewc_lam, best_si_c


def run_benchmark(bench_name: str, cfg, n_seeds: int) -> dict:
    """Run all methods × all seeds for one benchmark.

    Returns:
        Nested dict: {method: {seed: metrics_dict}}
    """
    seeds = list(range(n_seeds))
    best_ewc_lam, best_si_c = _grid_search_params(bench_name, cfg)

    all_results = {m: {} for m in ALL_METHODS}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        loaders = _load_data(bench_name, cfg, seed=seed)

        for method in ALL_METHODS:
            out_path = os.path.join(
                cfg.output_dir, bench_name, method, f"seed_{seed}", "metrics.json"
            )
            if os.path.exists(out_path):
                print(f"\n[{bench_name}] seed={seed}  method={method} — skipping (exists)")
                with open(out_path) as f:
                    all_results[method][seed] = json.load(f)
                continue

            print(f"\n[{bench_name}] seed={seed}  method={method}")
            model = _fresh_model(bench_name, cfg)
            result = train_one_method(
                method=method,
                model=model,
                task_loaders=loaders,
                cfg=cfg,
                seed=seed,
                best_ewc_lam=best_ewc_lam,
                best_si_c=best_si_c,
            )

            metrics = compute_all_metrics(
                acc_matrix=result["acc_matrix"],
                r_t_history=result["r_t_history"],
                total_replay_samples=result["total_replay_samples"],
                wall_clock=result["wall_clock_seconds"],
                n_classes_per_task=cfg.n_classes_per_task,
                pre_task_acc=result.get("pre_task_acc", []),
            )
            metrics["acc_matrix"] = result["acc_matrix"]
            metrics["r_t_history"] = result["r_t_history"]
            metrics["mv_proxy_history"] = result["mv_proxy_history"]
            metrics["dife_params_history"] = result["dife_params_history"]
            metrics["pre_task_acc"] = result.get("pre_task_acc", [])

            save_metrics(metrics, out_path)
            all_results[method][seed] = metrics
            print(
                f"  [{bench_name}] {method} seed={seed} "
                f"AA={metrics['avg_final_acc']:.3f} "
                f"AF={metrics['avg_forgetting']:.3f} "
                f"BWT={metrics['bwt']:.3f} "
                f"FWT={metrics['fwt']:.3f}"
            )

            # Free model and result tensors between jobs
            del model, result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del loaders
        gc.collect()

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
