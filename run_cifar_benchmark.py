"""
Split-CIFAR-10 (3 tasks, 2 classes each) benchmark for DIFE.

Tasks:
  T1: airplane vs automobile  (classes 0, 1)
  T2: bird vs cat             (classes 2, 3)
  T3: deer vs dog             (classes 4, 5)

Architecture: SmallCNN  (3→32→64 conv + 2 FC, ~750 K params, CPU-friendly)
Methods: Fine-tuning, EWC (λ=5000), SI (c=0.1)
Epochs: 5 per task
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from benchmark.data import split_cifar10
from benchmark.models import fresh_cnn
from benchmark.baselines import train_finetuning, train_ewc, train_si, evaluate
from benchmark.fitting import fit_dife, compute_metrics
from benchmark.plotting import (plot_forgetting_curves,
                                 plot_method_comparison,
                                 plot_accuracy_heatmap)
from dife import dife

RESULTS_DIR = "results/cifar"
N_TASKS = 3
EPOCHS = 5
LR = 1e-3
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

TASK_NAMES = ["airplane/auto", "bird/cat", "deer/dog"]


def header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_acc_matrix(name, acc_matrix):
    T = len(acc_matrix)
    print(f"\n  {name} accuracy matrix:")
    hdr = "       " + "  ".join(f"T{j+1:>5}" for j in range(T))
    print(hdr)
    for t, row in enumerate(acc_matrix):
        row_str = "  ".join(f"{v:.4f}" for v in row)
        print(f"  t={t+1}  {row_str}")


def print_metrics_table(all_metrics):
    print("\n  ┌─────────────┬────────────────┬─────────────────┬────────────┐")
    print("  │ Method      │ Avg Final Acc  │ Avg Forgetting  │    BWT     │")
    print("  ├─────────────┼────────────────┼─────────────────┼────────────┤")
    for method, m in all_metrics.items():
        print(f"  │ {method:<11} │ {m['avg_final_acc']:.4f}         │"
              f" {m['avg_forgetting']:.4f}           │ {m['bwt']:+.4f}     │")
    print("  └─────────────┴────────────────┴─────────────────┴────────────┘")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    header("Step 1 / 4 — Loading Split-CIFAR-10 (3 tasks)")
    # Use n_tasks=5 so each task has exactly 2 classes (output_dim=2).
    # We run only the first 3 tasks to keep CPU runtime manageable.
    task_loaders = split_cifar10(n_tasks=5, data_root="./data")[:N_TASKS]
    for i, (tr, te) in enumerate(task_loaders):
        print(f"  Task {i+1} ({TASK_NAMES[i]}): "
              f"{len(tr.dataset)} train / {len(te.dataset)} test")

    header("Step 2 / 4 — Training baselines")

    print("\n--- Fine-tuning ---")
    model_ft = fresh_cnn(output_dim=2)
    ft_matrix = train_finetuning(model_ft, task_loaders, epochs=EPOCHS, lr=LR)
    print_acc_matrix("FT", ft_matrix)

    print("\n--- EWC ---")
    model_ewc = fresh_cnn(output_dim=2)
    ewc_matrix = train_ewc(model_ewc, task_loaders, epochs=EPOCHS, lr=LR, lam=5000)
    print_acc_matrix("EWC", ewc_matrix)

    print("\n--- SI ---")
    model_si = fresh_cnn(output_dim=2)
    si_matrix = train_si(model_si, task_loaders, epochs=EPOCHS, lr=LR, c=0.1)
    print_acc_matrix("SI", si_matrix)

    acc_matrices = {"FT": ft_matrix, "EWC": ewc_matrix, "SI": si_matrix}

    header("Step 3 / 4 — Fitting DIFE")
    fit_ft  = fit_dife(ft_matrix)
    fit_ewc = fit_dife(ewc_matrix)
    fit_si  = fit_dife(si_matrix)

    print(f"\n  FT  fit: α={fit_ft['alpha']:.5f}  β={fit_ft['beta']:.5f}"
          f"  RMSE={fit_ft['rmse']:.5f}  (n={len(fit_ft['obs'])} obs)")
    print(f"  EWC fit: α={fit_ewc['alpha']:.5f}  β={fit_ewc['beta']:.5f}"
          f"  RMSE={fit_ewc['rmse']:.5f}")
    print(f"  SI  fit: α={fit_si['alpha']:.5f}  β={fit_si['beta']:.5f}"
          f"  RMSE={fit_si['rmse']:.5f}")

    # Residual detail for FT
    obs = fit_ft["obs"]
    alpha, beta = fit_ft["alpha"], fit_ft["beta"]
    print(f"\n  FT residuals:")
    print(f"  {'Q0':>6}  {'n':>3}  {'Q_obs':>7}  {'Q_pred':>7}  {'resid':>8}")
    for Q0, n, Q_obs in obs:
        pred = dife(n, Q_0=Q0, alpha=alpha, beta=beta)
        print(f"  {Q0:.4f}  {n:>3}  {Q_obs:.5f}  {pred:.5f}  {pred-Q_obs:+.5f}")

    header("Step 4 / 4 — Metrics & plots")
    all_metrics = {m: compute_metrics(mat) for m, mat in acc_matrices.items()}
    print_metrics_table(all_metrics)

    print("\n  Generating figures …")
    p1 = plot_forgetting_curves(acc_matrices, fit_ft, "Split-CIFAR-10", RESULTS_DIR)
    p2 = plot_method_comparison(acc_matrices, "Split-CIFAR-10", RESULTS_DIR)
    p3 = plot_accuracy_heatmap(acc_matrices, "Split-CIFAR-10", RESULTS_DIR)

    summary = {
        "dataset": "Split-CIFAR-10",
        "n_tasks": N_TASKS,
        "task_names": TASK_NAMES,
        "epochs_per_task": EPOCHS,
        "metrics": all_metrics,
        "dife_fit_ft":  {"alpha": fit_ft["alpha"],  "beta": fit_ft["beta"],  "rmse": fit_ft["rmse"]},
        "dife_fit_ewc": {"alpha": fit_ewc["alpha"], "beta": fit_ewc["beta"], "rmse": fit_ewc["rmse"]},
        "dife_fit_si":  {"alpha": fit_si["alpha"],  "beta": fit_si["beta"],  "rmse": fit_si["rmse"]},
        "acc_matrices": {m: mat for m, mat in acc_matrices.items()},
        "figures": [p1, p2, p3],
    }
    json_path = os.path.join(RESULTS_DIR, "cifar_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {json_path}")
    print(f"\n  Total time: {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
