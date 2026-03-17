"""
Seq-MNIST (Permuted MNIST, 5 tasks) benchmark for DIFE.

Runs Fine-tuning, EWC, and SI baselines, fits DIFE to FT forgetting
trajectories, and produces tables, plots, and residual statistics.
"""

import copy
import json
import os
import sys
import time

import numpy as np
import torch

# ── Make sure repo root is on path ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from benchmark.data import permuted_mnist
from benchmark.models import fresh_mlp
from benchmark.baselines import train_finetuning, train_ewc, train_si, evaluate
from benchmark.fitting import fit_dife, compute_metrics
from benchmark.plotting import (plot_forgetting_curves,
                                 plot_method_comparison,
                                 plot_accuracy_heatmap)
from dife import dife

RESULTS_DIR = "results/mnist"
N_TASKS = 5
EPOCHS = 5
LR = 1e-3
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Helpers ──────────────────────────────────────────────────────────────────

def header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_acc_matrix(name, acc_matrix):
    T = len(acc_matrix)
    print(f"\n  {name} accuracy matrix (row = after task t, col = task j):")
    header_row = "       " + "  ".join(f"T{j+1:>4}" for j in range(T))
    print(header_row)
    for t, row in enumerate(acc_matrix):
        row_str = "  ".join(f"{v:.4f}" if v is not None else "  -   " for v in row)
        print(f"  t={t+1}  {row_str}")


def print_metrics_table(all_metrics):
    print("\n  ┌─────────────┬────────────────┬─────────────────┬────────────┐")
    print("  │ Method      │ Avg Final Acc  │ Avg Forgetting  │    BWT     │")
    print("  ├─────────────┼────────────────┼─────────────────┼────────────┤")
    for method, m in all_metrics.items():
        print(f"  │ {method:<11} │ {m['avg_final_acc']:.4f}         │ {m['avg_forgetting']:.4f}           │ {m['bwt']:+.4f}     │")
    print("  └─────────────┴────────────────┴─────────────────┴────────────┘")


def print_fit_result(fit):
    print(f"\n  DIFE fit:  α = {fit['alpha']:.5f}   β = {fit['beta']:.5f}   RMSE = {fit['rmse']:.5f}")
    print(f"  Fitted on {len(fit['obs'])} (Q0, n, Q_obs) observations")


def print_residual_stats(fit, method_name="FT"):
    """Print residual table and stats."""
    alpha, beta = fit["alpha"], fit["beta"]
    obs = fit["obs"]
    print(f"\n  Residuals (DIFE prediction vs {method_name} observed):")
    print(f"  {'Q0':>6}  {'n':>3}  {'Q_obs':>7}  {'Q_pred':>7}  {'resid':>8}")
    residuals = []
    for Q0, n, Q_obs in obs:
        pred = dife(n, Q_0=Q0, alpha=alpha, beta=beta)
        resid = pred - Q_obs
        residuals.append(resid)
        print(f"  {Q0:.4f}  {n:>3}  {Q_obs:.5f}  {pred:.5f}  {resid:+.5f}")
    residuals = np.array(residuals)
    print(f"\n  Mean error: {residuals.mean():+.5f}")
    print(f"  RMSE:       {np.sqrt((residuals**2).mean()):.5f}")
    print(f"  Max |error|:{np.abs(residuals).max():.5f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    header("Step 1 / 4 — Loading Permuted MNIST (5 tasks)")
    task_loaders = permuted_mnist(n_tasks=N_TASKS, seed=SEED)
    print(f"  {N_TASKS} tasks loaded (each: 60 000 train / 10 000 test images)")

    # ── Fine-tuning ──────────────────────────────────────────────────────────
    header("Step 2 / 4 — Training baselines")
    print("\n--- Fine-tuning (no regularisation) ---")
    model_ft = fresh_mlp()
    ft_matrix = train_finetuning(model_ft, task_loaders, epochs=EPOCHS, lr=LR)
    print_acc_matrix("FT", ft_matrix)

    # ── EWC ─────────────────────────────────────────────────────────────────
    print("\n--- EWC ---")
    model_ewc = fresh_mlp()
    ewc_matrix = train_ewc(model_ewc, task_loaders, epochs=EPOCHS, lr=LR, lam=5000)
    print_acc_matrix("EWC", ewc_matrix)

    # ── SI ──────────────────────────────────────────────────────────────────
    print("\n--- SI ---")
    model_si = fresh_mlp()
    si_matrix = train_si(model_si, task_loaders, epochs=EPOCHS, lr=LR, c=0.1)
    print_acc_matrix("SI", si_matrix)

    acc_matrices = {"FT": ft_matrix, "EWC": ewc_matrix, "SI": si_matrix}

    # ── DIFE fit ─────────────────────────────────────────────────────────────
    header("Step 3 / 4 — Fitting DIFE to FT forgetting trajectories")
    print("  Running differential evolution + Nelder-Mead optimisation …")
    fit = fit_dife(ft_matrix)
    print_fit_result(fit)
    print_residual_stats(fit, method_name="FT")

    # ── Metrics ──────────────────────────────────────────────────────────────
    header("Step 4 / 4 — Metrics & plots")
    all_metrics = {m: compute_metrics(mat) for m, mat in acc_matrices.items()}
    print_metrics_table(all_metrics)

    # Also report DIFE residuals vs EWC and SI for comparison
    fit_ewc = fit_dife(ewc_matrix)
    fit_si  = fit_dife(si_matrix)
    print(f"\n  EWC forgetting – DIFE RMSE: {fit_ewc['rmse']:.5f}  "
          f"(α={fit_ewc['alpha']:.4f}, β={fit_ewc['beta']:.5f})")
    print(f"  SI  forgetting – DIFE RMSE: {fit_si['rmse']:.5f}  "
          f"(α={fit_si['alpha']:.4f}, β={fit_si['beta']:.5f})")

    # Plots
    print("\n  Generating figures …")
    p1 = plot_forgetting_curves(acc_matrices, fit, "Seq-MNIST", RESULTS_DIR)
    p2 = plot_method_comparison(acc_matrices, "Seq-MNIST", RESULTS_DIR)
    p3 = plot_accuracy_heatmap(acc_matrices, "Seq-MNIST", RESULTS_DIR)

    # Persist results
    summary = {
        "dataset": "Permuted-MNIST",
        "n_tasks": N_TASKS,
        "epochs_per_task": EPOCHS,
        "metrics": all_metrics,
        "dife_fit": {"alpha": fit["alpha"], "beta": fit["beta"], "rmse": fit["rmse"]},
        "dife_fit_ewc": {"alpha": fit_ewc["alpha"], "beta": fit_ewc["beta"], "rmse": fit_ewc["rmse"]},
        "dife_fit_si":  {"alpha": fit_si["alpha"],  "beta": fit_si["beta"],  "rmse": fit_si["rmse"]},
        "acc_matrices": {m: mat for m, mat in acc_matrices.items()},
        "figures": [p1, p2, p3],
    }
    json_path = os.path.join(RESULTS_DIR, "mnist_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {json_path}")
    print(f"\n  Total time: {time.time() - t0:.1f}s")

    return summary


if __name__ == "__main__":
    main()
