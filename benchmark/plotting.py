"""Plotting utilities for DIFE benchmark results."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dife import dife, dife_curve


# Consistent palette
COLORS = {
    "FT": "#e05c5c",
    "EWC": "#5c9ee0",
    "SI": "#5cc47a",
    "DIFE": "#e0a85c",
}


def plot_forgetting_curves(acc_matrices, fit_result, task_name, out_dir):
    """Plot per-task forgetting curves: observed vs DIFE fit.

    acc_matrices: dict method -> acc_matrix (lower-triangular list of lists)
    fit_result: output of fit_dife() applied to FT acc_matrix
    """
    os.makedirs(out_dir, exist_ok=True)
    alpha = fit_result["alpha"]
    beta = fit_result["beta"]
    ft_matrix = acc_matrices["FT"]
    T = len(ft_matrix)

    fig, axes = plt.subplots(1, T - 1, figsize=(4 * (T - 1), 3.5), sharey=True)
    if T - 1 == 1:
        axes = [axes]

    for j in range(T - 1):
        ax = axes[j]
        Q0 = ft_matrix[j][j]
        n_steps = T - j - 1
        xs = np.arange(0, n_steps + 1)

        # Observed FT forgetting
        observed = [ft_matrix[j + n][j] for n in range(n_steps + 1)]
        ax.plot(xs, observed, "o-", color=COLORS["FT"],
                label="FT (observed)", linewidth=2)

        # DIFE prediction
        predicted = [dife(n, Q_0=Q0, alpha=alpha, beta=beta) for n in xs]
        ax.plot(xs, predicted, "--", color=COLORS["DIFE"],
                label=f"DIFE fit (α={alpha:.3f}, β={beta:.4f})", linewidth=2)

        ax.set_title(f"Task {j+1}", fontsize=11)
        ax.set_xlabel("Tasks learned since")
        if j == 0:
            ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{task_name} – Per-task forgetting: FT vs DIFE fit\n"
                 f"RMSE = {fit_result['rmse']:.4f}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{task_name.lower().replace(' ', '_')}_forgetting_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_method_comparison(acc_matrices, task_name, out_dir):
    """Bar chart comparing avg final accuracy across methods."""
    os.makedirs(out_dir, exist_ok=True)
    from benchmark.fitting import compute_metrics

    methods = list(acc_matrices.keys())
    metrics = {m: compute_metrics(acc_matrices[m]) for m in methods}

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    metric_keys = ["avg_final_acc", "avg_forgetting", "bwt"]
    titles = ["Avg Final Accuracy ↑", "Avg Forgetting ↓", "BWT ↑"]
    good_high = [True, False, True]   # higher is better?

    for ax, key, title, hi in zip(axes, metric_keys, titles, good_high):
        vals = [metrics[m][key] for m in methods]
        bars = ax.bar(methods, vals,
                      color=[COLORS.get(m, "#999999") for m in methods],
                      edgecolor="black", linewidth=0.7)
        best_val = max(vals) if hi else min(vals)
        for bar, v in zip(bars, vals):
            h = bar.get_height()
            sign = "+" if v >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.005, f"{sign}{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
            if v == best_val:
                bar.set_edgecolor("gold")
                bar.set_linewidth(2.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(min(vals) - 0.05, max(vals) + 0.08)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{task_name} – Method Comparison", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{task_name.lower().replace(' ', '_')}_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_accuracy_heatmap(acc_matrices, task_name, out_dir):
    """Heatmap of the accuracy matrices side-by-side."""
    os.makedirs(out_dir, exist_ok=True)
    methods = list(acc_matrices.keys())
    T = len(list(acc_matrices.values())[0])

    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.5))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        matrix = acc_matrices[method]
        mat = np.full((T, T), np.nan)
        for t, row in enumerate(matrix):
            for j, v in enumerate(row):
                mat[t][j] = v
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_title(method, fontsize=11)
        ax.set_xlabel("Task j (eval)")
        ax.set_ylabel("After training task t")
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        for t in range(T):
            for j in range(T):
                if not np.isnan(mat[t][j]):
                    ax.text(j, t, f"{mat[t][j]:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="black")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"{task_name} – Accuracy matrices", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{task_name.lower().replace(' ', '_')}_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
