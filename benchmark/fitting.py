"""Fit DIFE parameters (alpha, beta) to an observed accuracy matrix.

The accuracy matrix acc_matrix[t][j] gives the accuracy on task j
immediately after training on task t (t >= j).

For each task j, the forgetting trajectory is:
    Q_observed[n] = acc_matrix[j + n][j]  for n = 0, 1, ..., T - j - 1

We fit a single (alpha, beta) across all tasks using least-squares.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution

from dife import dife


def _build_observations(acc_matrix):
    """Return list of (Q0, n, Q_observed) triples from the acc matrix."""
    T = len(acc_matrix)
    obs = []
    for j in range(T):
        Q0 = acc_matrix[j][j]          # accuracy right after learning task j
        for n in range(1, T - j):      # steps of additional tasks learned
            Q_obs = acc_matrix[j + n][j]
            obs.append((Q0, n, Q_obs))
    return obs


def _residuals_sq(params, obs):
    alpha, beta = params
    if not (0 < alpha < 1) or beta <= 0:
        return 1e9
    total = 0.0
    for Q0, n, Q_obs in obs:
        pred = dife(n, Q_0=Q0, alpha=alpha, beta=beta)
        total += (pred - Q_obs) ** 2
    return total


def fit_dife(acc_matrix):
    """Fit alpha and beta to the observed forgetting trajectories.

    Uses differential evolution (global) followed by Nelder-Mead (local).

    Returns:
        dict with keys: alpha, beta, rmse, obs (list of observation triples)
    """
    obs = _build_observations(acc_matrix)
    if not obs:
        return {"alpha": 0.95, "beta": 0.01, "rmse": float("nan"), "obs": obs}

    # Global search
    bounds = [(0.01, 0.9999), (1e-6, 5.0)]
    result_de = differential_evolution(_residuals_sq, bounds, args=(obs,),
                                       seed=42, maxiter=500, tol=1e-8,
                                       workers=1, polish=True)
    alpha, beta = result_de.x
    rmse = np.sqrt(result_de.fun / len(obs))
    return {"alpha": float(alpha), "beta": float(beta),
            "rmse": float(rmse), "obs": obs}


def compute_metrics(acc_matrix):
    """Compute standard CL metrics from a lower-triangular accuracy matrix.

    acc_matrix is a list-of-lists where acc_matrix[t] has t+1 entries.

    Returns dict with:
        avg_final_acc  – average accuracy on all tasks after the last task
        avg_forgetting – average max-accuracy drop
        bwt            – backward transfer (can be negative = forgetting)
    """
    T = len(acc_matrix)
    # Pad to T x T with None for missing entries
    full = [[None] * T for _ in range(T)]
    for t, row in enumerate(acc_matrix):
        for j, v in enumerate(row):
            full[t][j] = v

    # Final accuracies (after task T-1)
    final = [full[T - 1][j] for j in range(T)]
    avg_final_acc = float(np.mean(final))

    # Forgetting: for each j, max accuracy achieved - final accuracy
    forgetting = []
    for j in range(T - 1):
        best = max(full[t][j] for t in range(j, T) if full[t][j] is not None)
        forgetting.append(best - full[T - 1][j])
    avg_forgetting = float(np.mean(forgetting)) if forgetting else 0.0

    # BWT: average change in accuracy on old tasks (negative = catastrophic forgetting)
    bwt_vals = []
    for j in range(T - 1):
        bwt_vals.append(full[T - 1][j] - full[j][j])
    bwt = float(np.mean(bwt_vals)) if bwt_vals else 0.0

    return {"avg_final_acc": avg_final_acc,
            "avg_forgetting": avg_forgetting,
            "bwt": bwt}
