"""Unified training loop for all continual learning methods."""

import sys
import os
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory-vortex-dife-lab"))

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from benchmark.baselines import EWC, SI, evaluate
from eval.buffer import ReservoirBuffer
from eval.online_fitters import OnlineDIFEFitter, OnlineMVFitter
from eval.schedulers import get_replay_fraction, SchedulerState

# Controller trace CSV field names (written once per epoch per task)
_TRACE_FIELDS = [
    "seed", "method", "task_id", "epoch_in_task", "global_epoch",
    "alpha_fit", "beta_fit",
    "dife_envelope_value",
    "mv_proxy_value", "mv_operator_value",
    "final_replay_fraction_before_cap", "final_replay_fraction_after_cap",
    "replay_samples_this_epoch", "cumulative_replay_samples",
    "has_mv_fit", "r_max", "gamma",
]

# Methods that use regularization instead of replay
REGULARIZER_METHODS = {"FT", "EWC", "SI"}


def _get_input_shape(task_loaders) -> tuple:
    x, _ = next(iter(task_loaders[0][0]))
    return tuple(x.shape[1:])


def _mir_select(model, buffer, x_batch, y_batch, k, lr, loss_fn, n_candidates=200):
    """MIR (Maximally Interfered Retrieval) sample selection.

    Selects the k buffer samples that suffer the highest loss increase after a
    virtual gradient step on the current batch — i.e. the samples most
    "interfered with" by the upcoming update.

    Algorithm (Aljundi et al., NeurIPS 2019):
      1. Sample n_candidates from buffer randomly.
      2. Compute per-sample loss under current parameters: L(x; θ).
      3. Take a virtual gradient step on x_batch: θ_v = θ - lr * ∇L(x_batch; θ).
      4. Compute per-sample loss under virtual params: L(x; θ_v).
      5. Select top-k by interference = L(x; θ_v) - L(x; θ).
      6. Restore original parameters.
    """
    if buffer.size() == 0 or k == 0:
        return torch.zeros(0), torch.zeros(0, dtype=torch.long)

    n_cands = min(n_candidates, buffer.size())
    x_cands, y_cands = buffer.sample(n_cands)
    k = min(k, n_cands)

    # --- Step 1: loss under current params ---
    _ce_none = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    with torch.no_grad():
        loss_before = _ce_none(model(x_cands), y_cands)  # (n_cands,)

    # --- Steps 2-3: virtual gradient step on current batch ---
    model.train()
    saved_params = [p.data.clone() for p in model.parameters()]

    # Clear any stale gradients, compute gradient from current batch
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    virtual_loss = loss_fn(model(x_batch), y_batch)
    virtual_loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data.sub_(lr * p.grad)

    # --- Step 4: loss under virtual params ---
    model.eval()
    with torch.no_grad():
        loss_after = _ce_none(model(x_cands), y_cands)  # (n_cands,)

    # --- Step 5: restore params and clear gradients ---
    with torch.no_grad():
        for p, saved in zip(model.parameters(), saved_params):
            p.data.copy_(saved)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    model.train()

    # --- Step 6: select top-k by interference score ---
    scores = loss_after - loss_before
    top_idx = torch.topk(scores, k).indices
    return x_cands[top_idx], y_cands[top_idx]


def _compute_mv_proxy(model, buffer, n_samples: int) -> float:
    """Proxy forgetting signal = 1 - accuracy on n_samples from buffer."""
    x, y = buffer.sample(n_samples)
    if len(x) == 0:
        return 0.0
    loader = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=False)
    acc = evaluate(model, loader)
    return float(1.0 - acc)


def train_one_method(
    method: str,
    model: nn.Module,
    task_loaders: list,
    cfg,
    seed: int,
    best_ewc_lam: float,
    best_si_c: float,
    r_max: float = None,
    gamma: float = 1.0,
    injected_task_budgets: list = None,
    trace_path: str = None,
) -> dict:
    """Train model sequentially through all tasks with the given method.

    For MV-based methods (MV_only, DIFE_MV), replay is modulated per-epoch
    once the MV operator has been fitted (mv_fitter.has_fit is True).

    Per-epoch MV modulation:
      - DIFE sets the task-level envelope: r_dife = DIFE(task_index)
      - Each epoch, MV predicts: r_mv = MV_operator(global_epoch)
      - DIFE_MV per-epoch rate: r_epoch = r_dife × r_mv
      - MV_only per-epoch rate: r_epoch = r_mv
      - Before MV fits (cold-start): falls back to task-level r_t

    Returns dict with:
        acc_matrix, r_t_history, total_replay_samples, wall_clock_seconds,
        mv_proxy_history, dife_params_history, pre_task_acc
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)

    # Open controller trace file if requested
    _trace_file = None
    _trace_writer = None
    if trace_path is not None:
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        _trace_file = open(trace_path, "w", newline="")
        _trace_writer = csv.DictWriter(_trace_file, fieldnames=_TRACE_FIELDS)
        _trace_writer.writeheader()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    # Regularizer setup
    ewc_obj = EWC(model, lam=best_ewc_lam) if method == "EWC" else None
    si_obj = SI(model, c=best_si_c) if method == "SI" else None

    # Replay buffer setup
    uses_replay = method not in REGULARIZER_METHODS
    input_shape = _get_input_shape(task_loaders)
    buffer = ReservoirBuffer(cfg.buffer_capacity, input_shape) if uses_replay else None

    # Online fitter setup
    dife_fitter = OnlineDIFEFitter() if method in ("DIFE_only", "DIFE_MV") else None
    mv_fitter = OnlineMVFitter() if method in ("MV_only", "DIFE_MV") else None
    # DIFE_flatMatched uses replay buffer but no online fitters (injected budgets)
    if method == "DIFE_flatMatched":
        uses_replay = True

    acc_matrix = []
    r_t_history = []
    pre_task_acc = []
    total_replay_samples = 0
    replay_per_task = []
    mv_proxy_history = []
    dife_params_history = []
    global_epoch = 0

    for t, (train_loader, _) in enumerate(task_loaders):

        # Record zero-shot accuracy on task t before training (for FWT)
        if t > 0:
            _, test_t = task_loaders[t]
            pre_acc = evaluate(model, test_t)
            pre_task_acc.append(pre_acc)

        # Determine task-level r_t BEFORE training (uses fitted params from t-1)
        state = SchedulerState(
            task_index=t,
            total_epochs_so_far=global_epoch,
            dife_fitter=dife_fitter,
            mv_fitter=mv_fitter,
            rng=rng,
            r_max=r_max,
        )
        r_t = get_replay_fraction(method, state)
        if r_max is not None:
            r_t = float(np.clip(gamma * r_t, 0.0, r_max))
        r_t_history.append(r_t)

        # Task-level replay count (fallback when MV hasn't fit yet)
        n_replay_per_batch = int(r_t * train_loader.batch_size)

        # DIFE_flatMatched: override with injected flat per-batch count
        if method == "DIFE_flatMatched" and injected_task_budgets is not None and t < len(injected_task_budgets):
            n_batches = len(train_loader)
            denom = max(n_batches * cfg.epochs_per_task, 1)
            n_replay_per_batch = injected_task_budgets[t] // denom

        _task_replay = 0

        # Per-epoch MV modulation is active once the operator has been fitted.
        # Before that, we use the task-level n_replay_per_batch as the fallback.
        uses_per_epoch_mv = (mv_fitter is not None and mv_fitter.has_fit)

        # Training epochs
        for epoch_local in range(cfg.epochs_per_task):
            model.train()

            # Compute trace values before deciding replay rate
            _alpha_fit = dife_fitter.alpha if dife_fitter is not None else float("nan")
            _beta_fit = dife_fitter.beta if dife_fitter is not None else float("nan")
            _dife_envelope = (
                float(np.clip(dife_fitter.replay_fraction(t), 0.0, 1.0))
                if dife_fitter is not None else float("nan")
            )
            _mv_op_val = (
                float(mv_fitter._operator(global_epoch))
                if mv_fitter is not None else float("nan")
            )

            # Per-epoch replay rate: MV modulates within the task
            if uses_per_epoch_mv:
                r_mv = mv_fitter.replay_fraction(global_epoch)
                _mv_op_val = float(mv_fitter._operator(global_epoch))
                if method == "DIFE_MV" and dife_fitter is not None:
                    # Scale MV signal by the DIFE task-level envelope
                    _before_cap = r_mv * _dife_envelope
                    r_mv = _before_cap
                else:
                    _before_cap = r_mv
                if r_max is not None:
                    r_mv = float(np.clip(gamma * r_mv, 0.0, r_max))
                n_replay_this_epoch = int(r_mv * train_loader.batch_size)
                _after_cap = r_mv
            else:
                _before_cap = float(r_t) if method not in REGULARIZER_METHODS else 0.0
                _after_cap = float(r_t) if method not in REGULARIZER_METHODS else 0.0
                n_replay_this_epoch = n_replay_per_batch

            _epoch_replay_start = total_replay_samples
            for x_real, y_real in train_loader:
                # Select replay samples (MIR uses interference scoring; others random)
                if uses_replay and buffer.size() > 0 and n_replay_this_epoch > 0:
                    if method == "MIR":
                        x_rep, y_rep = _mir_select(
                            model, buffer, x_real, y_real,
                            n_replay_this_epoch, cfg.lr, loss_fn,
                        )
                    else:
                        x_rep, y_rep = buffer.sample(n_replay_this_epoch)
                    if len(x_rep) > 0:
                        x_all = torch.cat([x_real, x_rep], dim=0)
                        y_all = torch.cat([y_real, y_rep], dim=0)
                        total_replay_samples += len(x_rep)
                        _task_replay += len(x_rep)
                    else:
                        x_all, y_all = x_real, y_real
                else:
                    x_all, y_all = x_real, y_real

                optimizer.zero_grad()
                out = model(x_all)
                loss = loss_fn(out, y_all)

                if ewc_obj is not None:
                    loss = loss + ewc_obj.penalty(model)
                if si_obj is not None:
                    loss = loss + si_obj.penalty(model)

                loss.backward()

                if si_obj is not None:
                    si_obj.update_W()

                optimizer.step()

            _epoch_replay_count = total_replay_samples - _epoch_replay_start

            # After each epoch: record MV proxy if applicable
            _proxy_val = float("nan")
            if mv_fitter is not None and buffer.size() > 0:
                _proxy_val = _compute_mv_proxy(model, buffer, cfg.mv_proxy_eval_samples)
                mv_proxy_history.append(_proxy_val)
                mv_fitter.record_epoch(global_epoch, _proxy_val)

                # Update per-epoch flag: might become True mid-task after enough obs
                # (only affects subsequent epochs in this task)
                uses_per_epoch_mv = mv_fitter.has_fit

            # Write controller trace row
            if _trace_writer is not None:
                _trace_writer.writerow({
                    "seed": seed,
                    "method": method,
                    "task_id": t,
                    "epoch_in_task": epoch_local,
                    "global_epoch": global_epoch,
                    "alpha_fit": f"{_alpha_fit:.6f}" if not np.isnan(_alpha_fit) else "nan",
                    "beta_fit": f"{_beta_fit:.8e}" if not np.isnan(_beta_fit) else "nan",
                    "dife_envelope_value": f"{_dife_envelope:.6f}" if not np.isnan(_dife_envelope) else "nan",
                    "mv_proxy_value": f"{_proxy_val:.6f}" if not np.isnan(_proxy_val) else "nan",
                    "mv_operator_value": f"{_mv_op_val:.6f}" if not np.isnan(_mv_op_val) else "nan",
                    "final_replay_fraction_before_cap": f"{_before_cap:.6f}",
                    "final_replay_fraction_after_cap": f"{_after_cap:.6f}",
                    "replay_samples_this_epoch": _epoch_replay_count,
                    "cumulative_replay_samples": total_replay_samples,
                    "has_mv_fit": int(mv_fitter.has_fit) if mv_fitter is not None else 0,
                    "r_max": r_max if r_max is not None else "nan",
                    "gamma": gamma,
                })

            global_epoch += 1

        # Post-task: regularizer consolidation
        if ewc_obj is not None:
            ewc_obj.consolidate(train_loader)
        if si_obj is not None:
            si_obj.consolidate()

        # Post-task: update replay buffer with this task's data
        if uses_replay:
            for x_b, y_b in train_loader:
                buffer.update(x_b, y_b)

        # Post-task: evaluate all seen tasks
        row = []
        for j, (_, test_loader) in enumerate(task_loaders[: t + 1]):
            row.append(evaluate(model, test_loader))
        acc_matrix.append(row)
        replay_per_task.append(_task_replay)
        print(f"  [{method}] task {t+1}: {[f'{a:.3f}' for a in row]}")

        # Post-task: update online fitters (causal — uses only rows seen so far)
        if dife_fitter is not None and len(acc_matrix) > 0:
            alpha, beta = dife_fitter.update(acc_matrix)
            dife_params_history.append({"alpha": alpha, "beta": beta})

        if mv_fitter is not None:
            mv_fitter.update()

    if _trace_file is not None:
        _trace_file.close()

    return {
        "acc_matrix": acc_matrix,
        "r_t_history": r_t_history,
        "pre_task_acc": pre_task_acc,
        "total_replay_samples": total_replay_samples,
        "replay_per_task": replay_per_task,
        "wall_clock_seconds": time.time() - t0,
        "mv_proxy_history": mv_proxy_history,
        "dife_params_history": dife_params_history,
    }
