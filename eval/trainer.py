"""Unified training loop for all 9 continual learning methods."""

import sys
import os
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

# Methods that use regularization instead of replay
REGULARIZER_METHODS = {"FT", "EWC", "SI"}


def _get_input_shape(task_loaders) -> tuple:
    x, _ = next(iter(task_loaders[0][0]))
    return tuple(x.shape[1:])


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
) -> dict:
    """Train model sequentially through all tasks with the given method.

    Returns dict with:
        acc_matrix, r_t_history, total_replay_samples, wall_clock_seconds,
        mv_proxy_history, dife_params_history, pre_task_acc
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
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

    acc_matrix = []
    r_t_history = []
    pre_task_acc = []
    total_replay_samples = 0
    mv_proxy_history = []
    dife_params_history = []
    global_epoch = 0

    for t, (train_loader, _) in enumerate(task_loaders):

        # Record zero-shot accuracy on task t before training (for FWT)
        if t > 0:
            _, test_t = task_loaders[t]
            pre_acc = evaluate(model, test_t)
            pre_task_acc.append(pre_acc)

        # Determine r_t BEFORE training this task (uses fitted params from t-1)
        state = SchedulerState(
            task_index=t,
            total_epochs_so_far=global_epoch,
            dife_fitter=dife_fitter,
            mv_fitter=mv_fitter,
            rng=rng,
        )
        r_t = get_replay_fraction(method, state)
        if r_max is not None:
            r_t = float(np.clip(gamma * r_t, 0.0, r_max))
        r_t_history.append(r_t)
        n_replay_per_batch = int(r_t * train_loader.batch_size)

        # Training epochs
        for epoch_local in range(cfg.epochs_per_task):
            model.train()

            for x_real, y_real in train_loader:
                # Mix in replay samples
                if uses_replay and buffer.size() > 0 and n_replay_per_batch > 0:
                    x_rep, y_rep = buffer.sample(n_replay_per_batch)
                    x_all = torch.cat([x_real, x_rep], dim=0)
                    y_all = torch.cat([y_real, y_rep], dim=0)
                    total_replay_samples += len(x_rep)
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

            # After each epoch: record MV proxy if applicable
            if mv_fitter is not None and buffer.size() > 0:
                proxy = _compute_mv_proxy(model, buffer, cfg.mv_proxy_eval_samples)
                mv_proxy_history.append(proxy)
                mv_fitter.record_epoch(global_epoch, proxy)

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
        print(f"  [{method}] task {t+1}: {[f'{a:.3f}' for a in row]}")

        # Post-task: update online fitters (causal — uses only rows seen so far)
        if dife_fitter is not None and len(acc_matrix) > 0:
            alpha, beta = dife_fitter.update(acc_matrix)
            dife_params_history.append({"alpha": alpha, "beta": beta})

        if mv_fitter is not None:
            mv_fitter.update()

    return {
        "acc_matrix": acc_matrix,
        "r_t_history": r_t_history,
        "pre_task_acc": pre_task_acc,
        "total_replay_samples": total_replay_samples,
        "wall_clock_seconds": time.time() - t0,
        "mv_proxy_history": mv_proxy_history,
        "dife_params_history": dife_params_history,
    }
