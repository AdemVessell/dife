"""Hyperparameter grid search for EWC lambda and SI c on a validation split."""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from benchmark.baselines import EWC, SI, evaluate


def find_best_ewc_lambda(
    task_loaders: list,
    lambdas: list,
    epochs: int,
    lr: float,
    model_factory,
) -> float:
    """Find best EWC lambda using 80/20 train/val split on the first task.

    Args:
        task_loaders: list of (train_loader, test_loader) per task
        lambdas: list of lambda values to try
        epochs: training epochs per candidate
        lr: learning rate
        model_factory: callable returning a fresh model

    Returns:
        Best lambda value.
    """
    train_loader, _ = task_loaders[0]
    dataset = train_loader.dataset
    n_val = max(1, int(0.2 * len(dataset)))
    n_tr = len(dataset) - n_val
    tr_ds, val_ds = random_split(
        dataset, [n_tr, n_val], generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    best_lam, best_acc = None, -1.0
    for lam in lambdas:
        model = model_factory()
        ewc = EWC(model, lam=lam)
        opt = Adam(model.parameters(), lr=lr)
        tr_loader = DataLoader(tr_ds, batch_size=train_loader.batch_size, shuffle=True)

        for _ in range(epochs):
            model.train()
            for x, y in tr_loader:
                opt.zero_grad()
                loss = loss_fn(model(x), y) + ewc.penalty(model)
                loss.backward()
                opt.step()

        acc = evaluate(model, val_loader)
        print(f"  [grid_search EWC] lambda={lam:.0f} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_lam = acc, lam

    return best_lam


def find_best_si_c(
    task_loaders: list,
    cs: list,
    epochs: int,
    lr: float,
    model_factory,
) -> float:
    """Find best SI c using 80/20 train/val split on the first task.

    Args:
        task_loaders: list of (train_loader, test_loader) per task
        cs: list of c values to try
        epochs: training epochs per candidate
        lr: learning rate
        model_factory: callable returning a fresh model

    Returns:
        Best c value.
    """
    train_loader, _ = task_loaders[0]
    dataset = train_loader.dataset
    n_val = max(1, int(0.2 * len(dataset)))
    n_tr = len(dataset) - n_val
    tr_ds, val_ds = random_split(
        dataset, [n_tr, n_val], generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    best_c, best_acc = None, -1.0
    for c in cs:
        model = model_factory()
        si = SI(model, c=c)
        opt = Adam(model.parameters(), lr=lr)
        tr_loader = DataLoader(tr_ds, batch_size=train_loader.batch_size, shuffle=True)

        for _ in range(epochs):
            model.train()
            for x, y in tr_loader:
                opt.zero_grad()
                loss = loss_fn(model(x), y) + si.penalty(model)
                loss.backward()
                si.update_W()
                opt.step()

        acc = evaluate(model, val_loader)
        print(f"  [grid_search SI] c={c} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_c = acc, c

    return best_c
