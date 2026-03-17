"""EWC and SI continual learning baselines.

References:
  EWC: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural
       networks." PNAS 114(13):3521-3526.
  SI:  Zenke et al. (2017) "Continual Learning Through Synaptic Intelligence."
       ICML 2017.
"""

import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ---------------------------------------------------------------------------
# Shared training utilities
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, optimizer, loss_fn, extra_loss=None):
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        if extra_loss is not None:
            loss = loss + extra_loss(model)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


# ---------------------------------------------------------------------------
# Fine-tuning (SGD baseline, no regularisation)
# ---------------------------------------------------------------------------

def train_finetuning(model, task_loaders, epochs=5, lr=1e-3):
    """Train sequentially with no regularisation.

    Returns acc_matrix[i][j] = accuracy on task j after training on task i.
    """
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n_tasks = len(task_loaders)
    acc_matrix = []

    for t, (train_loader, _) in enumerate(task_loaders):
        for _ in range(epochs):
            _train_epoch(model, train_loader, optimizer, loss_fn)
        row = []
        for _, test_loader in task_loaders[: t + 1]:
            row.append(evaluate(model, test_loader))
        acc_matrix.append(row)
        print(f"  [FT] After task {t+1}: {[f'{a:.3f}' for a in row]}")

    return acc_matrix


# ---------------------------------------------------------------------------
# EWC
# ---------------------------------------------------------------------------

class EWC:
    """Elastic Weight Consolidation (online version, one Fisher per task)."""

    def __init__(self, model, lam=5000):
        self.model = model
        self.lam = lam
        # List of (param_mean, fisher_diag) per completed task
        self._anchors = []

    def _compute_fisher(self, loader, n_samples=200):
        """Diagonal Fisher over n_samples data points."""
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        count = 0
        for x, y in loader:
            if count >= n_samples:
                break
            self.model.zero_grad()
            out = self.model(x)
            log_prob = F.log_softmax(out, dim=1)
            # Sample labels from model's own distribution (standard EWC)
            sampled_y = torch.multinomial(log_prob.exp(), 1).squeeze()
            loss = F.nll_loss(log_prob, sampled_y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += len(x)
        for n in fisher:
            fisher[n] /= count
        return fisher

    def consolidate(self, loader):
        """Call after finishing a task to register its Fisher + weights."""
        fisher = self._compute_fisher(loader)
        means = {n: p.detach().clone()
                 for n, p in self.model.named_parameters() if p.requires_grad}
        self._anchors.append((means, fisher))

    def penalty(self, model):
        """EWC regularisation loss."""
        if not self._anchors:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for means, fisher in self._anchors:
            for n, p in model.named_parameters():
                if p.requires_grad and n in means:
                    loss = loss + (fisher[n] * (p - means[n]) ** 2).sum()
        return (self.lam / 2) * loss


def train_ewc(model, task_loaders, epochs=5, lr=1e-3, lam=5000):
    """Train with EWC regularisation.

    Returns acc_matrix[i][j] = accuracy on task j after training on task i.
    """
    ewc = EWC(model, lam=lam)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n_tasks = len(task_loaders)
    acc_matrix = []

    for t, (train_loader, _) in enumerate(task_loaders):
        for _ in range(epochs):
            _train_epoch(model, train_loader, optimizer, loss_fn,
                         extra_loss=ewc.penalty)
        ewc.consolidate(train_loader)
        row = []
        for _, test_loader in task_loaders[: t + 1]:
            row.append(evaluate(model, test_loader))
        acc_matrix.append(row)
        print(f"  [EWC] After task {t+1}: {[f'{a:.3f}' for a in row]}")

    return acc_matrix


# ---------------------------------------------------------------------------
# SI (Synaptic Intelligence)
# ---------------------------------------------------------------------------

class SI:
    """Synaptic Intelligence (Zenke et al., 2017)."""

    def __init__(self, model, c=0.1, xi=0.1):
        self.model = model
        self.c = c
        self.xi = xi
        # Cumulative importance Ω per parameter
        self.omega = {n: torch.zeros_like(p)
                      for n, p in model.named_parameters() if p.requires_grad}
        # θ* at the start of the current task
        self._prev_params = {n: p.detach().clone()
                             for n, p in model.named_parameters() if p.requires_grad}
        # Running sum of δloss/δθ · δθ (W in the paper) for current task
        self._W = {n: torch.zeros_like(p)
                   for n, p in model.named_parameters() if p.requires_grad}
        self._prev_grads = None

    def update_W(self):
        """Call after each backward pass to accumulate W."""
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                # δθ since last step (approximated as -lr·grad, but here
                # we directly track Δθ between consecutive steps)
                if self._prev_grads is not None and n in self._prev_grads:
                    delta = p.detach() - self._prev_grads[n]
                    self._W[n] -= p.grad.detach() * delta
                self._prev_grads = {n: p.detach().clone()
                                    for n, p in self.model.named_parameters()
                                    if p.requires_grad}

    def consolidate(self):
        """Call after finishing a task to update Ω."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                denom = (p.detach() - self._prev_params[n]) ** 2 + self.xi
                self.omega[n] = (self.omega[n] + self._W[n] / denom).clamp(min=0)
        # Reset for next task
        self._prev_params = {n: p.detach().clone()
                              for n, p in self.model.named_parameters()
                              if p.requires_grad}
        self._W = {n: torch.zeros_like(p)
                   for n, p in self.model.named_parameters() if p.requires_grad}
        self._prev_grads = None

    def penalty(self, model):
        """SI regularisation loss."""
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.omega:
                loss = loss + (self.omega[n] * (p - self._prev_params[n]) ** 2).sum()
        return self.c * loss


def train_si(model, task_loaders, epochs=5, lr=1e-3, c=0.1):
    """Train with SI regularisation.

    Returns acc_matrix[i][j] = accuracy on task j after training on task i.
    """
    si = SI(model, c=c)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    acc_matrix = []

    for t, (train_loader, _) in enumerate(task_loaders):
        for _ in range(epochs):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                out = model(x)
                loss = loss_fn(out, y) + si.penalty(model)
                loss.backward()
                si.update_W()
                optimizer.step()
        si.consolidate()
        row = []
        for _, test_loader in task_loaders[: t + 1]:
            row.append(evaluate(model, test_loader))
        acc_matrix.append(row)
        print(f"  [SI] After task {t+1}: {[f'{a:.3f}' for a in row]}")

    return acc_matrix
