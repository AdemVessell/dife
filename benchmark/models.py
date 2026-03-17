"""Shared neural network architectures for CL benchmarks."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two-hidden-layer MLP for Permuted MNIST."""

    def __init__(self, input_dim=784, hidden=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def fresh_mlp(hidden=256):
    """Return a freshly initialised MLP."""
    return MLP(hidden=hidden)
