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


class SmallCNN(nn.Module):
    """Small CNN for Split-CIFAR (binary classification per task).

    Input: (B, 3, 32, 32).
    Two conv blocks followed by two FC layers.
    output_dim should equal number of classes in this task (2 for binary).
    """

    def __init__(self, output_dim=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                           # → 16×16×32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                           # → 8×8×64
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


def fresh_cnn(output_dim=2):
    """Return a freshly initialised SmallCNN."""
    return SmallCNN(output_dim=output_dim)
