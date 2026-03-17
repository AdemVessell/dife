"""Reservoir-sampling experience replay buffer."""

import numpy as np
import torch


class ReservoirBuffer:
    """Fixed-capacity reservoir-sampling experience replay buffer (Algorithm R)."""

    def __init__(self, capacity: int, input_shape: tuple):
        self._capacity = capacity
        self._X = torch.zeros(capacity, *input_shape)
        self._y = torch.zeros(capacity, dtype=torch.long)
        self._n_seen = 0
        self._n_stored = 0

    def update(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> None:
        """Reservoir-sample x_batch/y_batch into buffer."""
        for x, y in zip(x_batch, y_batch):
            idx = self._n_seen
            self._n_seen += 1
            if idx < self._capacity:
                self._X[idx] = x.detach()
                self._y[idx] = y.detach()
                self._n_stored = min(self._n_stored + 1, self._capacity)
            else:
                j = np.random.randint(0, self._n_seen)
                if j < self._capacity:
                    self._X[j] = x.detach()
                    self._y[j] = y.detach()

    def sample(self, n: int) -> tuple:
        """Return n random samples (without exceeding stored count)."""
        if self._n_stored == 0 or n == 0:
            return torch.zeros(0), torch.zeros(0, dtype=torch.long)
        n = min(n, self._n_stored)
        idx = torch.randint(0, self._n_stored, (n,))
        return self._X[idx], self._y[idx]

    def size(self) -> int:
        return self._n_stored
