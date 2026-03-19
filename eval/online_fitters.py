"""Online (causal) fitters for DIFE and Memory Vortex parameters."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory-vortex-dife-lab"))

import numpy as np
from scipy.optimize import minimize


class OnlineDIFEFitter:
    """Fits (alpha, beta) online from acc_matrix rows seen so far.

    No peeking at future rows — causal by design.
    Falls back to default params when insufficient observations.
    """

    DEFAULT_ALPHA = 0.90
    DEFAULT_BETA = 0.01

    def __init__(self):
        self.alpha = self.DEFAULT_ALPHA
        self.beta = self.DEFAULT_BETA

    def update(self, acc_matrix: list) -> tuple:
        """Fit from lower-triangular acc_matrix seen so far.

        Returns (alpha, beta). Uses defaults if < 2 cross-task observations.
        """
        from benchmark.fitting import _build_observations
        from dife import dife

        obs = _build_observations(acc_matrix)
        if len(obs) < 2:
            return self.alpha, self.beta

        def residuals(params):
            a, b = params
            if not (0 < a < 1) or b <= 0:
                return 1e9
            return sum(
                (dife(n, Q_0=Q0, alpha=a, beta=b) - Q_obs) ** 2
                for Q0, n, Q_obs in obs
            )

        result = minimize(
            residuals,
            x0=[self.alpha, self.beta],
            method="Nelder-Mead",
            options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-7},
        )
        if result.success and 0 < result.x[0] < 1 and result.x[1] > 0:
            self.alpha, self.beta = float(result.x[0]), float(result.x[1])
        return self.alpha, self.beta

    def replay_fraction(self, task_index: int) -> float:
        """DIFE score at task_index, clipped to [0, 1]."""
        from dife import dife as _d

        return float(np.clip(_d(task_index, Q_0=1.0, alpha=self.alpha, beta=self.beta), 0.0, 1.0))


class OnlineMVFitter:
    """Accumulates per-epoch proxy observations and fits Memory Vortex operator.

    Falls back to MemoryVortexOperator.fallback() until MIN_OBS samples seen.
    """

    MIN_OBS = 15

    def __init__(self):
        from dife.controller import MemoryVortexOperator
        from memory_vortex.discovery import GCADiscoveryEngineV1

        self._engine = GCADiscoveryEngineV1()
        self._epoch_steps = []
        self._y_proxy = []
        self._operator = MemoryVortexOperator.fallback()
        self._total_epochs = 0

    def record_epoch(self, global_epoch: int, proxy_value: float) -> None:
        """Record one proxy observation (called after each training epoch)."""
        self._epoch_steps.append(global_epoch)
        self._y_proxy.append(float(np.clip(np.nan_to_num(proxy_value, nan=0.0), 0.0, 1.0)))
        self._total_epochs = global_epoch + 1

    def update(self):
        """Fit after completing a task. Returns updated MemoryVortexOperator."""
        from dife.controller import MemoryVortexOperator

        if len(self._epoch_steps) < self.MIN_OBS:
            return self._operator

        steps = np.array(self._epoch_steps, dtype=float)
        y = np.array(self._y_proxy, dtype=float)
        t_scale = float(max(self._total_epochs, 1))

        result_dict = self._engine.discover(
            task_n_raw=steps.astype(int),
            y_data=y,
            name="mv_online",
            scale=t_scale,
        )
        self._operator = MemoryVortexOperator(
            coef_raw=np.array(result_dict["coefficients_raw"], dtype=float),
            intercept_raw=float(result_dict["intercept_raw"]),
            t_scale=float(result_dict["t_scale"]),
        )
        return self._operator

    def replay_fraction(self, global_epoch: int) -> float:
        """Return replay fraction from current operator at this global epoch."""
        return float(np.clip(np.nan_to_num(self._operator(global_epoch), nan=0.0), 0.0, 1.0))
