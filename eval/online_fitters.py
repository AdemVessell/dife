"""Online (causal) fitters for DIFE and Memory Vortex parameters."""

import sys
import os

_MV_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "memory-vortex-dife-lab")
)
if _MV_PATH not in sys.path:
    sys.path.insert(0, _MV_PATH)

import numpy as np
from scipy.optimize import minimize


def _ensure_dife_package() -> None:
    """Ensure sys.modules['dife'] is the memory-vortex package, not the root dife.py.

    The project has two `dife` objects:
      - /home/.../dife.py           — root-level module (the DIFE equation)
      - memory-vortex-dife-lab/dife/ — package (exports same function + controller)

    benchmark/fitting.py imports `from dife import dife` at module level. If that
    runs before memory-vortex-dife-lab is in sys.path, dife.py gets cached.
    Subsequent `from dife.controller import ...` then fails.

    This function evicts the stale cache if needed so the package takes over.
    """
    cached = sys.modules.get("dife")
    if cached is not None and not hasattr(cached, "__path__"):
        # dife.py (a module, not a package) is cached — evict it
        if _MV_PATH not in sys.path:
            sys.path.insert(0, _MV_PATH)
        elif sys.path[0] != _MV_PATH:
            sys.path.remove(_MV_PATH)
            sys.path.insert(0, _MV_PATH)
        stale_keys = [k for k in sys.modules if k == "dife" or k.startswith("dife.")]
        for k in stale_keys:
            del sys.modules[k]


class OnlineDIFEFitter:
    """Fits (alpha, beta) online from acc_matrix rows seen so far.

    No peeking at future rows — causal by design.
    Falls back to default params when insufficient observations.

    Optimizer: L-BFGS-B with bounds to prevent beta collapse.
    Beta collapse happens when Nelder-Mead warm-starts near alpha→1,
    causing beta→0 (explains near-zero replay-assisted forgetting with a
    degenerate solution that under-estimates real interference strength).
    Bounded optimizer finds a better fit AND prevents the degenerate solution.
    """

    DEFAULT_ALPHA = 0.90
    DEFAULT_BETA = 0.01

    # Bounds prevent the degenerate alpha→1, beta→0 solution.
    # alpha_max=0.97 forces DIFE to model meaningful decay over 5 tasks.
    # beta_min=0.001 prevents the interference term from vanishing entirely.
    ALPHA_BOUNDS = (0.50, 0.97)
    BETA_BOUNDS = (0.001, 1.0)

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
            return sum(
                (dife(n, Q_0=Q0, alpha=a, beta=b) - Q_obs) ** 2
                for Q0, n, Q_obs in obs
            )

        result = minimize(
            residuals,
            x0=[self.alpha, self.beta],
            method="L-BFGS-B",
            bounds=[self.ALPHA_BOUNDS, self.BETA_BOUNDS],
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
        )
        if result.success or result.fun < 1e-4:
            a, b = float(result.x[0]), float(result.x[1])
            if self.ALPHA_BOUNDS[0] < a < self.ALPHA_BOUNDS[1] and b > 0:
                self.alpha, self.beta = a, b
        return self.alpha, self.beta

    def replay_fraction(self, task_index: int) -> float:
        """DIFE score at task_index, clipped to [0, 1]."""
        from dife import dife as _d

        return float(np.clip(_d(task_index, Q_0=1.0, alpha=self.alpha, beta=self.beta), 0.0, 1.0))


class OnlineMVFitter:
    """Accumulates per-epoch proxy observations and fits Memory Vortex operator.

    Falls back to MemoryVortexOperator.fallback() until MIN_OBS samples seen.

    MIN_OBS = 6: fits after 2 tasks × 3 epochs (the minimum for the fast-track
    split_cifar benchmark). With MIN_OBS=15 (old value), MV never fit on 3ep/task
    benchmarks since the buffer is empty for task 0 → only 12 proxy obs total.
    """

    MIN_OBS = 6

    def __init__(self):
        _ensure_dife_package()
        from dife.controller import MemoryVortexOperator
        from memory_vortex.discovery import GCADiscoveryEngineV1

        self._engine = GCADiscoveryEngineV1()
        self._epoch_steps = []
        self._y_proxy = []
        self._operator = MemoryVortexOperator.fallback()
        self._total_epochs = 0
        self._has_fit = False

    @property
    def has_fit(self) -> bool:
        """True once the operator has been fitted at least once from real proxy data."""
        return self._has_fit

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
        self._has_fit = True
        return self._operator

    def replay_fraction(self, global_epoch: int) -> float:
        """Return replay fraction from current operator at this global epoch."""
        return float(np.clip(np.nan_to_num(self._operator(global_epoch), nan=0.0), 0.0, 1.0))
