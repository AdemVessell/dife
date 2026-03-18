"""MemoryVortexScheduler: evaluate a discovered operator at inference time.

Source: AdemVessell/memory-vortex v1.0
"""

import json
import os
from typing import Dict, List

import numpy as np

from memory_vortex.basis import eval_basis_numeric

_FALLBACK_COEF = np.array([0.0, 0.0, 0.0, 0.01375, 0.798, 0.0, 0.0], dtype=float)


class MemoryVortexScheduler:
    """Load a discovered operator JSON and compute per-step replay fractions.

    Falls back to ``0.798·exp(-0.1t) + 0.01375·cos(2t)`` if JSON is absent.
    """

    def __init__(
        self,
        operator_file: str = "memory_vortex_operator.json",
        verbose: bool = True,
    ):
        self.modalities: List[str] = ["vision", "text", "audio"]

        if os.path.exists(operator_file):
            with open(operator_file) as f:
                data = json.load(f)
            schema = data.get("schema", "")
            if schema != "memory-vortex/operator-v1":
                raise ValueError(f"Unknown operator schema: {schema!r}")
            self._coef      = np.array(data["coefficients_raw"], dtype=float)
            self._intercept = float(data["intercept_raw"])
            self._t_scale   = float(data["t_scale"])
            if verbose:
                print(f"  MemoryVortexScheduler: loaded '{data.get('name', '?')}' "
                      f"from {operator_file}")
        else:
            self._coef      = _FALLBACK_COEF.copy()
            self._intercept = 0.0
            self._t_scale   = 100.0
            if verbose:
                print("  MemoryVortexScheduler: using safe fallback operator "
                      "(no JSON found at {operator_file!r})")

    def strength(self, task_n: int) -> float:
        """Replay-need strength in [0, 1]."""
        t = float(task_n) / self._t_scale
        raw = self._intercept + float(np.dot(self._coef, eval_basis_numeric(t)))
        return float(np.clip(raw, 0.0, 1.0))

    def __call__(self, task_n: int) -> Dict[str, float]:
        s = self.strength(task_n)
        return {m: s for m in self.modalities}
