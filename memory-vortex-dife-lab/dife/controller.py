"""
DIFE ∘ MemoryVortex combined controller.

The Memory Vortex operator expresses *how much replay is needed* (in [0,1])
as a learned function of the continual-learning step.  The DIFE envelope
models how much of that budget *remains* after n tasks of interference.
The product is the actual replay fraction served to the learner.

Typical usage
-------------
    from dife.controller import MemoryVortexOperator, DIFEParams, DIFE_MemoryVortexController

    op   = MemoryVortexOperator.from_json("operators/memory_vortex_operator.json")
    p    = DIFEParams(Q0=1.0, alpha=0.985, beta=0.004)
    ctrl = DIFE_MemoryVortexController(op=op, dife_params=p)

    for step in range(500):
        r = ctrl.replay_fraction(step)      # scalar in [0, 1]
        modality_dict = ctrl.per_modality(step)   # {"vision": r, "text": r, ...}
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from dife.core import dife as _dife


# ──────────────────────────────────────────────────────────────────────────────
# Basis (mirrors memory_vortex.basis exactly so JSON coefficients stay valid)
# ──────────────────────────────────────────────────────────────────────────────

BASIS_ORDER: List[str] = [
    "sin(t)", "cos(t)", "sin(2t)", "cos(2t)", "exp(-0.1t)", "t", "log(1+|sin(t)|)"
]


def _eval_basis(t: float) -> np.ndarray:
    t = float(t)
    return np.array([
        np.sin(t),
        np.cos(t),
        np.sin(2.0 * t),
        np.cos(2.0 * t),
        np.exp(-0.1 * t),
        t,
        np.log(1.0 + abs(np.sin(t))),
    ], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Memory Vortex operator (eval only — discovery lives in memory_vortex/)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryVortexOperator:
    """Evaluates a discovered Memory-Vortex operator loaded from JSON."""
    coef_raw:      np.ndarray
    intercept_raw: float = 0.0
    t_scale:       float = 1.0

    @classmethod
    def from_json(cls, path: str) -> "MemoryVortexOperator":
        with open(path) as f:
            data = json.load(f)
        coef = np.array(
            data.get("coefficients_raw", data.get("coefficients", [])), dtype=float
        )
        intercept = float(data.get("intercept_raw", data.get("intercept", 0.0)))
        t_scale   = float(data.get("t_scale", 1.0))
        if coef.shape[0] != len(BASIS_ORDER):
            raise ValueError(
                f"Expected {len(BASIS_ORDER)} coefficients, got {coef.shape[0]}"
            )
        return cls(coef_raw=coef, intercept_raw=intercept, t_scale=t_scale)

    @classmethod
    def fallback(cls) -> "MemoryVortexOperator":
        """Safe fallback: 0.798·exp(-0.1t) + 0.01375·cos(2t), t = step/100."""
        coef = np.array([0.0, 0.0, 0.0, 0.01375, 0.798, 0.0, 0.0], dtype=float)
        return cls(coef_raw=coef, intercept_raw=0.0, t_scale=100.0)

    def __call__(self, step: int) -> float:
        """Return need strength in [0, 1] at this step."""
        t = float(step) / self.t_scale
        raw = self.intercept_raw + float(np.dot(self.coef_raw, _eval_basis(t)))
        return float(np.clip(raw, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# DIFE parameters
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DIFEParams:
    Q0:    float = 1.0    # initial replay-budget scale
    alpha: float = 0.97   # retention rate  (0 < alpha < 1)
    beta:  float = 0.01   # interference slope (beta > 0)

    def __post_init__(self):
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0,1), got {self.alpha}")
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")


# ──────────────────────────────────────────────────────────────────────────────
# Combined controller
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DIFE_MemoryVortexController:
    """Compose the Memory Vortex operator with the DIFE forgetting envelope.

    replay_fraction(step) = clip(need(step) * dife_envelope(step), 0, r_max)

    where:
        need(step)          = MemoryVortexOperator(step)   ∈ [0, 1]
        dife_envelope(step) = clip(DIFE(step, Q0, α, β), 0, 1) ∈ [0, 1]
    """
    op:          MemoryVortexOperator
    dife_params: DIFEParams
    r_max:       float = 1.0
    modalities:  List[str] = field(default_factory=lambda: ["vision", "text", "audio"])

    def _dife_envelope(self, step: int) -> float:
        p = self.dife_params
        return float(np.clip(_dife(step, Q_0=p.Q0, alpha=p.alpha, beta=p.beta), 0.0, 1.0))

    def replay_fraction(self, step: int) -> float:
        """Scalar replay fraction in [0, r_max]."""
        need = self.op(step)
        envelope = self._dife_envelope(step)
        return float(np.clip(need * envelope, 0.0, self.r_max))

    def per_modality(self, step: int) -> Dict[str, float]:
        """Return per-modality replay fractions (currently uniform)."""
        r = self.replay_fraction(step)
        return {m: r for m in self.modalities}

    def breakdown(self, step: int) -> Dict[str, float]:
        """Diagnostic: return need, envelope, and product separately."""
        need     = self.op(step)
        envelope = self._dife_envelope(step)
        return {
            "need":     need,
            "envelope": envelope,
            "product":  float(np.clip(need * envelope, 0.0, self.r_max)),
        }
