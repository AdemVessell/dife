"""Replay fraction schedulers for all 9 evaluation methods."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SchedulerState:
    """Mutable state passed into every scheduler call."""
    task_index: int
    total_epochs_so_far: int
    dife_fitter: object
    mv_fitter: object
    rng: np.random.Generator
    r_max: Optional[float] = None  # governor cap; used for budget-aware MV blend


def get_replay_fraction(method: str, state: SchedulerState) -> float:
    """Dispatch to the correct r_t scheduler. Returns float in [0, 1]."""
    t = state.task_index

    if method == "FT":
        return 0.0
    elif method == "EWC":
        return 0.0
    elif method == "SI":
        return 0.0
    elif method == "ConstReplay_0.1":
        return 0.1
    elif method == "ConstReplay_0.3":
        return 0.3
    elif method == "RandReplay":
        return float(state.rng.uniform(0.05, 0.35))
    elif method == "DIFE_only":
        return state.dife_fitter.replay_fraction(t)
    elif method == "MV_only":
        return state.mv_fitter.replay_fraction(state.total_epochs_so_far)
    elif method == "DIFE_MV":
        d = state.dife_fitter.replay_fraction(t)
        # Budget-aware blend: at tight budgets (r_max <= 0.10) skip MV entirely
        # so DIFE_MV = DIFE_only exactly (no RNG divergence from proxy eval).
        # lambda ramps from 0 at r_max=0.10 to 1.0 at r_max=0.20.
        if state.r_max is not None:
            lam = float(np.clip((state.r_max - 0.10) / 0.10, 0.0, 1.0))
        else:
            lam = 1.0
        if lam == 0.0:
            return d  # pure DIFE, no MV call (preserves RNG state)
        m = state.mv_fitter.replay_fraction(state.total_epochs_so_far)
        raw = d * ((1.0 - lam) + lam * m)
        return float(np.clip(raw, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown method: {method}")
