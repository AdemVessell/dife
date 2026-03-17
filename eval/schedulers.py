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
        m = state.mv_fitter.replay_fraction(state.total_epochs_so_far)
        return float(np.clip(d * m, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown method: {method}")
