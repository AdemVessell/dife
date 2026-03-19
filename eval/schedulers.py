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
    r_max: Optional[float] = None  # governor cap (used by sweep runs)


def get_replay_fraction(method: str, state: SchedulerState) -> float:
    """Return the task-level replay fraction r_t for this method.

    For MV-based methods (MV_only, DIFE_MV), this is the *task-level* baseline.
    Per-epoch modulation happens in the trainer's epoch loop when mv_fitter.has_fit
    is True — not here. The scheduler only handles task-level planning.

    DIFE_MV design:
      - Task level: DIFE envelope sets the budget (no MV multiplication here)
      - Epoch level (trainer): r_epoch = DIFE(t) × MV_operator(global_epoch)
      This separation fixes two bugs from the old multiplicative formula:
        1. Cold-start deficit: old code multiplied by MV_fallback(0.81) even before
           MV had any data, cutting early-task replay by 19% vs DIFE_only.
        2. Never per-epoch: MV never modulated within a task; now it does.
    """
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
        # Task-level: pure DIFE envelope.
        # Per-epoch MV modulation is applied in the trainer loop.
        return state.dife_fitter.replay_fraction(t)
    elif method == "DIFE_flatMatched":
        # Budget is injected externally per task; r_t is overridden in trainer.
        return 0.0
    else:
        raise ValueError(f"Unknown method: {method}")
