"""Seven-function basis for the Memory Vortex symbolic operator."""

from typing import List

import numpy as np

BASIS_ORDER: List[str] = [
    "sin(t)", "cos(t)", "sin(2t)", "cos(2t)", "exp(-0.1t)", "t", "log(1+|sin(t)|)"
]


def eval_basis_numeric(t: float) -> np.ndarray:
    """Evaluate all 7 basis functions at scalar t. Returns shape (7,)."""
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
