"""DIFE: Decay-Interference Forgetting Equation

Q_n = max(0, Q_0 * alpha^n - beta * n * (1 - alpha^n))
"""

from dife.core import dife, dife_curve, forgetting_rate

__all__ = ["dife", "dife_curve", "forgetting_rate"]
