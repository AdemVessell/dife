"""
DIFE core equation.

Q_n = max(0, Q_0 * alpha^n - beta * n * (1 - alpha^n))
"""


def dife(n, Q_0=1.0, alpha=0.95, beta=0.01):
    """Compute DIFE quality at step n.

    Args:
        n:     step count (non-negative int or float)
        Q_0:   initial quality, default 1.0
        alpha: retention rate in (0, 1)
        beta:  interference strength > 0

    Returns:
        Predicted quality, clamped to [0, Q_0].
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if Q_0 < 0:
        raise ValueError(f"Q_0 must be >= 0, got {Q_0}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    decay = alpha ** n
    interference = beta * n * (1.0 - decay)
    return max(0.0, Q_0 * decay - interference)


def dife_curve(n_max, Q_0=1.0, alpha=0.95, beta=0.01):
    """Return list of DIFE values for steps 0 … n_max (inclusive)."""
    return [dife(n, Q_0=Q_0, alpha=alpha, beta=beta) for n in range(n_max + 1)]


def forgetting_rate(n, Q_0=1.0, alpha=0.95, beta=0.01):
    """Quality drop from step 0 to step n (0 = no forgetting)."""
    return Q_0 - dife(n, Q_0=Q_0, alpha=alpha, beta=beta)
