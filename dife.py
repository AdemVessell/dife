"""
DIFE: Decay-Interference Forgetting Equation

Q_n = max(0, Q_0 * alpha^n - beta * n * (1 - alpha^n))

- Q_0: Initial quality (e.g., 1.0 accuracy)
- alpha in (0,1): Decay rate (e.g., 0.95)
- beta > 0: Interference strength (e.g., 0.01)
- n: Task/step count
"""


def dife(n, Q_0=1.0, alpha=0.95, beta=0.01):
    """Compute DIFE forgetting score at task step n.

    Args:
        n: Task/step count (non-negative integer or float).
        Q_0: Initial quality. Default 1.0.
        alpha: Decay rate in (0, 1). Default 0.95.
        beta: Interference strength > 0. Default 0.01.

    Returns:
        Predicted quality at step n, clamped to [0, Q_0].

    Raises:
        ValueError: If parameters are out of valid range.
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
    interference = beta * n * (1 - decay)
    raw = Q_0 * decay - interference
    return max(0.0, raw)


def dife_curve(n_max, Q_0=1.0, alpha=0.95, beta=0.01):
    """Return a list of DIFE scores for steps 0..n_max (inclusive).

    Args:
        n_max: Maximum step count.
        Q_0: Initial quality.
        alpha: Decay rate in (0, 1).
        beta: Interference strength > 0.

    Returns:
        List of floats of length n_max + 1.
    """
    return [dife(n, Q_0=Q_0, alpha=alpha, beta=beta) for n in range(n_max + 1)]


def forgetting_rate(n, Q_0=1.0, alpha=0.95, beta=0.01):
    """Return the drop in quality from step 0 to step n.

    A value of 0 means no forgetting; a value of Q_0 means total forgetting.
    """
    return Q_0 - dife(n, Q_0=Q_0, alpha=alpha, beta=beta)
