
#### 2. **dife.py** (Core Module: O(1) Eval, Vectorized for Scale)
```python
"""
DIFE: Decay-Interference Forgetting Equation
Core implementation for simulation and loss regularization.
Author: Adem Vessell (@ademvessell)
License: MIT
"""

import numpy as np
from typing import Union, Tuple

def dife_equation(n: Union[int, np.ndarray], Q0: float = 1.0, alpha: float = 0.95, beta: float = 0.01) -> np.ndarray:
    """
    Compute DIFE: max(0, Q0 * alpha^n - beta * n * (1 - alpha^n))
    
    Args:
        n: Iterations/tasks (scalar or array).
        Q0: Initial quality.
        alpha: Decay rate (0 < alpha < 1).
        beta: Interference rate (>0).
    
    Returns:
        Retained quality Q_n (array).
    
    Example:
        >>> dife_equation(50, Q0=1.0, alpha=0.95, beta=0.01)
        0.04636555052743529
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0,1)")
    if beta <= 0:
        raise ValueError("beta must be >0")
    
    alpha_n = np.power(alpha, n)
    interference = beta * n * (1 - alpha_n)
    Q_n = np.maximum(0, Q0 * alpha_n - interference)
    return Q_n

def simulate_dife(n_steps: int, Q0: float = 1.0, alpha: float = 0.95, beta: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate curve over n=0 to n_steps."""
    n = np.arange(n_steps + 1)
    Q = dife_equation(n, Q0, alpha, beta)
    return n, Q

def fit_params(empirical_Q: np.ndarray, n: np.ndarray, method: str = 'lm') -> dict:
    """
    Fit DIFE params to data via SciPy optimize.
    
    Args:
        empirical_Q: Observed qualities.
        n: Corresponding steps.
        method: 'lm' (Levenberg-Marquardt) or 'nelder-mead'.
    
    Returns:
        Dict of fitted {Q0, alpha, beta}.
    """
    from scipy.optimize import curve_fit
    
    def model(n, Q0, alpha, beta):
        return dife_equation(n, Q0, alpha, beta)
    
    popt, _ = curve_fit(model, n, empirical_Q, p0=[1.0, 0.95, 0.01], method=method, bounds=([0, 0, 0], [np.inf, 1, np.inf]))
    return {'Q0': popt[0], 'alpha': popt[1], 'beta': popt[2]}

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n, Q = simulate_dife(200)
    plt.plot(n, Q, label='DIFE')
    plt.xlabel('n'); plt.ylabel('Q_n'); plt.title('Quick Sim'); plt.legend(); plt.show()
