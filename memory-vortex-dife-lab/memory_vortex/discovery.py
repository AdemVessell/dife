"""GCADiscoveryEngineV1: fixed-basis symbolic regression via ridge.

Discovers a Memory Vortex operator from (task_step, replay_need) pairs using
blocked 70/15/15 train/val/test splits and λ selected on val MAE.

Source: AdemVessell/memory-vortex v1.0
"""

from typing import Dict, Any

import numpy as np

try:
    import sympy as sp
    _SYMPY = True
except ImportError:
    _SYMPY = False

from memory_vortex.basis import BASIS_ORDER, eval_basis_numeric


class GCADiscoveryEngineV1:
    """Fixed-basis symbolic regression for replay-need operators.

    Usage
    -----
        engine = GCADiscoveryEngineV1()
        result = engine.discover(task_steps, replay_needs, name="my_op", scale=100.0)
        # result is a JSON-serialisable dict compatible with MemoryVortexScheduler
    """

    def __init__(self):
        if _SYMPY:
            t = sp.symbols("t")
            self._basis_sympy = [
                sp.sin(t),
                sp.cos(t),
                sp.sin(2 * t),
                sp.cos(2 * t),
                sp.exp(-sp.Rational(1, 10) * t),
                t,
                sp.log(1 + sp.Abs(sp.sin(t))),
            ]
            self._t_sym = t
        else:
            self._basis_sympy = None

    def discover(
        self,
        task_n_raw: np.ndarray,
        y_data: np.ndarray,
        name: str = "memory_vortex_v1",
        scale: float = 100.0,
    ) -> Dict[str, Any]:
        """Fit operator to (task_n_raw, y_data) pairs.

        Args:
            task_n_raw: integer step indices, shape (N,)
            y_data:     target replay-need values in [0,1], shape (N,)
            name:       operator name stored in JSON
            scale:      t = task_n / scale

        Returns:
            dict matching the memory-vortex/operator-v1 schema
        """
        t_vals = task_n_raw.astype(float) / float(scale)
        X_raw = np.vstack([eval_basis_numeric(float(tt)) for tt in t_vals])

        n = len(y_data)
        train_end = int(0.70 * n)
        val_end   = int(0.85 * n)

        # Standardise using train statistics only
        X_mean = X_raw[:train_end].mean(axis=0)
        X_std  = X_raw[:train_end].std(axis=0) + 1e-8
        X_std_arr = (X_raw - X_mean) / X_std
        X_aug = np.c_[X_std_arr, np.ones(n)]

        X_tr, y_tr = X_aug[:train_end],        y_data[:train_end]
        X_va, y_va = X_aug[train_end:val_end], y_data[train_end:val_end]
        X_te, y_te = X_aug[val_end:],          y_data[val_end:]

        # λ selection on val
        lambdas = np.logspace(-4, 2, 20)
        best_lam, best_val_mae = None, np.inf
        for lam in lambdas:
            I = np.eye(X_tr.shape[1])
            coef = np.linalg.solve(X_tr.T @ X_tr + lam * I, X_tr.T @ y_tr)
            mae = float(np.mean(np.abs(y_va - X_va @ coef)))
            if mae < best_val_mae:
                best_val_mae, best_lam = mae, float(lam)

        # Refit on train + val
        X_rv = np.vstack((X_tr, X_va))
        y_rv = np.concatenate((y_tr, y_va))
        I = np.eye(X_rv.shape[1])
        coef_aug = np.linalg.solve(X_rv.T @ X_rv + best_lam * I, X_rv.T @ y_rv)
        coef_aug[np.abs(coef_aug) < 1e-5] = 0.0

        # Convert standardised → raw operator
        beta_std    = coef_aug[:-1]
        b_std       = float(coef_aug[-1])
        coef_raw    = beta_std / X_std
        intercept_raw = b_std - float(np.dot(beta_std, X_mean / X_std))

        # Test metric (standardised space)
        test_mae = float(np.mean(np.abs(y_te - X_te @ coef_aug)))

        print(
            f"  Discovered '{name}'  val_MAE={best_val_mae:.4f}  "
            f"test_MAE={test_mae:.4f}  λ={best_lam:.2e}"
        )

        result: Dict[str, Any] = {
            "schema": "memory-vortex/operator-v1",
            "name": name,
            "basis_order": BASIS_ORDER,
            "coefficients_raw": coef_raw.tolist(),
            "intercept_raw": float(intercept_raw),
            "t_scale": float(scale),
            "fit": {
                "method": "ridge",
                "lambda": best_lam,
                "val_mae": float(best_val_mae),
                "test_mae": float(test_mae),
                "split": {"train_frac": 0.70, "val_frac": 0.15, "blocked": True},
            },
        }

        if _SYMPY and self._basis_sympy is not None:
            expr = float(intercept_raw) + sum(
                float(coef_raw[i]) * self._basis_sympy[i]
                for i in range(len(coef_raw))
            )
            simplified = sp.simplify(expr)
            result["symbolic"] = {
                "latex":  sp.latex(simplified),
                "sympy":  str(simplified),
            }

        return result
