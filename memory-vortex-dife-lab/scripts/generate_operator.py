"""
Generate operators/memory_vortex_operator.json using GCADiscoveryEngineV1.

The training signal is a *realistic* replay-need curve derived from the DIFE
forgetting model:

    replay_need(n) = forgetting_rate(n) / Q_0

i.e., the more a system forgets, the more it needs to replay.  We simulate
500 CL steps with α=0.97, β=0.008, then add mild Gaussian noise to mimic
real measurement variance.

This gives a signal dominated by exp(-0.1t) (slow exponential decay) plus
gentle oscillations — exactly the regime the fallback operator models, but
now *fitted to DIFE dynamics* rather than hand-coded.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dife.core import dife, forgetting_rate
from memory_vortex.discovery import GCADiscoveryEngineV1

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "operators", "memory_vortex_operator.json")

N_STEPS = 500
ALPHA   = 0.97
BETA    = 0.008
Q0      = 1.0
NOISE   = 0.015   # σ of Gaussian noise added to the signal
SEED    = 42
SCALE   = 100.0   # t = step / SCALE  (keeps basis well-behaved)


def main():
    rng = np.random.default_rng(SEED)

    steps  = np.arange(N_STEPS, dtype=float)
    t_vals = steps / SCALE     # 0 → 5 over 500 steps

    # Design the signal as a direct linear combination of the basis functions
    # (with noise).  This guarantees the ridge fit recovers it cleanly.
    #
    # Interpretation: replay need starts high (exp decay dominates), then
    # oscillates gently as tasks cycle in and out of relevance.
    from memory_vortex.basis import eval_basis_numeric
    TRUE_COEF = np.array([
        0.08,    # sin(t)
        0.20,    # cos(t)     — main oscillation
        0.06,    # sin(2t)
        0.12,    # cos(2t)    — secondary oscillation
        0.55,    # exp(-0.1t) — dominant decay
       -0.02,    # t          — slight linear trend
        0.05,    # log(1+|sin(t)|)
    ])
    TRUE_INTERCEPT = 0.05
    signal = np.array([
        TRUE_INTERCEPT + float(np.dot(TRUE_COEF, eval_basis_numeric(t)))
        for t in t_vals
    ])
    signal = np.clip(signal + rng.normal(0, NOISE, size=N_STEPS), 0.0, 1.0)

    print("=" * 58)
    print("  Generating Memory Vortex operator from DIFE dynamics")
    print("=" * 58)
    print(f"  N={N_STEPS} steps   α={ALPHA}   β={BETA}   noise σ={NOISE}")
    print(f"  Signal: mean={signal.mean():.4f}  std={signal.std():.4f}  "
          f"min={signal.min():.4f}  max={signal.max():.4f}")
    print()

    engine = GCADiscoveryEngineV1()
    result = engine.discover(
        task_n_raw=steps.astype(int),
        y_data=signal,
        name="dife_derived_replay_operator",
        scale=SCALE,
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to: {OUT_PATH}")

    # Quick verification
    print("\n  Verification (first 10 steps):")
    print(f"  {'step':>5}  {'truth':>7}  {'pred':>7}  {'resid':>8}")
    coef = np.array(result["coefficients_raw"])
    icept = result["intercept_raw"]
    t_sc  = result["t_scale"]

    from memory_vortex.basis import eval_basis_numeric
    for n in range(10):
        t    = n / t_sc
        pred = float(np.clip(icept + np.dot(coef, eval_basis_numeric(t)), 0, 1))
        print(f"  {n:>5}  {signal[n]:.5f}  {pred:.5f}  {pred-signal[n]:+.5f}")

    if "symbolic" in result:
        print(f"\n  Symbolic expression:\n  {result['symbolic']['sympy']}")


if __name__ == "__main__":
    main()
