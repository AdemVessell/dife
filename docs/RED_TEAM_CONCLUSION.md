# Red Team Conclusion

**Source:** canonical/audit-rebuild, results/canonical/split_cifar_rmax_0.30/
**No marketing language. Blunt internal-lab summary.**

---

## 1. Claims That Are Now Scientifically Safe

- MV_only shows a directional (not statistically strong) AF reduction vs ConstReplay_0.3 (ΔAF=+0.006, 0.5σ, 5 seeds).
- The MV proxy signal is non-degenerate on split-CIFAR-10 (buffer accuracy is not trivially 1.0 throughout training).
- DIFE_MV fails on perm-MNIST relative to ConstReplay baselines (5 seeds, large effect, consistent direction).
- Fine-tuning and regularization-only methods (EWC, SI) are significantly worse than any replay method — both benchmarks, multiple seeds.
- The offline DIFE equation fits observed forgetting curves with RMSE ~0.03 on perm-MNIST (benchmark/fitting.py, post-hoc analysis).
- beta collapses to the lower bound (0.001) in the bounded online fitter on split-CIFAR-10, meaning the DIFE envelope is >> r_max for all tasks.

---

## 2. Claims That Are Still Unsafe

- "DIFE_MV beats ConstReplay_0.3" — DIFE_MV ≈ MV_only in both performance and replay schedule. Any gap between DIFE_MV and ConstReplay_0.3 is attributable to MV, not DIFE.
- "DIFE_only beats ConstReplay_0.3" — the effect is within combined noise at 5 seeds.
- "DIFE adapts replay allocation across tasks" — the envelope has never dropped below r_max in any capped result. DIFE is computing numbers that are then discarded by the cap.
- "The crossover threshold is r_max=0.30" — the crossover claim requires DIFE to be adapting, which it is not.
- "DIFE_MV is a combined adaptive controller" — in the canonical run, DIFE_MV ≈ MV_only.
- "MV improves efficiency" — budget is identical for all capped methods at r_max=0.30.

---

## 3. What DIFE Is Currently Proven To Be

DIFE is a **curve-fitting tool** that can model post-hoc forgetting trajectories
with reasonable accuracy (RMSE ~0.03). As an **online adaptive replay controller**,
it is currently inoperative in all capped runs because:

1. Beta collapses to the lower bound (0.001) after the first two tasks on split-CIFAR.
2. Even at beta=0.001, dife(t=4, alpha=0.9, beta=0.001) ≈ 0.65 >> r_max=0.30.
3. The governor cap always binds, so DIFE's output is discarded every time.

DIFE is currently a **deterministic constant-rate scheduler** that happens to
run an expensive online optimizer in the background to compute values it never uses.

---

## 4. What MV Is Currently Proven To Be

MV is a **per-epoch replay modulator** that fits a 7-term basis function to a
proxy forgetting signal (1 - buffer accuracy) and uses this to vary replay
within tasks epoch by epoch. On split-CIFAR-10:

- MV_only reduces AF by 0.006 relative to ConstReplay_0.3 (0.5σ, 5 seeds).
- The proxy signal is real and non-flat on split-CIFAR-10.
- The benefit (if any) is from within-task epoch variation, not task-level adaptation.
- MV does NOT help on perm-MNIST (proxy is flat; MV oscillates randomly).
- MV is benchmark-sensitive: it requires a task where the proxy is informative.

---

## 5. Is DIFE_MV a Truly Combined Controller in the Current Code?

**No. But the reason is more specific than "DIFE_envelope ≈ 1.0."**

The DIFE envelope ranges from 0.697 to 1.000 (mean 0.868) — so it is NOT trivially 1.0.
However, it never drops below r_max=0.30. The result:

- When MV_operator > r_max: both DIFE_MV and MV_only clip to r_max. Identical.
- When MV_operator < r_max: DIFE_MV = MV_operator × DIFE_envelope < MV_operator = MV_only.

So DIFE_MV uses **less replay than MV_only** whenever MV is below the cap.
This is not task-adaptive behavior — it is a constant alpha^t decay with a collapsed beta
that systematically undercuts MV's already-low replay prescription.

In 18.7% of epochs, DIFE_MV and MV_only differ by >0.001 in replay fraction.
In those epochs, DIFE_MV always uses **less** replay than MV_only, not more.
The difference is small (mean 0.005) and has no statistically significant performance impact.

DIFE_MV is not a combined adaptive controller. It is MV_only with a fixed per-task
attenuation factor derived from alpha^t, producing marginally less replay at no
performance benefit.

---

## 6. The Single Strongest Next Experiment

**Fix beta initialization and re-run the canonical experiment.**

The root cause of DIFE's inactivity is that the online fitter fits beta → lower bound
because split-CIFAR-10 forgetting trajectories are well-modeled by pure exponential
decay (the interference term is not needed to fit 3-epoch task data).

The single experiment that would separate 'DIFE is broken' from 'DIFE is right':

1. Set `BETA_BOUNDS = (0.05, 1.0)` in `eval/online_fitters.py` (raise lower bound).
2. Re-run the canonical experiment with 5 seeds.
3. Check the controller trace: does dife_envelope_value vary across tasks?
4. If yes: compare DIFE_only vs ConstReplay_0.3 where DIFE is actually adapting.
5. If envelope still always hits r_max: DIFE is simply the wrong scale for this
   benchmark and needs a higher r_max or a harder benchmark to be activated.

Do NOT run this until the canonical run above is complete and committed.
The canonical run provides the inoperative-DIFE baseline against which any
future beta-fix run should be compared.

*Generated: 2026-03-19 | Source: /home/user/dife/results/canonical/split_cifar_rmax_0.30*
