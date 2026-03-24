# DIFE × Memory Vortex — Caveats and Open Questions

This file documents known limitations, open questions, and nuances that are not fully captured in the main README.

---

## Online vs Offline Behavior

**DIFE alone acts as an offline forgetting model** — it fits α and β from post-task accuracy histories and produces budget-proportional allocations for the *next* task, but the loop is one task-delayed. In budget-capped runs (r_max=0.30), DIFE alone saturates the cap every task (5/5 seeds at both β_min=0.05 and β_min=0.10), effectively degenerating to ConstReplay_0.3.

**Memory Vortex (MV) provides genuine online adaptation.** MV operates epoch-by-epoch within a task using a proxy signal (buffer accuracy drift). The beta-bound rerun confirms that MV is the component that enables DIFE_MV to fire below the budget cap — allocating less replay when forgetting pressure is low and more when it spikes.

**DIFE_MV is a confirmed adaptive controller.** With β_min ≥ 0.05, the combined system fires below the r_max cap in 10/10 seeds across two beta conditions, using 9–10% less replay budget than the fixed-rate baseline while matching or improving accuracy. The controller traces show clear task-by-task modulation (e.g., replay fraction dropping to 0.157 on low-forgetting tasks, then ramping back to cap on high-forgetting tasks).

---

## DIFE_MV vs Fixed Replay — Beta-Bound Rerun Results

The canonical beta-bound rerun (6 methods × 5 seeds × 2 beta conditions) is **complete**:

### β_min = 0.05
| Method | AA | AF | Replay Budget |
|---|---|---|---|
| ConstReplay_0.3 | 0.830 ± 0.005 | 0.104 ± 0.007 | 36,024 (fixed) |
| DIFE_only | 0.830 ± 0.005 | 0.104 ± 0.007 | 36,024 (at cap) |
| MV_only | 0.831 ± 0.009 | 0.099 ± 0.008 | 33,512 ± 1,562 |
| **DIFE_MV** | **0.833 ± 0.013** | **0.099 ± 0.017** | **32,674 ± 1,533** |

### β_min = 0.10
| Method | AA | AF | Replay Budget |
|---|---|---|---|
| ConstReplay_0.3 | 0.830 ± 0.005 | 0.104 ± 0.007 | 36,024 (fixed) |
| DIFE_only | 0.830 ± 0.005 | 0.104 ± 0.007 | 36,024 (at cap) |
| MV_only | 0.831 ± 0.009 | 0.099 ± 0.008 | 33,512 ± 1,562 |
| **DIFE_MV** | **0.837 ± 0.007** | **0.093 ± 0.009** | **32,422 ± 1,619** |

**Key observations:**
- DIFE_MV achieves the best accuracy and lowest forgetting at both β_min settings
- DIFE_MV uses ~3,350–3,600 fewer replay samples than the fixed-budget baseline (9–10% savings)
- The advantage is consistent across seeds (0/5 seeds at cap for DIFE_MV vs 5/5 for DIFE_only)
- β_min=0.10 yields tighter error bars and slightly better performance than β_min=0.05

---

## Beta Convergence Issue — Resolved

On split-CIFAR, fitted β converges near zero (β ≈ 8.9e-7 by task 5) when unconstrained. This causes DIFE to over-allocate replay globally — hitting the r_max cap every task.

**The fix:** Setting β_min ≥ 0.05 prevents β from collapsing. With this floor, DIFE's envelope value varies meaningfully across tasks (e.g., 0.90 → 0.77), providing a genuine modulation signal that MV can amplify at the epoch level.

---

## Crossover Threshold

An apparent crossover exists where MV adds value only at r_max ≥ 0.30 on split-CIFAR (3 epochs/task setup). This threshold:

- Is observed in one sweep configuration, not replicated across varied epoch counts or task sequences.
- May be an artifact of the 3-epoch setup limiting intra-task proxy signal.
- Is not a proven design property of MV — it is a descriptive observation of one experiment.

---

## DIFE Fit Quality

The fitted RMSE values (0.030–0.045 on perm_mnist) demonstrate the equation captures the shape of forgetting curves well as an offline fitting tool. This remains the strongest standalone claim for DIFE.

---

## What Would Strengthen the Claims Further

1. ~~**Canonical rerun results**~~ — **COMPLETE.** DIFE_MV fires below r_max cap, confirmed across 10 seeds.
2. **Harder benchmarks**: 5 epochs/task, more diverse task sequences — enough intra-task signal for MV to show epoch-level structure.
3. ~~**Ablation of β_init**~~ — **Partially addressed.** Two β_min values (0.05, 0.10) tested. A finer sweep could further characterize the transition.
4. **Larger seed counts**: ≥ 10 seeds on key comparisons to tighten error bars on the DIFE_MV vs ConstReplay gap.
5. **Different r_max values**: Confirm adaptive behavior persists at r_max < 0.30 with calibrated β_min.
