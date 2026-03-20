# DIFE × Memory Vortex — Caveats and Open Questions

This file documents known limitations, open questions, and nuances that are not fully captured in the main README.

---

## Online vs Offline Behavior

**DIFE is currently best supported as an offline forgetting model.**

DIFE fits α and β from post-task accuracy histories (the acc_matrix after each task completes). This produces reliable budget-proportional allocations for the *next* task, but the loop is one task-delayed: DIFE reacts to forgetting it has already observed, not forgetting as it is happening within the current task.

The claim that DIFE acts as a "true online adaptive controller" in budget-capped split-CIFAR runs has not been rigorously verified. A canonical rerun (6 methods × 5 seeds, beta-bound calibration) is in progress to test this.

**Memory Vortex (MV)** operates epoch-by-epoch within a task using a proxy signal (buffer accuracy drift), which is more genuinely online — but its epoch-level shaping is only demonstrably useful when the budget cap is loose enough to give it room to redistribute. At r_max ≤ 0.20 the λ-blend suppresses MV toward DIFE_only behavior.

---

## DIFE_MV vs Fixed Replay

The split-CIFAR r_max=0.30 sweep (4 seeds) shows DIFE_MV at AF=0.083 vs ConstReplay_0.3 at AF=0.097. This is a real numerical difference, but should not be read as conclusive:

- **Seed count is low** (4 seeds DIFE_MV, 2–3 seeds baselines). Error bars overlap in some configurations.
- **The budget is equalized by construction** at r_max=0.30, so this is a fair comparison in budget terms — but only at that specific cap.
- **DIFE_only performs worse than DIFE_MV at equal budget** (AF 0.106 vs 0.083), suggesting MV contributes something, but the contribution is not yet isolated from noise at this seed count.
- The canonical rerun is specifically designed to determine whether DIFE_MV's advantage over fixed replay is real and stable.

---

## Beta Convergence Issue

On split-CIFAR, fitted β converges near zero (β ≈ 8.9e-7 by task 5). This causes DIFE to over-allocate replay globally — hitting the r_max cap every task — rather than concentrating budget on volatile tasks. In capped runs this means DIFE effectively degrades to ConstReplay_r_max with DIFE's budget logic inert.

The canonical rerun uses a raised β_init (~0.05) to test whether better prior calibration allows DIFE to actually modulate below the cap, which is the condition required for it to function as a genuine adaptive controller rather than a passthrough.

---

## Crossover Threshold

An apparent crossover exists where MV adds value only at r_max ≥ 0.30 on split-CIFAR (3 epochs/task setup). This threshold:

- Is observed in one sweep configuration, not replicated across varied epoch counts or task sequences.
- May be an artifact of the 3-epoch setup limiting intra-task proxy signal.
- Is not a proven design property of MV — it is a descriptive observation of one experiment.

---

## DIFE Fit Quality

The fitted RMSE values (0.030–0.045 on perm_mnist) demonstrate the equation captures the shape of forgetting curves well as an offline fitting tool. This is the strongest current claim: DIFE is a good forgetting model.

Whether the fitted parameters produce useful online signals in real-time training depends on how quickly α/β stabilize across tasks — which is dataset-dependent and not yet fully characterized.

---

## What Would Strengthen the Claims

1. **Canonical rerun results** (in progress): 6 methods × 5 seeds with calibrated β_init, confirming whether DIFE fires below r_max cap in split-CIFAR.
2. **Harder benchmarks**: 5 epochs/task, more diverse task sequences — enough intra-task signal for MV to show epoch-level structure.
3. **Ablation of β_init**: Systematic sweep of β priors to characterize when DIFE transitions from cap-saturated to genuinely adaptive.
4. **Larger seed counts**: ≥ 10 seeds on key comparisons to distinguish signal from noise in the DIFE_MV vs ConstReplay gap.
