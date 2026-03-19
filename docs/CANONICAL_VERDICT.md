# Canonical Verdict

**Source:** `results/canonical/split_cifar_rmax_0.30/`
**Config:** split-CIFAR-10, 5 tasks, 3 epochs/task, r_max=0.30, seeds 0–4
**Branch:** canonical/audit-rebuild
**All 30 runs (6 methods × 5 seeds) generated at the same code state. No historical folder contamination.**

---

## Canonical Performance Table

| Method | Seeds | AA ↑ | AF ↓ | BWT | Replay Used |
|--------|-------|------|------|-----|-------------|
| FT | 5 | 0.700 ± 0.013 | 0.271 ± 0.013 | −0.271 ± 0.013 | 0 |
| ConstReplay_0.1 | 5 | 0.787 ± 0.007 | 0.160 ± 0.006 | −0.160 ± 0.006 | 11,376 |
| ConstReplay_0.3 | 5 | 0.830 ± 0.005 | 0.104 ± 0.007 | −0.104 ± 0.007 | 36,024 |
| DIFE_only | 5 | 0.830 ± 0.005 | 0.104 ± 0.007 | −0.104 ± 0.007 | 36,024 |
| MV_only | 5 | 0.831 ± 0.009 | 0.099 ± 0.008 | −0.099 ± 0.008 | 33,512 ± 1,562 |
| DIFE_MV | 5 | 0.835 ± 0.006 | 0.096 ± 0.008 | −0.096 ± 0.008 | 32,722 ± 1,554 |

---

## Q1. Does DIFE_only beat ConstReplay_0.3 at equal replay budget?

**NO — they are numerically identical.**

| | AA | AF | Replay |
|---|---|---|---|
| ConstReplay_0.3 | 0.830 ± 0.005 | 0.104 ± 0.007 | 36,024 |
| DIFE_only | 0.830 ± 0.005 | 0.104 ± 0.007 | 36,024 |

ΔAF = 0.000. Replay = identical. r_t_history is [0.300, 0.300, 0.300, 0.300, 0.300] for every
seed. DIFE_only and ConstReplay_0.3 are the same method in this run. DIFE's online optimizer
runs and produces envelope values (range 0.72–1.0), but the governor cap always discards them.

---

## Q2. Does MV_only beat ConstReplay_0.3 at equal replay budget?

**STATISTICALLY UNCLEAR — directionally yes, but below 1σ at 5 seeds.**

| | AA | AF | Replay |
|---|---|---|---|
| ConstReplay_0.3 | 0.830 ± 0.005 | 0.104 ± 0.007 | 36,024 |
| MV_only | 0.831 ± 0.009 | 0.099 ± 0.008 | 33,512 |

ΔAF = +0.005. Combined std = 0.011. Effect size = 0.47σ. Not statistically credible at 5 seeds.
MV_only uses less replay (33,512 vs 36,024) because its task-level r_t sometimes drops below
r_max when the MV operator reads the proxy signal as low. Directionally positive; not confirmable.

---

## Q3. Does DIFE_MV beat MV_only?

**NOT MEANINGFULLY.**

| | AA | AF | Replay |
|---|---|---|---|
| MV_only | 0.831 ± 0.009 | 0.099 ± 0.008 | 33,512 |
| DIFE_MV | 0.835 ± 0.006 | 0.096 ± 0.008 | 32,722 |
| Δ (DIFE_MV − MV_only) | +0.004 AA | −0.003 AF | −790 replay |

ΔAF = 0.003. Effect size = 0.27σ. This is noise.

**Trace-level comparison:**

- Comparable epoch pairs: 75 (5 seeds × 5 tasks × 3 epochs)
- Epochs where |DIFE_MV − MV_only| after-cap replay > 0.001: **14 / 75 (18.7%)**
- Max difference: 0.051
- Mean absolute difference: 0.005

The 18.7% discrepancy has a specific mechanical cause (see Q4). DIFE_MV uses systematically
**less** replay than MV_only (not more). The DIFE component is acting as a downscaler, not an amplifier.
Performance difference is within noise.

---

## Q4. Does DIFE actually influence the online replay controller in the canonical capped run?

**This is the core question. Answered directly from controller traces (75 epoch observations per method, 5 seeds).**

### DIFE_only

| Metric | Value |
|--------|-------|
| r_max | 0.30 |
| Total training epochs observed | 75 (5 seeds × 5 tasks × 3 epochs) |
| Tasks where DIFE envelope < r_max | **0 / 25 (0%)** |
| Epochs where DIFE envelope < r_max | **0 / 75 (0%)** |
| DIFE envelope range | [0.721, 1.000] |
| After-cap replay fraction | 0.300 ± 0.000 (never varies) |

**DIFE does not influence DIFE_only's replay in any epoch. DIFE_only = ConstReplay_0.3.**

### DIFE_MV

| Metric | Value |
|--------|-------|
| Tasks where DIFE envelope < r_max | **0 / 25 (0%)** |
| Epochs where DIFE envelope < r_max | **0 / 75 (0%)** |
| DIFE envelope range | [0.697, 1.000], mean 0.868 |
| After-cap replay fraction | [0.116, 0.300], varies (MV-driven) |

The DIFE envelope never drops below r_max, so it never serves as the binding constraint.
However, DIFE does affect DIFE_MV through a different mechanism:

**The DIFE envelope acts as a fixed per-task multiplicative downscaler.**

The formula is `r_epoch = clip(MV_operator(epoch) × DIFE_envelope(task), 0, r_max)`.
When MV_operator < r_max (e.g., 0.175 at task 3), DIFE_envelope (~0.90) reduces it
further (0.175 × 0.90 = 0.158). When MV_operator > r_max, the clip makes both DIFE_MV
and MV_only identical. The result: DIFE_MV systematically uses less replay than MV_only
whenever MV drops below r_max.

**Concrete example (seed_0, task 3, all 3 epochs):**

| Epoch | MV_op | DIFE_env | MV_only replay | DIFE_MV replay | Diff |
|-------|-------|----------|----------------|----------------|------|
| 0 | 0.175 | 0.901 | 0.175 | 0.158 | **−0.017** |
| 1 | 0.205 | 0.901 | 0.205 | 0.185 | **−0.020** |
| 2 | 0.237 | 0.901 | 0.237 | 0.213 | **−0.023** |

This is not adaptive task-level scheduling. DIFE_envelope at task 3 is 0.901 regardless of
whether task 3 has high or low forgetting — it is purely a function of alpha^t with t=3.
The interference-adaptive role of beta has been eliminated by beta collapse to 0.001.

---

## Q5. DIFE is not materially influencing replay online. Stated plainly:

**DIFE is not functioning as an adaptive replay controller in the current canonical run.**

1. **DIFE_only = ConstReplay_0.3.** Identical AA, AF, and total replay across all 5 seeds. The online fitter runs, but its output is always overridden by the cap.

2. **DIFE's contribution to DIFE_MV is a fixed downscaler (alpha^t), not adaptive scheduling.** Beta collapses to 0.001 after task 2. At beta=0.001, the beta term (interference) contributes < 0.001 to the envelope. The envelope reduces to approximately alpha^t, which decays from 1.0 to ~0.78 over 5 tasks regardless of actual forgetting severity. This is not what DIFE is designed to do.

3. **The current bounded fitter (beta_min=0.001) does not fix the problem.** Even at beta=0.001, the envelope always exceeds r_max=0.30. The lower bound needs to be high enough that the envelope eventually drops below r_max — which requires approximately beta > 0.05 given the current alpha and r_max values.

4. **The DIFE equation is valid as a curve-fitting tool. The controller is broken at this r_max.**

---

## Q6. Quantify DIFE's influence where it exists.

DIFE influences DIFE_MV's per-epoch replay in 18.7% of epochs (14/75), exclusively when
MV_operator drops below r_max. The influence is a fixed multiplicative reduction of
approximately 10–13% (DIFE_envelope mean = 0.868). This is not task-adaptive;
it is a consequence of the alpha^t decay formula with collapsed beta.

Total replay reduction: 790 samples (2.3%) over a full 5-seed run.
Performance impact: ΔAF = 0.003 (well within noise).

---

*Generated: 2026-03-19 | Branch: canonical/audit-rebuild | All 30 jobs from same code state.*
