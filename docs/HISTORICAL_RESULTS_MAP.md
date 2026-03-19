# Historical Results Map — Non-Canonical Split-CIFAR Families

**Status:** All entries below are NON-CANONICAL.
**Do not blend these tables with each other or with the canonical run.**
**Canonical results live in:** `results/canonical/split_cifar_rmax_0.30/`
**Created by:** canonical/audit-rebuild branch

---

## Why Historical Results Are Contaminated

Three distinct contamination axes make it impossible to compare these families to each other or to the canonical run:

1. **Fitter version** — The online `OnlineDIFEFitter` was updated in commit `d738e49` to use L-BFGS-B with `BETA_BOUNDS=(0.001, 1.0)`. Results generated before this commit used an unbounded fitter that allowed β → 1e-9. The job skip logic (`if os.path.exists(out_path): return`) prevented re-generation.

2. **Code state divergence** — The two `r_max_0.30` result sets (r_max_0.30 and r_max_0.30_full) were generated at different commits, meaning the DIFE_MV cold-start fix and λ=0 guard fix were applied to one but not the other. The same seeds (0–3) produce AF=0.083 in one run and AF=0.097 in the other — a direct contradiction from different code states.

3. **Missing MV_only baseline** — The r_max_0.30 (non-full) sweep contains only DIFE_only and DIFE_MV, making it impossible to isolate the DIFE contribution from the MV contribution without mixing in a different result family.

---

## Family-by-Family Registry

### 1. `results/fast_track/split_cifar/`

| Field | Value |
|-------|-------|
| Seed count | 2 (seeds 0–1) |
| Epochs/task | 3 |
| r_max / cap | **Uncapped** |
| Methods | FT, ConstReplay_0.1, ConstReplay_0.3, DIFE_only, MV_only, DIFE_MV |
| Fitter version | **Pre-fix** (unbounded; β may be as low as 1e-7) |
| Code state | Pre-`d738e49`, pre-cold-start fix, pre-λ=0 guard |
| Pre-fix or post-fix | Pre-fix |

**Why non-canonical:** Uncapped runs give DIFE_only 2–3× the replay budget of ConstReplay_0.3, making performance comparisons invalid. 2 seeds is underpowered. Old unbounded fitter. Cannot be blended with any capped result.

---

### 2. `results/split_cifar/` (root-level)

| Field | Value |
|-------|-------|
| Seed count | 1 (seed 0, FT only) |
| Epochs/task | Unknown (early exploratory run) |
| r_max / cap | Uncapped |
| Methods | FT only |
| Fitter version | N/A |
| Code state | Early exploratory |
| Pre-fix or post-fix | Pre-fix |

**Why non-canonical:** Single seed, single method, exploratory. Not a comparison experiment.

---

### 3. `results/sweep/split_cifar/r_max_0.05/`

| Field | Value |
|-------|-------|
| Seed count | 4 (seeds 0–3) |
| Epochs/task | 3 |
| r_max / cap | 0.05 |
| Methods | DIFE_only, DIFE_MV |
| Fitter version | **Pre-fix** — stored β: 1e-8 to 5e-8, far below BETA_BOUNDS lower bound of 0.001 |
| Code state | Pre-`d738e49` (bounded fitter not yet active) |
| Pre-fix or post-fix | Pre-fix |

**Why non-canonical:** No baseline methods (FT, ConstReplay, MV_only absent). Pre-fix fitter. Cannot answer whether MV or DIFE is responsible for any difference.

---

### 4. `results/sweep/split_cifar/r_max_0.10/`

| Field | Value |
|-------|-------|
| Seed count | 4 (seeds 0–3) |
| Epochs/task | 3 |
| r_max / cap | 0.10 |
| Methods | DIFE_only, DIFE_MV |
| Fitter version | **Pre-fix** — stored β: 1e-8 to 2e-7 |
| Code state | Pre-`d738e49` |
| Pre-fix or post-fix | Pre-fix |

**Why non-canonical:** Same as r_max_0.05. No baselines. Pre-fix fitter.

---

### 5. `results/sweep/split_cifar/r_max_0.20/`

| Field | Value |
|-------|-------|
| Seed count | 8 (seeds 0–7) |
| Epochs/task | 3 |
| r_max / cap | 0.20 |
| Methods | DIFE_only, DIFE_MV |
| Fitter version | **Pre-fix** — stored β: 5e-9 to 4e-7 |
| Code state | Pre-`d738e49` |
| Pre-fix or post-fix | Pre-fix |

**Why non-canonical:** No baselines. Pre-fix fitter. Despite 8 seeds, the absence of ConstReplay_0.3 and MV_only at the same r_max makes the crossover claim unverifiable.

---

### 6. `results/sweep/split_cifar/r_max_0.30/` ← **Conflicting run A**

| Field | Value |
|-------|-------|
| Seed count | 4 (seeds 0–3) |
| Epochs/task | 3 |
| r_max / cap | 0.30 |
| Methods | DIFE_only, DIFE_MV only |
| Fitter version | **Pre-fix** — stored β: 2e-8 to 1e-6 |
| Code state | Pre-`d738e49`, pre-λ=0 guard |
| Pre-fix or post-fix | Pre-fix |

**Why non-canonical:** DIFE_MV AF=0.083 in this run. No MV_only baseline. Pre-fix fitter. Directly contradicted by r_max_0.30_full (same seeds, different code state, DIFE_MV AF=0.097). The headline "DIFE_MV beats ConstReplay_0.3" was derived by mixing this run's DIFE_MV numbers with a different run's ConstReplay numbers — an invalid cross-family comparison.

---

### 7. `results/sweep/split_cifar/r_max_0.30_full/` ← **Conflicting run B**

| Field | Value |
|-------|-------|
| Seed count | 4 (seeds 0–3) |
| Epochs/task | 3 |
| r_max / cap | 0.30 |
| Methods | FT, ConstReplay_0.1, ConstReplay_0.3, DIFE_only, MV_only, DIFE_MV |
| Fitter version | **Pre-fix** — stored β: 5e-8 to 1e-6 |
| Code state | Post-`1d23488` (λ=0 guard fix), but still pre-β-bounds fix |
| Pre-fix or post-fix | Partial fix (λ guard fixed, β bounds not) |

**Why non-canonical:** More complete than r_max_0.30 but still uses old β-collapse fitter. Contradicts r_max_0.30 for DIFE_MV at the same seeds (0.097 vs 0.083 AF). Code state intermediate — neither fully old nor fully current.

---

### 8. `results/sweep_mix/split_cifar/` (all r_max levels)

| Field | Value |
|-------|-------|
| Seed count | **1 (seed 0 only)** |
| Epochs/task | 3 |
| r_max / cap | 0.05, 0.10, 0.20, 0.30 |
| Methods | DIFE_only, DIFE_MV |
| Fitter version | Pre-fix |
| Code state | Post-λ guard, pre-β-bounds fix |
| Pre-fix or post-fix | Partial fix |

**Why non-canonical:** Single seed throughout. Results in `SUMMARY_SWEEP_MIX.md` are 1-seed and cannot be treated as reliable estimates of mean or variance.

---

### 9. `results/replication_study/split_cifar/`

| Field | Value |
|-------|-------|
| Seed count | 2–3 (incomplete; ConstReplay_0.1 has 3; others have 2) |
| Epochs/task | 3 |
| r_max / cap | Capped at ConstReplay_0.3 budget |
| Methods | FT, ConstReplay_0.1, ConstReplay_0.3, DIFE_only, DIFE_MV (no MV_only) |
| Fitter version | **Post-fix** — β pinned at 0.001 (lower bound) |
| Code state | Post-`d738e49` (bounded fitter active) |
| Pre-fix or post-fix | Post-fix |

**Why non-canonical:** Incomplete — seeds missing for most methods. No MV_only baseline. Cannot answer the core DIFE vs MV question without MV_only.

---

### 10. `results/sweep/split_cifar/5ep_r_max_0.30/`

| Field | Value |
|-------|-------|
| Seed count | 1 (seed 0 for most methods; FT has seed 0+1) |
| Epochs/task | **5** (different from all other split-CIFAR sweep runs) |
| r_max / cap | 0.30 |
| Methods | FT, ConstReplay_0.1, ConstReplay_0.3, DIFE_only, MV_only, DIFE_MV |
| Fitter version | Mixed/pre-fix |
| Code state | Intermediate |
| Pre-fix or post-fix | Pre-fix |

**Why non-canonical:** 5 epochs/task (all canonical runs use 3). Different training length changes the forgetting dynamics. Single seed. Cannot be compared to 3-epoch runs.

---

## Blending Prohibition

The following pairs of result families explicitly cannot be blended into a single comparison:

- **r_max_0.30 and r_max_0.30_full**: Same seeds (0–3), same r_max (0.30), but different code states produce DIFE_MV AF=0.083 vs AF=0.097. Blending them implies both numbers are from the same code, which is false.
- **Any uncapped run with any capped run**: Uncapped DIFE_only uses 2–3× more replay. Comparing AA/AF across cap conditions is comparing different effective budgets.
- **Pre-β-fix results with replication_study results**: The fitter behavior is categorically different (β=1e-9 vs β=0.001). The DIFE envelope shape differs by ~4 orders of magnitude in β.
- **1-seed sweep_mix results with any multi-seed result**: Single seeds have no variance estimate and cannot be treated as a population mean.
- **5-epoch results with 3-epoch results**: Different total training exposure; forgetting dynamics differ by construction.

---

*Generated: 2026-03-19 | Branch: canonical/audit-rebuild*
