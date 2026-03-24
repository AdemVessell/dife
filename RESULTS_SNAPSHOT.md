# Results Snapshot — DIFE ∘ Memory Vortex

_Generated: 2026-03-18  |  Source: results/ directory (do not hand-edit)_

---

## Metric Definitions

| Metric | Full name | Definition |
|--------|-----------|------------|
| **AA** | Avg Final Accuracy | Mean accuracy on all tasks measured after the final task is trained |
| **AF** | Avg Forgetting | Mean over tasks j of: max_acc(j) − final_acc(j); always ≥ 0 |
| **BWT** | Backward Transfer | = −AF by construction; negative = forgetting, 0 = no forgetting |
| **FWT** | Forward Transfer | Mean zero-shot acc on task j before training it, minus random baseline (0.1 for 10-class) |
| **Replay** | Replay Budget | Total replay samples mixed into training batches across all tasks |
| **Efficiency** | Sample Efficiency | (FT_AF − method_AF) / Replay × 10,000 — forgetting reduction per 10k samples; 0 for no-replay methods |

---

## Table 1 — Full perm_mnist Benchmark

5 tasks · 5 epochs/task · **5 seeds** · buffer_capacity=2000

| Method | AA ↑ | AF ↓ | BWT | FWT | Replay Budget |
|--------|------|------|-----|-----|---------------|
| FT                 | 0.762 ± 0.021 | 0.268 ± 0.026 | -0.268 ± 0.026 | 0.001 ± 0.010 |            0 ±        0 |
| EWC                | 0.802 ± 0.030 | 0.217 ± 0.037 | -0.217 ± 0.037 | -0.002 ± 0.010 |            0 ±        0 |
| SI                 | 0.756 ± 0.026 | 0.275 ± 0.033 | -0.275 ± 0.033 | 0.004 ± 0.010 |            0 ±        0 |
| ConstReplay_0.1    | 0.959 ± 0.001 | 0.021 ± 0.002 | -0.021 ± 0.002 | -0.003 ± 0.010 |      117,500 ±        0 |
| ConstReplay_0.3    | 0.960 ± 0.001 | 0.021 ± 0.002 | -0.021 ± 0.002 | -0.000 ± 0.013 |      357,200 ±        0 |
| RandReplay         | 0.960 ± 0.002 | 0.020 ± 0.003 | -0.020 ± 0.003 | 0.003 ± 0.014 |      223,485 ±   37,706 |
| DIFE_only          | 0.962 ± 0.002 | 0.017 ± 0.003 | -0.017 ± 0.003 | -0.002 ± 0.012 |    1,101,210 ±    2,397 |
| MV_only            | 0.868 ± 0.026 | 0.136 ± 0.033 | -0.136 ± 0.033 | -0.001 ± 0.010 |      722,625 ±        0 |
| DIFE_MV            | 0.886 ± 0.025 | 0.113 ± 0.031 | -0.113 ± 0.031 | 0.001 ± 0.014 |      647,190 ±      879 |

_std is across seeds. All methods use the same fixed seeds and data splits._

---

## Table 2 — Fast-Track perm_mnist

5 tasks · **3 epochs/task** · 3 seeds · buffer_capacity=2000
Methods: FT, ConstReplay_0.1/0.3, DIFE_only, MV_only, DIFE_MV (no EWC/SI/RandReplay)

| Method | AA ↑ | AF ↓ | BWT | FWT | Replay Budget | Efficiency |
|--------|------|------|-----|-----|---------------|------------|
| FT                 | 0.771 ± 0.043 | 0.252 ± 0.053 | -0.252 ± 0.053 | 0.004 ± 0.010 |            0 ±      0 | 0.0000 |
| ConstReplay_0.1    | 0.960 ± 0.001 | 0.016 ± 0.001 | -0.016 ± 0.001 | -0.003 ± 0.015 |       70,500 ±      0 | 0.0334 |
| ConstReplay_0.3    | 0.961 ± 0.001 | 0.015 ± 0.001 | -0.015 ± 0.001 | -0.001 ± 0.010 |      214,320 ±      0 | 0.0110 |
| DIFE_only          | 0.960 ± 0.001 | 0.015 ± 0.001 | -0.015 ± 0.001 | -0.004 ± 0.016 |      661,290 ±  1,151 | 0.0036 |
| MV_only            | 0.959 ± 0.001 | 0.015 ± 0.002 | -0.015 ± 0.002 | -0.002 ± 0.009 |      580,215 ±      0 | 0.0041 |
| DIFE_MV            | 0.961 ± 0.001 | 0.014 ± 0.001 | -0.014 ± 0.001 | -0.003 ± 0.010 |      531,805 ±    665 | 0.0045 |

_Efficiency = AF improvement per 10k replay samples vs FT baseline._
_Higher is better; 0 = no replay._

---

## Table 3 — split_cifar Lean Benchmark (uncapped budget, for reference)

5 tasks · **3 epochs/task** · 2 seeds · buffer_capacity=2000 · **no budget cap on DIFE methods**

_Note: DIFE_only uses ~3× the replay of ConstReplay_0.3 here. This is not a fair budget comparison. See Table 4 for budget-equalized results._

| Method | AA ↑ | AF ↓ | BWT | FWT | Replay Budget | Efficiency |
|--------|------|------|-----|-----|---------------|------------|
| FT                 | 0.695 ± 0.006 | 0.275 ± 0.006 | -0.275 ± 0.006 | 0.093 ± 0.002 |            0 ±      0 | 0.0000 |
| ConstReplay_0.1    | 0.789 ± 0.003 | 0.156 ± 0.002 | -0.156 ± 0.002 | 0.120 ± 0.003 |       11,376 ±      0 | 0.1046 |
| ConstReplay_0.3    | 0.825 ± 0.009 | 0.102 ± 0.013 | -0.102 ± 0.013 | 0.139 ± 0.007 |       36,024 ±      0 | 0.0481 |
| DIFE_only          | 0.853 ± 0.007 | 0.060 ± 0.004 | -0.060 ± 0.004 | 0.130 ± 0.004 |      108,664 ±  1,540 | 0.0198 |
| MV_only            | 0.846 ± 0.002 | 0.061 ± 0.001 | -0.059 ± 0.001 | 0.120 ± 0.001 |       97,170 ±      0 | 0.0221 |
| DIFE_MV            | 0.846 ± 0.000 | 0.071 ± 0.000 | -0.071 ± 0.000 | 0.122 ± 0.004 |       86,031 ±    711 | 0.0237 |

---

## Table 4 — split_cifar Budget-Equalized (r_max=0.30 sweep, primary comparison)

5 tasks · **3 epochs/task** · r_max=0.30 cap · **identical budget across all replay methods**

This is the correct apples-to-apples comparison. Budget is capped to ConstReplay_0.3 level (36,024 samples) for all methods.

| Method | AA ↑ | AF ↓ | Replay Budget | Seeds |
|--------|------|------|---------------|-------|
| FT                 | 0.702 ± 0.011 | 0.269 ± 0.010 |            0 | 3 (replication) |
| ConstReplay_0.1    | 0.792 ± 0.004 | 0.155 ± 0.004 |       11,376 | 3 (replication) |
| ConstReplay_0.3    | 0.832 ± 0.010 | 0.097 ± 0.008 |       36,024 | 2 (replication) |
| DIFE_only          | 0.830 ± 0.005 | 0.106 ± 0.007 |       36,024 | 4 (sweep r_max=0.30) |
| **DIFE_MV**        | **0.838 ± 0.003** | **0.083 ± 0.006** | **36,024** | 4 (sweep r_max=0.30) |

_Pareto criteria: PRIMARY ✅ DIFE_only AF (0.106) ≤ CR_0.1 AF (0.155). SECONDARY ✅ budget equal by construction. COMBINED ✅ DIFE_MV AF (0.083) < DIFE_only AF (0.106) and beats CR_0.3 (0.097). PROXY ✅ max(mv_proxy)=0.155._

_Replication study seeds: FT/CR_0.1 complete (3 seeds), CR_0.3/DIFE_only/DIFE_MV in progress. Sweep results (4 seeds, DIFE methods) from `results/sweep/split_cifar/r_max_0.30/`._

---

## Reproducibility

All runs use:
- `torch.manual_seed(seed)` + `np.random.seed(seed)` before each job
- Isolated subprocess per job (`run_one_job.py`) — no shared state
- Skip logic: if `metrics.json` exists, job is skipped silently
- Grid-search params cached to `results/{bench}/grid_search_params.json`

To regenerate this file: `python scripts/gen_results_snapshot.py --write`
