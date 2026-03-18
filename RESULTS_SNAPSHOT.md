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

## Table 3 — split_cifar (Partial)

**12 jobs complete** as of snapshot date.

| Method | Seed | AA | AF | Replay |
|--------|------|----|----|--------|
| ConstReplay_0.1    | seed_0 | 0.786 | 0.154 |     11,376 |
| ConstReplay_0.1    | seed_1 | 0.793 | 0.158 |     11,376 |
| ConstReplay_0.3    | seed_0 | 0.816 | 0.115 |     36,024 |
| ConstReplay_0.3    | seed_1 | 0.834 | 0.089 |     36,024 |
| DIFE_MV            | seed_0 | 0.846 | 0.072 |     85,320 |
| DIFE_MV            | seed_1 | 0.847 | 0.071 |     86,742 |
| DIFE_only          | seed_0 | 0.845 | 0.065 |    110,205 |
| DIFE_only          | seed_1 | 0.860 | 0.056 |    107,124 |
| FT                 | seed_0 | 0.701 | 0.269 |          0 |
| FT                 | seed_1 | 0.689 | 0.282 |          0 |
| MV_only            | seed_0 | 0.844 | 0.061 |     97,170 |
| MV_only            | seed_1 | 0.848 | 0.060 |     97,170 |

_Do not draw conclusions from partial data. See RESUME.md to continue this run._

---

## Reproducibility

All runs use:
- `torch.manual_seed(seed)` + `np.random.seed(seed)` before each job
- Isolated subprocess per job (`run_one_job.py`) — no shared state
- Skip logic: if `metrics.json` exists, job is skipped silently
- Grid-search params cached to `results/{bench}/grid_search_params.json`

To regenerate this file: `python scripts/gen_results_snapshot.py --write`
