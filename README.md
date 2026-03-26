# DIFE × Memory Vortex

**Decay-Interference Forgetting Equation + Adaptive Replay Controller**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Tests: 77 passed](https://img.shields.io/badge/tests-77%20passed-brightgreen.svg)](#test-suite)

> Created by **Adem Vessell**

---

## Project Status

**DIFE × Memory Vortex is a confirmed adaptive replay controller.** The beta-bound rerun (6 methods × 5 seeds × 2 beta conditions = 60 jobs, 0 failures) and an independent replication study (5 fresh seeds, March 25, 2026) are both complete. 276+ experiment jobs have been run across two benchmarks, with 271 metrics files archived.

**Key finding (β_min=0.10, split-CIFAR-10):** DIFE_MV achieves AA=0.837±0.007 and AF=0.093±0.009 using 32,422 replay samples — higher accuracy, lower forgetting, and ~10% less replay than ConstReplay_0.3's 36,024 samples at the same budget cap.

**Honest scope:** Memory Vortex drives the epoch-level adaptation that enables below-cap modulation. DIFE provides the task-level planning envelope and fit signal. DIFE alone saturates the budget cap on split-CIFAR (5/5 seeds); the adaptive controller claim applies to the *combined* system, not DIFE in isolation. This is documented in detail in [CAVEATS.md](CAVEATS.md) and [PROJECT_FINAL_REPORT.md](PROJECT_FINAL_REPORT.md).

---

## What This Is

Neural networks forget. When you train a model on a new task, it damages what it learned on previous tasks — **catastrophic forgetting**, one of the core unsolved problems in AI.

The standard fix is experience replay: keep a buffer of old data and mix it back in during training. But existing replay methods use a fixed replay rate — the same fraction of old data, every epoch, regardless of whether the model is actually forgetting. That wastes compute and doesn't adapt to reality.

**This project does something different:**

1. **DIFE** — a novel closed-form equation that *models* how a neural network forgets, fitted in real time to the model's own accuracy history
2. **Memory Vortex** — a symbolic operator that learns *when within each task* forgetting pressure peaks, shaping the per-epoch replay schedule
3. **DIFE × Memory Vortex** — a combined adaptive controller that reads forgetting as it happens and adjusts replay accordingly

---

## The DIFE Equation

```
Q_n = max(0, Q_0 · α^n − β · n · (1 − α^n))
```

| Symbol | Meaning |
| --- | --- |
| `Q_n` | Knowledge retained after `n` interference events |
| `Q_0` | Initial quality (typically 1.0) |
| `α ∈ (0,1)` | Per-task retention rate (fitted from data) |
| `β > 0` | Cumulative interference strength (fitted from data) |
| `n` | Number of tasks seen since learning the skill |

Standard forgetting models use only exponential decay (`α^n`). DIFE adds a **linear interference term** — `β · n · (1 − α^n)` — that captures how multiple tasks compound their damage over time. This term grows with task count and saturates as α^n → 0, matching observed forgetting dynamics better than pure exponential baselines.

Fitted RMSE on perm_mnist: **0.030–0.045** across methods. Fitted α for EWC (0.958) > SI (0.908) > FT (0.871) correctly reflects EWC's superior per-weight retention.

---

## How the Controller Works

```
replay_fraction = clip(DIFE_envelope(task) × MV_operator(epoch), 0, 1)
```

**DIFE envelope** (task-level): After each new task, fit α and β from the observed accuracy matrix using L-BFGS-B with bounds (α ∈ [0.50, 0.97], β ∈ [β_min, 1.0]). Project future forgetting. Set a replay budget ceiling proportional to predicted forgetting severity. The β_min floor (default 0.10) prevents β from collapsing to near-zero on benchmarks where tasks share structure.

**Memory Vortex operator** (epoch-level): Fit a linear combination of 7 basis functions (sin, cos, exp, log, polynomial, sigmoid) to the per-epoch proxy forgetting signal — how well the model remembers buffer samples epoch to epoch. The result is a learned schedule of when replay should peak and trough within a task.

**Product**: DIFE sets the ceiling; MV shapes when to spend it. Two orthogonal, causally correct signals multiplied together. Controller traces show epoch-level replay fractions dropping to ~0.126–0.171 on low-forgetting tasks, then ramping back to the cap on high-forgetting tasks.

---

## Benchmark Results

### Split-CIFAR-10 — Best Configuration (β_min=0.10, r_max=0.30, 5 seeds)

Split-CIFAR has real, variable per-task forgetting — the harder benchmark that stress-tests adaptive scheduling.

| Method | Avg Accuracy ↑ | Avg Forgetting ↓ | Replay Budget |
| --- | --- | --- | --- |
| Fine-tuning | 0.700 ± 0.013 | 0.271 ± 0.013 | 0 |
| ConstReplay 10% | 0.787 ± 0.007 | 0.160 ± 0.006 | 11,376 |
| ConstReplay 30% | 0.830 ± 0.005 | 0.104 ± 0.007 | 36,024 |
| DIFE_only | 0.830 ± 0.005 | 0.104 ± 0.007 | 36,024 (at cap) |
| MV_only | 0.831 ± 0.009 | 0.099 ± 0.008 | 33,512 ± 1,562 |
| **DIFE_MV** | **0.837 ± 0.007** | **0.093 ± 0.009** | **32,422 ± 1,619** |

At equal budget cap, DIFE_MV achieves the lowest forgetting (0.093 AF vs ConstReplay_0.3's 0.104 — an 11% improvement) while using ~10% fewer replay samples. DIFE_MV fires below the r_max cap in **10/10 seeds** across both beta conditions tested.

### Split-CIFAR-10 — Cross-Beta Comparison

| Condition | DIFE_MV AA | DIFE_MV AF | DIFE_MV Replay | Cap Hits |
| --- | --- | --- | --- | --- |
| Default β | 0.835 ± 0.006 | 0.096 ± 0.008 | 32,722 ± 1,554 | 0/5 |
| β_min=0.05 | 0.833 ± 0.013 | 0.099 ± 0.017 | 32,674 ± 1,533 | 0/5 |
| **β_min=0.10** | **0.837 ± 0.007** | **0.093 ± 0.009** | **32,422 ± 1,619** | **0/5** |
| ConstReplay_0.3 | 0.830 ± 0.005 | 0.104 ± 0.007 | 36,024 | N/A (fixed) |

### Permuted MNIST (5 tasks, 5 epochs/task, 5 seeds)

| Method | Avg Accuracy ↑ | Avg Forgetting ↓ |
| --- | --- | --- |
| Fine-tuning | 0.762 ± 0.022 | 0.268 ± 0.026 |
| EWC | 0.802 ± 0.030 | 0.217 ± 0.038 |
| SI | 0.756 ± 0.026 | 0.275 ± 0.033 |
| ConstReplay 10% | 0.959 ± 0.002 | 0.021 ± 0.002 |
| ConstReplay 30% | 0.960 ± 0.001 | 0.021 ± 0.002 |
| RandReplay | 0.960 ± 0.002 | 0.020 ± 0.003 |
| **DIFE_only** | **0.962 ± 0.002** | **0.018 ± 0.003** |
| MV_only | 0.868 ± 0.026 | 0.136 ± 0.033 |
| DIFE_MV | 0.886 ± 0.026 | 0.113 ± 0.031 |

On perm_mnist, **DIFE_only** achieves the lowest forgetting. The simpler, more uniform forgetting dynamics don't benefit from MV's epoch-level shaping — benchmark-dependent behavior is expected and documented.

### Independent Replication (March 25, 2026)

5 fresh seeds on split-CIFAR-10, different from all canonical seeds. All values match canonical results within error bars:

| Method | AA ↑ | AF ↓ | Replay |
| --- | --- | --- | --- |
| FT | 0.700 ± 0.013 | 0.271 ± 0.013 | 0 |
| ConstReplay_0.3 | 0.829 ± 0.008 | 0.102 ± 0.010 | 36,024 |
| DIFE_only | 0.831 ± 0.004 | 0.101 ± 0.005 | 36,024 |
| **DIFE_MV** | **0.829 ± 0.010** | **0.099 ± 0.008** | **33,148 ± 580** |

DIFE_MV continues to save ~8% of replay budget with equal or better forgetting. 4/4 automated sanity checks pass. Full report: [RESULTS_REPLICATION.md](RESULTS_REPLICATION.md).

### Pareto Criteria — Split-CIFAR (Budget-Equalized, r_max=0.30)

| Criterion | Target | Result |
| --- | --- | --- |
| DIFE_only AF ≤ ConstReplay_0.1 AF | ≤ 0.160 | ✅ **0.104** |
| DIFE_MV budget ≤ ConstReplay_0.3 budget | ≤ 36,024 | ✅ **32,422** |
| DIFE_MV AF < ConstReplay_0.3 AF | < 0.104 | ✅ **0.093** |
| MV proxy non-degenerate | > 0.01 | ✅ **0.155** |
| Replication within error bars | — | ✅ Confirmed (5 fresh seeds) |

---

## Sanity Checks

8 verified checks ensuring mathematical integrity of all results:

| # | Check | Status |
| --- | --- | --- |
| 1 | AF = −BWT (by construction) | ✅ PASS |
| 2 | Replay budget accounting matches expected formula | ✅ PASS |
| 3 | Seed determinism (run twice → identical metrics.json) | ✅ PASS |
| 4 | MV proxy signal non-degenerate | ✅ PASS |
| 5 | DIFE causal ordering (future rows don't affect past params) | ✅ PASS |
| 6 | Buffer never exceeds capacity | ✅ PASS |
| 7 | r_t_history always in [0.0, 1.0] | ✅ PASS |
| 8 | Efficiency sign correct | ✅ PASS |

Run: `python scripts/run_sanity_checks.py`

---

## Test Suite

77 tests across 10 test classes, all passing. Tests use only synthetic tensors — no network access or dataset downloads required.

```bash
pytest tests/ -v
# 77 passed in ~10s
```

| Test Class | Count | Covers |
| --- | --- | --- |
| TestDife | 14 | Core equation: boundary conditions, known values, monotonicity, parameter validation, sensitivity |
| TestDifeCurve | 3 | Curve generation: length, first element, consistency with individual calls |
| TestForgettingRate | 3 | Forgetting rate: zero at start, increasing, bounded by Q₀ |
| TestConfig | 5 | Benchmark configurations: field values, unknown bench raises, device override |
| TestReservoirBuffer | 6 | Buffer: partial fill, capacity cap, sample shape, empty edge case, no NaN after overflow |
| TestOnlineDIFEFitter | 7 | DIFE fitter: defaults, fit thresholds, replay fraction range, causality (no future leakage) |
| TestOnlineMVFitter | 7 | MV fitter: fallback below MIN_OBS, fitting at threshold, has_fit state, proxy clipping |
| TestSchedulers | 12 | All 9 methods: correct delegation, return types, value ranges, unknown method raises |
| TestMetrics | 6 | AA/AF/BWT passthrough, FWT computation, JSON serialization, CSV summary |
| TestGridSearch | 2 | EWC lambda and SI c grid search return from candidate grid |
| TestTrainer | 12 | Integration: output keys, acc_matrix shape/range, replay behavior per method, all 9 methods complete |

---

## Quickstart

```bash
git clone https://github.com/AdemVessell/dife
cd dife
pip install -r requirements.txt

# Run the DIFE equation directly
python -c "from dife import dife; print(dife(3, Q_0=1.0, alpha=0.95, beta=0.01))"

# Integration demo (no training — runs in <5 seconds)
python demo_integration.py

# Run the full benchmark suite
python run_fast_track.py

# Run sanity checks
python scripts/run_sanity_checks.py

# Run tests
pytest tests/ -v
```

---

## Integration API

Drop DIFE × Memory Vortex into any existing continual learning training loop:

```python
from eval.schedulers import get_replay_fraction, SchedulerState
from eval.online_fitters import OnlineDIFEFitter, OnlineMVFitter

dife_fitter = OnlineDIFEFitter()
mv_fitter = OnlineMVFitter()

for task_index, task_data in enumerate(tasks):

    # Get replay fraction BEFORE training this task
    state = SchedulerState(
        task_index=task_index,
        total_epochs_so_far=task_index * epochs_per_task,
        dife_fitter=dife_fitter,
        mv_fitter=mv_fitter,
        rng=rng,
    )
    r = get_replay_fraction("DIFE_MV", state)

    # Train with r fraction of replay
    train_with_replay(task_data, replay_buffer, replay_fraction=r)

    # Update fitters after each task
    dife_fitter.update(acc_matrix)   # acc_matrix[i][j] = acc on task j after task i
    mv_fitter.update()
```

No hyperparameter tuning. No held-out validation. The controller learns the model's forgetting behavior from its own training history.

---

## Experimental Scale

| Metric | Count |
| --- | --- |
| Total metrics.json files | 271 |
| Total experiment jobs completed | 276+ |
| Split-CIFAR canonical runs (default β) | 30 (6 methods × 5 seeds) |
| Beta-bound rerun (β_min=0.05) | 30 (6 methods × 5 seeds) |
| Beta-bound rerun (β_min=0.10) | 30 (6 methods × 5 seeds) |
| Perm-MNIST runs | 45 (9 methods × 5 seeds) |
| Independent replication (Mar 25, 2026) | 25 (5 methods × 5 seeds) |
| r_max sweep configurations | 5 levels × multiple seeds |
| Beta-bound rerun failure rate | **0/60 (0%)** |
| Test suite | **77 tests, all passing** |

---

## Repository Structure

```
dife/
├── dife.py                         # Core DIFE equation (Q_n = max(0, Q_0·α^n − β·n·(1−α^n)))
├── demo_integration.py             # Standalone API demo (<5 sec, no training)
├── METHODOLOGY.md                  # AI-augmented development transparency statement
│
├── eval/                           # Evaluation & controller package
│   ├── schedulers.py               # get_replay_fraction() — the main controller API
│   ├── online_fitters.py           # OnlineDIFEFitter (L-BFGS-B, β_min), OnlineMVFitter
│   ├── metrics.py                  # AA, AF, BWT, FWT, replay_budget, efficiency
│   ├── trainer.py                  # Training loop with per-epoch MV modulation
│   ├── buffer.py                   # Replay buffer (reservoir sampling)
│   ├── runner.py                   # Benchmark orchestration
│   ├── config.py                   # Method and benchmark config
│   └── grid_search.py             # EWC lambda / SI c grid search
│
├── benchmark/                      # Neural architectures & data
│   ├── models.py                   # MLP (perm_mnist), SmallCNN (CIFAR)
│   ├── data.py                     # PermutedMNIST, SplitCIFAR loaders
│   ├── baselines.py                # EWC, SI implementations
│   ├── fitting.py                  # α/β fitting + CL metrics
│   └── plotting.py                 # Visualization utilities
│
├── memory-vortex-dife-lab/         # Memory Vortex standalone module
│   ├── memory_vortex/              # Basis discovery, operator fitting, scheduler
│   ├── dife/                       # DIFE controller integration
│   └── operators/                  # Symbolic operator definitions
│
├── tests/                          # 77 pytest tests (synthetic tensors, no network)
│   ├── conftest.py                 # Import ordering fix for dife.py vs dife/ collision
│   ├── test_dife.py                # Core equation tests (20 tests)
│   └── test_eval_suite.py          # Eval package tests (57 tests)
│
├── scripts/                        # Analysis and inspection tools
│   ├── run_sanity_checks.py        # 8 automated mathematical checks
│   ├── gen_results_snapshot.py     # Regenerate results tables from disk
│   ├── gen_canonical_results.py    # Canonical experiment analysis
│   ├── gen_canonical_verdict.py    # Canonical verdict generation
│   ├── gen_sweep_summary.py        # r_max sweep analysis
│   ├── gen_assessment_pdf.py       # PDF assessment generator
│   ├── analyze_beta_rerun.py       # Beta-bound rerun analysis
│   └── inspect_controller_trace.py # Epoch-level controller trace inspector
│
├── results/                        # All benchmark outputs (metrics.json per seed)
│   ├── canonical/                  # Default β, 6 methods × 5 seeds
│   ├── canonical_beta005/          # β_min=0.05, 6 methods × 5 seeds
│   ├── canonical_beta010/          # β_min=0.10, 6 methods × 5 seeds
│   ├── perm_mnist/                 # 9 methods × 5 seeds
│   ├── replication_study/          # Independent replication (5 fresh seeds)
│   └── sweep/ + sweep_repaired/    # r_max sweep results across budget levels
│
├── docs/
│   ├── CANONICAL_RESULTS.md        # Canonical experiment data tables
│   ├── CANONICAL_VERDICT.md        # Canonical experiment verdict
│   └── RED_TEAM_CONCLUSION.md      # Adversarial review (pre-beta-fix, see note below)
│
├── run_fast_track.py               # Main benchmark runner (perm_mnist + split_cifar)
├── run_canonical.py                # Canonical experiment runner
├── run_beta_bound_rerun.py         # Beta-bound experiment runner
├── run_bulletproof.py              # Fault-tolerant job runner
├── run_replication.py              # Independent replication runner
├── generate_replication_report.py  # Replication report generator
│
├── PROJECT_FINAL_REPORT.md         # Comprehensive final report (most authoritative)
├── CAVEATS.md                      # Honest limitations and open questions
├── SANITY_CHECKS.md                # 8 verified mathematical checks
├── RESULTS_REPLICATION.md          # Independent replication results
├── SUMMARY_SWEEP.md                # r_max sweep analysis
├── HARD_BENCH_PLAN.md              # Plan for harder future benchmarks
└── DIFE_MV_Assessment_AdemVessell.pdf  # Assessment document
```

---

## Key Documents

| Document | Purpose |
| --- | --- |
| [PROJECT_FINAL_REPORT.md](PROJECT_FINAL_REPORT.md) | Comprehensive final report with all results, analysis, and project rating |
| [CAVEATS.md](CAVEATS.md) | Honest limitations, resolved issues, open work |
| [RESULTS_REPLICATION.md](RESULTS_REPLICATION.md) | Independent replication study (5 fresh seeds, Mar 25 2026) |
| [SANITY_CHECKS.md](SANITY_CHECKS.md) | 8 mathematical verification checks |
| [SUMMARY_SWEEP.md](SUMMARY_SWEEP.md) | r_max sweep analysis across budget levels |
| [METHODOLOGY.md](METHODOLOGY.md) | AI-augmented development transparency statement |
| [docs/CANONICAL_VERDICT.md](docs/CANONICAL_VERDICT.md) | Canonical experiment verdict |
| [docs/RED_TEAM_CONCLUSION.md](docs/RED_TEAM_CONCLUSION.md) | Adversarial review (written pre-beta-fix; see PROJECT_FINAL_REPORT for current state) |
| [HARD_BENCH_PLAN.md](HARD_BENCH_PLAN.md) | Plan for harder future benchmarks |

---

## What's Next

1. **Harder benchmarks** — Split-CIFAR with 5 epochs/task, CIFAR-100 splits, TinyImageNet to stress-test both components at scale
2. **Lower r_max values** — Confirm adaptive behavior persists at r_max < 0.30 with calibrated β_min
3. **Larger seed counts** — ≥ 10 seeds on key comparisons to further tighten error bars
4. **Finer β_min sweep** — Characterize the transition from cap-saturated to adaptive more precisely
5. **DIFE standalone adaptation** — Investigate conditions under which DIFE alone can fire below the cap without MV assistance

---

## Version History

| Date | Milestone |
| --- | --- |
| Initial | Core DIFE equation + Memory Vortex implementation |
| — | Perm-MNIST benchmark complete (9 methods × 5 seeds) |
| — | Split-CIFAR canonical run (6 methods × 5 seeds) |
| — | r_max sweep across 5 budget levels |
| — | Honest documentation overhaul (CAVEATS.md, RED_TEAM) |
| Mar 19, 2026 | Test suite expanded to 77 tests (conftest fix, eval coverage) |
| Mar 24, 2026 | **Beta-bound rerun complete** (60/60 jobs, 0 failures) |
| Mar 24, 2026 | **DIFE_MV confirmed as adaptive controller** (10/10 seeds below cap) |
| Mar 25, 2026 | **Independent replication complete** (5 fresh seeds, all within error bars) |

---

## Citation

```
@misc{dife2025,
  title  = {DIFE × Memory Vortex: Adaptive Replay Scheduling via Online Forgetting Dynamics},
  author = {Adem Vessell},
  year   = {2025},
  url    = {https://github.com/AdemVessell/dife}
}
```

---

## License

MIT © Adem Vessell

Copyright (c) 2025-2026 Adem Vessell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

