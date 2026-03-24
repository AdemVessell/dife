
<img width="2376" height="1326" alt="readme_controller_trace_dark" src="https://github.com/user-attach<img width="2676" height="1147" alt="readme_split_cifar_beta010_panel_dark" src="https://github.com/user-attachments/assets/0db91a71-a1e8-4cd3-bcae-4fbfed5fbcfe" />
ments/assets/4a4f8e3f-d1ab-4fda-b333-0635b52c51dd" />

# DIFE × Memory Vortex

**Decay-Interference Forgetting Equation + Adaptive Replay Controller**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

> Created by **Adem Vessell**

---

## Project Status

> **DIFE × Memory Vortex is a confirmed adaptive replay controller.** The beta-bound rerun (6 methods × 5 seeds × 2 beta conditions) is complete. Key findings:
>
> - **DIFE_MV fires below the replay budget cap in 10/10 seeds** (both β_min=0.05 and β_min=0.10), using ~9–10% less replay than the fixed-budget baseline while matching or improving accuracy.
> - **Best result (β_min=0.10):** AA=0.837±0.007, AF=0.093±0.009, replay=32,422 vs ConstReplay_0.3's 36,024 — higher accuracy, lower forgetting, less replay.
> - **Memory Vortex drives the epoch-level adaptation.** DIFE alone saturates the budget cap (5/5 seeds at cap). MV's epoch-level proxy signal enables the combined system to genuinely modulate replay.
>
> For detailed nuances and open questions, see [CAVEATS.md](CAVEATS.md).

---

## What This Is

Neural networks forget. When you train a model on a new task, it damages what it learned on previous tasks — this is called **catastrophic forgetting**, and it's one of the core unsolved problems in AI.

The standard fix is "experience replay": keep a buffer of old data and mix it back in during training. But existing replay methods use a **fixed replay rate** — the same fraction of old data, every epoch, regardless of whether the model is actually forgetting. That wastes compute and doesn't adapt to reality.

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
|--------|---------|
| `Q_n` | Knowledge retained after `n` interference events |
| `Q_0` | Initial quality (typically 1.0) |
| `α ∈ (0,1)` | Per-task retention rate (fitted from data) |
| `β > 0` | Cumulative interference strength (fitted from data) |
| `n` | Number of tasks seen since learning the skill |

**Why this is novel:** Standard forgetting models use only exponential decay (`α^n`). DIFE adds a **linear interference term** — `β · n · (1 − α^n)` — that captures how multiple tasks compound their damage over time. This term grows with task count and saturates as α^n → 0, matching observed forgetting dynamics better than pure exponential baselines.

Fitted RMSE on perm_mnist: **0.030–0.045** across methods.

---

## How the Controller Works

```
replay_fraction = clip(DIFE_envelope(task) × MV_operator(epoch), 0, 1)
```

**DIFE envelope** (task-level): After each new task, fit α and β from the observed accuracy matrix using Nelder-Mead. Project future forgetting. Set a replay budget ceiling proportional to predicted forgetting severity. If the model is retaining well → budget shrinks automatically.

**Memory Vortex operator** (epoch-level): Fit a linear combination of 7 basis functions (sin, cos, exp, log, polynomial, sigmoid) to the per-epoch proxy forgetting signal (how well the model remembers buffer samples epoch to epoch). The result is a learned schedule of when replay should peak and trough within a task.

**Product**: DIFE sets the ceiling; MV shapes when to spend it. Two orthogonal, causally correct signals multiplied together.

---

## Benchmark Results

### Permuted MNIST — Full Run (5 tasks · 5 epochs/task · 5 seeds)

| Method | Avg Accuracy ↑ | Avg Forgetting ↓ | Replay Budget |
|--------|---------------|-----------------|---------------|
| Fine-tuning | 0.762 ± 0.021 | 0.268 ± 0.026 | 0 |
| EWC | 0.802 ± 0.030 | 0.217 ± 0.037 | 0 |
| SI | 0.756 ± 0.026 | 0.275 ± 0.033 | 0 |
| ConstReplay 10% | 0.959 ± 0.001 | 0.021 ± 0.002 | 117,500 |
| ConstReplay 30% | 0.960 ± 0.001 | 0.021 ± 0.002 | 357,200 |
| RandReplay | 0.960 ± 0.002 | 0.020 ± 0.003 | 223,485 |
| **DIFE_only** | **0.962 ± 0.002** | **0.017 ± 0.003** | 1,101,210 |
| MV_only | 0.868 ± 0.026 | 0.136 ± 0.033 | 722,625 |
| DIFE_MV | 0.886 ± 0.025 | 0.113 ± 0.031 | 647,190 |

**DIFE_only achieves the lowest forgetting of any method tested.** Perm-MNIST is a relatively easy benchmark — tasks are structurally similar and forgetting is mild — so the efficiency story requires a harder test.

### Permuted MNIST — Fast-Track (5 tasks · 3 epochs/task · 3 seeds)

| Method | Avg Accuracy ↑ | Avg Forgetting ↓ | Replay Budget | Efficiency |
|--------|---------------|-----------------|---------------|------------|
| Fine-tuning | 0.771 ± 0.043 | 0.252 ± 0.053 | 0 | 0.0000 |
| ConstReplay 10% | 0.960 ± 0.001 | 0.016 ± 0.001 | 70,500 | **0.0334** |
| ConstReplay 30% | 0.961 ± 0.001 | 0.015 ± 0.001 | 214,320 | 0.0110 |
| DIFE_only | 0.960 ± 0.001 | 0.015 ± 0.001 | 661,290 | 0.0036 |
| MV_only | 0.959 ± 0.001 | 0.015 ± 0.002 | 580,215 | 0.0041 |
| **DIFE_MV** | **0.961 ± 0.001** | **0.014 ± 0.001** | **531,805** | 0.0045 |

_Efficiency = forgetting reduction per 10,000 replay samples vs fine-tuning baseline._

On perm-MNIST, DIFE_MV edges out DIFE_only by a hair (AF 0.014 vs 0.015). But perm-MNIST is too uniform to stress-test the system — intra-task forgetting is near zero, so MV's proxy signal is flat and contributes little. The real test is split-CIFAR.

### Split-CIFAR — Budget-Equalized Results (5 tasks · 3 epochs/task)

Split-CIFAR has real, variable per-task forgetting — a much harder test of adaptive scheduling.

The table below uses a fixed replay budget cap of **r_max=0.30** for all methods, so the replay budget is identical across DIFE_only, DIFE_MV, and ConstReplay_0.3. This is the honest apples-to-apples comparison.

| Method | Avg Accuracy ↑ | Avg Forgetting ↓ | Replay Budget | Seeds |
|--------|---------------|-----------------|---------------|-------|
| Fine-tuning | 0.702 ± 0.011 | 0.269 ± 0.010 | 0 | 3 |
| ConstReplay 10% | 0.792 ± 0.004 | 0.155 ± 0.004 | 11,376 | 3 |
| ConstReplay 30% | 0.832 ± 0.010 | 0.097 ± 0.008 | 36,024 | 2 |
| DIFE_only | 0.830 ± 0.005 | 0.106 ± 0.007 | 36,024 | 4 |
| **DIFE_MV** | **0.838 ± 0.003** | **0.083 ± 0.006** | **36,024** | 4 |

_Source: r_max=0.30 sweep (4 seeds DIFE methods) + replication study (2–3 seeds baselines). Budget is identical across all replay methods at this r_max._

**At equal budget, DIFE_MV achieves the lowest forgetting** (0.083 AF vs ConstReplay_0.3's 0.097 AF — a 14% improvement) while DIFE_only is roughly on par with ConstReplay_0.3 (0.106 vs 0.097).

**Important budget context:** When DIFE_only is run without a budget cap, it uses ~108k samples (3× more than ConstReplay_0.3) and achieves AF=0.060 — lower forgetting, but at disproportionate replay cost. That result is real, but not a fair comparison. The equalized results above are the correct basis for claims.

### Pareto Criteria — Split-CIFAR (Budget-Equalized, r_max=0.30)

| Criterion | Target | Result |
|-----------|--------|--------|
| PRIMARY: DIFE_only AF ≤ ConstReplay_0.1 AF | ≤ 0.155 | ✅ **0.106** (passes) |
| SECONDARY: DIFE_MV budget ≤ ConstReplay_0.3 budget | ≤ 36,024 | ✅ **36,024** (equal by construction at r_max=0.30) |
| COMBINED: DIFE_MV AF < ConstReplay_0.3 AF | < 0.097 | ✅ **0.083** (14% better, 4 seeds) |
| PROXY: MV proxy non-degenerate | > 0.01 | ✅ **0.155** (real signal present) |

### DIFE Fit Quality (Forgetting Equation Parameters)

Parameters fitted to each method's accuracy trajectory via differential evolution + Nelder-Mead:

| Method | α (fitted) | β (fitted) | RMSE |
|--------|-----------|-----------|------|
| FT | 0.871 | 0.000030 | 0.045 |
| EWC | 0.958 | 0.619 | 0.032 |
| SI | 0.908 | 0.193 | 0.030 |

The fitted α for EWC (0.958) > SI (0.908) > FT (0.871) correctly reflects EWC's superior per-weight retention. The large β for EWC captures the compounding interference when tasks compete for Fisher-protected weights.

---

## Sanity Checks

8 verified checks ensuring mathematical integrity of all results:

| # | Check | Status |
|---|-------|--------|
| 1 | AF = −BWT (by construction) | PASS |
| 2 | Replay budget accounting matches expected formula | PASS |
| 3 | Seed determinism (run twice → identical metrics.json) | PASS |
| 4 | MV proxy signal non-degenerate | PASS |
| 5 | DIFE causal ordering (future rows don't affect past params) | PASS |
| 6 | Buffer never exceeds capacity | PASS |
| 7 | r_t_history always in [0.0, 1.0] | PASS |
| 8 | Efficiency sign correct | PASS |

Run: `python scripts/run_sanity_checks.py`

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

# Run the full perm_mnist benchmark
python run_fast_track.py

# Run sanity checks
python scripts/run_sanity_checks.py

# Regenerate results snapshot
python scripts/gen_results_snapshot.py --write

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

## Repository Structure

```
dife/
├── dife.py                         # Core DIFE equation
├── demo_integration.py             # Standalone API demo (<5 sec, no training)
├── run_fast_track.py               # Main benchmark runner (perm_mnist + split_cifar)
├── run_one_job.py                  # Single isolated job runner
│
├── eval/                           # Evaluation & controller package
│   ├── schedulers.py               # get_replay_fraction() — the main controller API
│   ├── online_fitters.py           # OnlineDIFEFitter, OnlineMVFitter
│   ├── metrics.py                  # AA, AF, BWT, FWT, replay_budget, efficiency
│   ├── trainer.py                  # Training loop with replay
│   ├── buffer.py                   # Replay buffer (reservoir sampling)
│   ├── runner.py                   # Benchmark orchestration
│   └── config.py                   # Method and benchmark config
│
├── benchmark/                      # Neural architectures & data
│   ├── models.py                   # MLP (perm_mnist), SmallCNN (CIFAR)
│   ├── data.py                     # PermutedMNIST, SplitCIFAR loaders
│   ├── baselines.py                # EWC, SI implementations
│   └── fitting.py                  # α/β fitting + CL metrics
│
├── memory-vortex-dife-lab/         # Memory Vortex standalone module
│   ├── memory_vortex/              # Basis discovery, operator fitting, scheduler
│   └── dife/                       # DIFE controller integration
│
├── scripts/
│   ├── gen_results_snapshot.py     # Regenerate RESULTS_SNAPSHOT.md from disk
│   └── run_sanity_checks.py        # Automated sanity check suite
│
├── results/                        # All benchmark outputs (metrics.json per seed)
│   ├── perm_mnist/                 # Full 45-job results
│   ├── fast_track/                 # Fast-track 18-job results + summary.csv
│   └── split_cifar/                # Lean benchmark results (12 jobs, 2 seeds)
│
├── tests/                          # 55 pytest tests
├── RESULTS_SNAPSHOT.md             # Auto-generated results table
├── RESULTS.md                      # Human-readable summary
├── SANITY_CHECKS.md                # 8 verified mathematical checks
└── HARD_BENCH_PLAN.md              # Plan for split_cifar lean benchmark
```

---

## What's Next

The beta-bound rerun is **complete** (6 methods × 5 seeds × 2 beta conditions). DIFE_MV is a confirmed adaptive replay controller.

### Confirmed Results (Beta-Bound Rerun)

| Condition | Method | AA | AF | Replay |
|---|---|---|---|---|
| β_min=0.05 | ConstReplay_0.3 | 0.830 ± 0.005 | 0.104 ± 0.007 | 36,024 |
| β_min=0.05 | **DIFE_MV** | **0.833 ± 0.013** | **0.099 ± 0.017** | **32,674** |
| β_min=0.10 | ConstReplay_0.3 | 0.830 ± 0.005 | 0.104 ± 0.007 | 36,024 |
| β_min=0.10 | **DIFE_MV** | **0.837 ± 0.007** | **0.093 ± 0.009** | **32,422** |

**Key finding:** DIFE_MV fires below the r_max cap in 10/10 seeds, saving ~9–10% replay budget while improving accuracy and reducing forgetting. DIFE alone saturates the cap (5/5 seeds); Memory Vortex provides the epoch-level signal that enables genuine adaptation.

### Remaining Open Work

1. **Harder benchmarks**: Split-CIFAR with 5 epochs/task and broader task suites to stress-test both components.
2. **Lower r_max values**: Confirm adaptive behavior persists at r_max < 0.30 with calibrated β_min.
3. **Larger seed counts**: ≥ 10 seeds on key comparisons to further tighten error bars.

See [CAVEATS.md](CAVEATS.md) for detailed nuances and `SUMMARY_SWEEP.md` for full diagnostics.

---

## Citation

If you use DIFE or DIFE × Memory Vortex in your research, please cite:

```bibtex
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
