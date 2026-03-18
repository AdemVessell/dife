# DIFE ∘ Memory Vortex — Colossus-2 Fast-Track Guide

This document explains the system architecture, what the fast-track evaluation shows,
and how to integrate the combined controller into a large training loop.

---

## What DIFE Is

**DIFE** (Decay-Interference Forgetting Equation) is an online diagnostic and scheduling signal.

The equation:

```
Q(n) = max(0, Q₀ · αⁿ  −  β · n · (1 − αⁿ))
```

where:
- `Q₀` — quality (accuracy) immediately after learning a task
- `α ∈ (0, 1)` — retention rate (how slowly the model forgets)
- `β > 0` — interference strength (how much new tasks hurt old ones)
- `n` — number of tasks learned since this task

**Key property**: α and β are fitted *causally* — only from the forgetting trajectories
already observed, with no future peeking. The `OnlineDIFEFitter` uses Nelder-Mead
minimisation over the lower-triangular accuracy matrix accumulated so far.

**What it produces**: a scalar `r_DIFE(t) = DIFE(t, Q₀=1, α, β)` per task, representing
the predicted fraction of quality already lost — which we use as the replay need signal.

**Stability**: α converges to ~0.994 and β to near-zero on perm_mnist within 3 tasks,
with low standard deviation across seeds. This means DIFE gives a consistent, interpretable
signal very early in training.

---

## What Memory Vortex Is

**Memory Vortex (MV)** is an adaptive operator/policy that governs replay scheduling at
the *within-task* (per-epoch) level.

Architecture:
1. A **7-basis function system**: `{sin(t), cos(t), sin(2t), cos(2t), exp(−0.1t), t, log(1+|sin(t)|)}`
2. A **GCA Discovery Engine** (ridge regression) that fits the operator coefficients online
   from `(epoch_index, proxy_value)` pairs observed during training
3. A **fallback operator** (pre-fitted JSON) used until ≥15 observations are available

The proxy signal MV is fitted to:
```
proxy(epoch) = 1 − accuracy_on_buffer_samples
```

This measures how much the model is currently forgetting buffered examples — a real-time
indicator of replay need that doesn't require access to old task test sets.

**What it produces**: a scalar `r_MV(epoch)` per training epoch, representing the
structural replay need at that moment in time.

---

## What the Combined Controller Does

**DIFE_MV** combines both signals:

```python
r(t, epoch) = clip(r_DIFE(t) × r_MV(epoch), 0, 1)
```

- **DIFE** sets the task-level envelope: how much replay this task *deserves* based on
  how forgettable the model has been so far
- **MV** modulates within-task: how urgently replay is needed *right now* based on the
  live forgetting proxy

The result is a two-timescale adaptive scheduler:
- Coarse control (task level) from DIFE — stable, low-variance, interpretable
- Fine control (epoch level) from MV — reactive, adapts to training dynamics

Neither component requires labels from future tasks or validation sets at inference time.

---

## Fast-Track Results

Run: `python run_fast_track.py`

Full results in `RESULTS.md`. Key findings:

| Method | What it shows |
|--------|--------------|
| FT | Catastrophic forgetting baseline — high AF, no replay cost |
| ConstReplay_0.1/0.3 | Fixed-budget replay ceiling — good AF, high replay cost |
| DIFE_only | Adaptive task-level scheduling — low AF, best efficiency |
| MV_only | Per-epoch adaptive scheduling — mid-range, high replay at start |
| DIFE_MV | Combined controller — best accuracy recovery, competitive AF |

**Efficiency metric**: AF improvement per 10,000 replay samples vs FT baseline.
DIFE_only consistently achieves the highest efficiency — it spends replay budget only
when the forgetting curve predicts it's needed.

See `results/fast_track/plots/af_vs_budget.png` for the visual comparison and
`results/fast_track/plots/rt_proxy_seed0.png` for the scheduling signals over time.

---

## Answering the Three Key Questions

### Q1 — Is DIFE a stable online scheduler signal?

Yes. Alpha and beta converge with low cross-seed variance by task 3 (of 5), with no
access to future task data. The fitted α ≈ 0.994 on perm_mnist reflects slow forgetting,
which matches the benchmark's structure (permuted tasks share the same network).
See the Q1 table in `RESULTS.md`.

### Q2 — Does MV add measurable value?

- vs fixed replay (ConstReplay_0.1/0.3): DIFE_only achieves lower AF with fewer replay
  samples — MV adds accuracy recovery on top of this.
- vs DIFE_only: DIFE_MV improves final accuracy (AA) at the cost of slightly more replay.
  The AF difference is small. MV's main benefit is keeping accuracy stable within tasks.

### Q3 — Proxy correlation with forgetting

The accuracy-on-buffer proxy shows positive correlation with subsequent forgetting across
seeds, confirming it's a valid signal. See the Q3 section in `RESULTS.md` for per-method
correlation coefficients.

---

## Next Integration Step for a Large Training Loop

To integrate this controller into Colossus-2 or any large-scale training run:

**1. Replace discrete task boundaries with window-based DIFE fits**

```python
# Every W gradient steps, update the DIFE fit using accuracy on a small
# held-out buffer from the last W steps:
if step % window_size == 0:
    dife_fitter.update(windowed_acc_matrix)
    alpha, beta = dife_fitter.alpha, dife_fitter.beta
```

**2. Load the pre-fitted MV operator**

The operator JSON at `memory-vortex-dife-lab/operators/memory_vortex_operator.json`
can be loaded once at startup. No online fitting is needed if training dynamics are
similar to the discovery run:

```python
from memory_vortex.scheduler import MemoryVortexScheduler
mv_sched = MemoryVortexScheduler()  # loads JSON automatically
```

**3. Use the 5-line controller API**

```python
from eval.schedulers import get_replay_fraction, SchedulerState

state = SchedulerState(
    task_index=current_window,
    total_epochs_so_far=global_step // steps_per_epoch,
    dife_fitter=dife_fitter,
    mv_fitter=mv_fitter,
    rng=rng,
)
r = get_replay_fraction("DIFE_MV", state)
n_replay_samples = int(r * batch_size)
```

**4. Use a streaming reservoir buffer**

`eval/buffer.py` implements reservoir sampling (Algorithm R). It works as a drop-in
stream buffer — call `buffer.update(x, y)` on each batch and `buffer.sample(n)` to
draw replay samples. The capacity is a single hyperparameter.

**5. Monitor DIFE parameters as a diagnostic**

`alpha` and `beta` after each window are meaningful training health indicators:
- Falling `alpha` → model is forgetting faster (more interference)
- Rising `beta` → interference is structurally increasing
- Use these as early-warning signals in a training dashboard

---

## File Reference

| File | Purpose |
|------|---------|
| `run_fast_track.py` | Main fast-track runner + analysis + plot generation |
| `eval/trainer.py` | Unified training loop for all methods |
| `eval/online_fitters.py` | OnlineDIFEFitter and OnlineMVFitter |
| `eval/schedulers.py` | get_replay_fraction dispatch + SchedulerState |
| `eval/buffer.py` | ReservoirBuffer (streaming reservoir sampling) |
| `memory-vortex-dife-lab/dife/core.py` | DIFE equation |
| `memory-vortex-dife-lab/memory_vortex/discovery.py` | GCADiscoveryEngineV1 |
| `memory-vortex-dife-lab/operators/memory_vortex_operator.json` | Pre-fitted operator |
| `results/fast_track/` | Fast-track result files |
| `RESULTS.md` | Auto-generated results table and analysis |
