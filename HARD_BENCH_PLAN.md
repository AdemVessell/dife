# Hard Benchmark Plan — split_cifar_lean (Colossus-2)

This document describes the replacement evaluation plan for split_cifar that is:
- Feasible in the current environment (processes die after ~30 min)
- Sufficient to answer the Colossus-2 integration questions
- Directly comparable to existing fast-track perm_mnist results

---

## Why the Original split_cifar Run Failed

**Root cause**: EWC on split_cifar requires computing the full Fisher information
matrix over the CIFAR dataset after each task. This takes 15–30 minutes per seed.
With 3 seeds × 5 tasks, the job time is 2–4 hours — longer than the environment
allows before killing processes.

**Not a fundamental issue**: EWC and SI are regularization baselines, not the
Colossus-2 story. The high-value comparison is DIFE/MV vs fixed-budget replay.

---

## Proposed Scope: split_cifar_lean

| Parameter | Full run | split_cifar_lean |
|-----------|----------|-----------------|
| Methods | 9 (inc. EWC, SI) | 6 (no EWC, SI, RandReplay) |
| Seeds | 3 | 2 |
| Epochs/task | 5 | 3 |
| Total jobs | 27 | 12 |
| Est. FT time | ~3 min | ~1.7 min |
| Est. DIFE/MV time | ~15–30 min | ~4–8 min |
| **Est. total** | **~2–4 hours** | **~40–60 min** |

**Methods kept**: FT, ConstReplay_0.1, ConstReplay_0.3, DIFE_only, MV_only, DIFE_MV

**Why 3 epochs/task**: perm_mnist FT took 165s at 5 epochs → ~99s at 3 epochs.
DIFE/MV overhead is proportional. All 12 jobs should fit in a single monitored session.

**Why 2 seeds (not 3)**: Reduces total time by 33%. With 2 seeds we still see
cross-seed variance. A third seed can be added later if time allows.

---

## Run Command

First, ensure `run_fast_track.py` has been updated with `--bench` support (see below).
Then:

```bash
# Run the lean split_cifar evaluation
python run_fast_track.py \
  --bench split_cifar \
  --seeds 0 1 \
  --epochs-per-task 3 \
  --methods FT ConstReplay_0.1 ConstReplay_0.3 DIFE_only MV_only DIFE_MV

# Monitor progress in another terminal:
tail -f /tmp/run_fast_track.log  # if run via setsid
```

Results will land in `results/fast_track/split_cifar/` — isolated from perm_mnist results.

**To survive session death** (use setsid):
```bash
setsid sh -c 'python run_fast_track.py --bench split_cifar --seeds 0 1 \
  --epochs-per-task 3 \
  --methods FT ConstReplay_0.1 ConstReplay_0.3 DIFE_only MV_only DIFE_MV \
  >> /tmp/fast_track_cifar.log 2>&1' &

# Check it's alive:
pgrep -f "run_fast_track.py"
tail -f /tmp/fast_track_cifar.log
```

---

## Pareto Success Criteria

The critical comparison is DIFE_only and DIFE_MV vs fixed-budget replay baselines.
On perm_mnist, DIFE_only used 9× more replay than ConstReplay_0.1 for the same AF —
this happened because perm_mnist is too easy (little real forgetting, so DIFE allocates
defensively for forgetting that barely occurs).

On split_cifar, the class-incremental structure creates real, variable forgetting.
DIFE should allocate more on volatile tasks and less on stable ones — achieving better
*forgetting-per-sample efficiency* than fixed budgets.

### Criteria (must meet PRIMARY to claim success; COMBINED is bonus)

```
PRIMARY   DIFE_only AF  ≤  ConstReplay_0.1 AF
          (matches or beats the tightest fixed-budget baseline on forgetting)

SECONDARY DIFE_only replay_budget  <  ConstReplay_0.3 replay_budget
          (achieves primary criterion with less replay than the generous fixed budget)

COMBINED  DIFE_MV AA  ≥  DIFE_only AA
          (MV adds accuracy recovery on top of DIFE's forgetting control)

PROXY     max(mv_proxy_history) > 0.01 for MV_only and DIFE_MV
          (proxy is non-degenerate — real forgetting dynamics visible in signal)
```

### Why these criteria make sense

| Criterion | What it proves |
|-----------|---------------|
| PRIMARY | DIFE is not just a high-budget replay method — it's an adaptive scheduler |
| SECONDARY | DIFE's schedule is efficient, not just throwing replay at the problem |
| COMBINED | MV's per-epoch modulation adds measurable value over task-level DIFE alone |
| PROXY | The MV proxy mechanism is actually seeing forgetting signal (validates Q3) |

### If PRIMARY fails (DIFE_only AF > ConstReplay_0.1 AF on split_cifar):

This would mean DIFE is under-allocating replay on CIFAR. Likely cause: the DIFE
equation's β term is calibrated for perm_mnist forgetting dynamics. The correct
response is not to abandon DIFE but to:
1. Inspect the fitted α and β values — if β is near-zero, DIFE underestimates interference
2. Consider initializing DIFE with a higher β prior (e.g., β=0.05 instead of 0.01)
3. This is itself a useful Colossus-2 finding: DIFE priors may need domain adaptation

---

## Expected Results (Hypothesis)

Based on perm_mnist results and split_cifar's harder forgetting dynamics:

| Method | Expected AA | Expected AF | Expected Replay |
|--------|-------------|-------------|-----------------|
| FT | ~0.55–0.65 | ~0.30–0.40 | 0 |
| ConstReplay_0.1 | ~0.80–0.85 | ~0.08–0.12 | ~50k |
| ConstReplay_0.3 | ~0.82–0.87 | ~0.07–0.10 | ~150k |
| DIFE_only | ~0.80–0.87 | ~0.06–0.10 | variable |
| MV_only | ~0.75–0.85 | ~0.08–0.12 | variable |
| DIFE_MV | ~0.82–0.88 | ~0.06–0.09 | variable |

These are rough hypotheses. The actual numbers, and whether DIFE's replay budget
is lower or higher than fixed baselines, is the key empirical question this run answers.

---

## After the Run

Once results land in `results/fast_track/split_cifar/`:

```bash
# Re-run snapshot generator to include split_cifar fast-track table
python scripts/gen_results_snapshot.py --write

# Re-run sanity checks (automatically picks up new files)
python scripts/run_sanity_checks.py

# Evaluate success criteria
python - <<'EOF'
import json, glob, numpy as np

results = {}
for f in glob.glob("results/fast_track/split_cifar/*/seed_*/metrics.json"):
    method = f.split("/")[3]
    data = json.load(open(f))
    results.setdefault(method, []).append(data)

ft_af  = np.mean([d["avg_forgetting"] for d in results["FT"]])
cr_af  = np.mean([d["avg_forgetting"] for d in results["ConstReplay_0.1"]])
cr3_budget = np.mean([d["total_replay_samples"] for d in results["ConstReplay_0.3"]])
dife_af = np.mean([d["avg_forgetting"] for d in results["DIFE_only"]])
dife_budget = np.mean([d["total_replay_samples"] for d in results["DIFE_only"]])
difemv_aa = np.mean([d["avg_final_acc"] for d in results["DIFE_MV"]])
dife_aa = np.mean([d["avg_final_acc"] for d in results["DIFE_only"]])
mv_proxy = max(x for d in results.get("MV_only",[]) for x in d.get("mv_proxy_history",[]) or [0])

print(f"PRIMARY:   DIFE_only AF ({dife_af:.3f}) <= CR_0.1 AF ({cr_af:.3f})? {'PASS' if dife_af <= cr_af else 'FAIL'}")
print(f"SECONDARY: DIFE_only budget ({dife_budget:,.0f}) < CR_0.3 budget ({cr3_budget:,.0f})? {'PASS' if dife_budget < cr3_budget else 'FAIL'}")
print(f"COMBINED:  DIFE_MV AA ({difemv_aa:.3f}) >= DIFE_only AA ({dife_aa:.3f})? {'PASS' if difemv_aa >= dife_aa else 'FAIL'}")
print(f"PROXY:     max(mv_proxy) ({mv_proxy:.4f}) > 0.01? {'PASS' if mv_proxy > 0.01 else 'FAIL'}")
EOF
```

---

## Connection to Colossus-2

If all four criteria pass on split_cifar_lean, the argument for Colossus-2 integration is:

> DIFE adaptively allocates replay based on observed forgetting dynamics, matching or
> outperforming fixed-budget baselines on both perm_mnist (easy) and split_cifar (hard)
> with variable and often lower total replay cost. MV adds within-task fine-tuning using
> a live proxy signal. The combined controller is a drop-in 5-line API requiring only a
> replay buffer and an accuracy evaluation hook.
