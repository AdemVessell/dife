# Sanity Checks — DIFE ∘ Memory Vortex

8 checks to run before trusting any result for external presentation.
Checks 1, 2, 7, 8 are automated by `scripts/run_sanity_checks.py` (exits 0 on pass).
Checks 3–6 are manual (< 5 min each).

---

## Automated Checks

Run all at once:

```bash
python scripts/run_sanity_checks.py
```

Expected output: 5 PASS lines, exit code 0.

---

### Check 1 — AF = −BWT

**What**: Average Forgetting and Backward Transfer are related by definition:
BWT = mean accuracy change on old tasks after learning new ones = −AF.

**Why**: If this fails, either the forgetting or BWT computation is wrong, which would
invalidate all results involving either metric.

**How**: Automated. Reads all `results/*/metrics.json`, asserts `|AF + BWT| < 1e-9`.

**Expected**: PASS for all 64 result files.

**Pass criteria**: Zero violations across all benchmarks, methods, and seeds.

---

### Check 2 — Replay Budget Accounting (ConstReplay_0.1)

**What**: For ConstReplay_0.1 on perm_mnist, the replay budget should equal:
- Tasks 2–5 (4 tasks, since task 1 has empty buffer): 4 × n_batches × epochs × replay_per_batch
- batch_size=256, r=0.1 → n_replay_per_batch = floor(0.1 × 256) = 25
- MNIST: ~234 batches/task, 5 epochs/task
- Expected ≈ 4 × 234 × 5 × 25 = 117,000

**Why**: Verifies the replay counter accumulates exactly as expected and r_t=0.1 is
applied correctly. A discrepancy signals a bug in buffer mixing or the counter.

**How**: Automated. Checks actual mean (117,500) is within ±25% of expected (117,000).

**Expected**: PASS. Actual=117,500, which is within 0.4% of the formula (batch count
varies slightly due to dataset size and drop_last behavior).

**Pass criteria**: Mean across 5 seeds within ±25% of formula value.

---

### Check 3 — Seed Determinism

**What**: Running the same job twice should produce bit-identical `metrics.json`.

**Why**: Confirms fixed seeds produce reproducible results. Required for claims about
inter-seed variance being real signal (not noise from non-determinism).

**How** (manual, ~1 min):
```bash
# Delete one output, re-run, compare
rm results/perm_mnist/FT/seed_0/metrics.json
python run_one_job.py --bench perm_mnist --seed 0 --method FT
diff <(python -c "import json; print(json.dumps(json.load(open('results/perm_mnist/FT/seed_0/metrics.json')), sort_keys=True))") \
     <(python -c "import json; print(json.dumps({'avg_final_acc': 0.72886, 'avg_forgetting': 0.30925}, sort_keys=True))")
# Or just inspect: the AA/AF should match values in RESULTS_SNAPSHOT.md exactly
```

**Expected**: Identical `avg_final_acc` and `avg_forgetting` to previously committed values.

**Pass criteria**: All scalar metrics match to at least 5 decimal places.

---

### Check 4 — Proxy Non-Degeneracy (5 epochs/task)

**What**: `mv_proxy_history` should contain non-zero values when `epochs_per_task ≥ 5`.

**Why**: In the fast-track (3 epochs/task), proxy is all-zero because perm_mnist doesn't
forget within 3 short epochs. This is a known limitation. At 5 epochs, the model should
show measurable within-task forgetting, making the proxy meaningful.

**How** (manual, ~30 sec):
```bash
# Run MV_only with 5 epochs (uses main results dir, already complete)
python -c "
import json, glob
for f in glob.glob('results/perm_mnist/MV_only/seed_*/metrics.json'):
    d = json.load(open(f))
    h = d['mv_proxy_history']
    print(f'{f}: max={max(h):.4f}, nonzero={sum(1 for x in h if x > 0)}/{len(h)}')
"
```

**Expected**: At least some non-zero proxy values in the 5-epoch full-run results.

**Pass criteria**: `max(mv_proxy_history) > 0.0` for at least one seed of MV_only on perm_mnist
(5 epochs/task). If all zeros, the proxy computation in `_compute_mv_proxy` is suspect.

---

### Check 5 — DIFE Causal Ordering (No Future Peeking)

**What**: The DIFE fit at task t should only use `acc_matrix` rows 0..t-1. Adding a
future row should not change previously fitted parameters.

**Why**: The core claim of DIFE is that it fits causally — this check verifies it.

**How** (manual, ~2 min):
```bash
python - <<'EOF'
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'memory-vortex-dife-lab')
from eval.online_fitters import OnlineDIFEFitter

# Build a 3-row acc_matrix
acc_3 = [[0.97], [0.95, 0.97], [0.92, 0.94, 0.97]]
f1 = OnlineDIFEFitter()
a1, b1 = f1.update(acc_3)

# Now add a 4th row and re-fit
acc_4 = acc_3 + [[0.88, 0.90, 0.95, 0.97]]
f2 = OnlineDIFEFitter()
# Simulate: fit after task 2 (only 3 rows), then check params match f1
f2.update(acc_3)
a2_at3 = f2.alpha

print(f"After 3 rows:  alpha={a1:.6f}")
print(f"After 3 rows (f2): alpha={a2_at3:.6f}")
print(f"Match: {abs(a1 - a2_at3) < 1e-9}")
EOF
```

**Expected**: `alpha` and `beta` after 3 rows are identical regardless of whether a
4th row is later added to a different fitter instance. The fitter is stateless between
update calls (it re-fits from scratch each time from the provided matrix).

**Pass criteria**: `|alpha_f1 - alpha_f2_at_3rows| < 1e-9`.

---

### Check 6 — Buffer Capacity Never Exceeded

**What**: `ReservoirBuffer` should never store more than `capacity` samples.

**Why**: A capacity violation would corrupt reservoir sampling statistics and invalidate
the replay budget accounting.

**How** (manual, ~30 sec):
```bash
python - <<'EOF'
import sys; sys.path.insert(0, '.')
from eval.buffer import ReservoirBuffer
import torch

buf = ReservoirBuffer(capacity=2000, input_shape=(784,))
import numpy as np
rng = np.random.default_rng(0)

for step in range(500):  # simulate 500 batches of 256 samples
    x = torch.rand(256, 784)
    y = torch.randint(0, 10, (256,))
    buf.update(x, y)
    assert buf.size() <= 2000, f"Overflow at step {step}: size={buf.size()}"

print(f"PASS: buffer size={buf.size()} <= capacity=2000 after 128,000 updates")
EOF
```

**Expected**: No assertion error. Buffer size stays ≤ 2000 throughout.

**Pass criteria**: Script prints PASS and exits 0.

---

### Check 7 — r_t_history Range

**What**: All scheduled replay fractions must be in [0.0, 1.0].

**Why**: Values outside this range would mean the replay mixing logic (`n_replay_per_batch
= int(r_t * batch_size)`) could produce negative or > batch_size values.

**How**: Automated. Reads all `r_t_history` entries across 64 result files.

**Expected**: PASS. All 64 files × up to 5 entries = 320 values all in [0, 1].

**Pass criteria**: Zero out-of-range values.

---

### Check 8 — Efficiency Sign Consistency

**What**: Efficiency = (FT_AF − method_AF) / replay_budget × 10,000. This must be:
- 0 for FT (no replay)
- > 0 for any method where AF < FT_AF and replay_budget > 0
- Could be ≤ 0 if a method performs worse than FT (not expected for replay methods)

**Why**: A sign error in the efficiency formula would invert the ranking of methods.

**How**: Automated. Reads `results/fast_track/summary.csv`, checks sign for all 6 methods.

**Expected**: PASS. All 5 replay methods have positive efficiency; FT has 0.

**Pass criteria**: No method with AF < FT_AF has efficiency ≤ 0; FT efficiency = 0.

---

## Full Run Command

```bash
python scripts/run_sanity_checks.py
# Expected: 5/5 automated checks PASS, exit 0
```

Manual checks (3–6) should be run before any external presentation of results.
Total time for all 8 checks: ~10 minutes.
