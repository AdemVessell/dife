"""
DIFE ∘ MemoryVortex controller demo.

Plots three signals over 500 CL steps:
  1. Memory Vortex need signal  (what the operator says you need)
  2. DIFE envelope              (how much budget remains under forgetting)
  3. Replay fraction = need * envelope  (what actually gets replayed)

Run from the repo root:
    python scripts/run_controller_demo.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dife.controller import (
    MemoryVortexOperator,
    DIFEParams,
    DIFE_MemoryVortexController,
)

OP_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "operators", "memory_vortex_operator.json")
OUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "dife_memory_vortex_demo.png")

# ── Load operator ────────────────────────────────────────────────────────────

if os.path.exists(OP_PATH):
    op = MemoryVortexOperator.from_json(OP_PATH)
    print(f"Loaded operator from {OP_PATH}")
else:
    op = MemoryVortexOperator.fallback()
    print("Using fallback operator (JSON not found)")

# ── DIFE params ──────────────────────────────────────────────────────────────
# For 500 continuous steps:
#   α=0.997 → 0.3% decay per step; at step 499: 0.997^499 ≈ 0.22
#   β=0.0001 → gentle interference; envelope ≈ 0.18 at step 499
# This keeps the envelope visibly non-zero across the full demo.
p = DIFEParams(Q0=1.0, alpha=0.997, beta=0.0001)

ctrl = DIFE_MemoryVortexController(op=op, dife_params=p, r_max=1.0)

# ── Evaluate ─────────────────────────────────────────────────────────────────

steps   = np.arange(0, 500)
need    = np.array([op(int(s))                 for s in steps])
envelope= np.array([ctrl._dife_envelope(int(s)) for s in steps])
replay  = np.array([ctrl.replay_fraction(int(s)) for s in steps])

# ── Print table ──────────────────────────────────────────────────────────────

print(f"\n{'step':>6}  {'need':>8}  {'envelope':>10}  {'replay':>8}")
checkpoints = [0, 50, 100, 150, 200, 300, 400, 499]
for s in checkpoints:
    bd = ctrl.breakdown(int(s))
    print(f"  {s:>4}   {bd['need']:.5f}   {bd['envelope']:.5f}    {bd['product']:.5f}")

print(f"\nSummary over 500 steps:")
print(f"  Avg replay fraction : {replay.mean():.4f}")
print(f"  Max replay fraction : {replay.max():.4f}")
print(f"  Final replay (499)  : {replay[-1]:.4f}")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

ax1 = axes[0]
ax1.fill_between(steps, need,     alpha=0.15, color="#5c9ee0")
ax1.fill_between(steps, envelope, alpha=0.15, color="#e0a85c")
ax1.plot(steps, need,     lw=2,   color="#5c9ee0", label="Memory Vortex need  (operator output)")
ax1.plot(steps, envelope, lw=2,   color="#e0a85c",
         label=f"DIFE envelope  (α={p.alpha}, β={p.beta}, Q₀={p.Q0})")
ax1.set_ylabel("Signal (0 – 1)")
ax1.set_ylim(-0.05, 1.1)
ax1.legend(fontsize=10)
ax1.set_title("DIFE ∘ MemoryVortex — component signals", fontsize=12)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.fill_between(steps, replay, alpha=0.25, color="#5cc47a")
ax2.plot(steps, replay, lw=2.5, color="#5cc47a",
         label="Replay fraction = need × DIFE envelope")
ax2.axhline(replay.mean(), color="#e05c5c", lw=1.5, ls="--",
            label=f"Mean = {replay.mean():.3f}")
ax2.set_xlabel("Continual-learning step")
ax2.set_ylabel("Replay fraction (0 – 1)")
ax2.set_ylim(-0.05, 1.1)
ax2.legend(fontsize=10)
ax2.set_title("Composed replay schedule", fontsize=12)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
print(f"\nSaved figure: {OUT_FILE}")
