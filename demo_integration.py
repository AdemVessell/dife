#!/usr/bin/env python3
"""DIFE × MV Integration Demo — Colossus-2 Controller API Walkthrough.

Demonstrates the exact 5-line controller API using SYNTHETIC data only.
No neural network is trained. Runtime < 5 seconds.

This script shows how to wire the DIFE_MV combined controller into a training
loop, log the scheduled replay fraction at each step, and adapt for large-scale
training without discrete task boundaries (see sliding-window note at bottom).

Usage:
    python demo_integration.py [--windows 5] [--epochs-per-window 5] [--batch-size 256]

Outputs:
    - Console table: per-window replay schedule
    - logs/demo_integration.csv: full epoch-level log
"""

import argparse
import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "memory-vortex-dife-lab"))

import numpy as np

from eval.online_fitters import OnlineDIFEFitter, OnlineMVFitter
from eval.schedulers import SchedulerState, get_replay_fraction

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def synthetic_accuracy(window: int, task_j: int, base: float = 0.97, decay: float = 0.02) -> float:
    """Simulate forgetting: accuracy on task j after training window w decreases with distance."""
    interference = window - task_j
    return max(0.5, base * ((1.0 - decay) ** interference))


def synthetic_proxy(epoch_local: int, epochs_per_window: int, noise_rng) -> float:
    """Simulate forgetting proxy: rises then falls over a training window.

    proxy = 1 - accuracy_on_buffer. Peaks mid-window, falls as model adapts.
    """
    phase = math.pi * epoch_local / max(epochs_per_window - 1, 1)
    signal = 0.05 + 0.12 * math.sin(phase)
    noise = float(noise_rng.uniform(-0.01, 0.01))
    return max(0.0, signal + noise)


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

def run_demo(n_windows: int, epochs_per_window: int, batch_size: int):
    rng = np.random.default_rng(42)
    noise_rng = np.random.default_rng(7)

    # --- CONTROLLER INITIALIZATION (5 lines in production) ---
    dife_fitter = OnlineDIFEFitter()   # fits alpha, beta from forgetting trajectories
    mv_fitter = OnlineMVFitter()       # fits per-epoch operator from proxy observations

    acc_matrix = []   # lower-triangular: acc_matrix[w][j] = accuracy on task j after window w
    log_rows = []

    print("=" * 72)
    print("DIFE × MV Integration Demo  |  Synthetic data, no real training")
    print(f"  {n_windows} windows · {epochs_per_window} epochs/window · batch_size={batch_size}")
    print("=" * 72)
    print()
    print(f"{'Win':>4}  {'Ep':>3}  {'DIFE_r':>8}  {'MV_r':>8}  {'r_combined':>11}  "
          f"{'replay/batch':>13}  {'α (fitted)':>11}  {'β (fitted)':>11}")
    print("-" * 72)

    for w in range(n_windows):
        global_epoch_start = w * epochs_per_window

        # --- STEP 1: QUERY CONTROLLER BEFORE TRAINING THIS WINDOW ---
        state = SchedulerState(
            task_index=w,
            total_epochs_so_far=global_epoch_start,
            dife_fitter=dife_fitter,
            mv_fitter=mv_fitter,
            rng=rng,
        )
        r_combined = get_replay_fraction("DIFE_MV", state)

        # Decompose for logging (mirrors what get_replay_fraction does internally)
        r_dife = float(np.clip(dife_fitter.replay_fraction(w), 0.0, 1.0))
        r_mv = float(np.clip(mv_fitter.replay_fraction(global_epoch_start), 0.0, 1.0))
        n_replay_per_batch = int(r_combined * batch_size)

        alpha = getattr(dife_fitter, "alpha", 0.90)
        beta = getattr(dife_fitter, "beta", 0.01)

        print(f"  {w:>2}  {'--':>3}  {r_dife:>8.4f}  {r_mv:>8.4f}  {r_combined:>11.4f}  "
              f"{n_replay_per_batch:>13}  {alpha:>11.5f}  {beta:>11.6f}   ← pre-window schedule")

        log_rows.append({
            "window": w, "global_epoch": global_epoch_start,
            "epoch_local": -1, "phase": "pre_window",
            "dife_r": round(r_dife, 5), "mv_r": round(r_mv, 5),
            "r_combined": round(r_combined, 5),
            "n_replay_per_batch": n_replay_per_batch,
            "alpha": round(alpha, 6), "beta": round(beta, 7),
            "proxy": "",
        })

        # --- STEP 2: SIMULATE TRAINING EPOCHS ---
        for epoch_local in range(epochs_per_window):
            global_epoch = global_epoch_start + epoch_local

            # Simulate: proxy = 1 - buffer accuracy (rises mid-task, falls as model adapts)
            proxy = synthetic_proxy(epoch_local, epochs_per_window, noise_rng)

            # Record proxy for MV fitting (called DURING training after each epoch)
            mv_fitter.record_epoch(global_epoch, proxy)

            log_rows.append({
                "window": w, "global_epoch": global_epoch,
                "epoch_local": epoch_local, "phase": "training",
                "dife_r": round(r_dife, 5), "mv_r": "", "r_combined": "",
                "n_replay_per_batch": n_replay_per_batch,
                "alpha": "", "beta": "", "proxy": round(proxy, 5),
            })

        # --- STEP 3: POST-WINDOW UPDATE (after task/window completes) ---
        # Build synthetic acc_matrix row: accuracy on each task j after window w
        new_row = [synthetic_accuracy(w, j) for j in range(w + 1)]
        acc_matrix.append(new_row)

        # Update DIFE fitter from full acc_matrix (causal — only sees rows 0..w)
        alpha_new, beta_new = dife_fitter.update(acc_matrix)

        # Update MV fitter (fits operator if >= 15 epoch observations available)
        mv_fitter.update()

        print(f"  {w:>2}  {'--':>3}  {'':>8}  {'':>8}  {'':>11}  "
              f"{'':>13}  {alpha_new:>11.5f}  {beta_new:>11.6f}   ← post-window fit "
              f"(acc_row={[f'{a:.3f}' for a in new_row]})")
        print()

    # --- OUTPUT CSV ---
    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", "demo_integration.csv")
    fieldnames = [
        "window", "global_epoch", "epoch_local", "phase",
        "dife_r", "mv_r", "r_combined", "n_replay_per_batch",
        "alpha", "beta", "proxy",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Log saved: {csv_path}  ({len(log_rows)} rows)")

    # --- SUMMARY ---
    pre_window_rows = [r for r in log_rows if r["phase"] == "pre_window"]
    print("\nSchedule summary (pre-window r_combined):")
    for r in pre_window_rows:
        bar = "█" * int(r["r_combined"] * 30)
        print(f"  window {r['window']}: {r['r_combined']:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Sliding-window note (for large training loops without discrete task boundaries)
# ---------------------------------------------------------------------------
#
# In a Colossus-scale training loop, "tasks" may not be clearly delineated.
# To use the DIFE_MV controller without task boundaries:
#
#   1. SLIDING-WINDOW DIFE:
#      Every W gradient steps, evaluate accuracy on a small held-out buffer
#      sampled from the last W steps. Treat each window as a "pseudo-task row":
#
#        if global_step % W == 0:
#            window_acc = [eval_on_buffer(model, past_buffer_j) for j in range(n_windows_so_far)]
#            pseudo_acc_matrix.append(window_acc)
#            dife_fitter.update(pseudo_acc_matrix)
#
#   2. MV PROXY (unchanged):
#      mv_fitter.record_epoch(global_step, 1 - accuracy_on_replay_buffer)
#      mv_fitter.update() every W steps (or whenever you want to refit the operator)
#
#   3. QUERY (unchanged, just replace task_index with window_index):
#        state = SchedulerState(
#            task_index=window_index,
#            total_epochs_so_far=global_step,
#            dife_fitter=dife_fitter,
#            mv_fitter=mv_fitter,
#            rng=rng,
#        )
#        r = get_replay_fraction("DIFE_MV", state)
#        n_replay = int(r * batch_size)
#
#   The rest of the controller API (OnlineDIFEFitter, OnlineMVFitter, SchedulerState,
#   get_replay_fraction) is completely unchanged. The only difference is the source
#   of the acc_matrix rows (windowed checkpoints vs discrete task boundaries).


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, default=5, help="Number of training windows (default 5)")
    parser.add_argument("--epochs-per-window", type=int, default=5, help="Epochs per window (default 5)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default 256)")
    args = parser.parse_args()

    run_demo(
        n_windows=args.windows,
        epochs_per_window=args.epochs_per_window,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
