#!/usr/bin/env python3
"""Entrypoint for the DIFE ∘ Memory Vortex end-to-end evaluation suite.

Usage:
    python run_all.py [--bench perm_mnist split_cifar] [--seeds N] [--device cpu]

Examples:
    python run_all.py
    python run_all.py --bench perm_mnist --seeds 3
    python run_all.py --bench split_cifar --seeds 1 --device cpu
"""

import argparse
import os
import sys

# Add repo root and memory-vortex-dife-lab to path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "memory-vortex-dife-lab"))

from eval.config import make_bench_config
from eval.runner import run_benchmark, build_summary_rows
from eval.metrics import write_summary_csv
from eval.plotting_ext import generate_all_plots
from eval.report import write_results_md

BENCH_DEFAULT_SEEDS = {
    "perm_mnist": 5,
    "split_cifar": 3,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run DIFE ∘ Memory Vortex continual learning evaluation suite."
    )
    parser.add_argument(
        "--bench",
        nargs="+",
        default=["perm_mnist", "split_cifar"],
        choices=list(BENCH_DEFAULT_SEEDS.keys()),
        help="Benchmarks to run (default: both)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override seed count for all benchmarks",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu)",
    )
    args = parser.parse_args()

    for bench in args.bench:
        n_seeds = args.seeds if args.seeds is not None else BENCH_DEFAULT_SEEDS[bench]
        cfg = make_bench_config(bench, device=args.device)

        print(f"\n{'='*60}")
        print(f"Benchmark: {bench}  |  seeds: {n_seeds}  |  device: {args.device}")
        print(f"{'='*60}")

        all_results = run_benchmark(bench, cfg, n_seeds)

        csv_path = write_summary_csv(bench, all_results, cfg.output_dir)
        print(f"\n[{bench}] Summary CSV: {csv_path}")

        summary_rows = build_summary_rows(all_results)
        generate_all_plots(bench, all_results, cfg, summary_rows)

    write_results_md(os.path.join(os.path.dirname(_HERE), "results")
                     if not os.path.exists(os.path.join(_HERE, "results"))
                     else os.path.join(_HERE, "results"))

    print("\nDone. Results written to results/ and RESULTS.md.")


if __name__ == "__main__":
    main()
