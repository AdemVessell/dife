"""Configuration dataclasses for the DIFE ∘ Memory Vortex evaluation suite."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchConfig:
    name: str
    n_tasks: int
    epochs_per_task: int
    lr: float
    batch_size: int
    buffer_capacity: int
    data_root: str
    device: str
    output_dir: str
    ewc_lambdas: List[float]
    si_cs: List[float]
    mv_proxy_eval_samples: int


def make_bench_config(name: str, device: str = "cpu") -> BenchConfig:
    """Factory returning default BenchConfig for the named benchmark."""
    if name == "perm_mnist":
        return BenchConfig(
            name=name,
            n_tasks=5,
            epochs_per_task=5,
            lr=1e-3,
            batch_size=256,
            buffer_capacity=2000,
            data_root="./data",
            device=device,
            output_dir="results",
            ewc_lambdas=[100.0, 500.0, 1000.0, 5000.0],
            si_cs=[0.01, 0.1, 1.0],
            mv_proxy_eval_samples=200,
        )
    elif name == "split_cifar":
        return BenchConfig(
            name=name,
            n_tasks=5,
            epochs_per_task=5,
            lr=1e-3,
            batch_size=128,
            buffer_capacity=5000,
            data_root="./data",
            device=device,
            output_dir="results",
            ewc_lambdas=[100.0, 500.0, 1000.0, 5000.0],
            si_cs=[0.01, 0.1, 1.0],
            mv_proxy_eval_samples=200,
        )
    else:
        raise ValueError(f"Unknown benchmark: {name}")
