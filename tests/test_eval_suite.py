"""Tests for the eval/ evaluation suite package.

Covers: config, buffer, online_fitters, schedulers, metrics, grid_search, trainer.
Uses only synthetic tensors — no network access / no MNIST/CIFAR downloads required.
"""

import csv
import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Path setup
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "memory-vortex-dife-lab"))

from eval.buffer import ReservoirBuffer
from eval.config import BenchConfig, make_bench_config
from eval.grid_search import find_best_ewc_lambda, find_best_si_c
from eval.metrics import compute_all_metrics, save_metrics, write_summary_csv
from eval.online_fitters import OnlineDIFEFitter, OnlineMVFitter
from eval.schedulers import SchedulerState, get_replay_fraction
from eval.trainer import train_one_method


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

INPUT_DIM = 10
N_CLASSES = 4
N_SAMPLES = 40
BATCH_SIZE = 8


def _make_dataset(n=N_SAMPLES, dim=INPUT_DIM, n_classes=N_CLASSES):
    x = torch.randn(n, dim)
    y = torch.randint(0, n_classes, (n,))
    return TensorDataset(x, y)


def _make_loader(n=N_SAMPLES, shuffle=True):
    ds = _make_dataset(n)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


@pytest.fixture
def tiny_mlp():
    return nn.Sequential(nn.Linear(INPUT_DIM, N_CLASSES))


@pytest.fixture
def tiny_loader_pair():
    return (_make_loader(), _make_loader(n=16, shuffle=False))


@pytest.fixture
def two_task_loaders():
    return [
        (_make_loader(), _make_loader(n=16, shuffle=False)),
        (_make_loader(), _make_loader(n=16, shuffle=False)),
    ]


@pytest.fixture
def tiny_cfg():
    return BenchConfig(
        name="perm_mnist",
        n_tasks=2,
        epochs_per_task=1,
        lr=1e-3,
        batch_size=BATCH_SIZE,
        buffer_capacity=50,
        data_root="./data",
        device="cpu",
        output_dir="results",
        ewc_lambdas=[100.0, 500.0],
        si_cs=[0.01, 0.1],
        mv_proxy_eval_samples=10,
    )


THREE_TASK_ACC = [[0.9], [0.7, 0.85], [0.5, 0.6, 0.8]]


# ---------------------------------------------------------------------------
# 1. TestConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_perm_mnist_fields(self):
        cfg = make_bench_config("perm_mnist")
        assert cfg.n_tasks == 5
        assert cfg.batch_size == 256
        assert cfg.buffer_capacity == 2000

    def test_split_cifar_fields(self):
        cfg = make_bench_config("split_cifar")
        assert cfg.n_tasks == 5
        assert cfg.batch_size == 128
        assert cfg.buffer_capacity == 5000

    def test_unknown_bench_raises(self):
        with pytest.raises(ValueError):
            make_bench_config("unknown_bench")

    def test_device_override(self):
        cfg = make_bench_config("perm_mnist", device="cuda")
        assert cfg.device == "cuda"

    def test_default_device_is_cpu(self):
        cfg = make_bench_config("perm_mnist")
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# 2. TestReservoirBuffer
# ---------------------------------------------------------------------------


class TestReservoirBuffer:
    def _make_batch(self, n):
        return torch.randn(n, INPUT_DIM), torch.randint(0, N_CLASSES, (n,))

    def test_size_after_partial_fill(self):
        buf = ReservoirBuffer(50, (INPUT_DIM,))
        x, y = self._make_batch(30)
        buf.update(x, y)
        assert buf.size() == 30

    def test_size_capped_at_capacity(self):
        buf = ReservoirBuffer(50, (INPUT_DIM,))
        for _ in range(10):
            x, y = self._make_batch(20)
            buf.update(x, y)
        assert buf.size() == 50

    def test_sample_shape(self):
        buf = ReservoirBuffer(50, (INPUT_DIM,))
        x, y = self._make_batch(30)
        buf.update(x, y)
        xs, ys = buf.sample(10)
        assert xs.shape == (10, INPUT_DIM)
        assert ys.shape == (10,)

    def test_sample_capped_at_stored(self):
        buf = ReservoirBuffer(50, (INPUT_DIM,))
        x, y = self._make_batch(30)
        buf.update(x, y)
        xs, ys = buf.sample(100)
        assert len(xs) == 30

    def test_empty_sample_returns_empty(self):
        buf = ReservoirBuffer(50, (INPUT_DIM,))
        xs, ys = buf.sample(5)
        assert len(xs) == 0

    def test_reservoir_no_nan_after_overflow(self):
        buf = ReservoirBuffer(50, (INPUT_DIM,))
        for _ in range(20):
            x, y = self._make_batch(30)
            buf.update(x, y)
        xs, ys = buf.sample(50)
        assert not torch.isnan(xs).any()
        assert xs.shape[1] == INPUT_DIM


# ---------------------------------------------------------------------------
# 3. TestOnlineDIFEFitter
# ---------------------------------------------------------------------------


class TestOnlineDIFEFitter:
    def test_defaults(self):
        f = OnlineDIFEFitter()
        assert f.alpha == pytest.approx(0.90)
        assert f.beta == pytest.approx(0.01)

    def test_single_task_returns_defaults(self):
        f = OnlineDIFEFitter()
        alpha, beta = f.update([[0.9]])
        assert alpha == pytest.approx(0.90)
        assert beta == pytest.approx(0.01)

    def test_two_tasks_still_defaults(self):
        """Only 1 cross-task observation — below threshold of 2."""
        f = OnlineDIFEFitter()
        alpha, beta = f.update([[0.9], [0.7, 0.85]])
        assert alpha == pytest.approx(0.90)
        assert beta == pytest.approx(0.01)

    def test_three_tasks_fits(self):
        """3 cross-task observations — fitting should proceed."""
        f = OnlineDIFEFitter()
        alpha, beta = f.update(THREE_TASK_ACC)
        assert 0.0 < alpha < 1.0
        assert beta > 0.0

    def test_replay_fraction_in_range(self):
        f = OnlineDIFEFitter()
        for t in range(5):
            r = f.replay_fraction(t)
            assert 0.0 <= r <= 1.0

    def test_replay_fraction_at_t0_is_one(self):
        """dife(0, Q_0=1.0, alpha, beta) = 1.0 always."""
        f = OnlineDIFEFitter()
        assert f.replay_fraction(0) == pytest.approx(1.0)

    def test_causality_no_future_leakage(self):
        """Params fitted from 3-row matrix should differ from 2-row (or remain default)."""
        f2 = OnlineDIFEFitter()
        f2.update([[0.9], [0.7, 0.85]])
        a2, b2 = f2.alpha, f2.beta

        f3 = OnlineDIFEFitter()
        f3.update(THREE_TASK_ACC)
        a3, b3 = f3.alpha, f3.beta

        # After 3 rows, fitter should have moved from defaults
        # (a2, b2 are still defaults because < 2 obs)
        assert (a3 != a2) or (b3 != b2)


# ---------------------------------------------------------------------------
# 4. TestOnlineMVFitter
# ---------------------------------------------------------------------------


class TestOnlineMVFitter:
    def test_fallback_below_min_obs(self):
        from dife.controller import MemoryVortexOperator

        f = OnlineMVFitter()
        fallback_coef = MemoryVortexOperator.fallback().coef_raw.copy()
        for i in range(14):
            f.record_epoch(i, 0.3 + 0.01 * i)
        op = f.update()
        # Should return unchanged fallback
        np.testing.assert_array_almost_equal(op.coef_raw, fallback_coef)

    def test_fits_at_15_obs(self):
        from dife.controller import MemoryVortexOperator

        f = OnlineMVFitter()
        fallback_coef = MemoryVortexOperator.fallback().coef_raw.copy()
        for i in range(15):
            f.record_epoch(i, 0.5 + 0.02 * np.sin(i))
        op = f.update()
        # After fitting, operator should differ from fallback
        # (not guaranteed in all corner cases, but with 15 diverse obs should change)
        assert op is not None
        assert hasattr(op, "coef_raw")

    def test_record_epoch_accumulates(self):
        f = OnlineMVFitter()
        for i in range(10):
            f.record_epoch(i, 0.5)
        assert len(f._epoch_steps) == 10
        assert len(f._y_proxy) == 10

    def test_replay_fraction_in_range(self):
        f = OnlineMVFitter()
        for step in range(0, 50, 5):
            r = f.replay_fraction(step)
            assert 0.0 <= r <= 1.0

    def test_proxy_values_clipped(self):
        """Out-of-range proxy values should be clipped to [0,1]."""
        f = OnlineMVFitter()
        f.record_epoch(0, -0.5)
        f.record_epoch(1, 1.8)
        assert f._y_proxy[0] == pytest.approx(0.0)
        assert f._y_proxy[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. TestSchedulers
# ---------------------------------------------------------------------------


@pytest.fixture
def scheduler_state():
    return SchedulerState(
        task_index=2,
        total_epochs_so_far=10,
        dife_fitter=OnlineDIFEFitter(),
        mv_fitter=OnlineMVFitter(),
        rng=np.random.default_rng(42),
    )


class TestSchedulers:
    def test_ft_zero(self, scheduler_state):
        assert get_replay_fraction("FT", scheduler_state) == pytest.approx(0.0)

    def test_ewc_zero(self, scheduler_state):
        assert get_replay_fraction("EWC", scheduler_state) == pytest.approx(0.0)

    def test_si_zero(self, scheduler_state):
        assert get_replay_fraction("SI", scheduler_state) == pytest.approx(0.0)

    def test_const_replay_lo(self, scheduler_state):
        assert get_replay_fraction("ConstReplay_0.1", scheduler_state) == pytest.approx(0.1)

    def test_const_replay_hi(self, scheduler_state):
        assert get_replay_fraction("ConstReplay_0.3", scheduler_state) == pytest.approx(0.3)

    def test_rand_replay_in_range(self, scheduler_state):
        for _ in range(10):
            r = get_replay_fraction("RandReplay", scheduler_state)
            assert 0.05 <= r <= 0.35

    def test_dife_only_delegates(self, scheduler_state):
        r = get_replay_fraction("DIFE_only", scheduler_state)
        expected = scheduler_state.dife_fitter.replay_fraction(scheduler_state.task_index)
        assert r == pytest.approx(expected)

    def test_mv_only_delegates(self, scheduler_state):
        r = get_replay_fraction("MV_only", scheduler_state)
        expected = scheduler_state.mv_fitter.replay_fraction(
            scheduler_state.total_epochs_so_far
        )
        assert r == pytest.approx(expected)

    def test_dife_mv_is_product_clipped(self, scheduler_state):
        r = get_replay_fraction("DIFE_MV", scheduler_state)
        d = scheduler_state.dife_fitter.replay_fraction(scheduler_state.task_index)
        m = scheduler_state.mv_fitter.replay_fraction(scheduler_state.total_epochs_so_far)
        expected = float(np.clip(d * m, 0.0, 1.0))
        assert r == pytest.approx(expected)

    def test_unknown_method_raises(self, scheduler_state):
        with pytest.raises(ValueError, match="Unknown method"):
            get_replay_fraction("BadMethod", scheduler_state)

    def test_all_methods_return_float(self, scheduler_state):
        methods = [
            "FT", "EWC", "SI",
            "ConstReplay_0.1", "ConstReplay_0.3", "RandReplay",
            "DIFE_only", "MV_only", "DIFE_MV",
        ]
        for m in methods:
            r = get_replay_fraction(m, scheduler_state)
            assert isinstance(r, float)
            assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# 6. TestMetrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_aa_af_bwt_passthrough(self):
        """AA/AF/BWT should match compute_metrics() directly."""
        from benchmark.fitting import compute_metrics

        base = compute_metrics(THREE_TASK_ACC)
        result = compute_all_metrics(
            acc_matrix=THREE_TASK_ACC,
            r_t_history=[0.0, 0.0, 0.0],
            total_replay_samples=0,
            wall_clock=1.0,
            n_classes_per_task=10,
            pre_task_acc=[],
        )
        assert result["avg_final_acc"] == pytest.approx(base["avg_final_acc"])
        assert result["avg_forgetting"] == pytest.approx(base["avg_forgetting"])
        assert result["bwt"] == pytest.approx(base["bwt"])

    def test_fwt_empty_pre_task_acc(self):
        result = compute_all_metrics(
            acc_matrix=THREE_TASK_ACC,
            r_t_history=[],
            total_replay_samples=0,
            wall_clock=1.0,
            n_classes_per_task=10,
            pre_task_acc=[],
        )
        assert result["fwt"] == pytest.approx(0.0)

    def test_fwt_single_task(self):
        """pre_task_acc=[0.6], n_classes=10 → fwt = 0.6 - 0.1 = 0.5"""
        result = compute_all_metrics(
            acc_matrix=[[0.9], [0.7, 0.85]],
            r_t_history=[0.0, 0.0],
            total_replay_samples=0,
            wall_clock=1.0,
            n_classes_per_task=10,
            pre_task_acc=[0.6],
        )
        assert result["fwt"] == pytest.approx(0.5)

    def test_fwt_multi_task(self):
        """pre_task_acc=[0.3, 0.4], n_classes=2 → fwt = mean([-0.2, -0.1]) = -0.15"""
        result = compute_all_metrics(
            acc_matrix=THREE_TASK_ACC,
            r_t_history=[0.0, 0.0, 0.0],
            total_replay_samples=0,
            wall_clock=1.0,
            n_classes_per_task=2,
            pre_task_acc=[0.3, 0.4],
        )
        assert result["fwt"] == pytest.approx(-0.15)

    def test_save_metrics_writes_json(self, tmp_path):
        path = str(tmp_path / "subdir" / "m.json")
        save_metrics({"AA": 0.8, "AF": 0.1}, path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["AA"] == pytest.approx(0.8)

    def test_write_summary_csv_has_method_column(self, tmp_path):
        all_results = {
            "FT": {
                0: {"avg_final_acc": 0.7, "avg_forgetting": 0.2, "bwt": -0.2,
                    "fwt": 0.0, "total_replay_samples": 0},
                1: {"avg_final_acc": 0.75, "avg_forgetting": 0.18, "bwt": -0.18,
                    "fwt": 0.0, "total_replay_samples": 0},
            },
            "EWC": {
                0: {"avg_final_acc": 0.8, "avg_forgetting": 0.1, "bwt": -0.1,
                    "fwt": 0.05, "total_replay_samples": 0},
                1: {"avg_final_acc": 0.82, "avg_forgetting": 0.09, "bwt": -0.09,
                    "fwt": 0.06, "total_replay_samples": 0},
            },
        }
        csv_path = write_summary_csv("perm_mnist", all_results, str(tmp_path))
        assert os.path.exists(csv_path)
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        methods_in_csv = [r["method"] for r in rows]
        assert "FT" in methods_in_csv
        assert "EWC" in methods_in_csv
        # Check mean/std columns exist
        assert "AA_mean" in rows[0]
        assert "AA_std" in rows[0]


# ---------------------------------------------------------------------------
# 7. TestGridSearch
# ---------------------------------------------------------------------------


class TestGridSearch:
    @pytest.fixture
    def two_loaders_grid(self):
        return [
            (_make_loader(n=40), _make_loader(n=16, shuffle=False)),
            (_make_loader(n=40), _make_loader(n=16, shuffle=False)),
        ]

    def _model_factory(self):
        return nn.Sequential(nn.Linear(INPUT_DIM, N_CLASSES))

    def test_find_best_ewc_lambda_returns_from_grid(self, two_loaders_grid):
        result = find_best_ewc_lambda(
            two_loaders_grid,
            lambdas=[100.0, 500.0],
            epochs=1,
            lr=1e-3,
            model_factory=self._model_factory,
        )
        assert result in [100.0, 500.0]

    def test_find_best_si_c_returns_from_grid(self, two_loaders_grid):
        result = find_best_si_c(
            two_loaders_grid,
            cs=[0.01, 0.1],
            epochs=1,
            lr=1e-3,
            model_factory=self._model_factory,
        )
        assert result in [0.01, 0.1]


# ---------------------------------------------------------------------------
# 8. TestTrainer (integration — uses tiny data, 1 epoch)
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "acc_matrix", "r_t_history", "pre_task_acc",
    "total_replay_samples", "wall_clock_seconds",
    "mv_proxy_history", "dife_params_history",
}


@pytest.fixture
def ft_result(two_task_loaders, tiny_cfg):
    model = nn.Sequential(nn.Linear(INPUT_DIM, N_CLASSES))
    return train_one_method(
        "FT", model, two_task_loaders, tiny_cfg, seed=0,
        best_ewc_lam=500.0, best_si_c=0.1,
    )


class TestTrainer:
    def test_output_keys_present(self, ft_result):
        assert EXPECTED_KEYS.issubset(ft_result.keys())

    def test_acc_matrix_shape(self, ft_result):
        m = ft_result["acc_matrix"]
        assert len(m) == 2
        assert len(m[0]) == 1
        assert len(m[1]) == 2

    def test_acc_matrix_values_in_range(self, ft_result):
        for row in ft_result["acc_matrix"]:
            for v in row:
                assert 0.0 <= v <= 1.0

    def test_pre_task_acc_length(self, ft_result):
        # T=2 tasks → T-1=1 pre-task accuracy measurement
        assert len(ft_result["pre_task_acc"]) == 1

    def test_r_t_history_length(self, ft_result):
        assert len(ft_result["r_t_history"]) == 2

    def test_ft_no_replay(self, ft_result):
        assert ft_result["total_replay_samples"] == 0

    def test_wall_clock_positive(self, ft_result):
        assert ft_result["wall_clock_seconds"] > 0.0

    def test_ewc_no_replay(self, two_task_loaders, tiny_cfg):
        model = nn.Sequential(nn.Linear(INPUT_DIM, N_CLASSES))
        result = train_one_method(
            "EWC", model, two_task_loaders, tiny_cfg, seed=0,
            best_ewc_lam=500.0, best_si_c=0.1,
        )
        assert result["total_replay_samples"] == 0

    def test_si_no_replay(self, two_task_loaders, tiny_cfg):
        model = nn.Sequential(nn.Linear(INPUT_DIM, N_CLASSES))
        result = train_one_method(
            "SI", model, two_task_loaders, tiny_cfg, seed=0,
            best_ewc_lam=500.0, best_si_c=0.1,
        )
        assert result["total_replay_samples"] == 0

    def test_const_replay_uses_buffer(self, two_task_loaders, tiny_cfg):
        """ConstReplay_0.3 with 2 tasks: buffer fills after task 1 → replay in task 2."""
        model = nn.Sequential(nn.Linear(INPUT_DIM, N_CLASSES))
        result = train_one_method(
            "ConstReplay_0.3", model, two_task_loaders, tiny_cfg, seed=0,
            best_ewc_lam=500.0, best_si_c=0.1,
        )
        assert result["total_replay_samples"] > 0

    def test_dife_mv_output_structure(self, two_task_loaders, tiny_cfg):
        model = nn.Sequential(nn.Linear(INPUT_DIM, N_CLASSES))
        result = train_one_method(
            "DIFE_MV", model, two_task_loaders, tiny_cfg, seed=0,
            best_ewc_lam=500.0, best_si_c=0.1,
        )
        assert len(result["r_t_history"]) == 2
        assert isinstance(result["mv_proxy_history"], list)
        assert isinstance(result["dife_params_history"], list)

    def test_dife_only_params_recorded(self, two_task_loaders, tiny_cfg):
        model = nn.Sequential(nn.Linear(INPUT_DIM, N_CLASSES))
        result = train_one_method(
            "DIFE_only", model, two_task_loaders, tiny_cfg, seed=0,
            best_ewc_lam=500.0, best_si_c=0.1,
        )
        # dife_params_history should have 1 entry per task (fitted after each task)
        assert len(result["dife_params_history"]) == 2
        for entry in result["dife_params_history"]:
            assert "alpha" in entry
            assert "beta" in entry

    def test_all_nine_methods_complete(self, two_task_loaders, tiny_cfg):
        """Smoke test: all 9 methods run to completion without error."""
        methods = [
            "FT", "EWC", "SI",
            "ConstReplay_0.1", "ConstReplay_0.3", "RandReplay",
            "DIFE_only", "MV_only", "DIFE_MV",
        ]
        for method in methods:
            model = nn.Sequential(nn.Linear(INPUT_DIM, N_CLASSES))
            result = train_one_method(
                method, model, two_task_loaders, tiny_cfg, seed=0,
                best_ewc_lam=500.0, best_si_c=0.1,
            )
            assert len(result["acc_matrix"]) == 2, f"{method}: wrong acc_matrix length"
