"""Tests for the DIFE ∘ MemoryVortex combined controller."""

import json
import os
import tempfile

import numpy as np
import pytest

from dife.core import dife, dife_curve, forgetting_rate
from dife.controller import (
    MemoryVortexOperator,
    DIFEParams,
    DIFE_MemoryVortexController,
    BASIS_ORDER,
    _eval_basis,
)
from memory_vortex.basis import eval_basis_numeric
from memory_vortex.discovery import GCADiscoveryEngineV1
from memory_vortex.scheduler import MemoryVortexScheduler


# ── dife.core ────────────────────────────────────────────────────────────────

def test_dife_step0():
    assert dife(0, Q_0=1.0, alpha=0.95, beta=0.01) == 1.0

def test_dife_monotone_decay():
    vals = dife_curve(10, Q_0=1.0, alpha=0.9, beta=0.02)
    assert all(vals[i] >= vals[i+1] for i in range(len(vals)-1))

def test_dife_clamp_zero():
    # With heavy interference, should hit 0 and stay there
    val = dife(200, Q_0=1.0, alpha=0.5, beta=1.0)
    assert val == 0.0

def test_dife_invalid_alpha():
    with pytest.raises(ValueError):
        dife(1, alpha=1.5)

def test_dife_invalid_beta():
    with pytest.raises(ValueError):
        dife(1, beta=-0.01)

def test_forgetting_rate():
    fr = forgetting_rate(5, Q_0=1.0, alpha=0.9, beta=0.01)
    assert 0.0 <= fr <= 1.0


# ── basis ────────────────────────────────────────────────────────────────────

def test_basis_length():
    x = eval_basis_numeric(1.23)
    assert x.shape == (7,)
    assert x.shape[0] == len(BASIS_ORDER)

def test_basis_controller_matches_mv():
    for t_val in [0.0, 0.5, 3.14, 10.0]:
        np.testing.assert_allclose(_eval_basis(t_val), eval_basis_numeric(t_val))


# ── MemoryVortexOperator ─────────────────────────────────────────────────────

def _make_tmp_json(**extra):
    coef = [0.0, 0.0, 0.0, 0.01375, 0.798, 0.0, 0.0]
    data = {
        "schema": "memory-vortex/operator-v1",
        "name": "test_op",
        "basis_order": BASIS_ORDER,
        "coefficients_raw": coef,
        "intercept_raw": 0.0,
        "t_scale": 100.0,
        **extra,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        return f.name

def test_operator_from_json_output_range():
    path = _make_tmp_json()
    try:
        op = MemoryVortexOperator.from_json(path)
        for step in range(0, 500, 50):
            val = op(step)
            assert 0.0 <= val <= 1.0, f"out of range at step {step}: {val}"
    finally:
        os.unlink(path)

def test_operator_fallback_output_range():
    op = MemoryVortexOperator.fallback()
    for step in range(0, 300, 30):
        assert 0.0 <= op(step) <= 1.0

def test_operator_wrong_coef_count():
    path = _make_tmp_json(coefficients_raw=[0.1, 0.2])
    try:
        with pytest.raises(ValueError, match="Expected 7"):
            MemoryVortexOperator.from_json(path)
    finally:
        os.unlink(path)


# ── DIFEParams ───────────────────────────────────────────────────────────────

def test_dife_params_invalid_alpha():
    with pytest.raises(ValueError):
        DIFEParams(alpha=0.0)

def test_dife_params_invalid_beta():
    with pytest.raises(ValueError):
        DIFEParams(beta=0.0)


# ── DIFE_MemoryVortexController ──────────────────────────────────────────────

@pytest.fixture
def controller():
    op = MemoryVortexOperator.fallback()
    p  = DIFEParams(Q0=1.0, alpha=0.97, beta=0.005)
    return DIFE_MemoryVortexController(op=op, dife_params=p)

def test_replay_fraction_range(controller):
    for step in range(0, 500, 10):
        r = controller.replay_fraction(step)
        assert 0.0 <= r <= 1.0

def test_replay_fraction_step0_nonzero(controller):
    # At step 0 DIFE envelope = Q0 = 1.0 and need > 0 (exp decay gives ~0.798)
    assert controller.replay_fraction(0) > 0.0

def test_replay_fraction_long_horizon_decreases(controller):
    r0   = controller.replay_fraction(0)
    r499 = controller.replay_fraction(499)
    # Should be lower at step 499 (both DIFE and exp decay decrease)
    assert r499 < r0

def test_per_modality_keys(controller):
    d = controller.per_modality(10)
    assert set(d.keys()) == {"vision", "text", "audio"}

def test_breakdown_keys(controller):
    bd = controller.breakdown(10)
    assert set(bd.keys()) == {"need", "envelope", "product"}
    assert bd["product"] == pytest.approx(bd["need"] * bd["envelope"], rel=1e-5)

def test_r_max_clamp():
    op = MemoryVortexOperator.fallback()
    p  = DIFEParams(Q0=1.0, alpha=0.999, beta=1e-6)
    ctrl = DIFE_MemoryVortexController(op=op, dife_params=p, r_max=0.3)
    assert ctrl.replay_fraction(0) <= 0.3


# ── GCADiscoveryEngineV1 ─────────────────────────────────────────────────────

def test_discovery_returns_valid_schema():
    rng    = np.random.default_rng(0)
    steps  = np.arange(200)
    signal = np.clip(0.8 * np.exp(-0.01 * steps) + rng.normal(0, 0.02, 200), 0, 1)
    engine = GCADiscoveryEngineV1()
    result = engine.discover(steps, signal, name="test_discovery", scale=100.0)
    assert result["schema"] == "memory-vortex/operator-v1"
    assert len(result["coefficients_raw"]) == 7
    assert result["t_scale"] == 100.0
    assert "fit" in result

def test_discovery_predictions_in_range():
    rng    = np.random.default_rng(1)
    steps  = np.arange(200)
    signal = np.clip(rng.uniform(0.1, 0.9, 200), 0, 1)
    engine = GCADiscoveryEngineV1()
    result = engine.discover(steps, signal, scale=100.0)
    coef   = np.array(result["coefficients_raw"])
    icept  = result["intercept_raw"]
    t_sc   = result["t_scale"]
    from memory_vortex.basis import eval_basis_numeric
    for s in range(0, 200, 20):
        t   = s / t_sc
        val = float(np.clip(icept + np.dot(coef, eval_basis_numeric(t)), 0, 1))
        assert 0.0 <= val <= 1.0
