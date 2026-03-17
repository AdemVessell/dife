"""Tests for the DIFE (Decay-Interference Forgetting Equation) module."""

import math
import pytest
from dife import dife, dife_curve, forgetting_rate


class TestDife:
    def test_n0_returns_Q0(self):
        """At step 0, quality should equal Q_0 (no forgetting yet)."""
        assert dife(0) == pytest.approx(1.0)
        assert dife(0, Q_0=0.8) == pytest.approx(0.8)

    def test_quality_decreases_with_steps(self):
        """Quality should monotonically decrease (or stay at 0) as n grows."""
        curve = dife_curve(20)
        for i in range(len(curve) - 1):
            assert curve[i] >= curve[i + 1], (
                f"Quality increased at step {i+1}: {curve[i]} -> {curve[i+1]}"
            )

    def test_clamps_at_zero(self):
        """Quality must never go below 0."""
        # Use aggressive interference to force clamping early
        for n in range(50):
            assert dife(n, alpha=0.9, beta=0.5) >= 0.0

    def test_known_value(self):
        """Manual calculation for n=1: Q_0*alpha - beta*1*(1-alpha)."""
        Q_0, alpha, beta = 1.0, 0.95, 0.01
        expected = Q_0 * alpha - beta * 1 * (1 - alpha)
        assert dife(1, Q_0=Q_0, alpha=alpha, beta=beta) == pytest.approx(expected)

    def test_known_value_n2(self):
        """Manual calculation for n=2."""
        Q_0, alpha, beta = 1.0, 0.95, 0.01
        expected = max(0.0, Q_0 * alpha**2 - beta * 2 * (1 - alpha**2))
        assert dife(2, Q_0=Q_0, alpha=alpha, beta=beta) == pytest.approx(expected)

    def test_invalid_alpha_low(self):
        with pytest.raises(ValueError, match="alpha"):
            dife(1, alpha=0.0)

    def test_invalid_alpha_high(self):
        with pytest.raises(ValueError, match="alpha"):
            dife(1, alpha=1.0)

    def test_invalid_beta(self):
        with pytest.raises(ValueError, match="beta"):
            dife(1, beta=0.0)

    def test_invalid_negative_beta(self):
        with pytest.raises(ValueError, match="beta"):
            dife(1, beta=-0.1)

    def test_invalid_negative_Q0(self):
        with pytest.raises(ValueError, match="Q_0"):
            dife(1, Q_0=-0.5)

    def test_invalid_negative_n(self):
        with pytest.raises(ValueError, match="n"):
            dife(-1)

    def test_higher_beta_more_forgetting(self):
        """Higher interference should result in more forgetting."""
        q_low = dife(10, beta=0.01)
        q_high = dife(10, beta=0.1)
        assert q_low > q_high

    def test_higher_alpha_less_forgetting(self):
        """Higher decay rate (closer to 1) means slower forgetting."""
        q_slow = dife(10, alpha=0.99)
        q_fast = dife(10, alpha=0.80)
        assert q_slow > q_fast

    def test_zero_Q0(self):
        """With zero initial quality, output should always be 0."""
        for n in range(10):
            assert dife(n, Q_0=0.0) == pytest.approx(0.0)


class TestDifeCurve:
    def test_length(self):
        curve = dife_curve(5)
        assert len(curve) == 6  # steps 0..5

    def test_first_element(self):
        curve = dife_curve(10)
        assert curve[0] == pytest.approx(1.0)

    def test_matches_individual_calls(self):
        curve = dife_curve(10)
        for n, val in enumerate(curve):
            assert val == pytest.approx(dife(n))


class TestForgettingRate:
    def test_zero_at_start(self):
        assert forgetting_rate(0) == pytest.approx(0.0)

    def test_increases_over_time(self):
        assert forgetting_rate(5) < forgetting_rate(10)

    def test_bounded_by_Q0(self):
        Q_0 = 0.9
        for n in range(20):
            assert forgetting_rate(n, Q_0=Q_0) <= Q_0 + 1e-9
