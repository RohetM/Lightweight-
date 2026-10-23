"""Tests for Module M8: Evaluation and Statistics.
Implements PRD section 6 (M8), Section 2 (S5, S6, S7), Section 0 (Rule 5).
"""
import os
import tempfile
import numpy as np
import pytest

from eval.stats import compute_paired_bootstrap_ci, compute_metrics_summary


def test_m8_bootstrap_reproduces_known_paired_result():
    """PRD M8: Bootstrap reproduces a known paired result and nominal coverage."""
    rng = np.random.default_rng(42)
    # 50 paired observations with true difference = 0.5
    y = rng.normal(1.0, 0.2, size=50)
    x = y + 0.5 + rng.normal(0.0, 0.1, size=50)

    mean_diff, ci_lower, ci_upper = compute_paired_bootstrap_ci(
        x, y, n_resamples=5000, alpha=0.05, seed=42
    )

    assert abs(mean_diff - 0.5) < 0.05
    assert ci_lower < 0.5 < ci_upper
    assert ci_lower < mean_diff < ci_upper


def test_m8_metrics_match_hand_computed_example():
    """PRD M8: Metrics summary matches hand-computed example."""
    runs = [
        {
            "utility": 0.8, "total_cost": 0.002, "total_latency": 1.0, "success": 1.0,
            "hops": [
                {"rate_limited": False, "decision_time_us": 20.0},
                {"rate_limited": True, "decision_time_us": 30.0},
                {"rate_limited": False, "decision_time_us": 40.0},
            ]
        },
        {
            "utility": 0.4, "total_cost": 0.004, "total_latency": 2.0, "success": 0.6,
            "hops": [
                {"rate_limited": False, "decision_time_us": 25.0},
                {"rate_limited": False, "decision_time_us": 35.0},
                {"rate_limited": True, "decision_time_us": 45.0},
            ]
        }
    ]

    summary = compute_metrics_summary(runs, random_u=0.2, clairvoyant_u=0.9)

    assert abs(summary["utility_mean"] - 0.6) < 1e-9
    assert abs(summary["cost_mean"] - 0.003) < 1e-9
    assert abs(summary["success_mean"] - 0.8) < 1e-9
    assert abs(summary["latency_mean"] - 1.5) < 1e-9
    assert summary["rate_limit_count"] == 2
    # Normalized performance: (0.6 - 0.2) / (0.9 - 0.2) = 0.4 / 0.7 = 0.57142857
    assert abs(summary["normalized_performance"] - (0.4 / 0.7)) < 1e-6


def test_m8_seed_determinism():
    """PRD M8: Same seed -> identical bootstrap CI numbers."""
    rng = np.random.default_rng(100)
    x = rng.normal(size=30)
    y = rng.normal(size=30)

    res1 = compute_paired_bootstrap_ci(x, y, n_resamples=1000, seed=777)
    res2 = compute_paired_bootstrap_ci(x, y, n_resamples=1000, seed=777)

    assert res1 == res2
