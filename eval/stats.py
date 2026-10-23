"""Statistical utilities for Restless Router evaluation.
Implements PRD section 6 (M8), Section 2 (S5, S6).
"""
from typing import Any
import numpy as np


def compute_paired_bootstrap_ci(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute 95% paired bootstrap CI on diff = x - y across paired task evaluations.

    Implements PRD Section 6 (M8) and Section 2 (S5, S6).
    Returns (mean_diff, ci_lower, ci_upper).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if len(x_arr) != len(y_arr):
        raise ValueError(f"x and y must have equal length, got {len(x_arr)} and {len(y_arr)}")

    diff = x_arr - y_arr
    mean_diff = float(np.mean(diff))

    if len(diff) == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    resampled_means = np.mean(diff[indices], axis=1)

    ci_lower = float(np.percentile(resampled_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(resampled_means, 100.0 * (1.0 - alpha / 2.0)))

    return mean_diff, ci_lower, ci_upper


def compute_metrics_summary(
    runs: list[dict[str, Any]],
    random_u: float = 0.0,
    clairvoyant_u: float = 1.0,
) -> dict[str, float]:
    """Compute summary metrics for a method across runs.

    Implements PRD Section 6 (M8).
    Metrics: utility, cost, success, mean and p95 latency,
    decision time (median, p99), 429 count, normalized performance.
    """
    if not runs:
        return {
            "utility_mean": 0.0,
            "utility_std": 0.0,
            "cost_mean": 0.0,
            "success_mean": 0.0,
            "latency_mean": 0.0,
            "latency_p95": 0.0,
            "decision_time_median": 0.0,
            "decision_time_p99": 0.0,
            "rate_limit_count": 0,
            "normalized_performance": 0.0,
        }

    utilities = np.array([r["utility"] for r in runs], dtype=np.float64)
    costs = np.array([r["total_cost"] for r in runs], dtype=np.float64)
    latencies = np.array([r["total_latency"] for r in runs], dtype=np.float64)
    successes = np.array([r["success"] for r in runs], dtype=np.float64)

    decision_times = [
        h["decision_time_us"] for r in runs for h in r.get("hops", []) if "decision_time_us" in h
    ]
    rate_limits = sum(
        sum(1 for h in r.get("hops", []) if h.get("rate_limited", False)) for r in runs
    )

    mean_u = float(np.mean(utilities))
    denom = clairvoyant_u - random_u
    norm_perf = (mean_u - random_u) / denom if abs(denom) > 1e-9 else 0.0

    return {
        "utility_mean": mean_u,
        "utility_std": float(np.std(utilities)),
        "cost_mean": float(np.mean(costs)),
        "success_mean": float(np.mean(successes)),
        "latency_mean": float(np.mean(latencies)),
        "latency_p95": float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0.0,
        "decision_time_median": float(np.median(decision_times)) if decision_times else 0.0,
        "decision_time_p99": float(np.percentile(decision_times, 99)) if decision_times else 0.0,
        "rate_limit_count": int(rate_limits),
        "normalized_performance": float(norm_perf),
    }


# Implementation Note:
# Implemented paired bootstrap CI (10,000 resamples) and summary metrics
# per PRD Section 6 (M8) and Section 2 (S5, S6).
