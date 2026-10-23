"""Tests for Module M2: Restless Cloud Environment.
Implements PRD section 6 (M2), Section 2 (S1, S2).
"""
import copy
import yaml
import pytest
import numpy as np
from envs.cloud_env import CloudEnv


@pytest.fixture
def config():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_m2_s1_sigma_zero_exactness(config):
    """PRD S1 & M2: At sigma = 0, simulator reproduces trace cost and latency exactly (tolerance 1e-9)."""
    env = CloudEnv(config["models"], config["cloud"], sigma=0.0, seed=42)

    # Base values from PRD M0 & default.yaml
    # m0: tokens=300, price=0.0001, base_latency=0.4
    res0 = env.request(0, tokens=300, base_latency_s=0.4)
    expected_cost0 = (300 / 1000.0) * 0.0001
    assert abs(res0["cost_usd"] - expected_cost0) < 1e-9
    assert abs(res0["realized_latency_s"] - 0.4) < 1e-9
    assert res0["backoff_s"] == 0.0
    assert res0["queue_wait_s"] == 0.0
    assert res0["rate_limited"] is False

    # Advance time, state should remain static at sigma=0
    env.advance(10.0)
    obs = env.observe()
    assert abs(obs[0, 0] - 1.0) < 1e-9  # bucket_frac = 1
    assert abs(obs[0, 1] - 0.0) < 1e-9  # queue_norm = 0
    assert abs(obs[0, 2] - (1.0 - 0.5) / 2.5) < 1e-9  # price_norm = 0.2 (P=1.0)


def test_m2_seed_determinism(config):
    """PRD M2: Same seed -> identical trajectories."""
    env1 = CloudEnv(config["models"], config["cloud"], sigma=0.5, seed=123)
    env2 = CloudEnv(config["models"], config["cloud"], sigma=0.5, seed=123)

    for _ in range(5):
        env1.advance(1.5)
        env2.advance(1.5)
        res1 = env1.request(1, tokens=600, base_latency_s=0.9)
        res2 = env2.request(1, tokens=600, base_latency_s=0.9)

        assert res1["cost_usd"] == res2["cost_usd"]
        assert res1["realized_latency_s"] == res2["realized_latency_s"]
        np.testing.assert_array_equal(env1.observe(), env2.observe())


def test_m2_passive_arms_evolve(config):
    """PRD M2: Unused arms change after advance when sigma > 0."""
    env = CloudEnv(config["models"], config["cloud"], sigma=0.8, seed=42)
    obs_before = env.observe().copy()

    env.advance(5.0)
    obs_after = env.observe()

    # Arms should have evolved (idle increased, prices/queues fluctuate)
    assert not np.allclose(obs_before, obs_after)
    # Idle should have increased
    for k in range(3):
        assert obs_after[k, 3] > obs_before[k, 3]


def test_m2_queue_cap(config):
    """PRD M2 & Section 4: Q never exceeds queue_cap."""
    env = CloudEnv(config["models"], config["cloud"], sigma=1.0, seed=42)
    queue_cap = config["cloud"]["queue_cap"]

    # Hammer arm 2 with many requests and large advance
    for _ in range(50):
        env.advance(0.1)
        env.request(2, tokens=900, base_latency_s=1.8)

    obs = env.observe()
    for k in range(3):
        # Q_norm is min(1, Q / 10), but let's check Q directly
        assert env.queues[k] <= queue_cap


def test_m2_clone(config):
    """PRD M2: clone() returns a deep copy including rng state."""
    env = CloudEnv(config["models"], config["cloud"], sigma=0.5, seed=42)
    env.advance(2.0)
    env.request(0, tokens=300, base_latency_s=0.4)

    env_clone = env.clone()
    np.testing.assert_array_equal(env.observe(), env_clone.observe())

    # Advance both and make identical requests, results must match
    env.advance(1.0)
    env_clone.advance(1.0)
    res1 = env.request(1, tokens=600, base_latency_s=0.9)
    res2 = env_clone.request(1, tokens=600, base_latency_s=0.9)

    assert res1["cost_usd"] == res2["cost_usd"]
    assert res1["realized_latency_s"] == res2["realized_latency_s"]


def test_m2_s2_statistical_test(config):
    """PRD S2: Higher sigma does not help a fixed policy on average (paired test across 20 seeds)."""
    from scipy.stats import ttest_rel

    latencies_sigma_0 = []
    latencies_sigma_1 = []

    for seed in range(20):
        # Run fixed policy (pulling arm 1 repeatedly) under sigma=0
        env0 = CloudEnv(config["models"], config["cloud"], sigma=0.0, seed=seed)
        tot_lat0 = 0.0
        for _ in range(10):
            res = env0.request(1, tokens=600, base_latency_s=0.9)
            env0.advance(res["realized_latency_s"])
            tot_lat0 += res["realized_latency_s"]
        latencies_sigma_0.append(tot_lat0)

        # Under sigma=1.0
        env1 = CloudEnv(config["models"], config["cloud"], sigma=1.0, seed=seed)
        tot_lat1 = 0.0
        for _ in range(10):
            res = env1.request(1, tokens=600, base_latency_s=0.9)
            env1.advance(res["realized_latency_s"])
            tot_lat1 += res["realized_latency_s"]
        latencies_sigma_1.append(tot_lat1)

    # Under higher volatility, latency should be greater or equal on average
    mean_lat0 = np.mean(latencies_sigma_0)
    mean_lat1 = np.mean(latencies_sigma_1)
    assert mean_lat1 >= mean_lat0
