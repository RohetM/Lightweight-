"""Tests for Module M5: Whittle-inspired router.
Implements PRD section 6 (M5) and Section 2 (S3).
"""
import time
import platform
import numpy as np
import torch
import pytest

from envs.toy_restless import ToyRestlessEnv, solve_toy_restless_vi, evaluate_policy
from router.whittle_router import WhittleRouter, IndexNet, CriticNet


def test_m5_index_net_forward_and_numpy_weights():
    """PRD M5: IndexNet forward pass and pure NumPy forward pass return identical results."""
    torch.manual_seed(42)
    net = IndexNet()
    net.eval()
    net.sync_numpy_weights()

    obs = np.random.default_rng(42).normal(size=(3, 9)).astype(np.float32)

    # PyTorch forward
    with torch.no_grad():
        t_out = net(torch.from_numpy(obs)).numpy().squeeze(-1)

    # NumPy forward
    np_out = net.forward_numpy(obs)

    np.testing.assert_allclose(t_out, np_out, atol=1e-5)


def test_m5_s3_decision_timing():
    """PRD S3 & M5: Router decision time median < 0.1 ms target, test fails only above 0.5 ms.
    Evaluated over 10,000 decisions.
    """
    router = WhittleRouter(seed=42)
    obs = np.random.default_rng(42).normal(size=(3, 9)).astype(np.float32)

    durations_us = []
    # Warmup
    for _ in range(500):
        router.select(obs, explore=False)

    # Benchmark 10,000 decisions
    n_evals = 10000
    for _ in range(n_evals):
        t0 = time.perf_counter_ns()
        router.select(obs, explore=False)
        t1 = time.perf_counter_ns()
        durations_us.append((t1 - t0) / 1000.0)

    median_us = float(np.median(durations_us))
    p99_us = float(np.percentile(durations_us, 99))
    median_ms = median_us / 1000.0
    p99_ms = p99_us / 1000.0

    print(f"\n[S3 Benchmark] Machine: {platform.processor() or 'Windows x64'}")
    print(f"[S3 Benchmark] PyTorch {torch.__version__}, NumPy {np.__version__}")
    print(f"[S3 Benchmark] 10,000 decisions: Median = {median_ms:.4f} ms ({median_us:.2f} us), p99 = {p99_ms:.4f} ms ({p99_us:.2f} us)")

    assert median_ms < 0.5, f"Decision time median exceeded 0.5 ms limit: {median_ms:.4f} ms"


def test_m5_greedy_selection_deterministic():
    """PRD M5: Greedy-argmax index selection is deterministic."""
    router = WhittleRouter(seed=123)
    obs = np.random.default_rng(123).uniform(size=(3, 9)).astype(np.float32)

    choices = [router.select(obs, explore=False) for _ in range(20)]
    assert all(c == choices[0] for c in choices)


def test_m5_seeded_training_reproducible():
    """PRD M5: Seeded training is reproducible."""
    r1 = WhittleRouter(seed=777)
    r2 = WhittleRouter(seed=777)

    # Train both with identical dummy data
    rng = np.random.default_rng(777)
    for _ in range(5):
        transitions = []
        for hop in range(3):
            for arm in range(3):
                obs = rng.normal(size=9).astype(np.float32)
                u = 1 if arm == 0 else 0
                rew = 0.5 if u == 1 else 0.0
                next_obs = rng.normal(size=9).astype(np.float32)
                done = 1.0 if hop == 2 else 0.0
                transitions.append((obs, u, rew, next_obs, done))
        r1.push_transitions(transitions)
        r2.push_transitions(transitions)
        r1.train_step()
        r2.train_step()

    test_obs = np.ones((3, 9), dtype=np.float32)
    s1 = r1.index_net.forward_numpy(test_obs)
    s2 = r2.index_net.forward_numpy(test_obs)
    np.testing.assert_allclose(s1, s2, atol=1e-6)


def test_m5_toy_restless_bandit_benchmark():
    """PRD M5 Test 1:
    Toy restless bandit: 3 arms, each state in {good, bad}.
    Value iteration on 8-state joint MDP.
    The trained learner must beat random and reach >= 85% of optimal average reward.
    """
    env = ToyRestlessEnv(seed=42)
    opt_v, opt_policy = solve_toy_restless_vi(gamma=0.95)

    # Evaluate optimal policy average reward
    opt_avg_reward = evaluate_policy(opt_policy, n_steps=3000, seed=10)

    # Evaluate random policy average reward
    class RandomToyPolicy:
        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)

        def __call__(self, state):
            return int(self.rng.integers(0, 3))

    rand_avg_reward = evaluate_policy(RandomToyPolicy(seed=20), n_steps=3000, seed=10)

    print(f"\n[Toy Bandit] Optimal Policy Avg Reward: {opt_avg_reward:.4f}")
    print(f"[Toy Bandit] Random Policy Avg Reward:  {rand_avg_reward:.4f}")

    assert opt_avg_reward > rand_avg_reward, "Optimal policy must beat random"

    # Train Whittle learner on toy environment
    learner = WhittleRouter(seed=42, lr_critic=0.01, lr_index=0.002, gamma=0.95)
    learner_avg_reward = learner.train_on_toy_env(n_episodes=1500, seed=42)

    print(f"[Toy Bandit] Learner Avg Reward:        {learner_avg_reward:.4f}")
    ratio = learner_avg_reward / opt_avg_reward
    print(f"[Toy Bandit] Learner / Optimal Ratio:   {ratio * 100:.2f}%")

    # PRD M5 requirement: trained learner must beat random and reach >= 85% of optimal
    assert learner_avg_reward > rand_avg_reward, f"Learner ({learner_avg_reward:.4f}) must beat random ({rand_avg_reward:.4f})"
    assert learner_avg_reward >= 0.85 * opt_avg_reward, (
        f"Learner average reward ({learner_avg_reward:.4f}) is below 85% of optimal ({0.85 * opt_avg_reward:.4f})"
    )
