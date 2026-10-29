"""Tests for Module M4: Baselines and Oracles.
Implements PRD section 6 (M4).
"""
import yaml
import pytest
import numpy as np

from envs.trace_cache import TraceCache
from envs.cloud_env import CloudEnv
from envs.pipeline_env import PipelineEnv
from router.baselines import (
    RandomPolicy,
    CheapestPolicy,
    StrongestPolicy,
    GreedyPolicy,
    RouteLLMPolicy,
    CurrentStateOracle,
    ClairvoyantOracle,
    BestStaticPathPolicy,
    find_best_static_path_on_train,
)


@pytest.fixture
def config():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def trace_cache():
    return TraceCache("data/traces/synthetic.json")


def test_m4_cheapest_lowest_cost_at_sigma_zero(config, trace_cache):
    """PRD M4: Cheapest policy has lowest cost at sigma = 0."""
    cloud_env_cheap = CloudEnv(config["models"], config["cloud"], sigma=0.0, seed=42)
    pipeline_cheap = PipelineEnv(trace_cache, cloud_env_cheap, config["reward"], config["models"])
    res_cheap = pipeline_cheap.run_task("task_000", CheapestPolicy(), method_name="cheapest")

    cloud_env_strong = CloudEnv(config["models"], config["cloud"], sigma=0.0, seed=42)
    pipeline_strong = PipelineEnv(trace_cache, cloud_env_strong, config["reward"], config["models"])
    res_strong = pipeline_strong.run_task("task_000", StrongestPolicy(), method_name="strongest")

    cloud_env_rand = CloudEnv(config["models"], config["cloud"], sigma=0.0, seed=42)
    pipeline_rand = PipelineEnv(trace_cache, cloud_env_rand, config["reward"], config["models"])
    res_rand = pipeline_rand.run_task("task_000", RandomPolicy(seed=42), method_name="random")

    assert res_cheap["total_cost"] <= res_strong["total_cost"]
    assert res_cheap["total_cost"] <= res_rand["total_cost"]


def test_m4_oracle_hierarchy_statistical(config, trace_cache):
    """PRD M4: Statistical test on average: Clairvoyant >= CurrentState >= Random."""
    tasks = trace_cache.task_ids()[:15]

    u_clairvoyant = []
    u_current_state = []
    u_random = []

    for task_id in tasks:
        # Common random numbers: each policy sees identical initial cloud env
        base_cloud = CloudEnv(config["models"], config["cloud"], sigma=0.5, seed=100)

        # 1. Clairvoyant Oracle
        env1 = base_cloud.clone()
        p1 = PipelineEnv(trace_cache, env1, config["reward"], config["models"])
        r1 = p1.run_task(task_id, ClairvoyantOracle(config["reward"], config["models"]), method_name="clairvoyant")
        u_clairvoyant.append(r1["utility"])

        # 2. Current State Oracle
        env2 = base_cloud.clone()
        p2 = PipelineEnv(trace_cache, env2, config["reward"], config["models"])
        r2 = p2.run_task(task_id, CurrentStateOracle(config["reward"], config["models"]), method_name="current_state")
        u_current_state.append(r2["utility"])

        # 3. Random Policy
        env3 = base_cloud.clone()
        p3 = PipelineEnv(trace_cache, env3, config["reward"], config["models"])
        r3 = p3.run_task(task_id, RandomPolicy(seed=42), method_name="random")
        u_random.append(r3["utility"])

    mean_clair = np.mean(u_clairvoyant)
    mean_curr = np.mean(u_current_state)
    mean_rand = np.mean(u_random)

    # Statistical hierarchy on average
    assert mean_clair >= mean_curr, f"Clairvoyant ({mean_clair:.4f}) should be >= CurrentState ({mean_curr:.4f})"
    assert mean_curr >= mean_rand, f"CurrentState ({mean_curr:.4f}) should be >= Random ({mean_rand:.4f})"


def test_m4_all_seven_policies_interface(config, trace_cache):
    """Verify all 7 baseline policies adhere to select interface."""
    policies = [
        RandomPolicy(seed=0),
        CheapestPolicy(),
        StrongestPolicy(),
        GreedyPolicy(config["models"]),
        RouteLLMPolicy(),
        CurrentStateOracle(config["reward"], config["models"]),
        ClairvoyantOracle(config["reward"], config["models"]),
        BestStaticPathPolicy((1, 1, 1)),
    ]

    for pol in policies:
        cloud = CloudEnv(config["models"], config["cloud"], sigma=0.25, seed=0)
        pipeline = PipelineEnv(trace_cache, cloud, config["reward"], config["models"])
        rec = pipeline.run_task("task_002", pol, method_name="test")
        assert len(rec["path"]) == 3
        assert len(rec["hops"]) == 3
        for h in rec["hops"]:
            assert h["arm"] in [0, 1, 2]


def test_m4_find_best_static_path_on_train(config, trace_cache):
    """PRD M4: Verify best static path search identifies a valid 3-hop path."""
    tasks = trace_cache.task_ids()[:10]
    best_p = find_best_static_path_on_train(
        train_task_ids=tasks,
        trace_cache=trace_cache,
        reward_config=config["reward"],
        models_config=config["models"],
        cloud_config=config["cloud"],
        sigma=0.0,
        seed=42,
    )
    assert len(best_p) == 3
    assert all(a in [0, 1, 2] for a in best_p)
