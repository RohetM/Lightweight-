"""Tests for Module M3: Pipeline Environment.
Implements PRD section 6 (M3), Section 5.2, Section 5.3.
"""
import os
import tempfile
import yaml
import pytest
import numpy as np

from envs.trace_cache import TraceCache
from envs.cloud_env import CloudEnv
from envs.pipeline_env import PipelineEnv


@pytest.fixture
def config():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def trace_cache():
    return TraceCache("data/traces/synthetic.json")


def test_m3_fixed_path_outcome_and_total_latency(config, trace_cache):
    """PRD M3: Fixed path returns cached outcome, total latency equals sum of hop latencies."""
    task_id = "task_000"
    cloud_env = CloudEnv(config["models"], config["cloud"], sigma=0.0, seed=42)
    pipeline = PipelineEnv(trace_cache, cloud_env, config["reward"], config["models"])

    # Fixed policy choosing arm 1 at each hop: ("m1", "m1", "m1")
    class FixedPolicy:
        def select(self, obs, hop, task_id, cache, cloud):
            return 1

    run_record = pipeline.run_task(task_id, FixedPolicy(), method_name="fixed_m1", seed=42)

    # Check cached outcome matches
    expected_outcome = trace_cache.outcome(task_id, ("m1", "m1", "m1"))
    assert run_record["success"] == expected_outcome

    # Total latency equals sum of hop latencies
    hop_latencies = [h["latency_s"] for h in run_record["hops"]]
    assert len(hop_latencies) == 3
    assert abs(run_record["total_latency"] - sum(hop_latencies)) < 1e-9

    # Total cost equals sum of hop costs
    hop_costs = [h["cost_usd"] for h in run_record["hops"]]
    assert abs(run_record["total_cost"] - sum(hop_costs)) < 1e-9

    # Utility formula: success - lambda_c * total_cost - lambda_l * total_latency
    expected_u = expected_outcome - (config["reward"]["lambda_c"] * run_record["total_cost"]) - (config["reward"]["lambda_l"] * run_record["total_latency"])
    assert abs(run_record["utility"] - expected_u) < 1e-9


def test_m3_log_schema(config, trace_cache):
    """PRD 5.3: Run log schema matches exact requirements."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "runs.jsonl")
        cloud_env = CloudEnv(config["models"], config["cloud"], sigma=0.25, seed=42)
        pipeline = PipelineEnv(trace_cache, cloud_env, config["reward"], config["models"], log_file=log_path)

        class MockPolicy:
            def select(self, obs, hop, task_id, cache, cloud):
                return hop % 3

        record = pipeline.run_task("task_001", MockPolicy(), method_name="mock", seed=42)

        # Check required fields in run log record
        expected_keys = {
            "run_id", "task_id", "method", "sigma", "seed", "path",
            "hops", "success", "total_cost", "total_latency", "utility"
        }
        assert expected_keys.issubset(set(record.keys()))
        assert record["path"] == ["m0", "m1", "m2"]
        assert len(record["hops"]) == 3

        hop_keys = {
            "arm", "obs_before", "tokens", "cost_usd", "latency_s",
            "backoff_s", "queue_wait_s", "rate_limited", "decision_time_us"
        }
        for h in record["hops"]:
            assert hop_keys.issubset(set(h.keys()))
            # obs_before shape (3, 9)
            obs_before = np.array(h["obs_before"])
            assert obs_before.shape == (3, 9)

        # Verify log file was written
        assert os.path.exists(log_path)
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
