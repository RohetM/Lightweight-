"""Tests for Module M0: Synthetic generator.
Implements PRD section 6 (M0) and Section 5.1.
"""
import os
import json
import tempfile
import pytest
import numpy as np

# We test that synth_traces.py implements M0 requirements.
from data.synth_traces import generate


def test_m0_task_counts_and_structure():
    """PRD 5.1 & M0: 100 synthetic tasks, 39 calls and 27 outcomes per task."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "synthetic.json")
        generate(json_path, n_tasks=100, seed=42)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert data["metadata"]["synthetic"] is True
        assert "tasks" in data
        tasks = data["tasks"]
        assert len(tasks) == 100

        for task_id, task_data in tasks.items():
            assert "calls" in task_data
            assert "outcomes" in task_data

            calls = task_data["calls"]
            outcomes = task_data["outcomes"]

            # Must have exactly 39 call records per task
            assert len(calls) == 39

            # Must have exactly 27 path outcomes per task
            assert len(outcomes) == 27

            # Check call fields: tokens_in, tokens_out, latency_s, routellm_score
            for call_key, call_val in calls.items():
                assert "tokens_in" in call_val
                assert "tokens_out" in call_val
                assert "latency_s" in call_val
                assert "routellm_score" in call_val
                assert isinstance(call_val["tokens_in"], int)
                assert isinstance(call_val["tokens_out"], int)
                assert isinstance(call_val["latency_s"], (float, int))
                assert 0.0 <= call_val["routellm_score"] <= 1.0

            # Check all 27 paths resolve and match hop-2 leaf records
            models = ["m0", "m1", "m2"]
            for m0_choice in models:
                for m1_choice in models:
                    for m2_choice in models:
                        path_key = f"{m0_choice}_{m1_choice}_{m2_choice}"
                        leaf_key = f"2_{path_key}"
                        assert path_key in outcomes
                        assert leaf_key in calls
                        assert 0.0 <= outcomes[path_key] <= 1.0
                        # Derived from hop-2 leaf record
                        assert outcomes[path_key] == calls[leaf_key]["success"]


def test_m0_determinism():
    """PRD M0: Same seed -> byte-identical file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = os.path.join(tmpdir, "synth1.json")
        path2 = os.path.join(tmpdir, "synth2.json")

        generate(path1, n_tasks=10, seed=123)
        generate(path2, n_tasks=10, seed=123)

        with open(path1, "rb") as f1, open(path2, "rb") as f2:
            assert f1.read() == f2.read()


def test_m0_model_cost_and_latency_ordering():
    """PRD M0: Stronger models cost more tokens and time (m0 < m1 < m2)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "synth.json")
        generate(json_path, n_tasks=5, seed=0)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        calls = data["tasks"]["task_000"]["calls"]

        # Hop 0 tokens & latency check
        c0 = calls["0_m0"]
        c1 = calls["0_m1"]
        c2 = calls["0_m2"]

        tokens_0 = c0["tokens_in"] + c0["tokens_out"]
        tokens_1 = c1["tokens_in"] + c1["tokens_out"]
        tokens_2 = c2["tokens_in"] + c2["tokens_out"]

        assert tokens_0 < tokens_1 < tokens_2
        assert c0["latency_s"] < c1["latency_s"] < c2["latency_s"]
        assert tokens_0 == 300
        assert tokens_1 == 600
        assert tokens_2 == 900
        assert c0["latency_s"] == 0.4
        assert c1["latency_s"] == 0.9
        assert c2["latency_s"] == 1.8


def test_m0_loo_neq_shapley():
    """PRD M0: Synergy condition causes LOO != Shapley on some tasks."""
    import itertools
    import math

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "synth.json")
        generate(json_path, n_tasks=10, seed=42)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Compute LOO and Shapley on path ("m2", "m2", "m2")
        # Coalition value v(S): roles in S use m2, roles not in S use m0
        found_diff = False
        for task_id, task in data["tasks"].items():
            outcomes = task["outcomes"]

            def v(s):
                path = tuple("m2" if i in s else "m0" for i in range(3))
                return outcomes["_".join(path)]

            # Shapley for 3 roles
            phi = [0.0] * 3
            for perm in itertools.permutations(range(3)):
                s = frozenset()
                for i in perm:
                    phi[i] += v(s | {i}) - v(s)
                    s |= {i}
            shapley = [p / math.factorial(3) for p in phi]

            # LOO for 3 roles
            full = frozenset(range(3))
            loo = [v(full) - v(full - {i}) for i in range(3)]

            # Check if LOO != Shapley for any role
            if any(abs(l - s) > 1e-4 for l, s in zip(loo, shapley)):
                found_diff = True
                break

        assert found_diff, "Synergy should cause LOO != Shapley on tasks with m1/m2 at hop 0 and hop 1"
