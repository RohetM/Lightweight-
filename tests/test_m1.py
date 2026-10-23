"""Tests for Module M1: TraceCache.
Implements PRD section 6 (M1) and Section 5.1.
"""
import os
import tempfile
import pytest

from data.synth_traces import generate
from envs.trace_cache import TraceCache


@pytest.fixture(scope="module")
def trace_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "synthetic.json")
        generate(path, n_tasks=10, seed=42)
        yield path


def test_m1_trace_cache_deterministic(trace_file):
    """PRD M1: TraceCache loads deterministically."""
    cache1 = TraceCache(trace_file)
    cache2 = TraceCache(trace_file)

    task_ids1 = cache1.task_ids()
    task_ids2 = cache2.task_ids()

    assert task_ids1 == task_ids2
    assert len(task_ids1) == 10

    for tid in task_ids1:
        assert cache1.outcome(tid, ("m0", "m1", "m2")) == cache2.outcome(tid, ("m0", "m1", "m2"))
        assert cache1.call(tid, 0, ("m0",)) == cache2.call(tid, 0, ("m0",))


def test_m1_missing_key_raises_error(trace_file):
    """PRD M1: Missing key raises a clear error."""
    cache = TraceCache(trace_file)

    with pytest.raises(KeyError):
        cache.call("non_existent_task", 0, ("m0",))

    with pytest.raises(KeyError):
        # Invalid model or prefix
        cache.call("task_000", 0, ("invalid_model",))

    with pytest.raises(KeyError):
        cache.outcome("non_existent_task", ("m0", "m1", "m2"))

    with pytest.raises(KeyError):
        cache.outcome("task_000", ("m0", "m1", "invalid_model"))


def test_m1_all_27_paths_resolve(trace_file):
    """PRD M1 & 5.1: All 27 paths resolve consistently with prefix calls."""
    cache = TraceCache(trace_file)
    models = ["m0", "m1", "m2"]

    for tid in cache.task_ids():
        for m0 in models:
            c0 = cache.call(tid, 0, (m0,))
            assert "tokens_in" in c0 and "tokens_out" in c0 and "latency_s" in c0

            for m1 in models:
                c1 = cache.call(tid, 1, (m0, m1))
                assert "tokens_in" in c1 and "tokens_out" in c1 and "latency_s" in c1

                for m2 in models:
                    c2 = cache.call(tid, 2, (m0, m1, m2))
                    assert "tokens_in" in c2 and "tokens_out" in c2 and "latency_s" in c2
                    assert "success" in c2

                    outcome_val = cache.outcome(tid, (m0, m1, m2))
                    assert isinstance(outcome_val, float)
                    assert 0.0 <= outcome_val <= 1.0
                    assert outcome_val == c2["success"]
