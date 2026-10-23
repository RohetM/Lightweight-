"""Tests for Module M6: Credit assignment math.
Implements PRD section 6 (M6).
"""
import pytest
from credit.credit_math import (
    compute_shapley,
    compute_loo,
    build_coalition_values,
)
from envs.trace_cache import TraceCache


@pytest.fixture
def trace_cache():
    return TraceCache("data/traces/synthetic.json")


def test_m6_shapley_efficiency():
    """PRD M6: Shapley efficiency sum(phi) == v(N) - v(empty)."""
    # Dummy coalition values with non-linear synergies
    v = {
        frozenset(): 0.2,
        frozenset([0]): 0.4,
        frozenset([1]): 0.35,
        frozenset([2]): 0.3,
        frozenset([0, 1]): 0.75,  # synergy
        frozenset([0, 2]): 0.5,
        frozenset([1, 2]): 0.6,
        frozenset([0, 1, 2]): 0.95,
    }

    phi = compute_shapley(v, n=3)
    full = frozenset([0, 1, 2])
    empty = frozenset()

    expected_gain = v[full] - v[empty]
    assert abs(sum(phi) - expected_gain) < 1e-9


def test_m6_null_role_zero_credit():
    """PRD M6: A null role gets 0 credit under both LOO and Shapley."""
    # Role 2 adds zero marginal value to all coalitions
    v = {
        frozenset(): 0.3,
        frozenset([0]): 0.6,
        frozenset([1]): 0.5,
        frozenset([2]): 0.3,  # no gain over empty
        frozenset([0, 1]): 0.8,
        frozenset([0, 2]): 0.6,  # no gain over {0}
        frozenset([1, 2]): 0.5,  # no gain over {1}
        frozenset([0, 1, 2]): 0.8,  # no gain over {0, 1}
    }

    shapley = compute_shapley(v, n=3)
    loo = compute_loo(v, n=3)

    assert abs(shapley[2]) < 1e-9, f"Null role Shapley credit should be 0, got {shapley[2]}"
    assert abs(loo[2]) < 1e-9, f"Null role LOO credit should be 0, got {loo[2]}"


def test_m6_symmetric_roles_equal_credit():
    """PRD M6: Symmetric roles get equal Shapley credit."""
    # Roles 0 and 1 are completely symmetric
    v = {
        frozenset(): 0.1,
        frozenset([0]): 0.4,
        frozenset([1]): 0.4,
        frozenset([2]): 0.2,
        frozenset([0, 1]): 0.7,
        frozenset([0, 2]): 0.5,
        frozenset([1, 2]): 0.5,
        frozenset([0, 1, 2]): 0.9,
    }

    shapley = compute_shapley(v, n=3)
    assert abs(shapley[0] - shapley[1]) < 1e-9


def test_m6_loo_neq_shapley_on_synergy(trace_cache):
    """PRD M6: LOO != Shapley on synergy tasks from trace cache."""
    # Find a task and path with synergy where LOO != Shapley
    diff_found = False

    for tid in trace_cache.task_ids()[:10]:
        chosen_path = ["m2", "m2", "m2"]
        v = build_coalition_values(tid, chosen_path, trace_cache)
        shapley = compute_shapley(v, n=3)
        loo = compute_loo(v, n=3)

        if any(abs(l - s) > 1e-4 for l, s in zip(loo, shapley)):
            diff_found = True
            break

    assert diff_found, "LOO and Shapley should produce different credit assignments on synergy paths"
