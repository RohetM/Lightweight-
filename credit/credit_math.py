"""Credit assignment mathematics (LOO and exact 3-role Shapley).
Implements PRD section 6 (M6) and Section 5.4.
"""
import itertools
import math
from envs.trace_cache import TraceCache


def compute_shapley(v: dict[frozenset[int], float], n: int = 3) -> list[float]:
    """Compute exact Shapley value credit assignment for n roles.

    Implements PRD Section 6 (M6).
    """
    phi = [0.0] * n
    for perm in itertools.permutations(range(n)):
        s = frozenset()
        for i in perm:
            phi[i] += v[s | {i}] - v[s]
            s |= {i}
    return [p / math.factorial(n) for p in phi]


def compute_loo(v: dict[frozenset[int], float], n: int = 3) -> list[float]:
    """Compute Leave-One-Out (LOO) credit assignment for n roles.

    Implements PRD Section 6 (M6).
    """
    full = frozenset(range(n))
    return [v[full] - v[full - {i}] for i in range(n)]


def build_coalition_values(
    task_id: str,
    chosen_path: list[str],
    trace_cache: TraceCache,
) -> dict[frozenset[int], float]:
    """Compute coalition values v(S) for all 2^3 subsets of roles.

    Implements PRD Section 6 (M6).
    Roles in S use the router's chosen arm; roles not in S use model 0 ('m0').
    Value = cache outcome of that path (success score in [0, 1]).
    Credit uses success only; cost and latency enter reward separately.
    """
    v: dict[frozenset[int], float] = {}

    for r in range(4):
        for subset in itertools.combinations(range(3), r):
            s_set = frozenset(subset)
            path = tuple(chosen_path[i] if i in s_set else "m0" for i in range(3))
            v[s_set] = float(trace_cache.outcome(task_id, path))

    return v


# Implementation Note:
# Implemented exact Shapley and LOO formulations per PRD Section 6 (M6).
# Coalition values correctly evaluate all 8 sub-paths using 'm0' baseline fallback.
