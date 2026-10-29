"""Baseline policies and oracles for Restless Router.
Implements PRD section 6 (M4).
"""
import itertools
from typing import Any, Optional
import numpy as np

from envs.trace_cache import TraceCache
from envs.cloud_env import CloudEnv


def compute_normalized_performance(method_u: float, random_u: float, clairvoyant_u: float) -> float:
    """Compute normalized performance: (method - random) / (clairvoyant - random).

    Implements PRD Section 6 (M4).
    """
    denom = clairvoyant_u - random_u
    if abs(denom) < 1e-9:
        return 0.0
    return float((method_u - random_u) / denom)


class RandomPolicy:
    """Random arm selection using an explicitly seeded generator.

    Implements PRD Section 6 (M4) - Policy 1.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select(
        self,
        obs: np.ndarray,
        hop: int,
        task_id: str,
        trace_cache: TraceCache,
        cloud_env: CloudEnv,
    ) -> int:
        return int(self.rng.integers(0, 3))


class CheapestPolicy:
    """Always chooses the cheapest arm (arm 0).

    Implements PRD Section 6 (M4) - Policy 2.
    """

    def select(
        self,
        obs: np.ndarray,
        hop: int,
        task_id: str,
        trace_cache: TraceCache,
        cloud_env: CloudEnv,
    ) -> int:
        return 0


class StrongestPolicy:
    """Always chooses the strongest arm (arm 2).

    Implements PRD Section 6 (M4) - Policy 3.
    """

    def select(
        self,
        obs: np.ndarray,
        hop: int,
        task_id: str,
        trace_cache: TraceCache,
        cloud_env: CloudEnv,
    ) -> int:
        return 2


class GreedyPolicy:
    """Greedy arm selection minimizing current_effective_price * (1 + queue_wait_s).

    Implements PRD Section 6 (M4) - Policy 4.
    """

    def __init__(self, models_config: Optional[list[dict[str, Any]]] = None):
        self.models_config = models_config

    def select(
        self,
        obs: np.ndarray,
        hop: int,
        task_id: str,
        trace_cache: TraceCache,
        cloud_env: CloudEnv,
    ) -> int:
        metrics = []
        for k in range(cloud_env.n_arms):
            eff_price = cloud_env.prices_per_1k[k] * cloud_env.price_mults[k]
            queue_wait_s = cloud_env.queues[k] / cloud_env.service_rates[k]
            metrics.append(eff_price * (1.0 + queue_wait_s))
        return int(np.argmin(metrics))


class RouteLLMPolicy:
    """Threshold routing based on prompt difficulty score: score >= 0.5 -> 2 else 0.

    Implements PRD Section 6 (M4) - Policy 5.
    """

    def __init__(self):
        self._current_prefix: list[str] = []

    def select(
        self,
        obs: np.ndarray,
        hop: int,
        task_id: str,
        trace_cache: TraceCache,
        cloud_env: CloudEnv,
    ) -> int:
        if hop == 0:
            self._current_prefix = []

        # Look up cached routellm_score for current hop prompt
        lookup_prefix = tuple(self._current_prefix + ["m0"])
        call_rec = trace_cache.call(task_id, hop, lookup_prefix)
        score = float(call_rec.get("routellm_score", 0.0))

        chosen_arm = 2 if score >= 0.5 else 0
        self._current_prefix.append(f"m{chosen_arm}")
        return chosen_arm


class CurrentStateOracle:
    """Myopic oracle simulating immediate cost/latency on cloned state plus downstream success.

    Implements PRD Section 6 (M4) - Policy 6.
    """

    def __init__(self, reward_config: dict[str, float], models_config: list[dict[str, Any]]):
        self.lambda_c = float(reward_config.get("lambda_c", 10.0))
        self.lambda_l = float(reward_config.get("lambda_l", 0.05))
        self.models_config = models_config
        self._current_prefix: list[str] = []

    def select(
        self,
        obs: np.ndarray,
        hop: int,
        task_id: str,
        trace_cache: TraceCache,
        cloud_env: CloudEnv,
    ) -> int:
        if hop == 0:
            self._current_prefix = []

        all_models = ["m0", "m1", "m2"]
        exp_utilities = []

        for k in range(cloud_env.n_arms):
            cand_prefix = tuple(self._current_prefix + [f"m{k}"])
            call_rec = trace_cache.call(task_id, hop, cand_prefix)
            tokens = int(call_rec["tokens_in"] + call_rec["tokens_out"])
            base_lat = float(call_rec["latency_s"])

            # Immediate cost and latency on a cloned cloud environment (without future noise)
            env_clone = cloud_env.clone()
            req_res = env_clone.request(k, tokens, base_lat)
            cost_usd = req_res["cost_usd"]
            latency_s = req_res["realized_latency_s"]

            # Compute average downstream success from trace cache
            if hop == 0:
                matching_paths = [
                    (f"m{k}", m1, m2) for m1 in all_models for m2 in all_models
                ]
            elif hop == 1:
                matching_paths = [
                    (self._current_prefix[0], f"m{k}", m2) for m2 in all_models
                ]
            else:
                matching_paths = [
                    (self._current_prefix[0], self._current_prefix[1], f"m{k}")
                ]

            avg_success = float(np.mean([trace_cache.outcome(task_id, p) for p in matching_paths]))
            exp_u = avg_success - (self.lambda_c * cost_usd) - (self.lambda_l * latency_s)
            exp_utilities.append(exp_u)

        best_arm = int(np.argmax(exp_utilities))
        self._current_prefix.append(f"m{best_arm}")
        return best_arm


class ClairvoyantOracle:
    """Upper bound oracle evaluating all 27 paths against exact seeded rollout.

    Implements PRD Section 6 (M4) - Policy 7.
    """

    def __init__(self, reward_config: dict[str, float], models_config: list[dict[str, Any]]):
        self.lambda_c = float(reward_config.get("lambda_c", 10.0))
        self.lambda_l = float(reward_config.get("lambda_l", 0.05))
        self.models_config = models_config
        self._planned_path: list[int] = []
        self.last_metrics: Optional[dict[str, float]] = None

    def select(
        self,
        obs: np.ndarray,
        hop: int,
        task_id: str,
        trace_cache: TraceCache,
        cloud_env: CloudEnv,
    ) -> int:
        if hop == 0:
            models = ["m0", "m1", "m2"]
            best_u = -float("inf")
            best_path = ("m0", "m0", "m0")
            best_metrics: dict[str, float] = {}

            # Evaluate all 27 paths against exact clone rollout
            for p in itertools.product(models, repeat=3):
                clone = cloud_env.clone()
                path_cost = 0.0
                path_lat = 0.0

                for t, m_id in enumerate(p):
                    arm_idx = int(m_id[1])
                    sub_prefix = p[: t + 1]
                    call_rec = trace_cache.call(task_id, t, sub_prefix)
                    tokens = int(call_rec["tokens_in"] + call_rec["tokens_out"])
                    base_lat = float(call_rec["latency_s"])

                    req_res = clone.request(arm_idx, tokens, base_lat)
                    path_cost += req_res["cost_usd"]
                    path_lat += req_res["realized_latency_s"]
                    clone.advance(req_res["realized_latency_s"])

                success = float(trace_cache.outcome(task_id, p))
                utility = success - (self.lambda_c * path_cost) - (self.lambda_l * path_lat)

                if utility > best_u:
                    best_u = utility
                    best_path = p
                    best_metrics = {
                        "utility": utility,
                        "cost": path_cost,
                        "success": success,
                        "latency": path_lat,
                    }

            self._planned_path = [int(m[1]) for m in best_path]
            self.last_metrics = best_metrics

        return self._planned_path[hop]


class BestStaticPathPolicy:
    """Choose, on TRAIN tasks only, the single model per hop (27 options) with the highest
    mean utility at each sigma; apply it deterministically on test.

    Implements PRD Section 6 (M4) - supplementary baseline.
    """

    def __init__(self, path: tuple[int, int, int] = (1, 1, 1)):
        self.path = tuple(int(x) for x in path)

    def select(
        self,
        obs: np.ndarray,
        hop: int,
        task_id: str,
        trace_cache: TraceCache,
        cloud_env: CloudEnv,
    ) -> int:
        return self.path[hop]


def find_best_static_path_on_train(
    train_task_ids: list[str],
    trace_cache: TraceCache,
    reward_config: dict[str, float],
    models_config: list[dict[str, Any]],
    cloud_config: dict[str, Any],
    sigma: float,
    seed: int = 42,
) -> tuple[int, int, int]:
    """Find the single path out of 27 with the highest mean utility on train tasks at sigma.

    Evaluates each candidate path (a0, a1, a2) on the training task stream under sigma.
    """
    from envs.pipeline_env import PipelineEnv

    best_p = (0, 0, 0)
    best_u = -float("inf")

    for a0 in range(len(models_config)):
        for a1 in range(len(models_config)):
            for a2 in range(len(models_config)):
                candidate = (a0, a1, a2)
                candidate_policy = BestStaticPathPolicy(candidate)
                cloud = CloudEnv(models_config, cloud_config, sigma=sigma, seed=seed)
                pipeline = PipelineEnv(trace_cache, cloud, reward_config, models_config)

                tot_u = 0.0
                for tid in train_task_ids:
                    rec = pipeline.run_task(tid, candidate_policy, seed=seed)
                    tot_u += rec["utility"]
                    cloud.advance(cloud_config.get("inter_task_gap_s", 1.0))

                mean_u = tot_u / max(1, len(train_task_ids))
                if mean_u > best_u:
                    best_u = mean_u
                    best_p = candidate

    return best_p


# Implementation Note:
# Implemented all 7 baseline policies and oracles per PRD Section 6 (M4).
# BestStaticPathPolicy added per user request: evaluates all 27 paths on train tasks
# under each target sigma and applies the top static path on test.
# ClairvoyantOracle evaluates exact seeded rollouts over all 27 paths.
# Normalized performance metric implemented per formula.
