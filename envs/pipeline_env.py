"""Pipeline environment executing 3-hop sequential multi-agent pipelines.
Implements PRD section 6 (M3), Section 5.2, Section 5.3.
"""
import json
import os
import time
import uuid
from typing import Any, Optional
import numpy as np

from envs.trace_cache import TraceCache
from envs.cloud_env import CloudEnv


class PipelineEnv:
    """Executes 3-hop agent pipelines with state-aware environment and logging.

    Implements PRD Section 6 (M3) and Section 5.2, 5.3.
    """

    def __init__(
        self,
        trace_cache: TraceCache,
        cloud_env: CloudEnv,
        reward_config: dict[str, float],
        models_config: list[dict[str, Any]],
        log_file: Optional[str] = None,
    ):
        """Initialize PipelineEnv with trace cache, cloud env and configurations.

        Implements PRD Section 6 (M3).
        """
        self.trace_cache = trace_cache
        self.cloud_env = cloud_env
        self.lambda_c = float(reward_config.get("lambda_c", 10.0))
        self.lambda_l = float(reward_config.get("lambda_l", 0.05))
        self.models_config = models_config
        self.log_file = log_file

        self.price_static = np.array(
            [m["price_per_1k_tokens"] / 0.0050 for m in models_config], dtype=np.float32
        )
        self.tier_static = np.array(
            [m.get("tier", i * 0.5) for i, m in enumerate(models_config)], dtype=np.float32
        )

    def build_obs(self, hop: int) -> np.ndarray:
        """Construct (3, 9) float32 observation vector for current hop.

        Implements PRD Section 5.2.
        Columns:
        [bucket_frac, queue_norm, price_norm, idle_norm,
         is_hop_0, is_hop_1, is_hop_2,
         price_static, tier_static]
        """
        dyn_obs = self.cloud_env.observe()  # (3, 4)
        obs = np.zeros((3, 9), dtype=np.float32)

        # Dynamic cloud features (cols 0-3)
        obs[:, :4] = dyn_obs

        # Hop one-hot indicator (cols 4-6)
        obs[:, 4 + hop] = 1.0

        # Static features (cols 7-8)
        obs[:, 7] = self.price_static
        obs[:, 8] = self.tier_static

        return obs

    def run_task(
        self,
        task_id: str,
        policy: Any,
        method_name: str = "unnamed",
        seed: int = 0,
        run_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute a 3-hop pipeline task using the provided routing policy.

        Implements PRD Section 6 (M3) and Section 5.3.
        """
        if run_id is None:
            run_id = f"run_{uuid.uuid4().hex[:12]}"

        hops_log = []
        chosen_arms = []
        chosen_prefix = []

        for hop in range(3):
            obs_before = self.build_obs(hop)

            # Time policy call in microseconds
            t_start = time.perf_counter_ns()
            arm = policy.select(obs_before, hop, task_id, self.trace_cache, self.cloud_env)
            t_end = time.perf_counter_ns()
            decision_time_us = (t_end - t_start) / 1000.0

            chosen_arms.append(arm)
            chosen_prefix.append(f"m{arm}")

            # Look up trace call record
            call_record = self.trace_cache.call(task_id, hop, tuple(chosen_prefix))
            tokens = int(call_record["tokens_in"] + call_record["tokens_out"])
            base_latency_s = float(call_record["latency_s"])

            # Send active request to CloudEnv
            req_res = self.cloud_env.request(arm, tokens, base_latency_s)

            # Record hop details per PRD 5.3
            hop_data = {
                "arm": int(arm),
                "obs_before": obs_before.tolist(),
                "tokens": tokens,
                "cost_usd": float(req_res["cost_usd"]),
                "latency_s": float(req_res["realized_latency_s"]),
                "backoff_s": float(req_res["backoff_s"]),
                "queue_wait_s": float(req_res["queue_wait_s"]),
                "rate_limited": bool(req_res["rate_limited"]),
                "decision_time_us": float(decision_time_us),
            }
            hops_log.append(hop_data)

            # Advance cloud environment by realized latency
            self.cloud_env.advance(dt=req_res["realized_latency_s"])

        # Final outcome from hop 2
        path_tuple = tuple(chosen_prefix)
        success = float(self.trace_cache.outcome(task_id, path_tuple))
        total_cost = float(sum(h["cost_usd"] for h in hops_log))
        total_latency = float(sum(h["latency_s"] for h in hops_log))
        utility = float(success - (self.lambda_c * total_cost) - (self.lambda_l * total_latency))

        run_record = {
            "run_id": run_id,
            "task_id": task_id,
            "method": method_name,
            "sigma": float(self.cloud_env.sigma),
            "seed": int(seed),
            "path": chosen_prefix,
            "hops": hops_log,
            "success": success,
            "total_cost": total_cost,
            "total_latency": total_latency,
            "utility": utility,
        }

        # Write to run log if configured (PRD 5.3)
        if self.log_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(run_record) + "\n")

        return run_record


# Implementation Note:
# Implemented exactly per PRD Section 6 (M3), Section 5.2, and Section 5.3.
# Obs vector (3, 9) float32 correctly combines dynamic cloud state, hop one-hot, and static price/tier.
