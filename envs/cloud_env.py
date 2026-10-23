"""Restless Cloud Environment simulator.
Implements PRD section 6 (M2), Section 2 (S1, S2), Section 4, Section 5.2.
"""
import copy
from typing import Any
import numpy as np


class CloudEnv:
    """Simulates multi-model cloud state dynamics (buckets, queues, prices, idle).

    Implements PRD Section 6 (M2) and Section 4.
    """

    def __init__(
        self,
        models_config: list[dict[str, Any]],
        cloud_config: dict[str, Any],
        sigma: float = 0.0,
        seed: int = 0,
    ):
        """Initialize restless cloud environment.

        Implements PRD Section 6 (M2).
        """
        self.sigma = float(sigma)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.models_config = models_config
        self.cloud_config = cloud_config

        self.n_arms = len(models_config)
        self.capacities = np.array([m["bucket_capacity"] for m in models_config], dtype=np.float64)
        self.refill_rates = np.array([m["refill_rate_per_sec"] for m in models_config], dtype=np.float64)
        self.service_rates = np.array([m["service_rate_per_sec"] for m in models_config], dtype=np.float64)
        self.prices_per_1k = np.array([m["price_per_1k_tokens"] for m in models_config], dtype=np.float64)

        self.queue_cap = float(cloud_config.get("queue_cap", 20.0))
        self.price_min = float(cloud_config.get("price_min", 0.5))
        self.price_max = float(cloud_config.get("price_max", 3.0))
        self.inter_task_gap_s = float(cloud_config.get("inter_task_gap_s", 1.0))

        # Initial state per arm
        self.buckets = self.capacities.copy()
        self.queues = np.zeros(self.n_arms, dtype=np.float64)
        self.price_mults = np.ones(self.n_arms, dtype=np.float64)
        self.idles = np.zeros(self.n_arms, dtype=np.float64)

    def advance(self, dt: float) -> None:
        """Passive time advance for all arms.

        Implements PRD Section 6 (M2).
        """
        if dt <= 0.0:
            return

        if self.sigma == 0.0:
            # Static control: B_k = cap_k, Q_k = 0, P_k = 1 always
            self.buckets = self.capacities.copy()
            self.queues[:] = 0.0
            self.price_mults[:] = 1.0
            self.idles += dt
            return

        # sigma > 0: passive advance for all arms
        for k in range(self.n_arms):
            eff_refill = self.refill_rates[k] / (1.0 + self.sigma)
            self.buckets[k] = min(self.capacities[k], self.buckets[k] + eff_refill * dt)
            arrivals = float(self.rng.poisson(2.0 * self.sigma * dt))
            departures = float(self.rng.poisson(self.service_rates[k] * dt))
            self.queues[k] = min(self.queue_cap, max(0.0, self.queues[k] + arrivals - departures))
            noise = float(self.rng.normal(0.0, 0.15 * self.sigma))
            self.price_mults[k] = float(
                np.clip(
                    self.price_mults[k] + 0.2 * (1.0 - self.price_mults[k]) * dt + noise,
                    self.price_min,
                    self.price_max,
                )
            )
            self.idles[k] += dt

    def request(self, arm: int, tokens: int, base_latency_s: float) -> dict[str, Any]:
        """Process active request on chosen arm.

        Implements PRD Section 6 (M2).
        """
        if self.sigma == 0.0:
            # Static control: exact trace cost and latency
            cost_usd = (tokens / 1000.0) * self.prices_per_1k[arm]
            realized_latency_s = base_latency_s
            backoff_s = 0.0
            queue_wait_s = 0.0
            rate_limited = False
            self.idles[arm] = 0.0
            return {
                "cost_usd": float(cost_usd),
                "realized_latency_s": float(realized_latency_s),
                "backoff_s": float(backoff_s),
                "queue_wait_s": float(queue_wait_s),
                "rate_limited": bool(rate_limited),
            }

        # sigma > 0
        eff_refill = self.refill_rates[arm] / (1.0 + self.sigma)
        if self.buckets[arm] >= tokens:
            backoff_s = 0.0
            self.buckets[arm] -= tokens
            rate_limited = False
        else:
            backoff_s = (tokens - self.buckets[arm]) / eff_refill
            self.buckets[arm] = 0.0
            rate_limited = True

        queue_wait_s = self.queues[arm] / self.service_rates[arm]
        self.queues[arm] = min(self.queue_cap, self.queues[arm] + 1.0)
        self.idles[arm] = 0.0

        realized_latency_s = base_latency_s + backoff_s + queue_wait_s
        cost_usd = (tokens / 1000.0) * self.prices_per_1k[arm] * self.price_mults[arm]

        return {
            "cost_usd": float(cost_usd),
            "realized_latency_s": float(realized_latency_s),
            "backoff_s": float(backoff_s),
            "queue_wait_s": float(queue_wait_s),
            "rate_limited": bool(rate_limited),
        }

    def observe(self) -> np.ndarray:
        """Return (3, 4) float32 observation of cloud dynamic features.

        Implements PRD Section 5.2 and Section 6 (M2).
        Columns: bucket_frac, queue_norm, price_norm, idle_norm.
        """
        obs = np.zeros((self.n_arms, 4), dtype=np.float32)
        for k in range(self.n_arms):
            bucket_frac = self.buckets[k] / self.capacities[k]
            queue_norm = min(1.0, self.queues[k] / 10.0)
            price_norm = (self.price_mults[k] - 0.5) / 2.5
            idle_norm = min(1.0, self.idles[k] / 10.0)
            obs[k] = [bucket_frac, queue_norm, price_norm, idle_norm]
        return obs

    def clone(self) -> "CloudEnv":
        """Return a deep copy including rng state (needed by oracles).

        Implements PRD Section 6 (M2).
        """
        return copy.deepcopy(self)


# Implementation Note:
# Implemented exactly per PRD Section 6 (M2), Section 2 (S1, S2), and Section 5.2.
# sigma=0 reproduces trace cost and latency exactly (tolerance 1e-9).
