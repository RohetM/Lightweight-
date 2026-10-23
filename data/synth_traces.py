"""Synthetic trace generator for Restless Router.
Implements PRD section 6 (M0) and Section 5.1.
"""
import json
import os
from typing import Any
import numpy as np


def generate(path: str, n_tasks: int = 100, seed: int = 0) -> None:
    """Generate synthetic trace cache and write to JSON.

    Implements PRD Section 6 (M0) and Section 5.1.
    Per task: difficulty ~ U[0.1, 0.9].
    Model mean quality m0: 0.45, m1: 0.72, m2: 0.92 (reduced by difficulty, plus noise);
    tokens 300/600/900; base latency 0.4/0.9/1.8 s.
    Synergy: if Hop 0 and Hop 1 both use m1 or m2, add +0.12 quality (so LOO and Shapley differ).
    routellm_score = clip(difficulty + N(0, 0.15), 0, 1) (invented proxy).
    Same seed produces byte-identical file.
    """
    rng = np.random.default_rng(seed)

    model_specs = {
        "m0": {"tokens": 300, "latency": 0.4, "quality": 0.45},
        "m1": {"tokens": 600, "latency": 0.9, "quality": 0.72},
        "m2": {"tokens": 900, "latency": 1.8, "quality": 0.92},
    }
    model_ids = ["m0", "m1", "m2"]

    tasks: dict[str, Any] = {}

    for i in range(n_tasks):
        task_id = f"task_{i:03d}"
        difficulty = float(rng.uniform(0.1, 0.9))

        calls: dict[str, dict[str, Any]] = {}
        outcomes: dict[str, float] = {}

        # Hop 0: 3 calls ("0_{m}")
        for m0 in model_ids:
            spec0 = model_specs[m0]
            r_score0 = float(np.clip(difficulty + rng.normal(0, 0.15), 0.0, 1.0))
            calls[f"0_{m0}"] = {
                "tokens_in": spec0["tokens"] // 2,
                "tokens_out": spec0["tokens"] - (spec0["tokens"] // 2),
                "latency_s": spec0["latency"],
                "routellm_score": round(r_score0, 4),
            }

        # Hop 1: 9 calls ("1_{m0}_{m1}")
        for m0 in model_ids:
            for m1 in model_ids:
                spec1 = model_specs[m1]
                r_score1 = float(np.clip(difficulty + rng.normal(0, 0.15), 0.0, 1.0))
                calls[f"1_{m0}_{m1}"] = {
                    "tokens_in": spec1["tokens"] // 2,
                    "tokens_out": spec1["tokens"] - (spec1["tokens"] // 2),
                    "latency_s": spec1["latency"],
                    "routellm_score": round(r_score1, 4),
                }

        # Hop 2: 27 calls ("2_{m0}_{m1}_{m2}") and 27 path outcomes ("{m0}_{m1}_{m2}")
        for m0 in model_ids:
            for m1 in model_ids:
                for m2 in model_ids:
                    spec2 = model_specs[m2]
                    r_score2 = float(np.clip(difficulty + rng.normal(0, 0.15), 0.0, 1.0))

                    # Compute path quality and success
                    mean_q = (model_specs[m0]["quality"] + model_specs[m1]["quality"] + model_specs[m2]["quality"]) / 3.0
                    noise = float(rng.normal(0.0, 0.02))
                    raw_q = mean_q - (0.25 * difficulty) + noise

                    # Synergy: if Hop 0 and Hop 1 both use m1 or m2, add +0.12 quality
                    if m0 in ("m1", "m2") and m1 in ("m1", "m2"):
                        raw_q += 0.12

                    success = float(np.clip(raw_q, 0.0, 1.0))
                    success = round(success, 4)

                    path_key = f"{m0}_{m1}_{m2}"
                    leaf_key = f"2_{path_key}"

                    calls[leaf_key] = {
                        "tokens_in": spec2["tokens"] // 2,
                        "tokens_out": spec2["tokens"] - (spec2["tokens"] // 2),
                        "latency_s": spec2["latency"],
                        "routellm_score": round(r_score2, 4),
                        "success": success,
                    }
                    outcomes[path_key] = success

        tasks[task_id] = {
            "difficulty": round(difficulty, 4),
            "calls": calls,
            "outcomes": outcomes,
        }

    data = {
        "metadata": {
            "model_names": ["cheap", "mid", "strong"],
            "model_versions": {
                "m0": "synthetic_cheap_v1",
                "m1": "synthetic_mid_v1",
                "m2": "synthetic_strong_v1",
            },
            "temperature": 0,
            "date": "2026-10-23",
            "synthetic": True,
            "git_hash": "HEAD",
            "n_tasks": n_tasks,
            "seed": seed,
        },
        "tasks": tasks,
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# Implementation Note:
# Implemented exactly as specified in PRD Section 6 (M0) and Section 5.1.
# All 39 calls and 27 outcomes generated deterministically per task.
