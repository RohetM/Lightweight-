"""Asynchronous credit assignment worker and background retraining process.
Implements PRD section 6 (M7), Section 5.4, and Section 2 (S4).
"""
import argparse
import json
import os
import sys
import time
from typing import Any
import numpy as np

from credit.credit_math import (
    compute_shapley,
    compute_loo,
    build_coalition_values,
)
from envs.trace_cache import TraceCache
from router.whittle_router import WhittleRouter


def run_worker(
    runs_file: str,
    credits_file: str,
    weights_file: str,
    trace_cache_path: str,
    credit_method: str = "shapley",
    retrain_every_n: int = 100,
    retrain_steps: int = 200,
    poll_interval_s: float = 0.05,
    stop_flag: str | None = None,
    lambda_c: float = 10.0,
    lambda_l: float = 0.05,
) -> None:
    """Watch runs.jsonl, compute offline credit, and retrain router every N runs.

    Implements PRD Section 6 (M7) and Section 5.4.
    retrain_steps: number of gradient steps taken from the replay buffer per trigger.
    """
    trace_cache = TraceCache(trace_cache_path)
    router = WhittleRouter(seed=42)

    os.makedirs(os.path.dirname(os.path.abspath(credits_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(weights_file)), exist_ok=True)

    processed_run_ids = set()
    run_counter = 0
    file_pos = 0

    while True:
        if stop_flag and os.path.exists(stop_flag):
            break

        if os.path.exists(runs_file):
            with open(runs_file, "r", encoding="utf-8") as f:
                f.seek(file_pos)
                lines = f.readlines()
                file_pos = f.tell()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                run_id = record["run_id"]
                if run_id in processed_run_ids:
                    continue

                processed_run_ids.add(run_id)
                task_id = record["task_id"]
                path = record["path"]
                hops = record["hops"]
                success = float(record["success"])

                # Compute credits according to configured method (PRD 5.4 & M6)
                if credit_method == "none":
                    # PRD M5 no-credit ablation: credit_t = terminal_success / 3
                    role_credits = [success / 3.0, success / 3.0, success / 3.0]
                elif credit_method == "loo":
                    v = build_coalition_values(task_id, path, trace_cache)
                    role_credits = compute_loo(v, n=3)
                elif credit_method == "shapley":
                    v = build_coalition_values(task_id, path, trace_cache)
                    role_credits = compute_shapley(v, n=3)
                else:
                    raise ValueError(f"Unknown credit method: {credit_method}")

                # Log to credits.jsonl (PRD 5.4)
                credit_record = {
                    "run_id": run_id,
                    "method": credit_method,
                    "role_credits": [round(c, 6) for c in role_credits],
                }
                with open(credits_file, "a", encoding="utf-8") as cf:
                    cf.write(json.dumps(credit_record) + "\n")

                # Build transitions for router replay buffer (PRD M5)
                # hop_reward = credit_t - lambda_c * cost_t - lambda_l * latency_t
                for t in range(3):
                    chosen_arm = hops[t]["arm"]
                    cost_t = hops[t]["cost_usd"]
                    lat_t = hops[t]["latency_s"]
                    credit_t = role_credits[t]
                    hop_reward = credit_t - (lambda_c * cost_t) - (lambda_l * lat_t)

                    obs_t = np.array(hops[t]["obs_before"], dtype=np.float32)
                    done = 1.0 if t == 2 else 0.0

                    if t < 2:
                        next_obs_t = np.array(hops[t + 1]["obs_before"], dtype=np.float32)
                    else:
                        next_obs_t = obs_t.copy()

                    for arm in range(3):
                        u_k = 1 if arm == chosen_arm else 0
                        rew_k = hop_reward if u_k == 1 else 0.0
                        router.replay.push(
                            obs_t[arm],
                            u_k,
                            rew_k,
                            next_obs_t[arm],
                            done,
                        )

                run_counter += 1

                # Every retrain_every_n runs, retrain router and save weights
                if run_counter % retrain_every_n == 0:
                    for _ in range(retrain_steps):
                        router.train_step()
                    router.save_weights(weights_file)

        time.sleep(poll_interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Async Credit Worker")
    parser.add_argument("--runs-file", type=str, required=True)
    parser.add_argument("--credits-file", type=str, required=True)
    parser.add_argument("--weights-file", type=str, required=True)
    parser.add_argument("--trace-cache", type=str, required=True)
    parser.add_argument("--credit-method", type=str, default="shapley", choices=["none", "loo", "shapley"])
    parser.add_argument("--retrain-every-n", type=int, default=100)
    parser.add_argument("--retrain-steps", type=int, default=200, help="Gradient steps per retrain trigger")
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--stop-flag", type=str, default=None)
    parser.add_argument("--lambda-c", type=float, default=10.0)
    parser.add_argument("--lambda-l", type=float, default=0.05)

    args = parser.parse_args()
    run_worker(
        runs_file=args.runs_file,
        credits_file=args.credits_file,
        weights_file=args.weights_file,
        trace_cache_path=args.trace_cache,
        credit_method=args.credit_method,
        retrain_every_n=args.retrain_every_n,
        retrain_steps=args.retrain_steps,
        poll_interval_s=args.poll_interval_s,
        stop_flag=args.stop_flag,
        lambda_c=args.lambda_c,
        lambda_l=args.lambda_l,
    )


if __name__ == "__main__":
    main()


# Implementation Note:
# Implemented async worker per PRD Section 6 (M7), Section 5.4, and Section 2 (S4).
# Operates as a decoupled background process via subprocess.Popen.
# retrain_steps (default 200) gradient steps are taken from the replay buffer per
# retrain trigger, up from the original 10; configurable via --retrain-steps CLI arg
# and configs/default.yaml training.retrain_steps.
# Periodically retrains router and serializes updated weights.pt.
