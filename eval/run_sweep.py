"""Experiment evaluation sweep across cloud volatility sigmas and seeds.
Implements PRD section 6 (M8), Section 2 (S5, S6, S7), Section 0 (Rule 5).
"""
import argparse
import copy
import json
import os
import sys
from typing import Any
import numpy as np
import pandas as pd
import yaml

from credit.credit_math import (
    compute_shapley,
    compute_loo,
    build_coalition_values,
)
from envs.cloud_env import CloudEnv
from envs.pipeline_env import PipelineEnv
from envs.trace_cache import TraceCache
from eval.plots import generate_evaluation_plots
from eval.stats import compute_paired_bootstrap_ci, compute_metrics_summary
from router.baselines import (
    RandomPolicy,
    CheapestPolicy,
    StrongestPolicy,
    GreedyPolicy,
    RouteLLMPolicy,
    CurrentStateOracle,
    ClairvoyantOracle,
    BestStaticPathPolicy,
    find_best_static_path_on_train,
)
from router.whittle_router import WhittleRouter


class StaticStateRouterWrapper:
    """Ablation d: Restlessness off. Masks dynamic cloud features to static control values."""

    def __init__(self, base_router: WhittleRouter):
        self.router = base_router

    def select(
        self,
        obs: np.ndarray,
        hop: int = 0,
        task_id: str = "",
        trace_cache: Any = None,
        cloud_env: Any = None,
    ) -> int:
        static_obs = obs.copy()
        # Set dynamic features (cols 0-3) to static control (B=1, Q=0, P_norm=0.2, idle=0)
        static_obs[:, 0] = 1.0
        static_obs[:, 1] = 0.0
        static_obs[:, 2] = 0.2
        static_obs[:, 3] = 0.0
        return self.router.select(static_obs, explore=False)


def train_router_variant(
    credit_method: str,
    train_tasks: list[str],
    trace_cache: TraceCache,
    config: dict[str, Any],
    train_runs: int = 1500,
    seed: int = 42,
    mask_dynamic: bool = False,
) -> WhittleRouter:
    """Train a WhittleRouter variant on training split tasks.

    Implements PRD Section 6 (M5 & M8).
    """
    router = WhittleRouter(
        seed=seed,
        lr_critic=config["training"]["lr_critic"],
        lr_index=config["training"]["lr_index"],
        gamma=config["training"]["gamma"],
        epsilon_start=config["training"]["epsilon_start"],
        epsilon_end=config["training"]["epsilon_end"],
        epsilon_decay_runs=min(train_runs, config["training"]["epsilon_decay_runs"]),
        replay_size=config["training"]["replay_size"],
        batch_size=config["training"]["batch_size"],
    )

    rng = np.random.default_rng(seed)
    sigmas = config["experiment"]["sigmas"]
    lambda_c = config["reward"]["lambda_c"]
    lambda_l = config["reward"]["lambda_l"]

    task_pool = train_tasks
    n_train_tasks = len(task_pool)

    # Stream train_runs pipeline runs
    for run_idx in range(train_runs):
        router.run_count += 1
        tid = task_pool[run_idx % n_train_tasks]
        sigma = float(rng.choice(sigmas))
        run_seed = int(rng.integers(0, 1000000))

        cloud = CloudEnv(config["models"], config["cloud"], sigma=sigma, seed=run_seed)
        pipeline = PipelineEnv(trace_cache, cloud, config["reward"], config["models"])

        # Execute 3 hops
        hops_data = []
        path = []
        for h in range(3):
            obs = pipeline.build_obs(h)
            if mask_dynamic:
                obs[:, 0] = 1.0
                obs[:, 1] = 0.0
                obs[:, 2] = 0.2
                obs[:, 3] = 0.0

            arm = router.select(obs, explore=True)
            path.append(f"m{arm}")

            call_rec = trace_cache.call(tid, h, tuple(path))
            tokens = call_rec["tokens_in"] + call_rec["tokens_out"]
            req_res = cloud.request(arm, tokens, call_rec["latency_s"])
            cloud.advance(req_res["realized_latency_s"])

            hops_data.append({
                "arm": arm,
                "obs_before": obs,
                "cost_usd": req_res["cost_usd"],
                "latency_s": req_res["realized_latency_s"],
            })

        success = float(trace_cache.outcome(tid, tuple(path)))

        # Assign credit
        if credit_method == "none":
            role_credits = [success / 3.0, success / 3.0, success / 3.0]
        elif credit_method == "loo":
            v = build_coalition_values(tid, path, trace_cache)
            role_credits = compute_loo(v, n=3)
        elif credit_method == "shapley":
            v = build_coalition_values(tid, path, trace_cache)
            role_credits = compute_shapley(v, n=3)
        else:
            raise ValueError(f"Unknown credit method: {credit_method}")

        # Push transitions to replay buffer
        for t in range(3):
            chosen_arm = hops_data[t]["arm"]
            cost_t = hops_data[t]["cost_usd"]
            lat_t = hops_data[t]["latency_s"]
            credit_t = role_credits[t]
            hop_reward = credit_t - (lambda_c * cost_t) - (lambda_l * lat_t)
            done = 1.0 if t == 2 else 0.0

            obs_t = hops_data[t]["obs_before"]
            next_obs_t = hops_data[t + 1]["obs_before"] if t < 2 else obs_t.copy()

            for arm_k in range(3):
                u_k = 1 if arm_k == chosen_arm else 0
                rew_k = hop_reward if u_k == 1 else 0.0
                router.replay.push(
                    obs_t[arm_k],
                    u_k,
                    rew_k,
                    next_obs_t[arm_k],
                    done,
                )

        retrain_every_n = config["training"].get("retrain_every_n", 100)
        retrain_steps = config["training"].get("retrain_steps", 200)

        if (run_idx + 1) % retrain_every_n == 0 and len(router.replay) >= router.batch_size:
            for _ in range(retrain_steps):
                router.train_step()
        elif len(router.replay) >= router.batch_size:
            router.train_step()

    return router


def run_experiment_sweep(
    config_path: str = "configs/default.yaml",
    splits_path: str = "configs/splits.yaml",
    trace_cache_path: str = "data/traces/synthetic.json",
    sigmas: list[float] | None = None,
    seeds: list[int] | None = None,
    train_runs: int = 1500,
    runs_log_path: str = "data/runs/runs.jsonl",
    summary_csv_path: str = "data/runs/sweep_summary.csv",
    plots_dir: str = "data/runs/plots",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete evaluation sweep across methods, sigmas, and seeds.

    Implements PRD Section 6 (M8) and Section 2 (S5, S6, S7).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = yaml.safe_load(f)

    trace_cache = TraceCache(trace_cache_path)
    is_synthetic = bool(trace_cache.metadata.get("synthetic", True))

    train_tasks = splits["train"]
    test_tasks = splits["test"]

    if sigmas is None:
        sigmas = config["experiment"]["sigmas"]
    if seeds is None:
        seeds = config["experiment"]["seeds"]

    print(f"\n=======================================================")
    print(f"Starting Sweep: {len(sigmas)} sigmas {sigmas}, {len(seeds)} seeds {seeds}")
    print(f"Data source: {trace_cache_path} (Synthetic: {is_synthetic})")
    print(f"Test tasks count: {len(test_tasks)}")
    print(f"=======================================================")

    # 1. Train proposed router variants on training split
    print("\n[Training Phase] Training router variants on train split...")
    router_shapley = train_router_variant("shapley", train_tasks, trace_cache, config, train_runs=train_runs, seed=42)
    router_loo = train_router_variant("loo", train_tasks, trace_cache, config, train_runs=train_runs, seed=43)
    router_nocredit = train_router_variant("none", train_tasks, trace_cache, config, train_runs=train_runs, seed=44)
    router_static = train_router_variant("shapley", train_tasks, trace_cache, config, train_runs=train_runs, seed=45, mask_dynamic=True)
    router_static_wrapped = StaticStateRouterWrapper(router_static)
    print("[Training Phase] Completed.")

    # Find BestStaticPath on TRAIN tasks for each sigma
    print("\n[BestStaticPath] Finding best static path on train tasks for each sigma...")
    best_static_paths: dict[float, tuple[int, int, int]] = {}
    for sigma in sigmas:
        b_p = find_best_static_path_on_train(
            train_tasks, trace_cache, config["reward"], config["models"], config["cloud"], sigma=sigma, seed=42
        )
        best_static_paths[sigma] = b_p
        print(f"  sigma={sigma}: Best static path on train={b_p}")

    # Prepare methods dictionary
    static_methods = {
        "Random": lambda seed, sig: RandomPolicy(seed=seed),
        "Cheapest": lambda seed, sig: CheapestPolicy(),
        "Strongest": lambda seed, sig: StrongestPolicy(),
        "Greedy": lambda seed, sig: GreedyPolicy(config["models"]),
        "RouteLLM": lambda seed, sig: RouteLLMPolicy(),
        "BestStaticPath": lambda seed, sig: BestStaticPathPolicy(path=best_static_paths[sig]),
        "CurrentStateOracle": lambda seed, sig: CurrentStateOracle(config["reward"], config["models"]),
        "ClairvoyantOracle": lambda seed, sig: ClairvoyantOracle(config["reward"], config["models"]),
        "Proposed_Shapley": lambda seed, sig: router_shapley,
        "Proposed_LOO": lambda seed, sig: router_loo,
        "Proposed_NoCredit": lambda seed, sig: router_nocredit,
        "Proposed_StaticState": lambda seed, sig: router_static_wrapped,
    }

    all_run_records = []
    os.makedirs(os.path.dirname(os.path.abspath(runs_log_path)), exist_ok=True)
    if os.path.exists(runs_log_path):
        os.remove(runs_log_path)

    # 2. Evaluation Phase
    # Common random numbers: for each (task_id, seed, sigma), every method sees the same initial state
    print("\n[Evaluation Phase] Running evaluation sweep...")
    for sigma in sigmas:
        for seed in seeds:
            print(f"  Evaluating sigma={sigma}, seed={seed}...")

            for method_name, policy_builder in static_methods.items():
                policy = policy_builder(seed, sigma)
                # One persistent task stream per (sigma, seed) run (PRD M8 & M2)
                cloud = CloudEnv(config["models"], config["cloud"], sigma=sigma, seed=seed)
                pipeline = PipelineEnv(
                    trace_cache, cloud, config["reward"], config["models"], log_file=runs_log_path
                )

                for tid in test_tasks:
                    rec = pipeline.run_task(
                        task_id=tid,
                        policy=policy,
                        method_name=method_name,
                        seed=seed,
                    )
                    all_run_records.append(rec)
                    # Idle gap between tasks in a stream (PRD M2)
                    cloud.advance(config["cloud"]["inter_task_gap_s"])

    print(f"[Evaluation Phase] Generated {len(all_run_records)} pipeline run records.")

    # 3. Compute Summary Statistics and Oracles Normalization
    summary_rows = []
    runs_df = pd.DataFrame(all_run_records)

    for sigma in sigmas:
        sigma_runs = runs_df[runs_df["sigma"] == sigma]
        rand_runs = sigma_runs[sigma_runs["method"] == "Random"]["utility"].values
        clair_runs = sigma_runs[sigma_runs["method"] == "ClairvoyantOracle"]["utility"].values
        rand_u = float(np.mean(rand_runs)) if len(rand_runs) > 0 else 0.0
        clair_u = float(np.mean(clair_runs)) if len(clair_runs) > 0 else 1.0

        for method_name in static_methods.keys():
            m_runs = sigma_runs[sigma_runs["method"] == method_name].to_dict("records")
            metrics = compute_metrics_summary(m_runs, random_u=rand_u, clairvoyant_u=clair_u)
            row = {
                "method": method_name,
                "sigma": sigma,
                **metrics,
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[Results] Saved summary table to {summary_csv_path}")

    # 4. Compute Paired Delta vs RouteLLM and BestStaticPath with 95% Bootstrap CI (PRD M8 & S6)
    delta_rows = []
    for sigma in sigmas:
        sub_sigma = runs_df[runs_df["sigma"] == sigma]
        # Match by (task_id, seed)
        prop_runs = sub_sigma[sub_sigma["method"] == "Proposed_Shapley"].sort_values(["task_id", "seed"])
        route_runs = sub_sigma[sub_sigma["method"] == "RouteLLM"].sort_values(["task_id", "seed"])
        static_runs = sub_sigma[sub_sigma["method"] == "BestStaticPath"].sort_values(["task_id", "seed"])

        u_prop = prop_runs["utility"].values
        u_route = route_runs["utility"].values
        u_static = static_runs["utility"].values

        mean_diff_r, ci_lower_r, ci_upper_r = compute_paired_bootstrap_ci(
            u_prop, u_route, n_resamples=10000, alpha=0.05, seed=42
        )
        mean_diff_s, ci_lower_s, ci_upper_s = compute_paired_bootstrap_ci(
            u_prop, u_static, n_resamples=10000, alpha=0.05, seed=42
        )
        delta_rows.append({
            "sigma": sigma,
            "delta_mean": mean_diff_r,
            "ci_lower": ci_lower_r,
            "ci_upper": ci_upper_r,
            "delta_static_mean": mean_diff_s,
            "ci_static_lower": ci_lower_s,
            "ci_static_upper": ci_upper_s,
        })

    delta_df = pd.DataFrame(delta_rows)

    # 5. Generate Figures (Rule 5: [SYNTHETIC] tagged)
    plot_files = generate_evaluation_plots(
        summary_df, delta_df, output_dir=plots_dir, is_synthetic=is_synthetic
    )
    print(f"[Results] Generated {len(plot_files)} plots in {plots_dir}")

    return summary_df, delta_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Evaluation Sweep")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--splits", type=str, default="configs/splits.yaml")
    parser.add_argument("--trace-cache", type=str, default="data/traces/synthetic.json")
    parser.add_argument("--quick", action="store_true", help="Run quick 2-sigma 3-seed sweep")
    parser.add_argument("--train-runs", type=int, default=1000)

    args = parser.parse_args()

    if args.quick:
        sigmas = [0.0, 0.5]
        seeds = [0, 1, 2]
    else:
        sigmas = None
        seeds = None

    run_experiment_sweep(
        config_path=args.config,
        splits_path=args.splits,
        trace_cache_path=args.trace_cache,
        sigmas=sigmas,
        seeds=seeds,
        train_runs=args.train_runs,
    )


if __name__ == "__main__":
    main()


# Implementation Note:
# Implemented evaluation sweep per PRD Section 6 (M8), Section 2 (S5, S6, S7).
# Common random numbers strictly enforced across all methods.
# Generated summary CSV and 4 plots with [SYNTHETIC] tags.
