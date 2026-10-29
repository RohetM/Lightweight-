"""Simulator sensitivity sweep at sigma=1.0 varying strong model service rate and arrival multiplier.
Evaluates Delta(Proposed_Shapley - RouteLLM) and Delta(Proposed_Shapley - BestStaticPath) across 5 seeds.
[SYNTHETIC]
"""
import copy
import yaml
import numpy as np
import pandas as pd

from envs.trace_cache import TraceCache
from envs.cloud_env import CloudEnv
from envs.pipeline_env import PipelineEnv
from router.baselines import RouteLLMPolicy, BestStaticPathPolicy, find_best_static_path_on_train
from router.whittle_router import WhittleRouter
from eval.run_sweep import train_router_variant
from eval.stats import compute_paired_bootstrap_ci


def run_sensitivity_sweep(
    config_path: str = "configs/default.yaml",
    splits_path: str = "configs/splits.yaml",
    trace_cache_path: str = "data/traces/synthetic.json",
    service_rates: list[float] = [1.0, 1.5, 3.0],
    arrival_multipliers: list[float] = [1.0, 2.0, 3.0],
    seeds: list[int] = [0, 1, 2, 3, 4],
    sigma: float = 1.0,
    train_runs: int = 1500,
) -> pd.DataFrame:
    """Run 3x3 sensitivity grid at sigma=1.0 over 5 seeds."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = yaml.safe_load(f)

    trace_cache = TraceCache(trace_cache_path)
    train_tasks = splits["train"]
    test_tasks = splits["test"]

    print("\n[Sensitivity Sweep] Training Proposed_Shapley router on train tasks...")
    router_shapley = train_router_variant("shapley", train_tasks, trace_cache, config, train_runs=train_runs, seed=42)
    print("[Sensitivity Sweep] Training complete.")

    results = []

    print(f"\nEvaluating 3x3 grid (service_rates={service_rates}, arrival_multipliers={arrival_multipliers}) at sigma={sigma} across {len(seeds)} seeds...")
    for s_rate in service_rates:
        for arr_mult in arrival_multipliers:
            print(f"\n--- Cell: strong service_rate={s_rate}, arrival_multiplier={arr_mult} ---")
            mod_cfg = copy.deepcopy(config["models"])
            mod_cfg[2]["service_rate_per_sec"] = float(s_rate)

            c_cfg = copy.deepcopy(config["cloud"])
            c_cfg["arrival_multiplier"] = float(arr_mult)

            # Fit BestStaticPath on train tasks for this specific simulator environment
            b_path = find_best_static_path_on_train(
                train_task_ids=train_tasks,
                trace_cache=trace_cache,
                reward_config=config["reward"],
                models_config=mod_cfg,
                cloud_config=c_cfg,
                sigma=sigma,
                seed=42,
            )
            print(f"  Best static path on train: {b_path}")

            methods = {
                "Proposed_Shapley": lambda s: router_shapley,
                "RouteLLM": lambda s: RouteLLMPolicy(),
                "BestStaticPath": lambda s: BestStaticPathPolicy(path=b_path),
            }

            records = []
            for seed in seeds:
                for m_name, pol_builder in methods.items():
                    pol = pol_builder(seed)
                    cloud = CloudEnv(mod_cfg, c_cfg, sigma=sigma, seed=seed)
                    pipeline = PipelineEnv(trace_cache, cloud, config["reward"], mod_cfg)

                    for tid in test_tasks:
                        rec = pipeline.run_task(task_id=tid, policy=pol, method_name=m_name, seed=seed)
                        records.append(rec)
                        cloud.advance(c_cfg.get("inter_task_gap_s", 1.0))

            df = pd.DataFrame(records)
            prop = df[df["method"] == "Proposed_Shapley"].sort_values(["task_id", "seed"])["utility"].values
            route = df[df["method"] == "RouteLLM"].sort_values(["task_id", "seed"])["utility"].values
            static = df[df["method"] == "BestStaticPath"].sort_values(["task_id", "seed"])["utility"].values

            d_route, r_lo, r_hi = compute_paired_bootstrap_ci(prop, route, n_resamples=10000, seed=42)
            d_static, s_lo, s_hi = compute_paired_bootstrap_ci(prop, static, n_resamples=10000, seed=42)

            u_prop_mean = float(np.mean(prop))
            u_route_mean = float(np.mean(route))
            u_static_mean = float(np.mean(static))

            print(f"  U(Proposed_Shapley): {u_prop_mean:.4f}")
            print(f"  U(RouteLLM):         {u_route_mean:.4f}")
            print(f"  U(BestStaticPath):   {u_static_mean:.4f}")
            print(f"  Delta vs RouteLLM:        {d_route:+.4f} [95% CI {r_lo:+.4f}, {r_hi:+.4f}]")
            print(f"  Delta vs BestStaticPath:  {d_static:+.4f} [95% CI {s_lo:+.4f}, {s_hi:+.4f}]")

            results.append({
                "service_rate": s_rate,
                "arrival_multiplier": arr_mult,
                "best_static_path": str(b_path),
                "u_proposed": u_prop_mean,
                "u_routellm": u_route_mean,
                "u_beststatic": u_static_mean,
                "delta_routellm": d_route,
                "delta_routellm_ci_lo": r_lo,
                "delta_routellm_ci_hi": r_hi,
                "delta_beststatic": d_static,
                "delta_beststatic_ci_lo": s_lo,
                "delta_beststatic_ci_hi": s_hi,
            })

    res_df = pd.DataFrame(results)
    out_csv = "data/runs/sensitivity_summary.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved sensitivity summary table to {out_csv}")
    return res_df


if __name__ == "__main__":
    run_sensitivity_sweep()
