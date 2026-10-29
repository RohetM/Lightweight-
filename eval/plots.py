"""Plotting utilities for Restless Router evaluation.
Implements PRD section 6 (M8), Section 0 (Rule 5).
"""
import os
from typing import Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def generate_evaluation_plots(
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    output_dir: str = "data/runs/plots",
    is_synthetic: bool = True,
) -> list[str]:
    """Generate the 4 required evaluation figures per PRD Section 6 (M8).

    Figures:
    1. fig1_utility_vs_sigma.png
    2. fig2_delta_vs_routellm.png
    3. fig3_credit_ablation.png
    4. fig4_latency_and_ratelimits.png
    Titles carry '[SYNTHETIC]' if is_synthetic is True (PRD Section 0 Rule 5).
    """
    os.makedirs(output_dir, exist_ok=True)
    created_plots = []
    tag = " [SYNTHETIC]" if is_synthetic else ""

    # Style configuration
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # 1. Fig 1: Utility vs Sigma
    fig, ax = plt.subplots(figsize=(8, 5))
    key_methods = [
        "ClairvoyantOracle",
        "CurrentStateOracle",
        "Proposed_Shapley",
        "BestStaticPath",
        "RouteLLM",
        "Greedy",
        "Cheapest",
        "Strongest",
        "Random",
    ]
    colors = {
        "ClairvoyantOracle": "black",
        "CurrentStateOracle": "purple",
        "Proposed_Shapley": "tab:blue",
        "BestStaticPath": "teal",
        "RouteLLM": "tab:red",
        "Greedy": "tab:orange",
        "Cheapest": "tab:green",
        "Strongest": "tab:brown",
        "Random": "tab:gray",
    }
    markers = {
        "ClairvoyantOracle": "*",
        "CurrentStateOracle": "s",
        "Proposed_Shapley": "o",
        "BestStaticPath": "P",
        "RouteLLM": "^",
        "Greedy": "v",
        "Cheapest": "x",
        "Strongest": "d",
        "Random": ".",
    }

    for method in key_methods:
        sub = summary_df[summary_df["method"] == method].sort_values("sigma")
        if not sub.empty:
            ax.plot(
                sub["sigma"],
                sub["utility_mean"],
                label=method,
                color=colors.get(method, "blue"),
                marker=markers.get(method, "o"),
                linewidth=2 if "Proposed" in method or "RouteLLM" in method else 1.5,
            )

    ax.set_xlabel(r"Cloud Volatility $\sigma$")
    ax.set_ylabel(r"Mean Pipeline Utility $U$")
    ax.set_title(f"Figure 1: Pipeline Utility vs. Volatility $\\sigma${tag}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    fig1_path = os.path.join(output_dir, "fig1_utility_vs_sigma.png")
    fig.savefig(fig1_path, dpi=200)
    plt.close(fig)
    created_plots.append(fig1_path)

    # 2. Fig 2: Delta vs Baselines (RouteLLM & BestStaticPath) with 95% Bootstrap CI
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if not delta_df.empty:
        delta_sorted = delta_df.sort_values("sigma")
        sigmas = delta_sorted["sigma"].values

        # 1. Delta vs RouteLLM
        if "delta_mean" in delta_sorted:
            deltas_r = delta_sorted["delta_mean"].values
            ci_lo_r = delta_sorted["ci_lower"].values
            ci_hi_r = delta_sorted["ci_upper"].values
            yerr_r = [deltas_r - ci_lo_r, ci_hi_r - deltas_r]
            ax.errorbar(
                sigmas,
                deltas_r,
                yerr=yerr_r,
                fmt="o-",
                color="tab:blue",
                capsize=5,
                capthick=1.5,
                linewidth=2,
                label=r"$\Delta(\sigma) = U(Proposed) - U(RouteLLM)$",
            )

        # 2. Delta vs BestStaticPath (if available)
        if "delta_static_mean" in delta_sorted:
            deltas_s = delta_sorted["delta_static_mean"].values
            ci_lo_s = delta_sorted["ci_static_lower"].values
            ci_hi_s = delta_sorted["ci_static_upper"].values
            yerr_s = [deltas_s - ci_lo_s, ci_hi_s - deltas_s]
            ax.errorbar(
                sigmas,
                deltas_s,
                yerr=yerr_s,
                fmt="s--",
                color="teal",
                capsize=5,
                capthick=1.5,
                linewidth=2,
                label=r"$\Delta(\sigma) = U(Proposed) - U(BestStaticPath)$",
            )

        ax.axhline(0.0, color="tab:red", linestyle=":", linewidth=1.2, label="Parity (0.0)")

    ax.set_xlabel(r"Cloud Volatility $\sigma$")
    ax.set_ylabel(r"$\Delta$ Utility (Proposed vs. Baseline)")
    ax.set_title(f"Figure 2: Performance Gap vs Baselines (95% CI){tag}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    fig2_path = os.path.join(output_dir, "fig2_delta_vs_routellm.png")
    fig.savefig(fig2_path, dpi=200)
    plt.close(fig)
    created_plots.append(fig2_path)

    # 3. Fig 3: Credit Ablations (No Credit, LOO, Shapley, Static State)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ablation_methods = [
        "Proposed_Shapley",
        "Proposed_LOO",
        "Proposed_NoCredit",
        "Proposed_StaticState",
    ]
    ab_labels = {
        "Proposed_Shapley": "Proposed (Exact Shapley)",
        "Proposed_LOO": "Ablation: Leave-One-Out (LOO)",
        "Proposed_NoCredit": "Ablation: No Credit (Terminal / 3)",
        "Proposed_StaticState": "Ablation: Static State (No Restlessness)",
    }

    for method in ablation_methods:
        sub = summary_df[summary_df["method"] == method].sort_values("sigma")
        if not sub.empty:
            ax.plot(
                sub["sigma"],
                sub["utility_mean"],
                marker="o",
                label=ab_labels.get(method, method),
                linewidth=1.8,
            )

    ax.set_xlabel(r"Cloud Volatility $\sigma$")
    ax.set_ylabel(r"Mean Pipeline Utility $U$")
    ax.set_title(f"Figure 3: Credit Assignment & Restlessness Ablations{tag}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    fig3_path = os.path.join(output_dir, "fig3_credit_ablation.png")
    fig.savefig(fig3_path, dpi=200)
    plt.close(fig)
    created_plots.append(fig3_path)

    # 4. Fig 4: Latency & Rate Limits
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for method in ["Proposed_Shapley", "RouteLLM", "Greedy"]:
        sub = summary_df[summary_df["method"] == method].sort_values("sigma")
        if not sub.empty:
            ax1.plot(sub["sigma"], sub["latency_mean"], marker="o", label=f"{method} (Mean)")
            ax2.plot(sub["sigma"], sub["rate_limit_count"], marker="s", label=method)

    ax1.set_xlabel(r"Cloud Volatility $\sigma$")
    ax1.set_ylabel(r"Mean Latency (s)")
    ax1.set_title(f"Mean Pipeline Latency vs $\\sigma${tag}")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="best", fontsize=9)

    ax2.set_xlabel(r"Cloud Volatility $\sigma$")
    ax2.set_ylabel(r"Rate Limit Incidents (429s)")
    ax2.set_title(f"Rate-Limit Incidents vs $\\sigma${tag}")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle(f"Figure 4: Latency and Cloud Congestion Dynamics{tag}")
    fig4_path = os.path.join(output_dir, "fig4_latency_and_ratelimits.png")
    fig.savefig(fig4_path, dpi=200)
    plt.close(fig)
    created_plots.append(fig4_path)

    return created_plots


# Implementation Note:
# Implemented 4 required evaluation figures per PRD Section 6 (M8).
# Rule 5 strictly enforced: '[SYNTHETIC]' tag included whenever synthetic cache is used.
