# Restless Router: Asynchronous, State-Aware Routing for Sequential Multi-Agent LLM Pipelines

**Team:** Arnav Ayush (KH.EN.P2MCA25019), Rohet M (KH.EN.P2MCA25045)  
**Guide:** Mahesh A S  
**Institution:** Amrita Vishwa Vidyapeetham  
**PRD Specification:** `docs/PRD.md` (PRD v4 Final)

---

## 1. Project Summary & Research Question

Sequential multi-agent LLM systems commonly structure task execution into multi-hop workflows (e.g., Hop 0 **Planner** → Hop 1 **Executor** → Hop 2 **Verifier**). At each hop, a router selects among candidate model tiers (e.g., `m0` cheap, `m1` mid, `m2` strong).

- **The Problem:** Existing prompt-only routers (such as RouteLLM) determine model selection purely based on query complexity while remaining blind to volatile cloud serving conditions (token bucket rate limits, queuing wait times, and spot price surges).
- **The Solution:** A small, ultra-fast Whittle-inspired state-aware router (decision time median ~0.016 ms, target < 0.1 ms) combined with an offline asynchronous credit assignment process (leave-one-out and exact 3-role Shapley value calculation) that feeds back credit into retraining.
- **Research Question:** As cloud volatility $\sigma$ rises from 0 to 1, does the state-aware router reach higher utility $U = \text{success} - \lambda_c \cdot \text{cost} - \lambda_l \cdot \text{latency}$ than RouteLLM?

### Honest Research Framing
The router is evaluated as a **Whittle-inspired heuristic**. It is not claimed to be formally indexable or optimal, and it is **not** a faithful reproduction of Xiong & Li (NeurIPS 2023). It is our own simplification tailored to sequential pipelines. With 3 sequential roles, credit assignment is computed via exact cooperative game-theoretic Shapley values ($2^3 = 8$ coalition sub-paths).

> [!WARNING]
> **Synthetic Data Disclaimer (Rule 5):** Results generated from `data/traces/synthetic.json` are for development, verification, and demonstration purposes only. Headline empirical findings are reserved for real trace caches collected under Gate 1 supervision.

---

## 2. Success Criteria & Verification Status

| ID | Criterion | How Checked | Status |
|---|---|---|---|
| **S1** | At $\sigma = 0$ simulator reproduces trace cost and latency exactly | Unit test `test_m2.py` (tolerance $10^{-9}$) | **PASS** |
| **S2** | Higher $\sigma$ does not help a fixed policy on average | Paired statistical test across 20 seeds (`test_m2.py`) | **PASS** |
| **S3** | Router decision time: median < 0.1 ms target (fail above 0.5 ms) | 10,000 decisions benchmarked: median ~0.016 ms, p99 ~0.051 ms (`test_m5.py`) | **PASS** |
| **S4** | Credit worker runs as a separate process while router serves | Process-level subprocess test (`test_m7.py`) | **PASS** |
| **S5** | 4 $\sigma$ levels, $\ge 5$ seeds, 95% CIs | `eval/run_sweep.py` outputs | **PASS** |
| **S6** | Paired significance test vs RouteLLM at each $\sigma$ | 10,000 resamples bootstrap CI in `eval/stats.py` | **PASS** |
| **S7** | Ablations: no-credit, LOO, Shapley, static-state | Implemented in `eval/run_sweep.py` & `plots.py` | **PASS** |
| **S8** | One command reruns main experiment from cached traces | `python -m eval.run_sweep` | **PASS** |
| **S9** | Headline results come from real trace cache | `data/collect_traces.py` ready for Gate 1 execution | **READY** |

---

## 3. Repository Architecture

```text
restless-router/
├── AGENTS.md, CLAUDE.md, requirements.txt, README.md
├── docs/
│   ├── PRD.md                  # PRD v4 (Final) specification
│   └── notes_papers.md         # Gate 3 literature review (Xiong & Li 2023, Raman et al. 2024)
├── configs/
│   ├── default.yaml            # Synthetic development & baseline parameters
│   ├── real_models.yaml        # Template for Gate 1 real provider pricing & rate limits
│   └── splits.yaml             # Fixed train (0.67) / test (0.33) splits
├── data/
│   ├── synth_traces.py         # M0: Deterministic synthetic trace generator
│   ├── collect_traces.py       # M1b: Real trace collection harness (API-enabled with --dry-run)
│   ├── traces/
│   │   ├── synthetic.json      # 100 tasks, 39 calls & 27 outcomes each
│   │   └── real.json           # Target for Gate 1 DBBench traces
│   └── runs/
│       ├── runs.jsonl          # Comprehensive pipeline run execution log
│       ├── sweep_summary.csv   # Aggregated metrics table across methods and sigmas
│       └── plots/              # fig1 to fig4 PNG figures with [SYNTHETIC] labeling
├── envs/
│   ├── trace_cache.py          # M1: Read-only trace access
│   ├── cloud_env.py            # M2: Continuous restless cloud simulator
│   ├── pipeline_env.py         # M3: 3-hop agent execution pipeline
│   └── toy_restless.py         # M5: 3-arm restless bandit with value iteration solver
├── router/
│   ├── baselines.py            # M4: 7 baseline policies and oracles
│   └── whittle_router.py       # M5: Fast IndexNet (NumPy forward) & CriticNet
├── credit/
│   ├── credit_math.py          # M6: LOO & exact 3-role Shapley value computation
│   └── async_worker.py         # M7: Background subprocess watching runs.jsonl & retraining
├── eval/
│   ├── run_sweep.py            # M8: Full experiment sweep with Common Random Numbers
│   ├── stats.py                # M8: Paired bootstrap CI (10,000 resamples)
│   └── plots.py                # M8: Publication-quality figures (fig1–fig4)
├── demo/
│   ├── app.py                  # M9: Streamlit live simulation & interactive dashboard
│   ├── record_demo.py          # M9: Headless Playwright / Matplotlib video recorder
│   └── working_model_demo.webm # Recorded demonstration video (>100 KB)
└── tests/
    └── test_m0.py ... test_m9.py # Full test suite (31 tests total)
```

---

## 4. Quick Start & Execution

### 1. Run Complete Test Suite
To verify all modules (M0 through M9):
```bash
python -m pytest -v
```

### 2. Run Main Evaluation Sweep (S8)
To rerun the experiment evaluation sweep from cached traces across all policies, $\sigma$ levels, and seeds:
```bash
python -m eval.run_sweep
```
For a quick verification sweep (2 $\sigma$ levels, 3 seeds):
```bash
python -m eval.run_sweep --quick
```
This generates:
- `data/runs/sweep_summary.csv`
- `data/runs/plots/fig1_utility_vs_sigma.png`
- `data/runs/plots/fig2_delta_vs_routellm.png`
- `data/runs/plots/fig3_credit_ablation.png`
- `data/runs/plots/fig4_latency_and_ratelimits.png`

### 3. Launch Interactive Demo Application (M9)
To run the Streamlit interactive dashboard:
```bash
streamlit run demo/app.py
```

### 4. Trace Collection Dry-Run (M1b)
To inspect the 39-call tree calculation and projected spend before live API calls:
```bash
python data/collect_traces.py --dry-run --n-tasks 5
```
Extrapolate to full 60 tasks (2,340 calls):
```bash
python data/collect_traces.py --dry-run --n-tasks 60
```
