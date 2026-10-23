# PRD v4 (Final): Restless Router

## Asynchronous, State-Aware Routing for Sequential Multi-Agent LLM Pipelines

**Team:** Arnav Ayush (KH.EN.P2MCA25019), Rohet M (KH.EN.P2MCA25045) | **Guide:** Mahesh A S
**Start:** Mon 26 Oct 2026 | **Duration:** 12 weeks | **Save as:** `docs/PRD.md`

**What this version is:** the exact-spec style of PRD v3 (fixed numbers, shapes, arithmetic) merged with the research safeguards of v2 (real-data path, human gates, honesty rules, risks). It tells the coding agent exactly what to build, and tells you honestly what remains uncertain. No document can make an agent literally hallucination-free, so every module has tests that catch deviations.

---

## 0. Rules for the AI coding agent (paste into `AGENTS.md` and `CLAUDE.md`)

```text
1. SPEC FIRST: Every equation, data schema, array shape and update rule is defined in docs/PRD.md.
   Do not invent formulas, add layers, rename keys, or add dependencies. If something is ambiguous
   or missing, STOP and ask. Do not guess.
2. TEST FIRST: One module per session. Write the pytest file first, then the module, until green.
3. NO LIVE API CALLS except in data/collect_traces.py. Everything else reads data/traces/.
4. EXPLICIT SEEDING: np.random.default_rng(seed) or torch.Generator().manual_seed(seed). No global state.
5. SYNTHETIC IS NOT EVIDENCE: results from data/traces/synthetic.json are for development and demos
   only and must be labelled "SYNTHETIC" in every output, plot title and dashboard banner.
6. Every function docstring names the PRD section it implements.
7. Each module ends with a short note listing anything it could not implement exactly as specified.
```

---

## 1. Project summary and research question

A 3-step pipeline (Hop 0 Planner → Hop 1 Executor → Hop 2 Verifier) chooses one of 3 candidate models (0 cheap, 1 mid, 2 strong) at each hop.

- **Problem:** prompt-only routers such as RouteLLM ignore live cloud conditions (rate-limit buckets, queue delays, price changes).
- **Solution:** a small Whittle-inspired, state-aware router (decision well under 1 ms) plus an asynchronous offline credit process (leave-one-out and exact 3-role Shapley) that scores which role caused success or failure and feeds that back into retraining.
- **Research question:** as cloud volatility σ rises from 0 to 1, does the state-aware router reach higher utility `U = success − λ_c·cost − λ_l·latency` than RouteLLM?
- **Honest framing (use in the report):** the router is a **Whittle-inspired heuristic**. It is not claimed to be indexable or optimal, and it is **not** a faithful reproduction of Xiong & Li (NeurIPS 2023). It is our own simplification inspired by it. Shapley-Whittle indices already exist (Raman, Shi, Fang, NeurIPS 2024) for simultaneous pulls. Our contribution is the sequential, asynchronous-credit, systems setting and its evaluation. With 3 roles, credit is "exact Shapley", not "hypergraph".

**Out of scope:** formal Whittle indexability proof; hypergraph-restricted Shapley; other benchmarks; live-cloud evaluation; production serving.

---

## 2. Success criteria

| ID | Criterion | How checked |
|---|---|---|
| S1 | At σ = 0 the simulator reproduces trace cost and latency exactly | Unit test, tolerance 1e-9 |
| S2 | Higher σ does not help a fixed policy on average | Paired one-sided test across σ over 20 seeds; never assert per-seed monotonicity |
| S3 | Router decision time: median < 0.1 ms target (test fails only above 0.5 ms) | 10,000 decisions; report median, p99, CPU model, library versions |
| S4 | Credit worker runs as a separate process while the router keeps serving | Process-level test |
| S5 | 4 σ levels, ≥ 5 seeds (aim 10), 95% CIs | `eval/` outputs |
| S6 | Paired significance test vs RouteLLM at each σ | `eval/stats.py` |
| S7 | Ablations: no-credit, LOO, Shapley, static-state | `eval/` outputs |
| S8 | One command reruns the main experiment from cached traces | README |
| S9 | Headline results come from the **real** trace cache | Checked at Gate 5 |

Not required: beating RouteLLM at every σ. At σ = 0 a tie is expected and is a good sanity check. The claim being tested is that the gap in your favour grows with σ. If it doesn't, report that and analyse why.

---

## 3. Repository layout

```
restless-router/
├── AGENTS.md, CLAUDE.md, requirements.txt
├── docs/ PRD.md, notes_papers.md
├── configs/ default.yaml (synthetic), real_models.yaml (HUMAN fills), splits.yaml
├── data/
│   ├── synth_traces.py     # M0
│   ├── collect_traces.py   # M1b: ONLY file allowed to call APIs
│   ├── traces/             # synthetic.json, real.json
│   └── runs/               # runs.jsonl, credits.jsonl, weights.pt, plots/, sweep_summary.csv
├── envs/ trace_cache.py (M1), cloud_env.py (M2), pipeline_env.py (M3), toy_restless.py (M5 test)
├── router/ baselines.py (M4), whittle_router.py (M5)
├── credit/ credit_math.py (M6), async_worker.py (M7)
├── eval/ run_sweep.py, stats.py, plots.py (M8)
├── demo/ app.py, record_demo.py (M9, optional, last)
└── tests/ test_m0.py ... test_m9.py
```

---

## 4. Fixed configuration (`configs/default.yaml`, synthetic development values)

```yaml
models:
  - {id: "m0", name: "cheap",  tier: 0.0, price_per_1k_tokens: 0.0001, bucket_capacity: 10000.0, refill_rate_per_sec: 2000.0, service_rate_per_sec: 5.0}
  - {id: "m1", name: "mid",    tier: 0.5, price_per_1k_tokens: 0.0008, bucket_capacity: 6000.0,  refill_rate_per_sec: 1000.0, service_rate_per_sec: 3.0}
  - {id: "m2", name: "strong", tier: 1.0, price_per_1k_tokens: 0.0050, bucket_capacity: 3000.0,  refill_rate_per_sec: 400.0,  service_rate_per_sec: 1.5}
cloud:
  queue_cap: 20.0          # Q_k never exceeds this (prevents unbounded waits)
  inter_task_gap_s: 1.0    # idle time advanced between consecutive tasks in a stream
  price_min: 0.5
  price_max: 3.0
reward: {lambda_c: 10.0, lambda_l: 0.05}
training:
  lr_critic: 0.005
  lr_index: 0.0005
  gamma: 0.95
  retrain_every_n: 100
  epsilon_start: 0.30
  epsilon_end: 0.05
  epsilon_decay_runs: 1500   # linear
  replay_size: 5000
  batch_size: 64
  train_runs: 3000           # pipeline runs streamed from train tasks
experiment: {sigmas: [0.0, 0.25, 0.5, 1.0], seeds: [0,1,2,3,4], train_ratio: 0.67}
```

**Real runs:** `configs/real_models.yaml` replaces `models:` with the 3 real models, their actual prices (cite provider docs) and rate limits. **HUMAN decision.** λ_c and λ_l must be re-tuned on training tasks only.

---

## 5. Exact data contracts

### 5.1 Trace cache (`data/traces/*.json`)
Single JSON with `"metadata"` (model names/versions, temperature 0, date, synthetic flag, git hash) and `"tasks"`.
- **39 call records per task**, keyed `"{hop}_{prefix}"`: hop 0 `"0_m0"`, hop 1 `"1_m0_m2"`, hop 2 `"2_m0_m2_m1"`. Fields: `tokens_in:int, tokens_out:int, latency_s:float, routellm_score:float∈[0,1]` (real cache also stores `output_text`).
- **27 path outcomes** keyed `"m0_m1_m2"` → success ∈ [0,1], **derived from the hop-2 leaf records** (the 39 call records are the only source of truth). Test: all 27 paths resolve with consistent prefixes.
- 100 synthetic tasks (60 for real); split by task id: train 0.67 / test 0.33, fixed in `configs/splits.yaml`.

### 5.2 Feature vector: `obs` shape `(3, 9)` float32 (row k = arm k at hop t)
```
[bucket_frac,        # B_k / cap_k
 queue_norm,         # min(1, Q_k / 10)
 price_norm,         # (P_k - 0.5) / 2.5
 idle_norm,          # min(1, idle_sec_k / 10)
 is_hop_0, is_hop_1, is_hop_2,
 price_static,       # price_per_1k / 0.0050
 tier_static]        # 0.0, 0.5, 1.0
```
No prompt text, so the comparison with prompt-only RouteLLM is clean.

### 5.3 Run log (`runs.jsonl`, one line per pipeline run)
```
run_id, task_id, method, sigma, seed, path:[a0,a1,a2],
hops:[{arm, obs_before:(3x9), tokens, cost_usd, latency_s, backoff_s, queue_wait_s, rate_limited, decision_time_us}],
success, total_cost, total_latency, utility
```
### 5.4 Credit log (`credits.jsonl`): `run_id, method:"loo"|"shapley", role_credits:[c0,c1,c2]`

---

## 6. Module specifications

### M0: Synthetic generator (`data/synth_traces.py`)
`generate(path, n_tasks=100, seed=0)`. Per task: difficulty ~ U[0.1, 0.9]. Model mean quality m0 0.45, m1 0.72, m2 0.92 (reduced by difficulty, plus noise); tokens 300/600/900; base latency 0.4/0.9/1.8 s. **Synergy:** if Hop 0 and Hop 1 both use m1 or m2, add +0.12 quality (so LOO and Shapley differ). `routellm_score = clip(difficulty + N(0, 0.15), 0, 1)` (documented as an invented proxy). Same seed → byte-identical file.
**Tests:** 39 calls and 27 outcomes per task; determinism; stronger models cost more tokens and time; LOO ≠ Shapley on some tasks.

### M1: TraceCache (`envs/trace_cache.py`, read-only)
```python
class TraceCache:
    def __init__(self, json_path: str): ...
    def call(self, task_id: str, hop: int, prefix: tuple[str, ...]) -> dict
    def outcome(self, task_id: str, path: tuple[str, str, str]) -> float
    def task_ids(self) -> list[str]
```
**Tests:** loads deterministically; missing key raises a clear error; all 27 paths resolve.

### M1b: Real trace collection (`data/collect_traces.py`, HUMAN-supervised)
- Task source: **AgentBench DBBench** (SQL), with our own fixed Planner/Executor/Verifier prompt template. Executor SQL is executed against the database (Docker MySQL) **at collection time only** and the result cached. If AgentBench setup blocks more than ~3 days, fall back to a code benchmark with unit tests (decide at Gate 1; state it in the report).
- Records the full 39-call tree per task at temperature 0. Resumable (skips saved keys), rate-limit aware, logs spend, `--dry-run` prints call count without calling anything.
- Computes `routellm_score` per hop prompt using a RouteLLM router that runs locally if possible (check which routers need a hosted-embedding API key). Mapping: score ≥ 0.5 → strong.
- **Run 5 tasks first, report cost, extrapolate to 60 tasks (~2,340 calls) before the full run.** Expect days, not an afternoon.

### M2: Restless cloud (`envs/cloud_env.py`)
State per arm: bucket B_k (starts at cap_k), queue Q_k (starts 0), price multiplier P_k (starts 1.0), idle_k (starts 0). **State persists across the whole task stream** for a given (σ, seed) run; it resets only at the start of each run. Between tasks call `advance(inter_task_gap_s)`.

**σ = 0 (static control):** B_k = cap_k, Q_k = 0, P_k = 1 always. `cost = tokens/1000 · price_per_1k`; `latency = base_latency_s`; `rate_limited = False`.

**σ > 0, passive advance** `advance(dt)` for ALL 3 arms:
```python
eff_refill = refill_rate / (1.0 + sigma)
B = min(cap, B + eff_refill * dt)
arrivals   = rng.poisson(2.0 * sigma * dt)
departures = rng.poisson(service_rate * dt)
Q = min(queue_cap, max(0, Q + arrivals - departures))
P = clip(P + 0.2 * (1.0 - P) * dt + rng.normal(0, 0.15 * sigma), price_min, price_max)
idle += dt
```
**Active request** `request(arm, tokens, base_latency_s)`:
```python
if B_arm >= tokens: backoff_s = 0.0; B_arm -= tokens; rate_limited = False
else: backoff_s = (tokens - B_arm) / (refill_rate / (1.0 + sigma)); B_arm = 0.0; rate_limited = True
queue_wait_s = Q_arm / service_rate
Q_arm = min(queue_cap, Q_arm + 1); idle_arm = 0.0
realized_latency_s = base_latency_s + backoff_s + queue_wait_s
cost_usd = tokens / 1000.0 * price_per_1k * P_arm
```
`observe() -> (3, 4)` of the first four feature columns; `clone()` returns a deep copy including the rng state (needed by the oracles).
**Tests:** S1 exactness; same seed → identical trajectories; unused arms change after `advance`; Q never exceeds the cap; S2 as a statistical test.

### M3: Pipeline (`envs/pipeline_env.py`)
For a task: for t = 0, 1, 2: build `obs`, time the policy call (`decision_time_us`), look up `TraceCache.call(task, t, prefix)`, `tokens = tokens_in + tokens_out`, `CloudEnv.request(...)`, then `CloudEnv.advance(dt=realized_latency_s)`. After hop 2: `success = TraceCache.outcome(task, path)`; `utility = success − λ_c·total_cost − λ_l·total_latency`. Writes the run log (5.3).
**Tests:** a fixed path returns the cached outcome; total latency equals the sum of hop latencies; log schema validated.

### M4: Baselines and oracles (`router/baselines.py`)
Interface: `select(obs, hop, task_id, trace_cache, cloud_env) -> int`.
1. `RandomPolicy` (seeded). 2. `CheapestPolicy` → 0. 3. `StrongestPolicy` → 2.
4. `GreedyPolicy`: arm minimizing `current_effective_price · (1 + queue_wait_s)`.
5. `RouteLLMPolicy`: cached `routellm_score ≥ 0.5` → 2 else 0; ignores CloudEnv.
6. `CurrentStateOracle` (myopic): on a `clone()` of CloudEnv, simulate each arm's immediate cost/latency (without future noise) and add the average cached downstream success of that arm; pick highest expected utility.
7. `ClairvoyantOracle` (upper bound): evaluates all 27 paths against the exact seeded CloudEnv rollout for that task and returns the max realized utility. Report utility, cost, success, latency.
Normalized performance: `(method − random) / (clairvoyant − random)` on utility.
**Tests (statistical, on average):** Clairvoyant ≥ CurrentState ≥ Random; Cheapest has lowest cost at σ = 0.

### M5: Whittle-inspired router (`router/whittle_router.py`)
**HUMAN review first:** before starting, one of you reads Xiong & Li (2023) and Raman et al. (2024) and writes 1 page in `docs/notes_papers.md` on how this heuristic differs from them. The guide reviews it at Gate 3. The code below is the **specified** algorithm; do not change it.

- `IndexNet f_θ`: `Linear(9,32) → ReLU → Linear(32,32) → ReLU → Linear(32,1)`. After each update, copy weights to NumPy; `select(obs)` runs a pure NumPy forward pass and returns `int(argmax(scores))` (or a random arm with probability ε during training only).
- `CriticNet Q_ψ`: `Linear(9,32) → ReLU → Linear(32,2)` → `[Q_passive, Q_active]`.
- **Exploration:** ε linear from `epsilon_start` to `epsilon_end` over `epsilon_decay_runs`; ε = 0 at evaluation.
- **Training:** stream `train_runs` pipeline runs from train tasks (random σ drawn from the σ list per run) into a replay buffer of per-arm transitions `(obs_k, u_k, reward, next_obs_k, done)`; `u_k = 1` if arm k was chosen at that hop. `next_obs_k` = row k of the next hop's `obs`; `done = 1` after hop 2. Update with batches of 64.
- **Fast critic step (lr 0.005):** chosen arm: `target = hop_reward + γ·(1−done)·max Q(next_obs_k)`. Passive arm: `target = w_k + γ·(1−done)·max Q(next_obs_k)` with `w_k = IndexNet(obs_k).detach()`. Minimize MSE of `Q(obs_k)[u_k]` to target.
- **Slow index step (lr 0.0005):** `target_index = (Q(obs_k)[1] − Q(obs_k)[0]).detach()`; minimize MSE of `IndexNet(obs_k)` to it.
- `hop_reward = credit_t − λ_c·cost_t − λ_l·latency_t`. No-credit ablation: `credit_t = terminal_success / 3`. This is the single main-method reward. Any blend is a separately labelled ablation.

**Tests:**
1. **Toy restless bandit** (`envs/toy_restless.py`): 3 arms, each state ∈ {good, bad}; pulling a good arm pays 1.0 then it turns bad with p = 0.5; pulling a bad arm pays 0.2; a passive bad arm recovers to good with p = 0.3. One pull per step, observed states. Compute the exact optimum by value iteration on the 8-state joint MDP. The trained learner must beat random and reach ≥ 85% of the optimal average reward. **If this fails, stop and debug before touching LLM routing.**
2. Seeded training is reproducible. 3. S3 timing. 4. Greedy-argmax index selection is deterministic.
**Fallback if it won't converge:** discretize state and use tabular Q first, then return to the network.

### M6: Credit math (`credit/credit_math.py`)
Coalition value `v(S)`: roles in S use the router's chosen arm; roles not in S use model 0 (cheapest); value = cache outcome of that path.
```python
import itertools, math
def compute_shapley(v: dict[frozenset[int], float], n: int = 3) -> list[float]:
    phi = [0.0] * n
    for perm in itertools.permutations(range(n)):
        S = frozenset()
        for i in perm:
            phi[i] += v[S | {i}] - v[S]
            S |= {i}
    return [p / math.factorial(n) for p in phi]
def compute_loo(v, n: int = 3) -> list[float]:
    full = frozenset(range(n))
    return [v[full] - v[full - {i}] for i in range(n)]
```
**Tests:** Shapley efficiency `sum(phi) == v(N) − v(∅)`; null role gets 0 under both; symmetric roles get equal Shapley credit; LOO ≠ Shapley on synergy tasks. Credit uses success only; cost and latency enter the reward separately from realized values.

### M7: Async worker (`credit/async_worker.py`)
Started with `subprocess.Popen`. Watches `runs.jsonl`, appends `credits.jsonl`, and every `retrain_every_n` = 100 runs retrains the router from the latest credits and saves `weights.pt`; the serving router reloads weights between runs. Credit method is configurable (`none|loo|shapley`).
**Tests:** S4; router keeps serving while the worker runs; with credit disabled the system still runs; retraining changes weights. Optional: p99 decision time with the worker busy vs idle.

### M8: Evaluation (`eval/`)
- **Common random numbers:** for each (task_id, seed, σ), every method sees the same initial cloud state and rng stream. One persistent task stream per (σ, seed) run.
- Test split only; σ ∈ {0, 0.25, 0.5, 1.0}; 5 seeds (aim 10).
- Methods: Random, Cheapest, Strongest, Greedy, RouteLLM, CurrentStateOracle, ClairvoyantOracle, Proposed (no credit), Proposed+LOO, Proposed+Shapley, Proposed static-state (ablation d: restlessness off).
- Metrics: utility, cost, success, mean and p95 latency, decision time (median, p99), 429 count, normalized performance.
- `Δ(σ) = mean_U(Proposed_Shapley) − mean_U(RouteLLM)` with 95% paired bootstrap CI (10,000 resamples, paired on (task, seed)). Report multiple-comparison caveat.
- Figures (titles carry "SYNTHETIC" if applicable) in `data/runs/plots/`: `fig1_utility_vs_sigma.png`, `fig2_delta_vs_routellm.png`, `fig3_credit_ablation.png`, `fig4_latency_and_ratelimits.png`. Also `data/runs/sweep_summary.csv`.
- Credit sanity: do credits correlate with the success drop when a role is replaced by the cheapest model?
**Tests:** same seed → identical numbers; metrics match a hand-computed example; bootstrap reproduces a known paired result.

### M9: Demo (OPTIONAL, do last, after real results exist)
`demo/app.py` (Streamlit): KPI banner (σ, mean utility proposed vs RouteLLM, median decision time), a button "▶ Run Live Multi-Agent Routing Simulation" animating 20 tasks with bars for bucket, queue, price and each router's choice, and tabs with fig1–fig4 and the summary table. **A banner must show "SYNTHETIC DATA" whenever the loaded cache is synthetic.** `demo/record_demo.py`: starts Streamlit headless on port 8501, records a 1280×720 Playwright video of ~15 s (click run, scroll charts), saves `demo/working_model_demo.webm` (and `.mp4` if ffmpeg exists); falls back to a Matplotlib animation if Chromium is unavailable. Do not run `playwright install` (Chromium is preinstalled in this environment; use `executablePath: /opt/pw-browsers/chromium` if needed). Test: video exists and is > 100 KB. This module must never delay M1b to M8.

---

## 7. Build order and human gates

| Weeks | Build | Human gate |
|---|---|---|
| 1 to 2 | Repo, AGENTS.md, config, M0, M1, M1b spec | **Gate 1:** pick 3 real models and prices; confirm AgentBench DBBench or fallback; 5-task dry run with cost estimate; start real collection (runs in background, days) |
| 3 to 4 | M2, M3 (on synthetic) | **Gate 2:** S1, S2 pass |
| 4 to 6 | M4, then M5 | **Gate 3:** notes_papers.md reviewed by guide; toy test ≥ 85% of optimum |
| 6 to 8 | M6, M7 | **Gate 4:** closed loop runs; Shapley tests pass; start NOC paperwork |
| 8 to 10 | M8 on **real** cache | **Gate 5:** human sanity-checks results (S9) before writing |
| 10 to 12 | Write-up, paper, optional M9 | Guide sign-off |

**Cut order if short on time:** M9, extra σ levels, greedy baseline, static-state ablation. **Never cut:** RouteLLM comparison, decision-time measurement, real-trace results.

---

## 8. Risks and decisions

| Risk | Plan |
|---|---|
| Router does not converge | Toy test first; tabular fallback; simplify state |
| Does not beat RouteLLM at high σ | Report honestly; confirm the simulator really charges backoff and queue waits; analyse why |
| Only 27 paths: small problem | Keep oracles and normalized performance; add models or hops only if time allows |
| Real collection cost, rate limits, AgentBench setup | 5-task dry run; D7 fallback decided at Gate 1 |
| One cached sample per path | Temperature 0, pinned model versions; state as limitation |
| Synthetic generator favours the router | Never headline; generator not keyed to router features |
| Credit lags live state | Report retrain interval as a sensitivity |
| Heuristic is not the published algorithm | Name it a Whittle-inspired heuristic; keep notes_papers.md |

**Humans must decide:** the 3 real models and prices; AgentBench environment vs fallback; λ_c, λ_l for real runs; RouteLLM router variant; retrain interval N.

**Definition of done:** S1 to S9 met; limitations section covers the points above; one command reproduces figures from cached traces; both of you can explain every module.

---

## 9. Agent prompts (run in order, one session each)

1. **M0 + M1:** "Read AGENTS.md and docs/PRD.md. Create configs/default.yaml, data/synth_traces.py and envs/trace_cache.py. Write tests/test_m0.py and tests/test_m1.py first (39 calls, 27 outcomes per task, determinism, path consistency). Run pytest."
2. **M2 + M3:** "Implement envs/cloud_env.py and envs/pipeline_env.py using the exact arithmetic in PRD section 6. Tests first: σ=0 exactness (1e-9), seed determinism, passive arms evolve, queue cap, log schema. Run pytest."
3. **M4:** "Implement router/baselines.py with all 7 policies. Tests: Clairvoyant ≥ CurrentState ≥ Random on average utility; Cheapest has lowest cost at σ=0."
4. **M5:** "Only after docs/notes_papers.md exists. Implement envs/toy_restless.py and router/whittle_router.py exactly as specified, including ε-greedy and replay. Write the toy-bandit test first (≥85% of value-iteration optimum, beats random). If it fails, stop and report; do not proceed."
5. **M6 + M7:** "Implement credit/credit_math.py and credit/async_worker.py. Tests: Shapley efficiency, null/symmetric roles, LOO≠Shapley on synergy tasks, worker runs as a separate subprocess while the router serves."
6. **M8:** "Implement eval/. Run a small synthetic sweep (2 σ, 3 seeds) to prove the pipeline, label everything SYNTHETIC, then stop and wait for the real cache."
7. **M1b (human-supervised):** "Implement data/collect_traces.py per section 6 with --dry-run, resume, and cost logging. Do not run it without confirmation."
8. **Real evaluation:** "Point the config at data/traces/real.json and real_models.yaml, run the full M8 protocol (σ list, 5 to 10 seeds), generate fig1 to fig4 and sweep_summary.csv."
9. **M9 (optional):** "Implement demo/app.py and demo/record_demo.py per section 6, show the SYNTHETIC banner when applicable, and record the video."
