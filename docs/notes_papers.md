# Literature Notes: Restless Bandits, Whittle Indices, and Credit Allocation

**Team:** Arnav Ayush (KH.EN.P2MCA25019), Rohet M (KH.EN.P2MCA25045)  
**Guide:** Mahesh A S  
**Date:** October 2026  
**Document:** `docs/notes_papers.md`  

---

## 1. Context and Objective

In multi-agent LLM pipelines (e.g., Planner $\to$ Executor $\to$ Verifier), selecting which model tier to invoke at each sequential stage (hop) presents a dynamic decision problem under volatile cloud conditions (token bucket rate limits, queuing delays, and dynamic pricing).

This document analyzes the theoretical foundations from two key papers:
1. **Xiong & Li (NeurIPS 2023)**: *Learning Whittle Index Policy for Restless Multi-Armed Bandits*
2. **Raman, Shi, and Fang (NeurIPS 2024)**: *Shapley-Whittle Indices for Restless Multi-Armed Bandits*

and clarifies how our **Whittle-Inspired Router** differs from these formal theoretical constructs.

---

## 2. Analysis of Xiong & Li (NeurIPS 2023)

### Core Framework
Xiong & Li study continuous-state restless multi-armed bandits (RMABs) where the underlying transition dynamics of passive and active arms are unknown. Under the standard Lagrangian relaxation of the RMAB constraint:
- A subsidy for passivity $W(s)$ is introduced.
- The Whittle index $W(s)$ is formally defined as the minimum subsidy that makes the agent indifferent between activating the arm and leaving it passive.
- An RMAB is *indexable* if the passive set monotonically grows with $W$.

### Learning Algorithm
Xiong & Li propose a two-timescale reinforcement learning architecture:
1. **Critic Network ($Q_\psi(s, a)$)**: Estimates the state-action values under the current policy and subsidy parameterization.
2. **Actor/Index Network ($W_\theta(s)$)**: Parametrizes the index function directly, trained on a slower timescale so that $W_\theta(s) \approx Q(s, 1) - Q(s, 0)$.

### Differences in Our Restless Router Implementation
- **Honest Heuristic Framing**: We do **not** claim or prove indexability of our multi-agent cloud routing environment. The transition dynamics of cloud rate-limiting buckets and queuing are non-linear and coupled through sequential budget constraints.
- **Sequential Pipeline Setting vs. Simultaneous Arms**: Standard RMAB pulls $K$ out of $N$ arms simultaneously at every discrete time step $t$. In our sequential pipeline, exactly one model is chosen at hop 0, one at hop 1, and one at hop 2.
- **Asynchronous Feedback**: Rewards in LLM pipelines are non-Markovian across hops (final success is only realized at hop 2). We couple the Whittle-inspired heuristic with an offline asynchronous credit worker (LOO and Shapley).
- **Simplified Update Rules**: As specified in PRD Section 6 (M5), our fast critic updates toward $hop\_reward + \gamma \max Q(s')$ for active arms and $w_k + \gamma \max Q(s')$ for passive arms, and the slow index network regresses directly to $(Q_{active} - Q_{passive})$. This provides an efficient, lightweight heuristic without the extensive technical machinery required for formal convergence proofs on continuous Markov chains.

---

## 3. Analysis of Raman, Shi, and Fang (NeurIPS 2024)

### Core Framework
Raman et al. address cooperative multi-agent restless bandits where pulling a subset of arms generates joint non-additive rewards. They define **Shapley-Whittle indices** by integrating cooperative game theory (Shapley values) into Whittle index calculations for simultaneous arm pulls.

### Differences in Our Setting
- **Role in Pipeline vs. Simultaneous Coalitions**: Raman et al. evaluate simultaneous subset selections per time step. In contrast, our pipeline is sequential across 3 distinct hops (Planner, Executor, Verifier).
- **Credit Assignment Decomposition**: We do not compute hypergraph-restricted coalitions or simultaneous pull valuations. Instead, we use exact 3-player cooperative game theory ($2^3 = 8$ coalitions) over the sequential path choices. The coalition value $v(S)$ measures pipeline success when roles in $S$ use the chosen model while roles outside $S$ fall back to model 0 (cheapest).
- **Decoupled Architecture**: Raman et al. evaluate theoretical bounds for joint policies. Our system decouples live, sub-millisecond inference (`IndexNet` forward pass in NumPy $< 0.1$ ms) from background credit computation running in a separate worker process.

---

## 4. Summary Table of Comparisons

| Dimension | Xiong & Li (NeurIPS 2023) | Raman et al. (NeurIPS 2024) | Restless Router (Ours) |
|---|---|---|---|
| **Problem Formulation** | Continuous-state RMAB | Cooperative RMAB | Sequential 3-hop Agent Pipeline |
| **Pulls per Step** | Top $M$ simultaneous arms | Subset of joint arms | Exactly 1 model per hop (3 hops total) |
| **Whittle Indexability** | Proven under assumptions | Assumed / extended | **Whittle-inspired heuristic (no indexability claim)** |
| **Credit Assignment** | Single-arm immediate reward | Joint cooperative reward | Offline exact Shapley / LOO across pipeline roles |
| **Inference Latency** | Offline / simulation focus | Theoretical bounds | Strict runtime budget: median $< 0.1$ ms (NumPy forward pass) |
| **Cloud Volatility** | Synthetic Markov state | Static / Markov reward | Realistic token bucket, queue cap, Poisson arrivals, dynamic pricing |

---

## 5. Review Sign-off
This document serves as the conceptual reference for Gate 3 review with the guide prior to experimental evaluations.
