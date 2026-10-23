"""Toy restless bandit environment and value-iteration solver.
Implements PRD section 6 (M5 - Test 1).
"""
import itertools
from typing import Callable, Any
import numpy as np


class ToyRestlessEnv:
    """3-arm Restless Bandit environment.

    Implements PRD Section 6 (M5 - Test 1).
    State per arm: 1 (good) or 0 (bad).
    Active pull:
      - good pays 1.0, turns bad with p = 0.5 (stays good with 0.5).
      - bad pays 0.2, stays bad with p = 1.0.
    Passive arm:
      - bad recovers to good with p = 0.3 (stays bad with 0.7).
      - good stays good with p = 1.0.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_arms = 3
        # Initial states: all arms start good
        self.state = np.ones(self.n_arms, dtype=np.int32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = np.ones(self.n_arms, dtype=np.int32)
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        arm = action
        arm_state = self.state[arm]

        # Reward
        reward = 1.0 if arm_state == 1 else 0.2

        # State transitions
        next_state = self.state.copy()

        # Active arm transition
        if arm_state == 1:
            # Turns bad with p = 0.5
            next_state[arm] = 0 if self.rng.random() < 0.5 else 1
        else:
            # Stays bad
            next_state[arm] = 0

        # Passive arms transition
        for j in range(self.n_arms):
            if j != arm:
                if self.state[j] == 0:
                    # Recovers with p = 0.3
                    next_state[j] = 1 if self.rng.random() < 0.3 else 0
                else:
                    next_state[j] = 1

        self.state = next_state
        return self.state.copy(), reward, False, {}

    def get_obs(self) -> np.ndarray:
        """Construct (3, 9) observation representation for IndexNet compatibility.

        Arm k row: [s_k, 1 - s_k, is_arm_0, is_arm_1, is_arm_2, 0, 0, 0, 0]
        """
        obs = np.zeros((3, 9), dtype=np.float32)
        for k in range(3):
            s_k = float(self.state[k])
            obs[k, 0] = s_k
            obs[k, 1] = 1.0 - s_k
            obs[k, 2 + k] = 1.0
        return obs


def solve_toy_restless_vi(gamma: float = 0.95, tol: float = 1e-8) -> tuple[np.ndarray, dict[tuple[int, ...], int]]:
    """Solve the exact optimum for the 8-state joint MDP via Value Iteration.

    Implements PRD Section 6 (M5 - Test 1).
    """
    states = list(itertools.product([0, 1], repeat=3))  # 8 states
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    # Transition probability matrix P[s_idx, action, s_next_idx] and R[s_idx, action]
    P = np.zeros((n_states, 3, n_states), dtype=np.float64)
    R = np.zeros((n_states, 3), dtype=np.float64)

    for s_idx, s in enumerate(states):
        for a in range(3):
            # Reward
            R[s_idx, a] = 1.0 if s[a] == 1 else 0.2

            # Compute transition probabilities for each arm
            arm_probs = []
            for j in range(3):
                p_next = {}
                if j == a:
                    # Active arm
                    if s[j] == 1:
                        p_next[0] = 0.5
                        p_next[1] = 0.5
                    else:
                        p_next[0] = 1.0
                        p_next[1] = 0.0
                else:
                    # Passive arm
                    if s[j] == 0:
                        p_next[1] = 0.3
                        p_next[0] = 0.7
                    else:
                        p_next[1] = 1.0
                        p_next[0] = 0.0
                arm_probs.append(p_next)

            # Joint transition probability
            for s_next in states:
                prob = arm_probs[0][s_next[0]] * arm_probs[1][s_next[1]] * arm_probs[2][s_next[2]]
                s_next_idx = state_to_idx[s_next]
                P[s_idx, a, s_next_idx] = prob

    # Value Iteration
    V = np.zeros(n_states, dtype=np.float64)
    while True:
        Q = R + gamma * np.sum(P * V[np.newaxis, np.newaxis, :], axis=2)
        V_new = np.max(Q, axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new

    # Optimal policy
    Q_final = R + gamma * np.sum(P * V[np.newaxis, np.newaxis, :], axis=2)
    opt_actions = np.argmax(Q_final, axis=1)
    opt_policy = {states[i]: int(opt_actions[i]) for i in range(n_states)}

    return V, opt_policy


def evaluate_policy(
    policy_fn: Any,
    n_steps: int = 3000,
    seed: int = 42,
) -> float:
    """Evaluate average reward of a policy on ToyRestlessEnv."""
    env = ToyRestlessEnv(seed=seed)
    total_reward = 0.0

    for _ in range(n_steps):
        state = tuple(env.state.tolist())
        if isinstance(policy_fn, dict):
            action = policy_fn[state]
        elif callable(policy_fn):
            try:
                action = policy_fn(state)
            except TypeError:
                action = policy_fn(env.state)
        else:
            raise TypeError("policy_fn must be a callable or dict mapping states to actions")

        _, rew, _, _ = env.step(action)
        total_reward += rew

    return total_reward / float(n_steps)


# Implementation Note:
# Implemented exactly per PRD Section 6 (M5 - Test 1).
# Solves exact 8-state joint MDP via Value Iteration with gamma=0.95.
