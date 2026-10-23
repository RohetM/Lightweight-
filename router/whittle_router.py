"""Whittle-inspired router for multi-agent LLM pipelines.
Implements PRD section 6 (M5) and Section 4.
"""
import copy
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.trace_cache import TraceCache
from envs.cloud_env import CloudEnv


class IndexNet(nn.Module):
    """Index network f_theta computing Whittle-inspired arm scores.

    Implements PRD Section 6 (M5):
    Linear(9,32) -> ReLU -> Linear(32,32) -> ReLU -> Linear(32,1).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None
        self.W3: Optional[np.ndarray] = None
        self.b3: Optional[np.ndarray] = None
        self.sync_numpy_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def sync_numpy_weights(self) -> None:
        """Copy weights to NumPy for ultra-fast, pure NumPy forward passes."""
        self.W1 = self.net[0].weight.detach().cpu().numpy().copy()
        self.b1 = self.net[0].bias.detach().cpu().numpy().copy()
        self.W2 = self.net[2].weight.detach().cpu().numpy().copy()
        self.b2 = self.net[2].bias.detach().cpu().numpy().copy()
        self.W3 = self.net[4].weight.detach().cpu().numpy().copy()
        self.b3 = self.net[4].bias.detach().cpu().numpy().copy()

    def forward_numpy(self, obs: np.ndarray) -> np.ndarray:
        """Pure NumPy forward pass for sub-0.1ms inference.

        Implements PRD Section 6 (M5) and Section 2 (S3).
        obs shape: (3, 9) -> returns (3,)
        """
        h1 = np.maximum(0.0, obs @ self.W1.T + self.b1)
        h2 = np.maximum(0.0, h1 @ self.W2.T + self.b2)
        out = h2 @ self.W3.T + self.b3
        return out.squeeze(-1)


class CriticNet(nn.Module):
    """Critic network Q_psi estimating Q_passive and Q_active.

    Implements PRD Section 6 (M5):
    Linear(9,32) -> ReLU -> Linear(32,2) -> [Q_passive, Q_active].
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-capacity ring buffer of per-arm transitions."""

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer: list[tuple[np.ndarray, int, float, np.ndarray, float]] = []
        self.position = 0

    def push(
        self,
        obs_k: np.ndarray,
        u_k: int,
        reward: float,
        next_obs_k: np.ndarray,
        done: float,
    ) -> None:
        item = (obs_k, u_k, reward, next_obs_k, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        indices = rng.integers(0, len(self.buffer), size=batch_size)
        obs_b, u_b, r_b, next_obs_b, done_b = zip(*[self.buffer[i] for i in indices])
        return {
            "obs": np.array(obs_b, dtype=np.float32),
            "u": np.array(u_b, dtype=np.int64),
            "reward": np.array(r_b, dtype=np.float32),
            "next_obs": np.array(next_obs_b, dtype=np.float32),
            "done": np.array(done_b, dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self.buffer)


class WhittleRouter:
    """Whittle-inspired state-aware router with decoupled IndexNet and CriticNet.

    Implements PRD Section 6 (M5) and Section 4.
    """

    def __init__(
        self,
        seed: int = 0,
        lr_critic: float = 0.005,
        lr_index: float = 0.0005,
        gamma: float = 0.95,
        epsilon_start: float = 0.30,
        epsilon_end: float = 0.05,
        epsilon_decay_runs: int = 1500,
        replay_size: int = 5000,
        batch_size: int = 64,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.lr_critic = lr_critic
        self.lr_index = lr_index
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_runs = epsilon_decay_runs
        self.batch_size = batch_size

        self.run_count = 0

        self.index_net = IndexNet()
        self.critic_net = CriticNet()

        self.optimizer_critic = torch.optim.Adam(self.critic_net.parameters(), lr=lr_critic)
        self.optimizer_index = torch.optim.Adam(self.index_net.parameters(), lr=lr_index)

        self.replay = ReplayBuffer(capacity=replay_size)

    @property
    def epsilon(self) -> float:
        """Current exploration rate epsilon decayed linearly."""
        frac = min(1.0, self.run_count / float(self.epsilon_decay_runs))
        return self.epsilon_start - frac * (self.epsilon_start - self.epsilon_end)

    def select(
        self,
        obs: np.ndarray,
        hop: int = 0,
        task_id: str = "",
        trace_cache: Optional[TraceCache] = None,
        cloud_env: Optional[CloudEnv] = None,
        explore: bool = False,
    ) -> int:
        """Select arm using pure NumPy forward pass on obs (3, 9).

        Implements PRD Section 6 (M5) and Section 2 (S3).
        """
        if explore and (self.rng.random() < self.epsilon):
            return int(self.rng.integers(0, 3))

        scores = self.index_net.forward_numpy(obs)
        return int(np.argmax(scores))

    def push_transitions(
        self,
        transitions: list[tuple[np.ndarray, int, float, np.ndarray, float]],
    ) -> None:
        """Push a batch of per-arm transitions to replay buffer."""
        for item in transitions:
            self.replay.push(*item)

    def train_step(self) -> dict[str, float]:
        """Perform one fast critic step and one slow index step.

        Implements PRD Section 6 (M5).
        """
        if len(self.replay) < self.batch_size:
            return {"loss_critic": 0.0, "loss_index": 0.0}

        batch = self.replay.sample(self.batch_size, self.rng)
        obs_t = torch.from_numpy(batch["obs"])
        u_t = torch.from_numpy(batch["u"])
        reward_t = torch.from_numpy(batch["reward"])
        next_obs_t = torch.from_numpy(batch["next_obs"])
        done_t = torch.from_numpy(batch["done"])

        # 1. Fast critic step
        # Chosen arm: target = hop_reward + gamma * (1 - done) * max Q(next_obs_k)
        # Passive arm: target = w_k + gamma * (1 - done) * max Q(next_obs_k), w_k = IndexNet(obs_k).detach()
        with torch.no_grad():
            q_next = self.critic_net(next_obs_t)  # (B, 2)
            max_q_next = torch.max(q_next, dim=-1).values  # (B,)
            w_k = self.index_net(obs_t).squeeze(-1)  # (B,)

            target_chosen = reward_t + self.gamma * (1.0 - done_t) * max_q_next
            target_passive = w_k + self.gamma * (1.0 - done_t) * max_q_next
            target = torch.where(u_t == 1, target_chosen, target_passive)

        q_pred = self.critic_net(obs_t)  # (B, 2)
        q_chosen = q_pred.gather(1, u_t.unsqueeze(1)).squeeze(1)  # (B,)

        loss_critic = F.mse_loss(q_chosen, target)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # 2. Slow index step
        # target_index = (Q(obs_k)[1] - Q(obs_k)[0]).detach()
        with torch.no_grad():
            q_vals = self.critic_net(obs_t)
            target_index = q_vals[:, 1] - q_vals[:, 0]

        index_pred = self.index_net(obs_t).squeeze(-1)
        loss_index = F.mse_loss(index_pred, target_index)

        self.optimizer_index.zero_grad()
        loss_index.backward()
        self.optimizer_index.step()

        # Copy weights to NumPy after each update
        self.index_net.sync_numpy_weights()

        return {
            "loss_critic": float(loss_critic.item()),
            "loss_index": float(loss_index.item()),
        }

    def train_on_toy_env(self, n_episodes: int = 1500, seed: int = 42) -> float:
        """Train router on ToyRestlessEnv and return evaluation average reward.

        Implements PRD Section 6 (M5 - Test 1).
        """
        from envs.toy_restless import ToyRestlessEnv

        env = ToyRestlessEnv(seed=seed)
        env.reset()

        for ep in range(n_episodes):
            self.run_count += 1
            obs_before = env.get_obs()

            # Exploration during training
            if self.rng.random() < self.epsilon:
                action = int(self.rng.integers(0, 3))
            else:
                scores = self.index_net.forward_numpy(obs_before)
                action = int(np.argmax(scores))

            _, reward, _, _ = env.step(action)
            obs_after = env.get_obs()

            # Create per-arm transitions
            for arm in range(3):
                u_k = 1 if arm == action else 0
                rew_k = reward if u_k == 1 else 0.0
                self.replay.push(
                    obs_before[arm],
                    u_k,
                    rew_k,
                    obs_after[arm],
                    0.0,  # continuous MDP in toy bandit
                )

            if len(self.replay) >= self.batch_size:
                self.train_step()

        # Evaluate trained policy greedily (epsilon = 0)
        eval_env = ToyRestlessEnv(seed=seed + 100)
        eval_env.reset()
        total_eval_reward = 0.0
        n_eval_steps = 3000

        for _ in range(n_eval_steps):
            obs = eval_env.get_obs()
            scores = self.index_net.forward_numpy(obs)
            act = int(np.argmax(scores))
            _, r, _, _ = eval_env.step(act)
            total_eval_reward += r

        return total_eval_reward / float(n_eval_steps)

    def save_weights(self, path: str) -> None:
        """Save network weights to disk."""
        torch.save(
            {
                "index_state": self.index_net.state_dict(),
                "critic_state": self.critic_net.state_dict(),
                "run_count": self.run_count,
            },
            path,
        )

    def load_weights(self, path: str) -> None:
        """Load network weights from disk."""
        data = torch.load(path, map_location="cpu")
        self.index_net.load_state_dict(data["index_state"])
        self.critic_net.load_state_dict(data["critic_state"])
        self.run_count = data.get("run_count", 0)
        self.index_net.sync_numpy_weights()


# Implementation Note:
# Implemented exactly per PRD Section 6 (M5).
# IndexNet: Linear(9,32)->ReLU->Linear(32,32)->ReLU->Linear(32,1).
# CriticNet: Linear(9,32)->ReLU->Linear(32,2)->[Q_passive, Q_active].
# Two-timescale update rules strictly adhere to fast critic / slow index equations.
