"""Experience replay buffers and rollout storage for RL agents.

Provides:
  - ReplayBuffer   : uniform experience replay for off-policy agents (DQN).
  - RolloutStorage : on-policy trajectory buffer for policy-gradient agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Transition tuple used by ReplayBuffer
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs: np.ndarray
    action: int | np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool


# ---------------------------------------------------------------------------
# Replay Buffer (DQN / off-policy)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular uniform experience replay buffer.

    Parameters
    ----------
    capacity:
        Maximum number of transitions to store.
    obs_dim:
        Dimensionality of observations.
    action_dim:
        1 for discrete actions; >1 for continuous action vectors.
    device:
        Torch device to use when sampling batches.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int | np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition."""
        self._obs[self._ptr] = obs
        if self.action_dim == 1:
            self._actions[self._ptr, 0] = float(action)
        else:
            self._actions[self._ptr] = np.asarray(action, dtype=np.float32)
        self._rewards[self._ptr] = float(reward)
        self._next_obs[self._ptr] = next_obs
        self._dones[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a random batch of transitions.

        Returns a dict with keys:
          obs, actions, rewards, next_obs, dones
        """
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": torch.from_numpy(self._obs[idx]).to(self.device),
            "actions": torch.from_numpy(self._actions[idx]).to(self.device),
            "rewards": torch.from_numpy(self._rewards[idx]).to(self.device),
            "next_obs": torch.from_numpy(self._next_obs[idx]).to(self.device),
            "dones": torch.from_numpy(self._dones[idx]).to(self.device),
        }

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        """True when at least one batch worth of data is available."""
        return self._size > 0


# ---------------------------------------------------------------------------
# Rollout Storage (PPO / REINFORCE / on-policy)
# ---------------------------------------------------------------------------

@dataclass
class RolloutStorage:
    """Fixed-length on-policy rollout buffer.

    Stores complete trajectories for on-policy algorithms such as
    REINFORCE and PPO.  Call ``reset()`` before each new rollout.
    """

    obs_dim: int
    max_steps: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # Per-step storage (pre-allocated)
    _obs: np.ndarray = field(init=False)
    _actions: np.ndarray = field(init=False)
    _rewards: np.ndarray = field(init=False)
    _log_probs: np.ndarray = field(init=False)
    _values: np.ndarray = field(init=False)
    _dones: np.ndarray = field(init=False)
    _ptr: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._allocate()

    def _allocate(self) -> None:
        n = self.max_steps
        self._obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros(n, dtype=np.int64)
        self._rewards = np.zeros(n, dtype=np.float32)
        self._log_probs = np.zeros(n, dtype=np.float32)
        self._values = np.zeros(n, dtype=np.float32)
        self._dones = np.zeros(n, dtype=np.float32)
        self._ptr = 0

    def reset(self) -> None:
        """Clear the buffer without re-allocating memory."""
        self._ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        """Append a single timestep."""
        if self._ptr >= self.max_steps:
            raise RuntimeError("RolloutStorage is full; call reset() first.")
        i = self._ptr
        self._obs[i] = obs
        self._actions[i] = int(action)
        self._rewards[i] = float(reward)
        self._log_probs[i] = float(log_prob)
        self._values[i] = float(value)
        self._dones[i] = float(done)
        self._ptr += 1

    def compute_returns(
        self, gamma: float = 0.99, last_value: float = 0.0
    ) -> np.ndarray:
        """Compute discounted Monte-Carlo returns (reversed cumsum)."""
        n = self._ptr
        returns = np.zeros(n, dtype=np.float32)
        running = last_value
        for t in reversed(range(n)):
            running = self._rewards[t] + gamma * running * (1.0 - self._dones[t])
            returns[t] = running
        return returns

    def compute_gae(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        last_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and bootstrapped returns.

        Returns
        -------
        advantages : np.ndarray, shape (n,)
        returns    : np.ndarray, shape (n,)
        """
        n = self._ptr
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        next_val = last_value
        for t in reversed(range(n)):
            mask = 1.0 - self._dones[t]
            delta = self._rewards[t] + gamma * next_val * mask - self._values[t]
            last_gae = delta + gamma * lam * mask * last_gae
            advantages[t] = last_gae
            next_val = self._values[t]
        returns = advantages + self._values[:n]
        return advantages, returns

    def as_tensors(self) -> dict[str, torch.Tensor]:
        """Convert stored data to tensors on ``self.device``."""
        n = self._ptr
        return {
            "obs": torch.from_numpy(self._obs[:n]).to(self.device),
            "actions": torch.from_numpy(self._actions[:n]).to(self.device),
            "rewards": torch.from_numpy(self._rewards[:n]).to(self.device),
            "log_probs": torch.from_numpy(self._log_probs[:n]).to(self.device),
            "values": torch.from_numpy(self._values[:n]).to(self.device),
            "dones": torch.from_numpy(self._dones[:n]).to(self.device),
        }

    def __len__(self) -> int:
        return self._ptr
