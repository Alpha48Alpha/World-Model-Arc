"""Experience replay buffer for off-policy algorithms (e.g. DQN).

Supports:
- Uniform random sampling
- Efficient circular buffer (pre-allocated numpy arrays)
- Serialisation / deserialisation for checkpointing
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch


class Batch(NamedTuple):
    """A mini-batch of transitions ready for training."""

    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    """Circular replay buffer storing (obs, action, reward, next_obs, done) tuples.

    Args:
        capacity: Maximum number of transitions to store.
        obs_shape: Shape of a single observation (excluding batch dimension).
        device: Torch device for sampled tensors.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        device: torch.device | str = "cpu",
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.device = torch.device(device)

        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

        self._ptr = 0
        self._size = 0

    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self._obs[self._ptr] = obs
        self._next_obs[self._ptr] = next_obs
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Sample a random mini-batch.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            A :class:`Batch` of torch tensors on ``self.device``.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has only {self._size} samples, but {batch_size} were requested."
            )
        idx = np.random.randint(0, self._size, size=batch_size)
        return Batch(
            obs=torch.as_tensor(self._obs[idx], device=self.device),
            action=torch.as_tensor(self._actions[idx], device=self.device),
            reward=torch.as_tensor(self._rewards[idx], device=self.device),
            next_obs=torch.as_tensor(self._next_obs[idx], device=self.device),
            done=torch.as_tensor(self._dones[idx], device=self.device),
        )

    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the buffer contents to a compressed numpy archive."""
        np.savez_compressed(
            str(path),
            obs=self._obs[: self._size],
            next_obs=self._next_obs[: self._size],
            actions=self._actions[: self._size],
            rewards=self._rewards[: self._size],
            dones=self._dones[: self._size],
            ptr=np.array(self._ptr),
            size=np.array(self._size),
        )

    def load(self, path: str | Path) -> None:
        """Load buffer contents from a numpy archive."""
        data = np.load(str(path))
        n = int(data["size"])
        self._obs[:n] = data["obs"]
        self._next_obs[:n] = data["next_obs"]
        self._actions[:n] = data["actions"]
        self._rewards[:n] = data["rewards"]
        self._dones[:n] = data["dones"]
        self._ptr = int(data["ptr"])
        self._size = n
