"""Abstract base class for all RL agents in World-Model-Arc."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


class BaseAgent(abc.ABC):
    """Abstract base class that all agents must implement.

    Subclasses override:
    - :meth:`select_action` – choose an action given an observation.
    - :meth:`observe` – store a transition (used by off-policy agents).
    - :meth:`update` – perform one training update.
    - :meth:`state_dict` / :meth:`load_state_dict` – for checkpointing.
    """

    def __init__(self, obs_shape: tuple[int, ...], n_actions: int, device: str = "cpu") -> None:
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.device = torch.device(device)
        self._train_steps: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Return an action index for the given observation.

        Args:
            obs: Current environment observation.
            deterministic: If *True*, use the greedy/deterministic policy
                (used during evaluation; no exploration).

        Returns:
            Integer action index.
        """

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition.

        Off-policy agents (e.g. DQN) push to a replay buffer here.
        On-policy agents may choose to accumulate episodes instead.
        Default implementation is a no-op.
        """

    @abc.abstractmethod
    def update(self) -> dict[str, float]:
        """Perform one gradient update step.

        Returns:
            Dictionary of scalar training metrics (e.g. ``{"loss": 0.12}``).
        """

    def end_episode(self) -> dict[str, float]:
        """Called at the end of every episode.

        On-policy agents can trigger their update here.
        Default implementation returns an empty dict.
        """
        return {}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serializable agent state (network weights, optimizer, etc.)."""

    @abc.abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore agent state from a dictionary returned by :meth:`state_dict`."""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert a numpy observation to a batched tensor on *self.device*."""
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
