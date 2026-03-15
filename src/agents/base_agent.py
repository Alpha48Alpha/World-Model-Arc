"""Abstract base class for all RL agents."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import numpy as np
import torch


class BaseAgent(abc.ABC):
    """Abstract interface that every RL agent must implement.

    Sub-classes must implement:
      - ``select_action``  : choose an action given an observation.
      - ``update``         : perform a learning step with collected data.
      - ``state_dict``     : return serialisable agent state.
      - ``load_state_dict``: restore agent state.
    """

    def __init__(self, obs_dim: int, action_dim: int, device: str = "cpu") -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self._total_steps: int = 0

    @abc.abstractmethod
    def select_action(
        self, obs: np.ndarray, *, deterministic: bool = False
    ) -> Any:
        """Select an action for the given observation.

        Parameters
        ----------
        obs:
            Current environment observation as a numpy array.
        deterministic:
            If ``True``, use the greedy/mode action (for evaluation).
        """

    @abc.abstractmethod
    def update(self, **kwargs: Any) -> dict[str, float]:
        """Perform one learning update.

        Returns a dict of scalar metrics (e.g. ``{"loss": 0.23}``).
        """

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the agent's state."""

    @abc.abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the agent from a previously saved state dict."""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert a numpy observation to a float tensor on ``self.device``."""
        return torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(
            self.device
        )

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def increment_steps(self, n: int = 1) -> None:
        self._total_steps += n
