"""Base environment interface for World-Model-Arc."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class StepResult:
    """Container returned by :meth:`BaseEnv.step`.

    Attributes:
        observation: Next observation after taking the action.
        reward: Scalar reward signal.
        done: Whether the episode has ended.
        truncated: Whether the episode was cut short (e.g. time limit).
        info: Optional diagnostic dictionary.
    """

    observation: np.ndarray
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any]


class BaseEnv(abc.ABC):
    """Abstract base class for all environments.

    Subclasses must implement :meth:`reset`, :meth:`step`,
    :meth:`observation_space`, and :meth:`action_space`.

    This interface is intentionally kept minimal so that it is easy to
    wrap third-party simulators (e.g. Gymnasium, MuJoCo) or plug in
    custom world models.
    """

    # ------------------------------------------------------------------
    # Properties that every environment must expose
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of a single observation array."""

    @property
    @abc.abstractmethod
    def action_size(self) -> int:
        """Number of discrete actions (for continuous envs use the dim)."""

    @property
    def is_continuous(self) -> bool:
        """Whether the action space is continuous."""
        return False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment and return the initial observation.

        Args:
            seed: Optional RNG seed for reproducibility.

        Returns:
            Initial observation as a numpy array.
        """

    @abc.abstractmethod
    def step(self, action: Any) -> StepResult:
        """Apply *action* and advance the simulation by one step.

        Args:
            action: The action to take.  Type depends on the environment.

        Returns:
            A :class:`StepResult` containing the transition data.
        """

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns an RGB array (H, W, 3) if rendering is supported,
        otherwise returns *None*.
        """
        return None

    def close(self) -> None:
        """Clean up any resources held by the environment."""
