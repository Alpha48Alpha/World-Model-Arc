"""Environment wrappers that augment or transform observations/rewards."""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NormalizeObservation(gym.ObservationWrapper):
    """Running-mean / running-std normalization of observations.

    Maintains a Welford online estimator and normalizes incoming
    observations to approximately zero mean / unit variance.  This is
    particularly useful for continuous-state environments where raw
    feature magnitudes differ by orders of magnitude.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8) -> None:
        super().__init__(env)
        self.epsilon = epsilon
        obs_shape = self.observation_space.shape
        self._mean = np.zeros(obs_shape, dtype=np.float64)
        self._var = np.ones(obs_shape, dtype=np.float64)
        self._count: float = 0.0

        # Keep the same dtype as the wrapped env
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=obs_shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._update_stats(obs)
        normalized = (obs - self._mean) / np.sqrt(self._var + self.epsilon)
        return normalized.astype(np.float32)

    def _update_stats(self, obs: np.ndarray) -> None:
        """Welford online update."""
        self._count += 1.0
        delta = obs - self._mean
        self._mean += delta / self._count
        delta2 = obs - self._mean
        self._var += delta * delta2


class FrameStack(gym.ObservationWrapper):
    """Stack the last *n* observations along the first axis.

    The resulting observation has shape ``(n * obs_dim,)`` for 1-D
    observations (or ``(n, H, W)`` for 2-D image observations).
    Useful for providing temporal context to agents that operate on
    Markov-style flattened inputs.
    """

    def __init__(self, env: gym.Env, n_frames: int = 4) -> None:
        super().__init__(env)
        self.n_frames = n_frames
        self._frames: deque = deque(maxlen=n_frames)

        orig_shape = self.observation_space.shape
        new_shape = (orig_shape[0] * n_frames,)
        self.observation_space = spaces.Box(
            low=np.tile(self.observation_space.low, n_frames),
            high=np.tile(self.observation_space.high, n_frames),
            shape=new_shape,
            dtype=self.observation_space.dtype,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.n_frames):
            self._frames.append(obs)
        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._frames.append(obs)
        return np.concatenate(list(self._frames), axis=0)
