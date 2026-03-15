"""Deterministic evaluation of trained agents."""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym

from src.agents.base_agent import BaseAgent


class Evaluator:
    """Runs a trained agent in deterministic mode and collects statistics.

    Parameters
    ----------
    env:
        Gymnasium-compatible environment.
    agent:
        A trained agent whose ``select_action(obs, deterministic=True)``
        is called at every step.
    """

    def __init__(self, env: gym.Env, agent: BaseAgent) -> None:
        self.env = env
        self.agent = agent

    def evaluate(
        self,
        n_episodes: int = 10,
        max_steps: int = 500,
        render: bool = False,
    ) -> dict[str, Any]:
        """Run *n_episodes* deterministic evaluation episodes.

        Returns
        -------
        dict with keys:
          ``mean_return``  – mean total episode return  
          ``std_return``   – std of episode returns  
          ``mean_length``  – mean episode length  
          ``min_return``   – minimum episode return  
          ``max_return``   – maximum episode return  
          ``episode_returns`` – raw per-episode returns  
        """
        returns: list[float] = []
        lengths: list[int] = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            ep_return = 0.0
            for step in range(max_steps):
                if render:
                    self.env.render()
                action = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                ep_return += float(reward)
                if terminated or truncated:
                    break
            returns.append(ep_return)
            lengths.append(step + 1)

        returns_arr = np.array(returns)
        return {
            "mean_return": float(returns_arr.mean()),
            "std_return": float(returns_arr.std()),
            "mean_length": float(np.mean(lengths)),
            "min_return": float(returns_arr.min()),
            "max_return": float(returns_arr.max()),
            "episode_returns": returns,
        }
