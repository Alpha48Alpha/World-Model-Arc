"""Evaluation utilities for World-Model-Arc agents."""

from __future__ import annotations

from typing import Any

import numpy as np

from world_model_arc.agents.base import BaseAgent
from world_model_arc.envs.base import BaseEnv


class Evaluator:
    """Runs deterministic rollouts and collects evaluation statistics.

    Args:
        env: Environment to evaluate in.
        agent: The agent to evaluate.
    """

    def __init__(self, env: BaseEnv, agent: BaseAgent) -> None:
        self.env = env
        self.agent = agent

    # ------------------------------------------------------------------

    def evaluate(self, n_episodes: int = 10, seed_offset: int = 10_000) -> dict[str, Any]:
        """Run *n_episodes* deterministic episodes and return aggregate stats.

        Args:
            n_episodes: Number of episodes to run.
            seed_offset: Starting seed for deterministic resets.

        Returns:
            Dictionary with keys:
            - ``mean_reward`` – mean total episode reward.
            - ``std_reward`` – std of total episode rewards.
            - ``min_reward``, ``max_reward``
            - ``mean_length`` – mean episode length.
            - ``success_rate`` – fraction of episodes reaching a goal.
        """
        rewards: list[float] = []
        lengths: list[int] = []
        successes: list[bool] = []

        for i in range(n_episodes):
            obs = self.env.reset(seed=seed_offset + i)
            total_reward = 0.0
            step = 0
            success = False

            while True:
                action = self.agent.select_action(obs, deterministic=True)
                result = self.env.step(action)
                total_reward += result.reward
                step += 1
                obs = result.observation

                if result.info.get("goal_reached"):
                    success = True

                if result.done or result.truncated:
                    break

            rewards.append(total_reward)
            lengths.append(step)
            successes.append(success)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean(lengths)),
            "success_rate": float(np.mean(successes)),
            "episode_rewards": rewards,
        }

    def run_episode(self, seed: int = 0) -> dict[str, Any]:
        """Run a single deterministic episode and return full trajectory.

        Useful for visualisation or manual inspection.

        Returns:
            Dictionary with ``reward``, ``length``, ``success``, and ``frames``
            (list of RGB arrays if the env supports ``render()``).
        """
        obs = self.env.reset(seed=seed)
        total_reward = 0.0
        step = 0
        success = False
        frames: list[np.ndarray] = []

        while True:
            frame = self.env.render()
            if frame is not None:
                frames.append(frame)

            action = self.agent.select_action(obs, deterministic=True)
            result = self.env.step(action)
            total_reward += result.reward
            step += 1
            obs = result.observation

            if result.info.get("goal_reached"):
                success = True

            if result.done or result.truncated:
                break

        return {
            "reward": total_reward,
            "length": step,
            "success": success,
            "frames": frames,
        }
