"""Main training loop for all supported agent types.

The ``Trainer`` class wraps the interaction between an environment,
an agent, a metrics logger, and a checkpointer into a single
``train()`` call that supports all three agent types:

  - DQNAgent       (off-policy, step-based updates)
  - REINFORCEAgent (on-policy, episode-based updates)
  - PPOAgent       (on-policy, rollout-based updates)

The caller passes a ``config`` dict that controls hyperparameters.
All training artefacts (checkpoints, metrics CSV, plots) are written
under ``config["run_dir"]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym

from src.agents.base_agent import BaseAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.reinforce_agent import REINFORCEAgent
from src.agents.ppo_agent import PPOAgent
from src.utils.logging import MetricsLogger
from src.utils.checkpointing import Checkpointer
from src.utils.visualization import plot_training_curves


class Trainer:
    """Orchestrates environment–agent interaction and logging.

    Parameters
    ----------
    env:
        A Gymnasium-compatible environment.
    agent:
        An RL agent (DQN, REINFORCE, or PPO).
    config:
        Training configuration dict.  Expected keys:

        ``n_episodes``          – total training episodes  
        ``max_steps_per_episode`` – per-episode step budget  
        ``run_dir``             – where to write artefacts  
        ``save_every``          – checkpoint frequency (episodes)  
        ``eval_every``          – evaluation frequency (episodes)  
        ``n_eval_episodes``     – episodes per evaluation run  
        ``use_tensorboard``     – write TensorBoard events  
        ``print_every``         – console print frequency  
    """

    def __init__(
        self,
        env: gym.Env,
        agent: BaseAgent,
        config: dict[str, Any],
    ) -> None:
        self.env = env
        self.agent = agent
        self.config = config

        run_dir = Path(config.get("run_dir", "runs/default"))
        run_dir.mkdir(parents=True, exist_ok=True)

        self.logger = MetricsLogger(
            log_dir=run_dir,
            use_tensorboard=bool(config.get("use_tensorboard", False)),
            print_every=int(config.get("print_every", 10)),
        )
        self.checkpointer = Checkpointer(
            checkpoint_dir=run_dir / "checkpoints",
            save_every=int(config.get("save_every", 50)),
        )

        self._episode_returns: list[float] = []
        self._episode_lengths: list[float] = []
        self._losses: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns
        -------
        dict with keys ``episode_returns``, ``episode_lengths``, ``losses``.
        """
        n_episodes = int(self.config.get("n_episodes", 500))
        max_steps = int(self.config.get("max_steps_per_episode", 500))
        eval_every = int(self.config.get("eval_every", 50))
        n_eval_eps = int(self.config.get("n_eval_episodes", 5))

        for ep in range(1, n_episodes + 1):
            ep_return, ep_len, metrics = self._run_episode(max_steps)
            self._episode_returns.append(ep_return)
            self._episode_lengths.append(float(ep_len))
            if "loss" in metrics:
                self._losses.append(metrics["loss"])

            self.logger.log(
                episode=ep,
                episode_return=round(ep_return, 4),
                episode_length=ep_len,
                total_steps=self.agent.total_steps,
                **{k: round(v, 6) for k, v in metrics.items()},
            )

            self.checkpointer.maybe_save(
                state={
                    "agent": self.agent.state_dict(),
                    "episode": ep,
                    "config": self.config,
                },
                step=ep,
            )

            if eval_every > 0 and ep % eval_every == 0:
                eval_return = self._evaluate(n_eval_eps, max_steps)
                self.logger.log(
                    episode=ep,
                    eval_return=round(eval_return, 4),
                )

        self.logger.close()
        self._save_plots()
        return {
            "episode_returns": self._episode_returns,
            "episode_lengths": self._episode_lengths,
            "losses": self._losses,
        }

    def resume(self) -> None:
        """Resume from the latest checkpoint if one exists."""
        if self.checkpointer.latest_exists():
            state = self.checkpointer.load("latest")
            self.agent.load_state_dict(state["agent"])
            ep = state.get("episode", 0)
            print(f"[Trainer] Resumed from episode {ep}.")
        else:
            print("[Trainer] No checkpoint found; starting from scratch.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_episode(
        self, max_steps: int
    ) -> tuple[float, int, dict[str, float]]:
        """Run a single episode and return (total_return, length, metrics)."""
        obs, _ = self.env.reset()
        ep_return = 0.0
        last_metrics: dict[str, float] = {}

        if isinstance(self.agent, DQNAgent):
            for step in range(max_steps):
                action = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.agent.observe(obs, action, float(reward), next_obs, done)
                last_metrics = self.agent.update()
                ep_return += float(reward)
                obs = next_obs
                if done:
                    return ep_return, step + 1, last_metrics
            return ep_return, max_steps, last_metrics

        elif isinstance(self.agent, REINFORCEAgent):
            for step in range(max_steps):
                action = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.agent.store_reward(float(reward), done)
                ep_return += float(reward)
                obs = next_obs
                if done:
                    break
            last_metrics = self.agent.update()
            return ep_return, step + 1, last_metrics

        elif isinstance(self.agent, PPOAgent):
            last_obs = obs
            for step in range(max_steps):
                action = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.agent.store_reward(float(reward), done)
                ep_return += float(reward)
                last_obs = next_obs
                obs = next_obs
                rollout_full = (
                    len(self.agent.rollout) >= self.agent.rollout_steps
                )
                if done or rollout_full:
                    last_metrics = self.agent.update(
                        last_obs=None if done else last_obs
                    )
                    if done:
                        return ep_return, step + 1, last_metrics
                    obs = next_obs
            if len(self.agent.rollout) > 0:
                last_metrics = self.agent.update(last_obs=last_obs)
            return ep_return, max_steps, last_metrics

        else:
            raise TypeError(f"Unknown agent type: {type(self.agent)}")

    def _evaluate(self, n_episodes: int, max_steps: int) -> float:
        """Run deterministic evaluation episodes and return mean return."""
        from src.training.evaluator import Evaluator
        evaluator = Evaluator(self.env, self.agent)
        results = evaluator.evaluate(n_episodes=n_episodes, max_steps=max_steps)
        return float(results["mean_return"])

    def _save_plots(self) -> None:
        run_dir = Path(self.config.get("run_dir", "runs/default"))
        metrics: dict[str, list[float]] = {
            "Episode Return": self._episode_returns,
        }
        if self._episode_lengths:
            metrics["Episode Length"] = self._episode_lengths
        if self._losses:
            metrics["Loss"] = self._losses
        plot_training_curves(
            metrics=metrics,
            title=f"Training — {self.config.get('agent_type', 'agent')}",
            save_path=run_dir / "training_curves.png",
        )
