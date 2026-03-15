"""Main training loop for World-Model-Arc agents.

The :class:`Trainer` orchestrates:
- Episode rollouts
- Agent updates (on-policy end-of-episode, or off-policy per-step)
- Periodic evaluation
- Metrics logging (CSV + TensorBoard)
- Checkpoint saving

Usage::

    from world_model_arc.training import Trainer
    from world_model_arc.envs import GridWorld
    from world_model_arc.agents import DQNAgent

    env = GridWorld(8, 8)
    agent = DQNAgent(env.observation_shape, env.action_size)
    trainer = Trainer(env, agent)
    trainer.train(n_episodes=500)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from world_model_arc.agents.base import BaseAgent
from world_model_arc.envs.base import BaseEnv
from world_model_arc.training.evaluator import Evaluator
from world_model_arc.utils.checkpointing import CheckpointManager
from world_model_arc.utils.logger import MetricsLogger


class Trainer:
    """Trains an agent in an environment.

    Args:
        env: Training environment.
        agent: The RL agent to train.
        eval_env: Optional separate environment for evaluation.
            If *None*, *env* is used for both train and eval.
        log_dir: Root directory for logs and checkpoints.
        experiment_name: Sub-directory name for this run.
        eval_interval: Evaluate every N episodes.
        eval_episodes: Number of evaluation episodes per evaluation.
        log_interval: Print metrics every N episodes.
        checkpoint_interval: Save a checkpoint every N episodes.
        max_to_keep: Number of recent checkpoints to retain.
        use_tensorboard: Whether to log to TensorBoard.
        seed: RNG seed for the training environment resets.
    """

    def __init__(
        self,
        env: BaseEnv,
        agent: BaseAgent,
        eval_env: Optional[BaseEnv] = None,
        log_dir: str | Path = "runs",
        experiment_name: str = "experiment",
        eval_interval: int = 50,
        eval_episodes: int = 10,
        log_interval: int = 10,
        checkpoint_interval: int = 100,
        max_to_keep: int = 5,
        use_tensorboard: bool = True,
        seed: int = 42,
    ) -> None:
        self.env = env
        self.agent = agent
        self.eval_env = eval_env or env
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.seed = seed

        log_dir = Path(log_dir)
        self.logger = MetricsLogger(
            log_dir=log_dir,
            experiment_name=experiment_name,
            use_tensorboard=use_tensorboard,
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=log_dir / experiment_name / "checkpoints",
            max_to_keep=max_to_keep,
        )
        self.evaluator = Evaluator(self.eval_env, self.agent)

        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []

    # ------------------------------------------------------------------

    def train(self, n_episodes: int) -> None:
        """Run the training loop for *n_episodes* episodes.

        Args:
            n_episodes: Total number of episodes to train for.
        """
        print(f"[Trainer] Starting training for {n_episodes} episodes.")
        t0 = time.time()

        for episode in range(1, n_episodes + 1):
            ep_reward, ep_length, train_metrics = self._run_episode(episode)

            self._episode_rewards.append(ep_reward)
            self._episode_lengths.append(ep_length)

            # Logging
            if episode % self.log_interval == 0:
                avg_rew = float(np.mean(self._episode_rewards[-self.log_interval :]))
                avg_len = float(np.mean(self._episode_lengths[-self.log_interval :]))
                log_row = {
                    "episode_reward": ep_reward,
                    "avg_reward": avg_rew,
                    "episode_length": ep_length,
                    "avg_length": avg_len,
                    **train_metrics,
                }
                self.logger.log(step=episode, **log_row)
                elapsed = time.time() - t0
                print(
                    f"  Ep {episode:5d}/{n_episodes} | "
                    f"Reward {avg_rew:7.3f} | "
                    f"Len {avg_len:6.1f} | "
                    f"Elapsed {elapsed:6.1f}s"
                )

            # Evaluation
            if episode % self.eval_interval == 0:
                eval_stats = self.evaluator.evaluate(n_episodes=self.eval_episodes)
                self.logger.log(
                    step=episode,
                    eval_mean_reward=eval_stats["mean_reward"],
                    eval_success_rate=eval_stats["success_rate"],
                )
                print(
                    f"  [EVAL] ep {episode} | "
                    f"mean_reward={eval_stats['mean_reward']:.3f} | "
                    f"success_rate={eval_stats['success_rate']:.2f}"
                )

                # Checkpoint best model
                self.checkpoint_manager.save(
                    state={
                        "episode": episode,
                        "agent": self.agent.state_dict(),
                    },
                    step=episode,
                    metric=eval_stats["mean_reward"],
                )

            elif episode % self.checkpoint_interval == 0:
                self.checkpoint_manager.save(
                    state={"episode": episode, "agent": self.agent.state_dict()},
                    step=episode,
                )

        self.logger.close()
        print(f"[Trainer] Done. Total time: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------

    def _run_episode(self, episode: int) -> tuple[float, int, dict[str, float]]:
        """Execute a single episode and return (total_reward, length, train_metrics)."""
        obs = self.env.reset(seed=self.seed + episode)
        total_reward = 0.0
        step = 0
        last_train_metrics: dict[str, float] = {}

        while True:
            action = self.agent.select_action(obs, deterministic=False)
            result = self.env.step(action)

            self.agent.observe(obs, action, result.reward, result.observation, result.done)

            # Off-policy update after every step
            metrics = self.agent.update()
            if metrics:
                last_train_metrics = metrics

            total_reward += result.reward
            step += 1
            obs = result.observation

            if result.done or result.truncated:
                break

        # On-policy end-of-episode update
        ep_metrics = self.agent.end_episode()
        if ep_metrics:
            last_train_metrics = ep_metrics

        return total_reward, step, last_train_metrics
