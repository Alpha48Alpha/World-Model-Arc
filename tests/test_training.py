"""Tests for training infrastructure (Trainer, Evaluator, replay buffer, etc.)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from world_model_arc.agents import DQNAgent, PolicyGradientAgent
from world_model_arc.envs import GridWorld
from world_model_arc.training import Evaluator, Trainer
from world_model_arc.utils import ReplayBuffer


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(capacity=100, obs_shape=(4,))
        assert len(buf) == 0
        buf.add(np.zeros(4, dtype=np.float32), 0, 1.0, np.ones(4, dtype=np.float32), False)
        assert len(buf) == 1

    def test_sample_shape(self):
        buf = ReplayBuffer(capacity=100, obs_shape=(4,))
        for _ in range(50):
            buf.add(np.random.randn(4).astype(np.float32), 0, 0.5, np.random.randn(4).astype(np.float32), False)
        batch = buf.sample(16)
        assert batch.obs.shape == (16, 4)
        assert batch.action.shape == (16,)
        assert batch.reward.shape == (16,)
        assert batch.next_obs.shape == (16, 4)
        assert batch.done.shape == (16,)

    def test_sample_raises_when_not_enough(self):
        buf = ReplayBuffer(capacity=100, obs_shape=(4,))
        buf.add(np.zeros(4, dtype=np.float32), 0, 0.0, np.zeros(4, dtype=np.float32), False)
        with pytest.raises(ValueError):
            buf.sample(10)

    def test_circular_overflow(self):
        buf = ReplayBuffer(capacity=10, obs_shape=(2,))
        for i in range(20):
            buf.add(np.array([i, i], dtype=np.float32), 0, float(i), np.zeros(2, dtype=np.float32), False)
        assert len(buf) == 10  # capacity enforced

    def test_save_load(self, tmp_path):
        buf = ReplayBuffer(capacity=100, obs_shape=(4,))
        for _ in range(30):
            buf.add(
                np.random.randn(4).astype(np.float32),
                np.random.randint(0, 4),
                float(np.random.randn()),
                np.random.randn(4).astype(np.float32),
                False,
            )
        save_path = tmp_path / "buf"
        buf.save(save_path)

        buf2 = ReplayBuffer(capacity=100, obs_shape=(4,))
        buf2.load(str(save_path) + ".npz")
        assert len(buf2) == 30


# ---------------------------------------------------------------------------
# Trainer – smoke tests
# ---------------------------------------------------------------------------


class TestTrainer:
    def _make_env_and_agent(self):
        env = GridWorld(height=4, width=4, max_steps=20)
        agent = DQNAgent(
            obs_shape=env.observation_shape,
            n_actions=env.action_size,
            buffer_capacity=100,
            batch_size=8,
            epsilon_decay_steps=50,
        )
        return env, agent

    def test_trainer_runs_without_error(self, tmp_path):
        env, agent = self._make_env_and_agent()
        trainer = Trainer(
            env=env,
            agent=agent,
            log_dir=str(tmp_path),
            experiment_name="test_run",
            eval_interval=5,
            eval_episodes=2,
            log_interval=2,
            checkpoint_interval=5,
            use_tensorboard=False,
            seed=0,
        )
        trainer.train(n_episodes=10)

    def test_trainer_pg_agent(self, tmp_path):
        env = GridWorld(height=4, width=4, max_steps=20)
        agent = PolicyGradientAgent(
            obs_shape=env.observation_shape,
            n_actions=env.action_size,
        )
        trainer = Trainer(
            env=env,
            agent=agent,
            log_dir=str(tmp_path),
            experiment_name="pg_test",
            eval_interval=5,
            eval_episodes=2,
            log_interval=2,
            checkpoint_interval=5,
            use_tensorboard=False,
            seed=1,
        )
        trainer.train(n_episodes=10)

    def test_metrics_csv_created(self, tmp_path):
        env, agent = self._make_env_and_agent()
        trainer = Trainer(
            env=env,
            agent=agent,
            log_dir=str(tmp_path),
            experiment_name="csv_test",
            eval_interval=5,
            eval_episodes=2,
            log_interval=2,
            checkpoint_interval=10,
            use_tensorboard=False,
            seed=2,
        )
        trainer.train(n_episodes=10)
        csv_path = tmp_path / "csv_test" / "metrics.csv"
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class TestEvaluator:
    def test_evaluate_returns_stats(self):
        env = GridWorld(height=4, width=4, max_steps=20)
        agent = DQNAgent(obs_shape=env.observation_shape, n_actions=env.action_size)
        evaluator = Evaluator(env, agent)
        stats = evaluator.evaluate(n_episodes=3)
        assert "mean_reward" in stats
        assert "success_rate" in stats
        assert 0.0 <= stats["success_rate"] <= 1.0

    def test_run_episode_returns_trajectory(self):
        env = GridWorld(height=4, width=4, max_steps=20)
        agent = DQNAgent(obs_shape=env.observation_shape, n_actions=env.action_size)
        evaluator = Evaluator(env, agent)
        traj = evaluator.run_episode(seed=0)
        assert "reward" in traj
        assert "length" in traj
        assert "frames" in traj
        assert traj["length"] > 0
