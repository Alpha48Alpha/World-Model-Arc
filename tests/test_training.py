"""Integration tests for training loop, evaluator, checkpointing, and logging."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.envs.grid_world import GridWorld
from src.agents.dqn_agent import DQNAgent
from src.agents.reinforce_agent import REINFORCEAgent
from src.agents.ppo_agent import PPOAgent
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.logging import MetricsLogger
from src.utils.checkpointing import Checkpointer


def _small_config(run_dir, agent_type="DQN"):
    return {
        "agent_type": agent_type,
        "n_episodes": 5,
        "max_steps_per_episode": 20,
        "eval_every": 5,
        "n_eval_episodes": 2,
        "run_dir": str(run_dir),
        "save_every": 3,
        "use_tensorboard": False,
        "print_every": 5,
        "device": "cpu",
    }


def _small_env():
    return GridWorld(height=4, width=4, n_goals=1, n_hazards=0, n_walls=0, max_steps=20)


def _small_dqn():
    env = _small_env()
    return env, DQNAgent(
        obs_dim=16, action_dim=4,
        hidden_sizes=(16,), replay_capacity=50, batch_size=8,
        epsilon_decay_steps=20,
    )


def _small_reinforce():
    env = _small_env()
    return env, REINFORCEAgent(
        obs_dim=16, action_dim=4, hidden_sizes=(16,), max_episode_steps=30
    )


def _small_ppo():
    env = _small_env()
    return env, PPOAgent(
        obs_dim=16, action_dim=4, hidden_sizes=(16,), rollout_steps=16,
        batch_size=8, n_epochs=2
    )


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

class TestTrainer:
    def test_dqn_training_runs(self, tmp_path):
        env, agent = _small_dqn()
        config = _small_config(tmp_path, "DQN")
        trainer = Trainer(env, agent, config)
        results = trainer.train()
        assert "episode_returns" in results
        assert len(results["episode_returns"]) == 5

    def test_reinforce_training_runs(self, tmp_path):
        env, agent = _small_reinforce()
        config = _small_config(tmp_path, "REINFORCE")
        trainer = Trainer(env, agent, config)
        results = trainer.train()
        assert len(results["episode_returns"]) == 5

    def test_ppo_training_runs(self, tmp_path):
        env, agent = _small_ppo()
        config = _small_config(tmp_path, "PPO")
        trainer = Trainer(env, agent, config)
        results = trainer.train()
        assert len(results["episode_returns"]) == 5

    def test_training_saves_metrics_csv(self, tmp_path):
        env, agent = _small_dqn()
        config = _small_config(tmp_path, "DQN")
        trainer = Trainer(env, agent, config)
        trainer.train()
        assert (tmp_path / "metrics.csv").exists()

    def test_training_saves_checkpoint(self, tmp_path):
        env, agent = _small_dqn()
        config = _small_config(tmp_path, "DQN")
        trainer = Trainer(env, agent, config)
        trainer.train()
        assert (tmp_path / "checkpoints" / "latest.pt").exists()

    def test_training_saves_plot(self, tmp_path):
        env, agent = _small_dqn()
        config = _small_config(tmp_path, "DQN")
        trainer = Trainer(env, agent, config)
        trainer.train()
        assert (tmp_path / "training_curves.png").exists()

    def test_resume_from_checkpoint(self, tmp_path):
        env, agent = _small_dqn()
        config = _small_config(tmp_path, "DQN")
        trainer = Trainer(env, agent, config)
        trainer.train()
        # Now resume
        env2, agent2 = _small_dqn()
        config2 = _small_config(tmp_path, "DQN")
        trainer2 = Trainer(env2, agent2, config2)
        trainer2.resume()  # Should not raise


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_evaluate_returns_correct_keys(self):
        env, agent = _small_dqn()
        evaluator = Evaluator(env, agent)
        results = evaluator.evaluate(n_episodes=3, max_steps=20)
        for key in ("mean_return", "std_return", "mean_length", "min_return",
                    "max_return", "episode_returns"):
            assert key in results

    def test_evaluate_n_episodes(self):
        env, agent = _small_dqn()
        evaluator = Evaluator(env, agent)
        results = evaluator.evaluate(n_episodes=4, max_steps=20)
        assert len(results["episode_returns"]) == 4

    def test_evaluate_returns_floats(self):
        env, agent = _small_ppo()
        evaluator = Evaluator(env, agent)
        results = evaluator.evaluate(n_episodes=2, max_steps=10)
        assert isinstance(results["mean_return"], float)
        assert isinstance(results["std_return"], float)


# ---------------------------------------------------------------------------
# MetricsLogger tests
# ---------------------------------------------------------------------------

class TestMetricsLogger:
    def test_creates_csv(self, tmp_path):
        logger = MetricsLogger(log_dir=tmp_path, print_every=0)
        logger.log(episode=1, episode_return=-2.5, loss=0.1)
        logger.close()
        assert (tmp_path / "metrics.csv").exists()

    def test_csv_contains_data(self, tmp_path):
        logger = MetricsLogger(log_dir=tmp_path, print_every=0)
        logger.log(episode=1, episode_return=-1.0)
        logger.log(episode=2, episode_return=-0.5)
        logger.close()
        rows = logger.read_all()
        assert len(rows) == 2
        assert rows[0]["episode"] == "1"

    def test_context_manager(self, tmp_path):
        with MetricsLogger(log_dir=tmp_path, print_every=0) as logger:
            logger.log(episode=1, val=42.0)
        assert (tmp_path / "metrics.csv").exists()


# ---------------------------------------------------------------------------
# Checkpointer tests
# ---------------------------------------------------------------------------

class TestCheckpointer:
    def test_save_and_load(self, tmp_path):
        ckpt = Checkpointer(checkpoint_dir=tmp_path)
        state = {"weights": np.array([1.0, 2.0, 3.0]), "step": 10}
        ckpt.save(state, step=10)
        loaded = ckpt.load("latest")
        np.testing.assert_array_equal(loaded["weights"], state["weights"])

    def test_maybe_save_creates_latest(self, tmp_path):
        ckpt = Checkpointer(checkpoint_dir=tmp_path, save_every=10)
        state = {"x": 1}
        ckpt.maybe_save(state, step=1)
        assert (tmp_path / "latest.pt").exists()

    def test_named_snapshot_created_at_interval(self, tmp_path):
        ckpt = Checkpointer(checkpoint_dir=tmp_path, save_every=2, keep_last=5)
        state = {"x": 1}
        result = None
        for i in range(1, 5):
            result = ckpt.maybe_save(state, step=i)
        # After 4 calls with save_every=2, we should have a snapshot at step 4
        assert result is not None

    def test_latest_exists_false_initially(self, tmp_path):
        ckpt = Checkpointer(checkpoint_dir=tmp_path)
        assert not ckpt.latest_exists()

    def test_eviction_of_old_snapshots(self, tmp_path):
        ckpt = Checkpointer(checkpoint_dir=tmp_path, save_every=1, keep_last=2)
        for i in range(1, 6):
            ckpt.maybe_save({"x": i}, step=i)
        # Only 2 snapshots should remain besides latest.pt
        snapshots = list(tmp_path.glob("step_*.pt"))
        assert len(snapshots) <= 2

    def test_load_nonexistent_raises(self, tmp_path):
        ckpt = Checkpointer(checkpoint_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            ckpt.load("nonexistent")
