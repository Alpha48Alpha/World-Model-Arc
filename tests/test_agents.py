"""Tests for RL agents: DQN, REINFORCE, PPO."""

import pytest
import numpy as np
import torch

from src.agents.dqn_agent import DQNAgent
from src.agents.reinforce_agent import REINFORCEAgent
from src.agents.ppo_agent import PPOAgent
from src.envs.grid_world import GridWorld


OBS_DIM = 16
ACT_DIM = 4


def make_obs(n=1):
    return np.random.randn(OBS_DIM).astype(np.float32)


# ---------------------------------------------------------------------------
# DQNAgent
# ---------------------------------------------------------------------------

class TestDQNAgent:
    def _agent(self, **kwargs):
        defaults = dict(
            obs_dim=OBS_DIM,
            action_dim=ACT_DIM,
            hidden_sizes=(32, 32),
            replay_capacity=200,
            batch_size=16,
            epsilon_decay_steps=100,
        )
        defaults.update(kwargs)
        return DQNAgent(**defaults)

    def test_select_action_range(self):
        agent = self._agent()
        for _ in range(20):
            action = agent.select_action(make_obs())
            assert 0 <= action < ACT_DIM

    def test_deterministic_action_stable(self):
        agent = self._agent()
        agent.epsilon = 0.0
        obs = make_obs()
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent.select_action(obs, deterministic=True)
        assert a1 == a2

    def test_update_before_buffer_ready(self):
        agent = self._agent()
        metrics = agent.update()
        assert metrics["loss"] == 0.0

    def test_update_after_fill(self):
        agent = self._agent()
        obs = make_obs()
        next_obs = make_obs()
        for i in range(20):
            agent.observe(obs, i % ACT_DIM, float(i), next_obs, i == 19)
        metrics = agent.update()
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_epsilon_annealing(self):
        agent = self._agent(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10)
        eps_start = agent.epsilon
        obs = make_obs()
        next_obs = make_obs()
        for i in range(20):
            agent.observe(obs, 0, 0.0, next_obs, False)
        for _ in range(5):
            agent.update()
        assert agent.epsilon <= eps_start

    def test_state_dict_roundtrip(self):
        agent = self._agent()
        obs = make_obs()
        next_obs = make_obs()
        for i in range(20):
            agent.observe(obs, i % ACT_DIM, 0.1, next_obs, False)
        agent.update()
        sd = agent.state_dict()
        agent2 = self._agent()
        agent2.load_state_dict(sd)
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent2.select_action(obs, deterministic=True)
        assert a1 == a2

    def test_target_net_sync(self):
        agent = self._agent(target_update_freq=2, batch_size=4, replay_capacity=50)
        obs = make_obs()
        next_obs = make_obs()
        for i in range(10):
            agent.observe(obs, i % ACT_DIM, 0.1, next_obs, False)
        # Run exactly target_update_freq updates so the last update triggers a sync
        for _ in range(2):
            agent.update()
        # After sync, target weights should equal online weights
        import torch
        for p_online, p_target in zip(
            agent.q_net.parameters(), agent.target_net.parameters()
        ):
            assert torch.allclose(p_online, p_target)


# ---------------------------------------------------------------------------
# REINFORCEAgent
# ---------------------------------------------------------------------------

class TestREINFORCEAgent:
    def _agent(self, **kwargs):
        defaults = dict(
            obs_dim=OBS_DIM,
            action_dim=ACT_DIM,
            hidden_sizes=(32, 32),
            max_episode_steps=50,
        )
        defaults.update(kwargs)
        return REINFORCEAgent(**defaults)

    def test_select_action_range(self):
        agent = self._agent()
        for _ in range(10):
            action = agent.select_action(make_obs())
            assert 0 <= action < ACT_DIM

    def test_update_empty_rollout(self):
        agent = self._agent()
        metrics = agent.update()
        assert metrics["policy_loss"] == 0.0

    def test_update_after_episode(self):
        agent = self._agent()
        obs = make_obs()
        for i in range(10):
            action = agent.select_action(obs)
            agent.store_reward(float(i), done=(i == 9))
        metrics = agent.update()
        assert "policy_loss" in metrics
        assert "entropy" in metrics

    def test_rollout_reset_after_update(self):
        agent = self._agent()
        obs = make_obs()
        for i in range(5):
            agent.select_action(obs)
            agent.store_reward(1.0, done=(i == 4))
        agent.update()
        assert len(agent.rollout) == 0

    def test_state_dict_roundtrip(self):
        agent = self._agent()
        obs = make_obs()
        for i in range(5):
            agent.select_action(obs)
            agent.store_reward(1.0, done=(i == 4))
        agent.update()
        sd = agent.state_dict()
        agent2 = self._agent()
        agent2.load_state_dict(sd)
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent2.select_action(obs, deterministic=True)
        assert a1 == a2

    def test_no_baseline_mode(self):
        agent = REINFORCEAgent(
            obs_dim=OBS_DIM, action_dim=ACT_DIM, use_baseline=False,
            hidden_sizes=(32,), max_episode_steps=20
        )
        obs = make_obs()
        for i in range(5):
            agent.select_action(obs)
            agent.store_reward(1.0, done=(i == 4))
        metrics = agent.update()
        assert "policy_loss" in metrics


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

class TestPPOAgent:
    def _agent(self, **kwargs):
        defaults = dict(
            obs_dim=OBS_DIM,
            action_dim=ACT_DIM,
            hidden_sizes=(32, 32),
            rollout_steps=32,
            batch_size=16,
            n_epochs=2,
        )
        defaults.update(kwargs)
        return PPOAgent(**defaults)

    def test_select_action_range(self):
        agent = self._agent()
        for _ in range(10):
            action = agent.select_action(make_obs())
            assert 0 <= action < ACT_DIM

    def test_update_empty(self):
        agent = self._agent()
        metrics = agent.update()
        assert metrics["policy_loss"] == 0.0

    def test_update_after_rollout(self):
        agent = self._agent()
        obs = make_obs()
        for i in range(32):
            agent.select_action(obs)
            agent.store_reward(1.0, done=(i == 31))
        metrics = agent.update()
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics

    def test_rollout_reset_after_update(self):
        agent = self._agent()
        obs = make_obs()
        for i in range(5):
            agent.select_action(obs)
            agent.store_reward(1.0, done=(i == 4))
        agent.update()
        assert len(agent.rollout) == 0

    def test_state_dict_roundtrip(self):
        agent = self._agent()
        obs = make_obs()
        for i in range(5):
            agent.select_action(obs)
            agent.store_reward(1.0, done=(i == 4))
        agent.update()
        sd = agent.state_dict()
        agent2 = self._agent()
        agent2.load_state_dict(sd)
        # Deterministic actions should match (no rollout side-effects in deterministic mode)
        agent.ac.eval()
        agent2.ac.eval()
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent2.select_action(obs, deterministic=True)
        assert a1 == a2

    def test_deterministic_action_stable(self):
        agent = self._agent()
        obs = make_obs()
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent.select_action(obs, deterministic=True)
        assert a1 == a2
